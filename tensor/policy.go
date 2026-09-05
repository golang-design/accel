// Copyright 2026 The golang.design Initiative Authors.
// All rights reserved. Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

package tensor

import (
	"fmt"
	"math"

	"golang.design/x/accel"
	"golang.design/x/accel/internal/kernels"
)

// MinTemperature is the smallest positive temperature [SamplingOptions.Validate]
// accepts.
//
// A guardrail and only a guardrail. It is not the point at which the arithmetic
// stops working -- that cliff is far below it and is refused separately -- it is
// the point below which a caller almost certainly meant to ask for greedy
// decoding and should be told to say so.
const MinTemperature = 1e-3

// SamplingOptions is one sequence's policy.
//
// The zero value is greedy: argmax, no penalties, no truncation. That is the
// policy a caller gets by passing nothing, and it is the one with no way to
// surprise them.
//
// # There is no generator in here
//
// specs/039-sampling-policy.md section 2. Copying this struct copies numbers.
// The draw comes from a [Stream] the caller holds per sequence and reaches the
// graph as a tensor, because it is a per-row value and
// specs/043-per-row-values.md makes those tensors rather than scalars.
type SamplingOptions struct {
	// Temperature of 0 is greedy, and is a different graph rather than a small
	// number: see [SamplingOptions.Validate].
	Temperature float32

	// TopK and TopP are off at 0. "Off" removes the node from the graph, which
	// is the only way to turn truncation off -- a TopK equal to the vocabulary
	// is not off, because the mask's round count is bounded and it would
	// silently keep [TopMaxRounds].
	TopK int
	TopP float32

	// Repetition divides a positive logit and multiplies a non-positive one.
	// Both 0 and 1 are off.
	Repetition float32

	// Presence is subtracted once from any token that occurred, Frequency once
	// per occurrence. 0 is off for both.
	Presence  float32
	Frequency float32
}

// Penalised reports whether any penalty is configured.
//
// Exported because it is the question a caller has to answer before deciding
// whether to declare the history and counts states at all, and re-deriving it
// from three fields at the call site is how the two answers drift apart.
func (o SamplingOptions) Penalised() bool {
	return (o.Repetition != 0 && o.Repetition != 1) || o.Presence != 0 || o.Frequency != 0
}

// Greedy reports whether this policy takes the argmax rather than a draw.
func (o SamplingOptions) Greedy() bool { return o.Temperature == 0 }

// Validate refuses a policy rather than clamping it.
//
// This is the first layer that can report an error at all: a kernel cannot,
// which is why [TopMaxRounds] truncates down there and the draw clamps. So the
// refusals belong here, and they refuse rather than repair, because a policy
// quietly changed into a different one produces plausible tokens and no
// evidence.
func (o SamplingOptions) Validate() error {
	t := o.Temperature
	switch {
	case math.IsNaN(float64(t)):
		return fmt.Errorf("accel: sampling: Temperature is NaN")
	case t < 0:
		return fmt.Errorf("accel: sampling: Temperature is %v; it is a divisor, and the "+
			"way to ask for greedy decoding is 0", t)
	case t > 0 && t < MinTemperature:
		// A guardrail, and stated as one. It is not a numerical guarantee: with
		// the max subtracted, the runner-up's weight is exp(-gap/T), which
		// underflows f32 only past gap/T = 87, so at T = 1e-3 a gap of 0.02
		// still leaves a weight near 2e-9 and the walk can still take it.
		return fmt.Errorf("accel: sampling: Temperature is %v, below %v; any positive "+
			"temperature is a stochastic policy that can disagree with argmax at close "+
			"logits, so a very small one is almost always a Temperature of 0 written "+
			"the wrong way", t, float32(MinTemperature))
	}
	// The numerical refusal, which is a different rule from the guardrail and
	// is checked separately so neither is mistaken for the other. Below roughly
	// 2.9e-39 the reciprocal is +Inf, the scaled logits are +/-Inf, the softmax
	// is NaN across the whole vocabulary, every comparison against the draw is
	// false, and the walk returns the last index: a plausible token id from a
	// completely broken step.
	if t > 0 {
		if inv := float32(1) / t; math.IsInf(float64(inv), 0) || math.IsNaN(float64(inv)) {
			return fmt.Errorf("accel: sampling: Temperature is %v and its reciprocal is "+
				"%v; the scaled logits would be infinite and the step would return a "+
				"plausible token from an all-NaN distribution", t, inv)
		}
	}

	if o.TopK < 0 {
		return fmt.Errorf("accel: sampling: TopK is %d", o.TopK)
	}
	if o.TopK > TopMaxRounds {
		return fmt.Errorf("accel: sampling: TopK is %d and the bound is %d; the mask "+
			"walks one entry per round, so a larger k would keep %d and report nothing",
			o.TopK, TopMaxRounds, TopMaxRounds)
	}
	if p := o.TopP; math.IsNaN(float64(p)) || p < 0 || p > 1 {
		return fmt.Errorf("accel: sampling: TopP is %v; it is a fraction of the "+
			"distribution's mass, and 0 is off", p)
	}
	for _, c := range []struct {
		name string
		v    float32
	}{{"Repetition", o.Repetition}, {"Presence", o.Presence}, {"Frequency", o.Frequency}} {
		if math.IsNaN(float64(c.v)) {
			return fmt.Errorf("accel: sampling: %s is NaN", c.name)
		}
	}
	if o.Repetition < 0 {
		return fmt.Errorf("accel: sampling: Repetition is %v; it multiplies a "+
			"non-positive logit, so a negative one flips the sign of every repeated "+
			"token's score", o.Repetition)
	}
	return nil
}

// Scalars is what one decode step binds.
//
// Every value here is a number a step changes without changing the plan:
// specs/039-sampling-policy.md section 7 prices them at nothing, because Submit
// rewrites every uniform on every submission anyway. What is *structural* --
// which nodes exist, and the k and p the masks were recorded with -- is not
// here, and changing one of those is a different plan.
//
// n is how much of the history ring is filled: min(tokens so far, capacity).
// It is refused above the capacity rather than clamped, because a clamp turns a
// caller's off-by-one into a penalty over a window they did not ask for.
func (o SamplingOptions) Scalars(prefix string, n, historyCap uint32) (map[string]ScalarValue, error) {
	if err := o.Validate(); err != nil {
		return nil, err
	}
	out := map[string]ScalarValue{}
	if !o.Greedy() {
		out[prefix+".invT"] = F32(1 / o.Temperature)
	}
	if o.Penalised() {
		if historyCap == 0 {
			return nil, fmt.Errorf("accel: sampling: a penalty is configured and the " +
				"history capacity is 0")
		}
		if n > historyCap {
			return nil, fmt.Errorf("accel: sampling: the history holds %d of a capacity "+
				"of %d; the window is the last %d tokens, and a count past the end would "+
				"read whatever the ring held before it", n, historyCap, historyCap)
		}
		out[prefix+".n"] = U32(n)
		out[prefix+".rep"] = F32(o.Repetition)
		out[prefix+".pres"] = F32(o.Presence)
		out[prefix+".freq"] = F32(o.Frequency)
	}
	return out, nil
}

// DeclareSamplingScalars declares the scalars [Sample] reads for this policy.
//
// It exists so that the names [SamplingOptions.Scalars] produces and the names
// the graph reads come from one place. Declaring them by hand at the call site
// is how a step ends up binding a value nobody reads, which is silent.
func DeclareSamplingScalars(b *Builder, o SamplingOptions, prefix string) {
	if !o.Greedy() {
		Scalar(b, ScalarDesc{Name: prefix + ".invT", Kind: ScalarF32})
	}
	if o.Penalised() {
		Scalar(b, ScalarDesc{Name: prefix + ".n", Kind: ScalarU32})
		Scalar(b, ScalarDesc{Name: prefix + ".rep", Kind: ScalarF32})
		Scalar(b, ScalarDesc{Name: prefix + ".pres", Kind: ScalarF32})
		Scalar(b, ScalarDesc{Name: prefix + ".freq", Kind: ScalarF32})
	}
}

// Sample records one sequence's sampling policy and returns the chosen token.
//
// # The order, and why not another
//
//	logits ─▶ penalties ─▶ ×1/T ─▶ softmax ─▶ top-k ─▶ top-p ─▶ draw ─▶ token
//	           (optional)                     (optional) (optional)
//	       └─▶ argmax ─▶ token   when Temperature is 0
//
// Penalties act before temperature because subtracting a after scaling by 1/T
// is subtracting a·T before it: two knobs a caller turns independently must not
// multiply. Truncation acts after the softmax because top-p is a mass
// threshold, and top-k joins it there so both masks act on the values the
// sampler will actually walk -- f32 rounding can make two distinct logits equal
// probabilities, and a top-k over logits would then keep a different boundary
// entry than the walk sees. Top-k precedes top-p because top-p is relative to
// its own input's total, so each bound is one the other cannot violate.
//
// Nothing renormalizes and there is never a second softmax. A mask leaves the
// weights summing below one, which invites a fix; specs/028-sampling.md deleted
// that fix and made the walk compare against draw × total instead. A softmax
// over a mask's output is near-uniform over the whole vocabulary, because
// exp(0) = 1 for every dropped entry.
//
// # What the caller owns
//
// draws is one uniform per row, from a [Stream]. history and counts are
// caller-owned state, required exactly when a penalty is configured and refused
// when one is not, so a caller cannot bind storage nothing reads.
//
// counts is [vocab]u32 and is rebuilt from the history every step rather than
// carried: this returns the version after that rebuild, so a caller holding the
// old one is reading what was there before, which specs/007-tensor-layer.md
// makes meaningful rather than a mistake.
func Sample(b *Builder, logits, draws *Tensor, history, counts *State,
	o SamplingOptions, prefix string) *Tensor {

	if poisoned(logits) {
		return b.poison()
	}
	if err := o.Validate(); err != nil {
		return b.fail(1, "Sample", "%s", err)
	}
	if why := checkLogits(logits); why != "" {
		return b.fail(1, "Sample", "%s", why)
	}
	_, vocab, _ := sampleShape(logits)

	x := logits
	if o.Penalised() {
		if history == nil || counts == nil {
			return b.fail(1, "Sample", "a penalty is configured and %s is nil; the "+
				"counts are rebuilt from the history every step, so both are storage "+
				"this graph writes rather than reads",
				map[bool]string{true: "history", false: "counts"}[history == nil])
		}
		var ok bool
		if x, ok = b.penalise(logits, history, counts, vocab, prefix); !ok {
			return b.poison()
		}
	} else if history != nil || counts != nil {
		return b.fail(1, "Sample", "no penalty is configured and history or counts was "+
			"given; storage nothing reads is the defect this refusal exists to catch, "+
			"not a spare argument")
	}

	// T = 0 is a different graph, not a small number. With T = 1e-6 and a logit
	// gap smaller than T the softmax is not one-hot at all and the walk takes
	// the runner-up; and at an exact tie -- ordinary at saturation -- argmax
	// returns the lowest index while the walk returns wherever the draw lands.
	// Clamping T to an epsilon gives decoding that is greedy almost always and
	// silently emits the second-best token when the top two logits are close,
	// reproducibly under the seed, so it reads as a model quirk.
	if o.Greedy() {
		if draws != nil {
			return b.fail(1, "Sample", "Temperature is 0 and draws was given; greedy "+
				"decoding consumes no randomness, and a draw nobody reads is a caller "+
				"believing this step is stochastic")
		}
		return Argmax(b, x)
	}
	if draws == nil {
		return b.fail(1, "Sample", "Temperature is %v and draws is nil; every sequence "+
			"draws against its own uniform", o.Temperature)
	}

	x = Scale(b, x, prefix+".invT")
	x = Softmax(b, x, SoftmaxOptions{})
	if o.TopK > 0 {
		x = TopKMask(b, x, o.TopK)
	}
	if o.TopP > 0 {
		x = TopPMask(b, x, o.TopP)
	}
	return SampleCategorical(b, x, draws)
}

// penalise records the three penalty nodes and returns the penalised logits.
//
// Clear, then count, then apply. The counts are accumulated with an integer
// atomic, so the clear is what stops each step being penalised by every earlier
// one -- a sequence that skipped it still decodes, and decodes increasingly
// wrongly.
//
// The counting is integer for the reason specs/039-sampling-policy.md section 4
// gives: the history holds duplicate ids by construction, so subtracting from
// the logits directly has several invocations writing one address, and with
// AddF32 the result is numerics class E, which the conformance harness excludes
// from bit comparison. The bug would then be invisible to the differential and
// visible only as a different token on rerun, and only when a token repeats.
func (b *Builder) penalise(logits *Tensor, history, counts *State, vocab int,
	prefix string) (*Tensor, bool) {

	if counts.desc.DType != accel.U32 {
		b.fail(2, "Sample", "the counts state is %v and the penalty counts tokens, "+
			"which is u32", counts.desc.DType)
		return nil, false
	}
	if got := counts.shape.Elements(); got != vocab {
		b.fail(2, "Sample", "the counts state holds %d entries and the vocabulary is "+
			"%d; there is one count per token id", got, vocab)
		return nil, false
	}
	if history.desc.DType != accel.U32 {
		b.fail(2, "Sample", "the history state is %v and it holds token ids, which "+
			"are u32", history.desc.DType)
		return nil, false
	}
	historyCap := history.shape.Elements()
	if historyCap == 0 {
		b.fail(2, "Sample", "the history state is empty; its capacity is the window "+
			"the penalties act over")
		return nil, false
	}

	dims := func(vals map[string]ScalarValue) any {
		return kernels.PenaltyDims{
			Vocab: uint32(vocab), History: uint32(historyCap),
			Count:      vals[prefix+".n"].U32,
			Repetition: vals[prefix+".rep"].F32,
			Presence:   vals[prefix+".pres"].F32,
			Frequency:  vals[prefix+".freq"].F32,
		}
	}
	reads := []string{prefix + ".n", prefix + ".rep", prefix + ".pres", prefix + ".freq"}
	overVocab := func(k *accel.Kernel) grid {
		return func(*Tensor) accel.WorkgroupCount {
			wg := int(k.WorkgroupSize.X)
			return accel.WorkgroupCount{X: (vocab + wg - 1) / wg}
		}
	}

	// The clear takes no input at all, which is what it is: it writes a
	// constant into caller-owned storage and reads nothing. Giving it a
	// nominal input so that it looks like the others would be a dependency the
	// planner has to order and nothing has to obey.
	//
	// Both writes advance the counts state's version, which is what orders them
	// against each other and against the read below. Recording two raw nodes
	// against one port and trusting their order in the slice would compile and
	// order nothing: specs/007-tensor-layer.md makes the version chain the
	// statement of which contents a reader meant.
	cleared := b.writeCounts(counts, node{
		op: "PenaltyClear", kernel: &kernels.PenaltyClearKernel,
		uniform: dims, reads: reads, grid: overVocab(&kernels.PenaltyClearKernel),
		reason: "a store of zero over the counts, because the accumulation below is an " +
			"atomic add and a reused buffer would carry every earlier step's history",
	})

	counted := b.writeCounts(cleared, node{
		op: "PenaltyCount", inputs: []*Tensor{readState(b, history)},
		kernel:  &kernels.PenaltyCountKernel,
		uniform: dims, reads: reads,
		grid: func(*Tensor) accel.WorkgroupCount {
			wg := int(kernels.PenaltyCountKernel.WorkgroupSize.X)
			return accel.WorkgroupCount{X: (historyCap + wg - 1) / wg}
		},
		reason: "one invocation per history entry with an integer atomic increment, " +
			"which is exact and order-independent where an f32 accumulation would be " +
			"non-deterministic against itself",
		rejected: []string{"one invocation per history entry subtracting from the logits: " +
			"the history holds duplicates by construction, so several invocations write " +
			"one address"},
	})

	return b.record(node{
		op:      "PenaltyApply",
		inputs:  []*Tensor{logits, readState(b, counted)},
		kernel:  &kernels.PenaltyApplyKernel,
		uniform: dims, reads: reads,
		grid: overVocab(&kernels.PenaltyApplyKernel),
		reason: "one invocation per vocabulary entry, applying one update from a count " +
			"that is already final",
	}, accel.F32, logits.shape), true
}

// writeCounts records a node that rewrites the counts state and returns the
// version after it.
//
// The version rather than the tensor, because two nodes here write one buffer
// and the second must see the first's result. That ordering is the state
// version chain's job, and a node recorded straight against the port would have
// nothing expressing it.
func (b *Builder) writeCounts(s *State, n node) *State {
	n.outPort = s.desc.Name
	n.outOff = s.offset
	out := b.record(n, s.desc.DType, s.shape)
	next := *s
	next.version = s.version + 1
	next.producer = out.node
	if b.stateVersion == nil {
		b.stateVersion = map[window]int{}
	}
	b.stateVersion[s.window()] = next.version
	return &next
}
