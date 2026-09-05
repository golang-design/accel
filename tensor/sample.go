// Copyright 2026 The golang.design Initiative Authors.
// All rights reserved. Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

package tensor

import (
	"math"

	"golang.design/x/accel"
	"golang.design/x/accel/internal/kernels"
)

// Sampling: the operators that turn a row of logits into a token id.
//
// specs/028-sampling.md designed and built the four kernels; this file is the
// promotion specs/039-sampling-policy.md section 1 asks for. Until it existed no
// [Plan] could produce a token, so a consumer read the whole logits vector back
// to the host every decode step -- 128 KiB for a 32k vocabulary against the 4
// bytes a token id costs -- and ran the sampler in Go.
//
// # The three decisions this file makes, and where each comes from
//
// **The random draw is an input.** specs/028-sampling.md section 1 settles it
// and the reasoning is not repeated here: a token is reproducible only if the
// caller supplies the randomness, and two backends agree on a token only if
// neither runs a PRNG.
//
// **The draw is a tensor, not a scalar.** specs/043-per-row-values.md draws the
// line -- a scalar is a value every row of a dispatch shares, and a value that
// differs per row is device data. One draw across a batch keeps
// reproducibility and destroys *independence*: two sequences with similar
// distributions emit the same token, their contexts converge, and they become
// more similar still. Every reproducibility test passes, because reproducibility
// is exactly what is preserved.
//
// **The truncations are masks composed before the draw, not fused into it.**
// specs/039-sampling-policy.md section 5 fixes the order:
//
//	penalties -> Scale by 1/T -> Softmax -> TopKMask -> TopPMask -> SampleCategorical
//
// with T = 0 lowering to [Argmax] instead. Nothing renormalizes between the mask
// and the draw and there is never a second softmax: [SampleCategorical] compares
// against draw times the total for that reason. A softmax over a mask output is
// worse than useless, because exp(0) = 1 for every dropped entry.
//
// # Batched from the start
//
// Every operator here reads the last axis as the vocabulary and every axis
// before it as a row, so a batch of one is [vocab] and a batch of B is
// [B, vocab]. There is no single-sequence path beside a batched one, which is
// specs/043-per-row-values.md section 3's orthogonality test: after this there
// is no question of the form "which of the two do I use?".
//
// # k and p are recorded values, not named scalars
//
// A deviation from specs/039-sampling-policy.md section 7, which prices k and p
// as per-step scalars that "cost nothing" because [Plan.Submit] rewrites every
// uniform. They cost nothing to *bind*; what they cost is the refusal.
// [Plan.bind] validates a scalar's kind and never its range, so a k of vocab
// bound as a scalar reaches the kernel, is clamped to [TopMaxRounds], and the
// caller runs top-128 believing they turned truncation off -- which
// specs/039-sampling-policy.md section 9 names as a mutation a test must catch
// and which no tensor-layer test could catch through a scalar.
//
// Recorded at build time, both are refusable in one line each, the way
// [RMSNorm] already refuses a non-positive eps. The price is a plan per (k, p),
// which is a real cost a serving process pays; it is the same cost
// specs/039-sampling-policy.md section 7 already accepts for top-k being
// present or absent, and a policy layer that wants a plan per shape has one
// anyway.
//
// # k and p are shared by the batch, and the draw is not
//
// specs/039-sampling-policy.md section 10 leaves open "whether per-sequence k
// and p are worth their bindings or a batch shares one policy". This takes the
// second: a truncation parameter is a property of the request's policy and a
// draw is a property of the sequence's position in its own random stream, so
// only the draw has to differ per row for two sequences to decode
// independently. Moving k and p to bindings later adds an operand and removes
// nothing, so it is not a decision this forecloses.

// TopMaxRounds is how many entries either truncation can keep.
//
// Re-exported from the corpus so a caller can write the check their own
// configuration has to pass, rather than discovering the bound by watching a
// top-k keep fewer entries than it was asked for. specs/028-sampling.md states
// it: both masks walk the distribution one entry per round, so this is a real
// limit and not a buffer size.
const TopMaxRounds = kernels.TopMaxRounds

// sampleShape splits a logits tensor into its rows and its vocabulary.
//
// The result shape a sampler writes is the leading axes, and a rank-1 input has
// none -- so it gets [1] rather than a shape with no elements. That is what
// makes a batch of one the same path: a single sequence binds a one-element
// token buffer for the same reason it binds a one-element draws buffer.
func sampleShape(x *Tensor) (rows, vocab int, out Shape) {
	vocab = x.shape[len(x.shape)-1]
	rows = x.shape.Elements() / vocab
	// Copied rather than sliced. The result shares no backing array with the
	// operand's shape, so an operator that later grows one -- the way
	// GatherRows appends a width -- cannot write through into the tensor it
	// read from.
	out = append(Shape(nil), x.shape[:len(x.shape)-1]...)
	if len(out) == 0 {
		out = Shape{1}
	}
	return rows, vocab, out
}

// checkLogits reports why a logits tensor is not samplable, or "".
func checkLogits(x *Tensor) string {
	if x.dtype != accel.F32 {
		// specs/007-tensor-layer.md has no silent promotion, so an f16 logits
		// tensor needs an explicit Cast. Named here rather than left to the
		// generic dtype message, because the caller has a model whose head is
		// f16 and needs the route rather than the rule.
		return "logits are " + x.dtype.String() + " and the registered kernel reads f32; " +
			"a narrow head needs an explicit Cast, which specs/007-tensor-layer.md " +
			"requires so the conversion is where you wrote it"
	}
	if len(x.shape) < 1 {
		return "logits are " + x.shape.String() + "; the last axis is the vocabulary and " +
			"every axis before it is a sequence"
	}
	// Slice can produce empty views even though declarations require positive
	// dimensions. Reject them before sampleShape divides by the vocabulary or
	// a per-row kernel records a dispatch with no rows.
	for _, d := range x.shape {
		if d <= 0 {
			return "logits are " + x.shape.String() + "; sampling needs non-empty rows and vocabulary"
		}
	}
	return ""
}

// Argmax writes the index of the largest logit in each row.
//
// Greedy decoding, and the operator specs/039-sampling-policy.md section 3 puts
// behind a temperature of zero: T = 0 selects this rather than a softmax at a
// tiny T, because with a logit gap smaller than T the softmax is not one-hot at
// all and the walk samples the runner-up.
//
// # Ties go to the lowest index
//
// Equal logits are ordinary -- an untrained model produces them everywhere and a
// trained one produces them at saturation. specs/028-sampling.md section 3
// states the rule because the alternative is not "some index" but a *different*
// index on each backend: a tree reduction's answer depends on which lane
// compared which pair, and two backends reducing at different widths would
// disagree about a token.
//
// The result is u32, which needs nothing new: [GatherRows] and [ScatterRows]
// already require u32 ids, so a sampled token feeds the next step's embedding
// lookup directly.
func Argmax(b *Builder, logits *Tensor) *Tensor {
	if poisoned(logits) {
		return b.poison()
	}
	if why := checkLogits(logits); why != "" {
		return b.fail(1, "Argmax", "%s", why)
	}
	rows, vocab, out := sampleShape(logits)

	return b.record(node{
		op: "Argmax", inputs: []*Tensor{logits}, kernel: &kernels.SampleArgmaxKernel,
		uniform: func(map[string]ScalarValue) any {
			return kernels.SampleDims{Vocab: uint32(vocab), Rows: uint32(rows)}
		},
		// One workgroup per row rather than per output element: the whole
		// vocabulary reduces together, because a maximum split across two
		// workgroups would need a second pass and a tie rule that survived it.
		// perRow cannot be used here -- it counts rows of the *result*, and the
		// result is one token per row rather than one row per row.
		grid: func(*Tensor) accel.WorkgroupCount {
			return accel.WorkgroupCount{X: rows}
		},
		reason: "the cooperative argmax reduction, one workgroup per row; equal logits " +
			"go to the lowest index so the two backends name the same token",
		rejected: []string{"a two-pass reduction over several workgroups per row: it " +
			"would need a tie rule that survived the second pass, and " +
			"specs/028-sampling.md takes reproducibility over the split"},
	}, accel.U32, out)
}

// SampleCategorical draws one index per row from the weights that row holds.
//
//	out[r] = min{ i : Σⱼ≤ᵢ weights[r][j] > draws[r] × Σⱼ weights[r][j] }
//
// # The draws are a tensor
//
// One per row, and specs/043-per-row-values.md section 2 is the reason: a value
// that differs per row is device data. A batch sharing one draw stays perfectly
// reproducible and stops being independent, which is a failure no
// reproducibility test can see.
//
// Each draw is a uniform in [0, 1). A draw outside it is clamped rather than
// refused, because a kernel cannot report an error and an unclamped draw reads
// past the end of the row; specs/028-sampling.md section 4 states that and the
// clamp is unconditional, so correctness does not depend on a build mode.
//
// # The weights need not be normalized
//
// The walk compares against the draw times the row's own total, so this is
// correct for any vector of non-negative weights and not only for a
// distribution. That is what lets [TopKMask] and [TopPMask] compose in front of
// it without a renormalizing pass -- and specs/039-sampling-policy.md section 5
// forbids putting a second [Softmax] there, since exp(0) = 1 for every dropped
// entry would make the mask do nothing.
//
// A zero-weight entry can never be drawn, because its partial sum does not
// increase and the comparison is strict. That is what makes a mask a mask.
//
// # The walk is in index order
//
// A parallel prefix sum would be faster and would put the boundary somewhere
// else when two weights are equal. specs/028-sampling.md section 3 takes the
// reproducible answer over the fast one, which is the trade
// specs/008-numerics.md makes when it forbids a tolerance.
func SampleCategorical(b *Builder, weights, draws *Tensor) *Tensor {
	if poisoned(weights, draws) {
		return b.poison()
	}
	if why := checkLogits(weights); why != "" {
		return b.fail(1, "SampleCategorical", "%s", why)
	}
	if draws.dtype != accel.F32 {
		return b.fail(1, "SampleCategorical", "draws are %v and the kernel reads f32",
			draws.dtype)
	}
	rows, vocab, out := sampleShape(weights)
	if got := draws.shape.Elements(); got != rows {
		return b.fail(1, "SampleCategorical", "the weights hold %d rows and draws holds %d; "+
			"every sequence draws against its own uniform, so there is exactly one per row "+
			"(specs/043-per-row-values.md). A shared draw stays reproducible and makes two "+
			"sequences with similar distributions emit the same token", rows, got)
	}

	return b.record(node{
		op: "SampleCategorical", inputs: []*Tensor{weights, draws},
		kernel: &kernels.SampleCategoricalKernel,
		uniform: func(map[string]ScalarValue) any {
			return kernels.SampleDims{Vocab: uint32(vocab), Rows: uint32(rows)}
		},
		// One invocation per row, which is what the kernel bounds itself by. A
		// row's walk is sequential, so there is nothing for a second lane in it
		// to do.
		grid: perElement(int(kernels.SampleCategoricalKernel.WorkgroupSize.X)),
		reason: "the sequential cumulative walk, one invocation per row; the draw scales " +
			"by the row's own total, so a masked distribution needs no renormalizing pass",
		rejected: []string{"a parallel prefix scan: it is faster and places the boundary " +
			"elsewhere when two weights are equal, which specs/028-sampling.md refuses"},
	}, accel.U32, out)
}

// TopKMask keeps the k largest entries of each row and zeroes the rest.
//
// A mask over weights rather than a step fused into the draw: kept entries carry
// their input value and dropped ones carry zero, so the result feeds
// [SampleCategorical] directly. specs/039-sampling-policy.md section 5 places it
// after [Softmax] and before [TopPMask], and that order is not re-derived here:
// top-p is a mass threshold so it must see probabilities, and top-p is relative
// to its input's own total so it composes after a k that has already cut.
//
// # Exactly k, including at a tie
//
// The comparison is lexicographic on (value, index) descending, so an entry ties
// only with itself and "the k largest" is exactly k entries whatever the data.
// A threshold search would be fewer rounds and would keep however many happened
// to sit above wherever the bisection stopped -- which is k only when nothing
// ties, and ties near the tail of a distribution are the normal case.
//
// # k is refused rather than clamped
//
// The kernel clamps its round count to [TopMaxRounds] because a kernel cannot
// report an error. This is the first layer that can, so a k above the bound is
// refused here: silently running top-128 when a caller asked for top-vocab
// changes what a model samples without changing what it reports, which
// specs/039-sampling-policy.md section 5 calls out by name.
func TopKMask(b *Builder, weights *Tensor, k int) *Tensor {
	if poisoned(weights) {
		return b.poison()
	}
	if why := checkLogits(weights); why != "" {
		return b.fail(1, "TopKMask", "%s", why)
	}
	_, vocab, _ := sampleShape(weights)
	if k <= 0 {
		return b.fail(1, "TopKMask", "k is %d; a truncation keeps at least one entry, and "+
			"the way to say \"no truncation\" is to leave this operator out of the graph "+
			"rather than to ask for zero", k)
	}
	if k > TopMaxRounds {
		return b.fail(1, "TopKMask", "k is %d and the bound is %d; the mask walks the "+
			"distribution one entry per round, so a larger k would keep %d and report "+
			"nothing (specs/028-sampling.md)", k, TopMaxRounds, TopMaxRounds)
	}
	if k > vocab {
		return b.fail(1, "TopKMask", "k is %d over a vocabulary of %d; a top-k wider than "+
			"the distribution keeps all of it, which is the graph without this operator",
			k, vocab)
	}

	return b.record(node{
		op: "TopKMask", inputs: []*Tensor{weights}, kernel: &kernels.TopKMaskKernel,
		uniform: func(map[string]ScalarValue) any {
			return kernels.TopDims{Vocab: uint32(vocab), K: uint32(k)}
		},
		// One workgroup per row, and perRow reads that off the result because
		// a mask's result is shaped like its input.
		grid: perRow,
		reason: "the repeated extraction, one workgroup per row and one round per kept " +
			"entry; the comparison is lexicographic on (value, index) so exactly k " +
			"entries survive a tie at the boundary",
		rejected: []string{"a full sort of the row: thousands of comparisons to answer a " +
			"question about its first few dozen entries",
			"a bisection on the value range: fewer rounds, and it keeps however many " +
				"entries happen to sit above wherever it stopped rather than k"},
		// k is recorded rather than bound, so it is part of what this plan *is*
		// and [Builder.Identity] has to carry it. Without it a [PlanCache] keyed
		// on the digest answers a request for top-5 with the top-40 plan it
		// compiled earlier, and the token it returns is plausible.
		attrs: []uint64{uint64(k)},
	}, accel.F32, weights.shape)
}

// TopPMask keeps the smallest set of largest entries whose mass reaches p.
//
// The nucleus. The same walk [TopKMask] performs with a different stopping rule:
// it accumulates weight rather than counting entries, and the entry that crosses
// the threshold is kept -- which is what makes the set the *smallest* one
// reaching p rather than the largest one below it.
//
// The mass is a fraction of the row's own total rather than of one, so this
// composes after a [TopKMask] and on unnormalized weights, for the same reason
// [SampleCategorical] scales its draw by the total.
//
// # p is refused rather than clamped, and zero is not "off"
//
// A p of zero makes the threshold zero, so the walk never advances its frontier,
// the mask keeps nothing, and [SampleCategorical] over an all-zero row finds no
// partial sum above its target and returns the last index. That is a plausible
// token id from a row that was entirely erased. The way to say "no truncation"
// is to leave the operator out of the graph, which specs/039-sampling-policy.md
// section 5 states as "off means the node is absent".
//
// # It costs the same whatever p is
//
// All [TopMaxRounds] rounds run. Stopping the loop early would put a barrier in
// non-uniform control flow, which specs/002-compute-model.md section 3.1 and
// specs/018-cooperative-lowering.md forbid, so only the frontier's advance is
// conditional. p = 0.5 costs what p = 0.99 costs, and a top-k followed by a
// top-p costs k + 128 workgroup reductions. Enabling this by default on a false
// cost model is a design mistake rather than only a slow step.
func TopPMask(b *Builder, weights *Tensor, p float32) *Tensor {
	if poisoned(weights) {
		return b.poison()
	}
	if why := checkLogits(weights); why != "" {
		return b.fail(1, "TopPMask", "%s", why)
	}
	_, vocab, _ := sampleShape(weights)
	if !(p > 0) || p > 1 || math.IsNaN(float64(p)) {
		return b.fail(1, "TopPMask", "p is %v and a nucleus is a mass in (0, 1]; p = 0 "+
			"keeps nothing, and a draw over a row of zeros returns the last index rather "+
			"than reporting anything. The way to say \"no truncation\" is to leave this "+
			"operator out of the graph", p)
	}

	return b.record(node{
		op: "TopPMask", inputs: []*Tensor{weights}, kernel: &kernels.TopPMaskKernel,
		uniform: func(map[string]ScalarValue) any {
			return kernels.TopDims{Vocab: uint32(vocab), P: p}
		},
		grid: perRow,
		reason: "the repeated extraction with a mass threshold, one workgroup per row; " +
			"the threshold is a fraction of that row's own total, so this composes " +
			"after a top-k",
		rejected: []string{"an early exit once the mass is reached: it puts a barrier in " +
			"non-uniform control flow, which specs/002-compute-model.md section 3.1 " +
			"forbids, so every round runs whatever p is",
			"a sort followed by a prefix sum: the same cost objection as top-k's"},
		// As with k, and through the bit pattern rather than a conversion: 0.9
		// and 0.5 are both zero as integers.
		attrs: []uint64{uint64(math.Float32bits(p))},
	}, accel.F32, weights.shape)
}
