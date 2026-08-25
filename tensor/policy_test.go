// Copyright 2026 The golang.design Initiative Authors.
// All rights reserved. Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

package tensor_test

import (
	"math"
	"strings"
	"testing"

	"golang.design/x/accel"
	"golang.design/x/accel/tensor"
)

// policyRun compiles and runs one sampling step and returns the token.
//
// vocab is small and the logits are given, so every assertion below is about
// which token the policy picks rather than about a model.
func policyRun(t *testing.T, o tensor.SamplingOptions, logits []float32,
	history []uint32, n uint32, draw float32) uint32 {

	t.Helper()
	rt := newRuntime(t)
	d := rt.Device()
	vocab := len(logits)
	b := rt.NewBuilder("policy")
	tensor.DeclareSamplingScalars(b, o, "s")

	in := tensor.Input(b, tensor.ValueDesc{
		Name: "logits", DType: accel.F32, Shape: tensor.Shape{vocab},
	})
	var draws *tensor.Tensor
	if !o.Greedy() {
		draws = tensor.Input(b, tensor.ValueDesc{
			Name: "draws", DType: accel.F32, Shape: tensor.Shape{1},
		})
	}
	var hist, counts *tensor.State
	if o.Penalised() {
		hist = tensor.NewState(b, tensor.StateDesc{
			Name: "history", DType: accel.U32, Shape: tensor.Shape{len(history)},
		})
		counts = tensor.NewState(b, tensor.StateDesc{
			Name: "counts", DType: accel.U32, Shape: tensor.Shape{vocab},
		})
	}

	tensor.Output(b, "token", tensor.Sample(b, in, draws, hist, counts, o, "s"))

	plan, err := b.Compile(rt, tensor.CompileOptions{Label: "policy"})
	if err != nil {
		t.Fatalf("compile: %v", err)
	}
	defer plan.Close()

	scalars, err := o.Scalars("s", n, uint32(len(history)))
	if err != nil {
		t.Fatalf("scalars: %v", err)
	}
	out := u32Buffer(t, d, "token", make([]uint32, 1))
	bufs := map[string]accel.BufferView{
		"logits": f32Buffer(t, d, "logits", logits),
		"token":  out,
	}
	if !o.Greedy() {
		bufs["draws"] = f32Buffer(t, d, "draws", []float32{draw})
	}
	if o.Penalised() {
		bufs["history"] = u32Buffer(t, d, "history", history)
		bufs["counts"] = u32Buffer(t, d, "counts", make([]uint32, vocab))
	}
	f := plan.Submit(d.Queue(), tensor.Bindings{Buffers: bufs, Scalars: scalars})
	if err := f.Wait(); err != nil {
		t.Fatalf("submit: %v", err)
	}
	got := make([]uint32, 1)
	if err := d.Queue().ReadBuffer(out.Buffer, 0, got); err != nil {
		t.Fatalf("read: %v", err)
	}
	return got[0]
}

// Penalties act before temperature, which is the one ordering decision 039 owns.
//
// With the coefficients fixed, the penalised *ranking* must be the same at
// T = 0.5 and at T = 2. Moving the penalty nodes after the scale makes the
// effective penalty alpha*T, and the two rankings then differ -- which is the
// mutation this catches, and the reason the two knobs must not multiply.
func TestPenaltiesActBeforeTemperature(t *testing.T) {
	// Token 0 leads before the penalty and token 1 takes over after it, but
	// only if the penalty keeps its full strength. The other two entries are
	// far enough down to hold no mass, so the draw resolves between the first
	// two.
	//
	// The true penalty on token 0 is 0.2 + 0.1x4 = 0.6, giving [2.4, 2.6]: token
	// 1 leads at every temperature. Applied after the scale instead, the
	// effective penalty is 0.6T, which is 0.3 at T=0.5 -- not enough, so token 0
	// still leads -- and 1.2 at T=2, which is more than enough. That is the
	// disagreement this reads.
	logits := []float32{3.0, 2.6, -20, -20}
	history := []uint32{0, 0, 0, 0}
	o := tensor.SamplingOptions{Presence: 0.2, Frequency: 0.1}

	// A draw of 0.5 resolves to whichever of the two holds more than half the
	// mass, which is the ranking question asked through the sampler that
	// exists. A draw of 0 would answer a different question: the walk compares
	// cumulative mass against draw x total, so 0 returns the first entry with
	// any mass rather than the largest.
	const draw = 0.5
	o.Temperature = 0.5
	cold := policyRun(t, o, logits, history, 4, draw)
	o.Temperature = 2
	warm := policyRun(t, o, logits, history, 4, draw)

	if cold != warm {
		t.Fatalf("the penalised leader is token %d at T=0.5 and token %d at T=2; the "+
			"penalty is being applied after the scale, so its strength is alpha*T",
			cold, warm)
	}
	if cold != 1 {
		t.Fatalf("the penalised leader is token %d, want 1: token 0 occurs four times "+
			"and should fall behind", cold)
	}
}

// Temperature 0 returns the lowest index on a plateau.
//
// A tie at the top is ordinary at saturation. Argmax returns the lowest index;
// a softmax at a tiny temperature returns whichever index the draw lands in.
// Replacing the greedy branch with T = 1e-6 fails this.
func TestGreedyReturnsTheLowestIndexOnAPlateau(t *testing.T) {
	logits := []float32{1, 5, 5, 5, 2}
	if got := policyRun(t, tensor.SamplingOptions{}, logits, nil, 0, 0); got != 1 {
		t.Fatalf("greedy returned token %d over a plateau at 1, 2 and 3, want 1: the "+
			"lowest index is the only tie rule two backends can agree on", got)
	}
}

// A token nobody generated is not penalised, so the unpenalised leader wins.
func TestAnUnpenalisedStepPicksTheLargestLogit(t *testing.T) {
	logits := []float32{0.5, 4.0, 1.0}
	history := []uint32{2, 2, 2, 2}
	o := tensor.SamplingOptions{Presence: 1, Frequency: 1}
	if got := policyRun(t, o, logits, history, 4, 0); got != 1 {
		t.Fatalf("token %d won, want 1: only token 2 was in the history", got)
	}
}

// The counts are rebuilt each step rather than accumulated.
//
// Running the same graph twice with the same history must give the same token.
// Without the clear pass the second step is penalised by the first step's
// history as well as its own, and the sequence decodes increasingly wrongly
// rather than failing.
func TestAStepIsNotPenalisedByThePreviousStep(t *testing.T) {
	logits := []float32{3.0, 2.9, 0.1}
	history := []uint32{0, 0}
	o := tensor.SamplingOptions{Frequency: 0.06}
	first := policyRun(t, o, logits, history, 2, 0)
	second := policyRun(t, o, logits, history, 2, 0)
	if first != second {
		t.Fatalf("the same step gave token %d and then %d: the counts carried over",
			first, second)
	}
}

// Validate refuses rather than clamps, and each refusal names its own rule.
func TestValidateRefusesAPolicyRatherThanRepairingIt(t *testing.T) {
	for _, c := range []struct {
		name string
		o    tensor.SamplingOptions
		want string
	}{
		{"a negative temperature", tensor.SamplingOptions{Temperature: -1}, "divisor"},
		{"a temperature under the guardrail",
			tensor.SamplingOptions{Temperature: 1e-6}, "written the wrong way"},
		{"a temperature whose reciprocal is not finite",
			// The guardrail catches this one first, and that is the point of
			// the case below: the two rules are not independent.
			tensor.SamplingOptions{Temperature: 1e-45}, "Temperature is"},
		{"a top-k past the round bound",
			tensor.SamplingOptions{Temperature: 1, TopK: tensor.TopMaxRounds + 1}, "bound is"},
		{"a top-p above one", tensor.SamplingOptions{Temperature: 1, TopP: 1.5}, "fraction"},
		{"a negative repetition penalty",
			tensor.SamplingOptions{Temperature: 1, Repetition: -1}, "flips the sign"},
	} {
		t.Run(c.name, func(t *testing.T) {
			err := c.o.Validate()
			if err == nil {
				t.Fatalf("%+v was accepted", c.o)
			}
			if !strings.Contains(err.Error(), c.want) {
				t.Fatalf("refused with %q, which does not mention %q", err, c.want)
			}
		})
	}
}

// The guardrail shadows the numerical refusal, and that is checked rather than
// left to be discovered.
//
// specs/039-sampling-policy.md section 3 states the two as separate rules, and
// they are, but they are not independent: every temperature whose reciprocal
// overflows is around 3e-39, which is far below the 1e-3 guardrail, so the
// guardrail always answers first. The numerical check is therefore unreachable
// at the current constant.
//
// It is kept, and this test is why it can be. The check exists so that lowering
// MinTemperature -- a policy change someone could reasonably make -- cannot
// silently start admitting a temperature that produces an all-NaN distribution
// and a plausible token from it. This asserts the shadowing, so the day the
// constant moves, the relationship is written down rather than rediscovered.
func TestTheGuardrailShadowsTheNumericalRefusal(t *testing.T) {
	// The smallest accepted temperature is accepted.
	if err := (tensor.SamplingOptions{Temperature: tensor.MinTemperature}).Validate(); err != nil {
		t.Fatalf("the smallest accepted temperature was refused: %v", err)
	}
	// A temperature whose reciprocal genuinely overflows is refused by the
	// guardrail, not by the arithmetic, because it is far below it.
	err := tensor.SamplingOptions{Temperature: 1e-45}.Validate()
	if err == nil {
		t.Fatal("a temperature whose reciprocal is infinite was accepted")
	}
	if strings.Contains(err.Error(), "reciprocal") {
		t.Fatalf("the numerical refusal answered first: %v. It is meant to be "+
			"unreachable while MinTemperature is %v, and a reader who found it "+
			"reachable would conclude the guardrail had a hole", err, float32(tensor.MinTemperature))
	}
	tiny := float32(1e-45)
	if !math.IsInf(float64(1/tiny), 1) {
		t.Fatal("1e-45 no longer overflows on inversion, so this test stopped " +
			"describing the case it names")
	}
}

// Storage nothing reads is refused, in both directions.
func TestSampleRefusesStateItWouldNotRead(t *testing.T) {
	rt := newRuntime(t)
	b := rt.NewBuilder("refusal")
	in := tensor.Input(b, tensor.ValueDesc{
		Name: "logits", DType: accel.F32, Shape: tensor.Shape{4},
	})
	// A penalty configured with no history to count.
	tensor.Output(b, "token",
		tensor.Sample(b, in, nil, nil, nil, tensor.SamplingOptions{Presence: 1}, "s"))
	_, err := b.Compile(rt, tensor.CompileOptions{Label: "refusal"})
	if err == nil {
		t.Fatal("a penalty with no history state was accepted, so the step would " +
			"penalise nothing and report nothing")
	} else if !strings.Contains(err.Error(), "history") {
		t.Fatalf("refused with %q, which does not name the missing state", err)
	}
}

// A sequence of tokens reproduces bit for bit from one seed, with a second
// sequence interleaved between the two runs.
//
// specs/039-sampling-policy.md section 9's first assertion, and the interleave
// is the whole of it. One token sampled twice proves nothing: it passes for a
// design that reseeds every step. A whole sequence sampled twice proves little
// more: it passes for a design holding a generator that happens to be advanced
// the same way both times. Running a *different* sequence in between is what
// fails for any design with a mutable generator anywhere, because the second
// sequence's draws would advance it.
//
// Every step is compared exactly. There is no bounded tier here: a token id is
// an integer, and "nearly the same token" is not a thing.
func TestASequenceReproducesFromOneSeedAcrossAnInterleave(t *testing.T) {
	logits := []float32{2.0, 1.6, 1.1, 0.4, -0.5, -20}
	o := tensor.SamplingOptions{Temperature: 0.9, TopK: 4}

	sequence := func(seed uint64, n int) []uint32 {
		s := tensor.Stream{Seed: seed}
		out := make([]uint32, n)
		for i := range out {
			out[i] = policyRun(t, o, logits, nil, 0, s.Draw(uint64(i)))
		}
		return out
	}

	const n = 12
	first := sequence(1234, n)

	// A different seed, and a different length, so nothing about the second run
	// lines up with the first.
	other := sequence(999, n+5)

	again := sequence(1234, n)
	for i := range first {
		if first[i] != again[i] {
			t.Fatalf("token %d of the sequence was %d and then %d after another "+
				"sequence ran in between: something holds a generator that the "+
				"interleaved run advanced", i, first[i], again[i])
		}
	}

	// The two seeds must not agree everywhere, or the assertion above is about
	// a constant rather than about a stream.
	same := 0
	for i := range first {
		if first[i] == other[i] {
			same++
		}
	}
	if same == len(first) {
		t.Fatalf("two different seeds produced the same %d tokens: the draw is not "+
			"reaching the walk", len(first))
	}
}
