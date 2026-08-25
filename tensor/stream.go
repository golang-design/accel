// Copyright 2026 The golang.design Initiative Authors.
// All rights reserved. Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

package tensor

// The draws a sampler consumes, defined here rather than taken from
// math/rand.
//
// specs/039-sampling-policy.md section 2 is the argument. The short form is
// that a golden test pinning token ids for a seed would otherwise pin the
// standard library, and neither version of it will hold still: math/rand is
// frozen to the Go 1 value stream, and math/rand/v2 promises nothing about
// which values a seed produces and computes Float32 by a different method.
// This is twenty lines of arithmetic, and the promise is accel's.

// golden is 2^64 / phi, the odd multiplier SplitMix64 advances by.
//
// Odd so that adding it repeatedly visits every uint64 before repeating, and
// this particular odd number because its bits are as unlike a small stride as
// an increment gets, which is what keeps adjacent steps from producing
// adjacent-looking states before the avalanche runs.
const golden = 0x9E3779B97F4A7C15

// finalize is SplitMix64's avalanche: it maps a counter to a value whose bits
// each depend on all of the input's.
//
// The avalanche is the part that cannot be skipped. Without it the draw is a
// function of seed+j that drifts monotonically as j advances, which produces
// output that looks random in a histogram and is badly biased as a sequence.
func finalize(z uint64) uint64 {
	z ^= z >> 30
	z *= 0xBF58476D1CE4E5B9
	z ^= z >> 27
	z *= 0x94D049BB133111EB
	z ^= z >> 31
	return z
}

// Stream is one sequence's source of draws.
//
// It is a value, and that is the design rather than an accident. The shape
// everyone writes first is a policy struct holding a *rand.Rand, and Go copies
// a struct silently -- on assignment, by value, into a map -- so two sequences
// end up sharing one generator, interleaving their draws, and neither
// reproduces. A *rand.Rand is also not safe for concurrent use, so two
// goroutines holding copies of that struct are a data race the detector finds
// only if a test happens to run them together.
//
// Copying a Stream copies a number. There is nothing to share.
type Stream struct{ Seed uint64 }

// Derive gives sequence seq of a batch its own stream from one root seed.
//
// seq+1 rather than seq so that sequence 0 is not the root seed itself, which
// would make a one-sequence batch and an unbatched run of the same seed draw
// the same numbers -- true today and silently false the moment the derivation
// changes.
func Derive(root, seq uint64) Stream {
	return Stream{Seed: finalize(root + (seq+1)*golden)}
}

// Draw returns the uniform in [0,1) for token step of this sequence.
//
// # The step is the token index, not a draw counter
//
// The caller already holds this number: it is the position the KV cache writes
// at. So the sampler stores nothing per sequence, resuming a sequence at token
// N costs nothing, and exactly one draw is defined per token whether or not
// that step used it. Turning temperature off for one step does not shift every
// later token, which it would if the stream had a position to advance.
//
// # Why the result can never be 1.0
//
// Twenty-four bits of the finalized state, divided by 2^24. Both the numerator
// and the divisor are exact in f32, the largest numerator is 2^24-1, and so the
// largest result is exactly 0.99999994.
//
// The obvious spelling, float32(rng.Float64()), rounds up to exactly 1.0 for
// about one input in 2^24. specs/028-sampling.md's walk clamps a draw of 1.0
// down, so nothing crashes: the last token in the vocabulary silently receives
// the extra mass, on one step in sixteen million, and every differential still
// passes because both backends clamp identically. This division makes the
// generator and that backstop agree by construction rather than by luck.
func (s Stream) Draw(step uint64) float32 {
	return float32(finalize(s.Seed+step*golden)>>40) / (1 << 24)
}
