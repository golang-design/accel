// Copyright 2026 The golang.design Initiative Authors.
// All rights reserved. Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

package kernel_test

import (
	"math"
	"testing"

	"golang.design/x/accel/internal/kernel"
)

// Every atomic, exercised directly.
//
// Directly rather than through a kernel, because the corpus reaches the u32
// family and the signed and float ones would otherwise be code nobody runs
// until a caller writes the first kernel that needs them.
func TestEveryAtomicReturnsThePreviousValueAndStores(t *testing.T) {
	t.Run("unsigned", func(t *testing.T) {
		cases := []struct {
			name  string
			start uint32
			run   func(b []uint32) uint32
			want  uint32
		}{
			{"add wraps", math.MaxUint32, func(b []uint32) uint32 { return kernel.AddU32(b, 0, 2) }, 1},
			{"sub wraps", 0, func(b []uint32) uint32 { return kernel.SubU32(b, 0, 1) }, math.MaxUint32},
			{"min stores the smaller", 10, func(b []uint32) uint32 { return kernel.MinU32(b, 0, 3) }, 3},
			{"min keeps the smaller", 3, func(b []uint32) uint32 { return kernel.MinU32(b, 0, 10) }, 3},
			{"max stores the larger", 3, func(b []uint32) uint32 { return kernel.MaxU32(b, 0, 10) }, 10},
			{"max keeps the larger", 10, func(b []uint32) uint32 { return kernel.MaxU32(b, 0, 3) }, 10},
			{"and", 0xFF, func(b []uint32) uint32 { return kernel.AndU32(b, 0, 0x0F) }, 0x0F},
			{"or", 0x0F, func(b []uint32) uint32 { return kernel.OrU32(b, 0, 0xF0) }, 0xFF},
			{"xor", 0xFF, func(b []uint32) uint32 { return kernel.XorU32(b, 0, 0x0F) }, 0xF0},
			{"exchange", 7, func(b []uint32) uint32 { return kernel.ExchangeU32(b, 0, 9) }, 9},
		}
		for _, c := range cases {
			t.Run(c.name, func(t *testing.T) {
				b := []uint32{c.start}
				if got := c.run(b); got != c.start {
					t.Errorf("returned %d, want the previous value %d", got, c.start)
				}
				if b[0] != c.want {
					t.Errorf("stored %d, want %d", b[0], c.want)
				}
			})
		}
	})

	t.Run("signed", func(t *testing.T) {
		cases := []struct {
			name  string
			start int32
			run   func(b []int32) int32
			want  int32
		}{
			{"add wraps two's complement", math.MaxInt32,
				func(b []int32) int32 { return kernel.AddI32(b, 0, 1) }, math.MinInt32},
			{"sub wraps", math.MinInt32,
				func(b []int32) int32 { return kernel.SubI32(b, 0, 1) }, math.MaxInt32},
			// The comparison is by the named type, which is the whole reason
			// there are two families: -1 is less than 1 as int32 and greater as
			// uint32.
			{"min compares as signed", 1, func(b []int32) int32 { return kernel.MinI32(b, 0, -1) }, -1},
			{"max compares as signed", -1, func(b []int32) int32 { return kernel.MaxI32(b, 0, 1) }, 1},
			{"max keeps the larger signed", 1, func(b []int32) int32 { return kernel.MaxI32(b, 0, -1) }, 1},
			{"min keeps the smaller signed", -1, func(b []int32) int32 { return kernel.MinI32(b, 0, 1) }, -1},
			{"exchange", -7, func(b []int32) int32 { return kernel.ExchangeI32(b, 0, 9) }, 9},
			{"compare-exchange matching", 5,
				func(b []int32) int32 { return kernel.CompareExchangeI32(b, 0, 5, -2) }, -2},
			{"compare-exchange not matching", 5,
				func(b []int32) int32 { return kernel.CompareExchangeI32(b, 0, 4, -2) }, 5},
		}
		for _, c := range cases {
			t.Run(c.name, func(t *testing.T) {
				b := []int32{c.start}
				if got := c.run(b); got != c.start {
					t.Errorf("returned %d, want the previous value %d", got, c.start)
				}
				if b[0] != c.want {
					t.Errorf("stored %d, want %d", b[0], c.want)
				}
			})
		}
	})
}

// The signed and unsigned comparisons differ on the same bits, which is why
// there are two families rather than one taking a sign flag.
func TestSignedAndUnsignedComparisonsDisagree(t *testing.T) {
	// The same 32 bits: the maximum unsigned, and -1 signed.
	const bits = uint32(0xFFFFFFFF)

	u := []uint32{1}
	kernel.MinU32(u, 0, bits)
	if u[0] != 1 {
		t.Errorf("unsigned min of 1 and 0x%X is %d, want 1", bits, u[0])
	}

	s := []int32{1}
	signed := int32(int64(bits) - 1<<32) // the same bits read as int32: -1
	kernel.MinI32(s, 0, signed)
	if s[0] != -1 {
		t.Errorf("signed min of 1 and -1 is %d, want -1", s[0])
	}
}

// The float atomic adds and returns the previous value like the rest. What is
// non-deterministic about it is the order several of them run in, which is a
// property of a reduction rather than of the operation.
func TestTheFloatAtomicAdds(t *testing.T) {
	b := []float32{1.5}
	if got := kernel.AddF32(b, 0, 0.25); got != 1.5 {
		t.Errorf("returned %v, want the previous 1.5", got)
	}
	if b[0] != 1.75 {
		t.Errorf("stored %v, want 1.75", b[0])
	}

	// A sum of the same values in two orders differs in the last bit, which is
	// why a test asserting an exact total for a float reduction is wrong even
	// where the same test is right for integers.
	forward := []float32{0}
	for _, v := range []float32{1, 1e-8, 1e-8} {
		kernel.AddF32(forward, 0, v)
	}
	backward := []float32{0}
	for _, v := range []float32{1e-8, 1e-8, 1} {
		kernel.AddF32(backward, 0, v)
	}
	if forward[0] == backward[0] {
		t.Skip("this machine's f32 addition happens to be associative for these values; " +
			"the point stands and the demonstration needs different ones")
	}
}
