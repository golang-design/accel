package kmath_test

import (
	"math"
	"testing"

	"golang.design/x/accel/kmath"
)

func TestToI32Boundaries(t *testing.T) {
	nan := float32(math.NaN())
	for _, c := range []struct {
		in   float32
		want int32
	}{
		{nan, 0},
		{float32(math.Inf(1)), math.MaxInt32},
		{float32(math.Inf(-1)), math.MinInt32},
		{0, 0}, {1.9, 1}, {-1.9, -1},
		{-2147483648, math.MinInt32},
		{-2147483904, math.MinInt32},
		{2147483648, math.MaxInt32},
		{1e30, math.MaxInt32},
	} {
		if got := kmath.ToI32(c.in); got != c.want {
			t.Errorf("ToI32(%v) = %d, want %d", c.in, got, c.want)
		}
	}
}

func TestToU32Boundaries(t *testing.T) {
	nan := float32(math.NaN())
	for _, c := range []struct {
		in   float32
		want uint32
	}{
		{nan, 0},
		{float32(math.Inf(1)), math.MaxUint32},
		{float32(math.Inf(-1)), 0},
		{-1, 0}, {-1e30, 0},
		{0, 0}, {1.9, 1},
		{4294967296, math.MaxUint32},
		{1e30, math.MaxUint32},
	} {
		if got := kmath.ToU32(c.in); got != c.want {
			t.Errorf("ToU32(%v) = %d, want %d", c.in, got, c.want)
		}
	}
}
