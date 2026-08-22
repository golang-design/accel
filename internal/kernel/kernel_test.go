// Copyright 2026 The golang.design Initiative Authors.
// All rights reserved. Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

package kernel_test

import (
	"strings"
	"testing"

	"golang.design/x/accel/internal/kernel"
)

func id(x, y, z uint32) kernel.ID3 { return kernel.ID3{X: x, Y: y, Z: z} }

// TestThreadIDs covers the accessors a kernel body reads.
func TestThreadIDs(t *testing.T) {
	th := kernel.NewThread(id(9, 2, 0), id(1, 2, 0), id(2, 0, 0), id(4, 4, 1), id(3, 1, 1))
	if got := th.GlobalID(); got != id(9, 2, 0) {
		t.Errorf("GlobalID = %+v", got)
	}
	if got := th.LocalID(); got != id(1, 2, 0) {
		t.Errorf("LocalID = %+v", got)
	}
	if got := th.GroupID(); got != id(2, 0, 0) {
		t.Errorf("GroupID = %+v", got)
	}
}

// TestLinearIndicesAreXFastest is spec 002 section 1.4: linearization is
// x-fastest and that is guaranteed rather than incidental, because a kernel
// that indexes a buffer by a linear id and a host that fills that buffer have
// to agree on the order.
func TestLinearIndicesAreXFastest(t *testing.T) {
	size := id(4, 2, 1)
	seen := make(map[uint32]kernel.ID3)
	for z := range size.Z {
		for y := range size.Y {
			for x := range size.X {
				th := kernel.NewThread(id(0, 0, 0), id(x, y, z), id(0, 0, 0), size, id(1, 1, 1))
				got := th.LocalIndex()
				if prev, dup := seen[got]; dup {
					t.Fatalf("LocalIndex %d is produced by both %+v and %+v", got, prev, id(x, y, z))
				}
				seen[got] = id(x, y, z)
			}
		}
	}
	// x-fastest means (1,0,0) is index 1 and (0,1,0) is index 4, not the reverse.
	if got := kernel.NewThread(id(0, 0, 0), id(1, 0, 0), id(0, 0, 0), size, id(1, 1, 1)).LocalIndex(); got != 1 {
		t.Errorf("LocalIndex of (1,0,0) is %d, want 1: linearization is x-fastest", got)
	}
	if got := kernel.NewThread(id(0, 0, 0), id(0, 1, 0), id(0, 0, 0), size, id(1, 1, 1)).LocalIndex(); got != 4 {
		t.Errorf("LocalIndex of (0,1,0) is %d, want 4: linearization is x-fastest", got)
	}
	if len(seen) != 8 {
		t.Errorf("%d distinct indices over 8 invocations", len(seen))
	}
}

// TestGlobalIndexIsWorkgroupContiguous is the distinction spec 002 section 1.4
// calls out as the trap: GlobalIndex is not the grid linearization of GlobalID,
// and the two agree only when the grid is one workgroup wide.
func TestGlobalIndexIsWorkgroupContiguous(t *testing.T) {
	// Two workgroups of four along x. Workgroup 1, local 0 is global id 4.
	size, count := id(4, 1, 1), id(2, 1, 1)
	th := kernel.NewThread(id(4, 0, 0), id(0, 0, 0), id(1, 0, 0), size, count)
	if got := th.GlobalIndex(); got != 4 {
		t.Errorf("GlobalIndex = %d, want 4", got)
	}

	// The case where the two definitions diverge: a 2x2 grid of 2x2 groups. The
	// invocation at group (1,0) local (0,0) has global id (2,0,0), whose grid
	// linearization over a width of 4 is 2. Its workgroup-contiguous index is 4,
	// because workgroup 0's four invocations come first.
	size, count = id(2, 2, 1), id(2, 2, 1)
	th = kernel.NewThread(id(2, 0, 0), id(0, 0, 0), id(1, 0, 0), size, count)
	if got := th.GlobalIndex(); got != 4 {
		t.Errorf("GlobalIndex = %d, want 4: the index is workgroup-contiguous, not grid-linearized", got)
	}

	// Every invocation in a 2x2 grid of 2x2 groups gets a distinct index, and
	// the set is exactly [0, 16).
	seen := make(map[uint32]bool)
	for gz := range count.Z {
		for gy := range count.Y {
			for gx := range count.X {
				for lz := range size.Z {
					for ly := range size.Y {
						for lx := range size.X {
							th := kernel.NewThread(
								id(gx*size.X+lx, gy*size.Y+ly, gz*size.Z+lz),
								id(lx, ly, lz), id(gx, gy, gz), size, count)
							i := th.GlobalIndex()
							if seen[i] {
								t.Fatalf("GlobalIndex %d appears twice", i)
							}
							seen[i] = true
						}
					}
				}
			}
		}
	}
	if len(seen) != 16 {
		t.Fatalf("%d distinct global indices over 16 invocations", len(seen))
	}
	for i := range uint32(16) {
		if !seen[i] {
			t.Errorf("no invocation has GlobalIndex %d", i)
		}
	}
}

// TestBindChecksOnce is spec 012's rule that the argument set is validated
// against the declared bindings before the invocation loop rather than inside
// it: the signature is the binding layout, so a mismatch is something
// generation already proved.
func TestBindChecksOnce(t *testing.T) {
	k := &kernel.Kernel{
		Name:          "Scale",
		WorkgroupSize: id(64, 1, 1),
		Generator:     kernel.ABIVersion,
		Bindings: []kernel.Binding{
			{Name: "in", DType: kernel.F32, Access: kernel.Read},
			{Name: "out", DType: kernel.F32, Access: kernel.Write},
		},
	}

	if err := k.Bind(kernel.Args{Slices: []any{[]float32{1}, []float32{0}}}); err != nil {
		t.Fatalf("a correct argument set was rejected: %v", err)
	}

	for _, tc := range []struct {
		name string
		args kernel.Args
		want []string
	}{
		{"too few", kernel.Args{Slices: []any{[]float32{1}}}, []string{"takes 2 bindings and got 1"}},
		{"too many", kernel.Args{Slices: []any{[]float32{1}, []float32{1}, []float32{1}}}, []string{"got 3"}},
		{"wrong dtype", kernel.Args{Slices: []any{[]float32{1}, []int32{0}}}, []string{`"out"`, "f32", "[]float32", "[]int32"}},
		{"not a slice", kernel.Args{Slices: []any{[]float32{1}, 42}}, []string{`"out"`, "int"}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			err := k.Bind(tc.args)
			if err == nil {
				t.Fatal("was accepted")
			}
			for _, want := range tc.want {
				if !strings.Contains(err.Error(), want) {
					t.Errorf("error %q does not carry %q", err, want)
				}
			}
		})
	}
}

// TestBindNamesEveryDTypesHostSlice checks that a mismatch tells a caller which
// Go slice each dtype binds to, since guessing is how someone passes the right
// number of bytes with the wrong meaning.
func TestBindNamesEveryDTypesHostSlice(t *testing.T) {
	for _, tc := range []struct {
		dt   kernel.DType
		want string
	}{
		{kernel.F32, "[]float32"},
		{kernel.F16, "[]uint16"},
		{kernel.BF16, "[]uint16"},
		{kernel.I32, "[]int32"},
		{kernel.U32, "[]uint32"},
		{kernel.I8, "[]int8"},
		{kernel.U8, "[]uint8"},
		{kernel.DType(99), "an unknown slice type"},
	} {
		k := &kernel.Kernel{
			Name: "K", Generator: kernel.ABIVersion,
			Bindings: []kernel.Binding{{Name: "b", DType: tc.dt}},
		}
		// A []complex64 matches no dtype, so every case takes the naming path.
		err := k.Bind(kernel.Args{Slices: []any{[]complex64{0}}})
		if err == nil {
			t.Errorf("%v accepted a []complex64", tc.dt)
			continue
		}
		if !strings.Contains(err.Error(), tc.want) {
			t.Errorf("%v: error %q does not name %s", tc.dt, err, tc.want)
		}
	}
}

// TestBindRejectsAStaleABI is why the ABI version is a field rather than a
// comment: an adapter generated against one shape of the runtime and loaded by
// another is a wrong-answer bug, and there is no version of it that is a
// compile error.
func TestBindRejectsAStaleABI(t *testing.T) {
	k := &kernel.Kernel{Name: "Old", Generator: kernel.ABIVersion - 1}
	err := k.Bind(kernel.Args{})
	if err == nil {
		t.Fatal("a kernel generated against an older ABI was accepted")
	}
	for _, want := range []string{"Old", "go generate"} {
		if !strings.Contains(err.Error(), want) {
			t.Errorf("error %q does not carry %q", err, want)
		}
	}
}

// TestSlicePanicsWithoutBind documents why Slice does not return an error: Bind
// has already proved the type, so a generated caller would be handling an error
// that cannot happen, at every binding.
func TestSlicePanicsWithoutBind(t *testing.T) {
	defer func() {
		r := recover()
		if r == nil {
			t.Fatal("Slice on a mistyped argument did not panic")
		}
		if !strings.Contains(r.(string), "Bind was not called") {
			t.Errorf("panic %q does not say what was skipped", r)
		}
	}()
	kernel.Slice[float32](kernel.Args{Slices: []any{[]int32{1}}}, 0)
}

func TestSliceRecoversBoundArguments(t *testing.T) {
	a := kernel.Args{Slices: []any{[]float32{1, 2}, []uint32{3}}}
	if got := kernel.Slice[float32](a, 0); len(got) != 2 || got[1] != 2 {
		t.Errorf("Slice[float32] = %v", got)
	}
	if got := kernel.Slice[uint32](a, 1); len(got) != 1 || got[0] != 3 {
		t.Errorf("Slice[uint32] = %v", got)
	}
}

func TestStrings(t *testing.T) {
	for _, tc := range []struct {
		got, want string
	}{
		{kernel.F32.String(), "f32"},
		{kernel.BF16.String(), "bf16"},
		{kernel.U8.String(), "u8"},
		{kernel.DType(99).String(), "DType(99)"},
		{kernel.Read.String(), "read"},
		{kernel.Write.String(), "write"},
		{(kernel.Read | kernel.Write).String(), "read-write"},
		{kernel.Access(0).String(), "no access"},
	} {
		if tc.got != tc.want {
			t.Errorf("got %q, want %q", tc.got, tc.want)
		}
	}

	k := &kernel.Kernel{
		Name: "Scale", WorkgroupSize: id(64, 1, 1),
		Bindings: []kernel.Binding{{Name: "in", DType: kernel.F32, Access: kernel.Read}},
	}
	if got := k.String(); !strings.Contains(got, "Scale workgroup=64,1,1") || !strings.Contains(got, "in:f32/read") {
		t.Errorf("Kernel.String = %q", got)
	}
}
