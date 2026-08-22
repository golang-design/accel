// Copyright 2026 The golang.design Initiative Authors.
// All rights reserved. Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

package kernel_test

import (
	"testing"

	"golang.design/x/accel/internal/kernel"
)

// FuzzBind checks the gate between a caller's buffers and a generated kernel.
//
// Everything past Bind assumes the argument set matches the record, and
// [kernel.Slice] panics rather than returning an error on that assumption. So
// the property is exact: Bind accepts an argument set only when every binding's
// slice is the type its dtype names, and after it accepts, every Slice call the
// generated code makes must succeed.
func FuzzBind(f *testing.F) {
	f.Add(uint(0), uint(0), uint(2))
	f.Add(uint(3), uint(1), uint(1))
	f.Add(uint(6), uint(6), uint(0))

	dtypes := []kernel.DType{kernel.F32, kernel.F16, kernel.BF16, kernel.I32, kernel.U32, kernel.I8, kernel.U8}
	supply := []any{
		[]float32{1}, []uint16{1}, []int32{1}, []uint32{1}, []int8{1}, []uint8{1},
		[]float64{1}, 42, nil, []any{},
	}

	f.Fuzz(func(t *testing.T, rawDType, rawSupply, rawCount uint) {
		n := int(rawCount%4) + 1
		k := &kernel.Kernel{Name: "K", Generator: kernel.ABIVersion}
		args := kernel.Args{}
		for i := range n {
			dt := dtypes[(rawDType+uint(i))%uint(len(dtypes))]
			k.Bindings = append(k.Bindings, kernel.Binding{Name: "b", DType: dt})
			args.Slices = append(args.Slices, supply[(rawSupply+uint(i))%uint(len(supply))])
		}

		err := k.Bind(args)
		if err != nil {
			if err.Error() == "" {
				t.Fatal("a rejection with no message")
			}
			return
		}

		// It accepted, so every generated Slice call must succeed. This is the
		// assumption Slice's panic rests on, and the only place it is checked.
		for i, b := range k.Bindings {
			func() {
				defer func() {
					if r := recover(); r != nil {
						t.Fatalf("Bind accepted binding %d (%v) and Slice panicked: %v", i, b.DType, r)
					}
				}()
				switch b.DType {
				case kernel.F32:
					_ = kernel.Slice[float32](args, i)
				case kernel.F16, kernel.BF16:
					_ = kernel.Slice[uint16](args, i)
				case kernel.I32:
					_ = kernel.Slice[int32](args, i)
				case kernel.U32:
					_ = kernel.Slice[uint32](args, i)
				case kernel.I8:
					_ = kernel.Slice[int8](args, i)
				case kernel.U8:
					_ = kernel.Slice[uint8](args, i)
				}
			}()
		}
	})
}

// FuzzThreadIndices checks the linearizations a kernel indexes buffers with.
//
// Spec 002 guarantees x-fastest ordering and a workgroup-contiguous global
// index, and a kernel that reads one and a host that fills the buffer have to
// agree. The property is that within one dispatch the indices are a bijection
// onto a contiguous range: a collision silently drops an invocation's result,
// and a gap leaves a buffer element nobody wrote.
func FuzzThreadIndices(f *testing.F) {
	f.Add(uint(4), uint(1), uint(1), uint(2), uint(1), uint(1))
	f.Add(uint(2), uint(2), uint(1), uint(2), uint(2), uint(1))
	f.Add(uint(1), uint(1), uint(1), uint(1), uint(1), uint(1))

	f.Fuzz(func(t *testing.T, sx, sy, sz, cx, cy, cz uint) {
		clamp := func(v uint) uint32 { return uint32(v%4) + 1 }
		size := kernel.ID3{X: clamp(sx), Y: clamp(sy), Z: clamp(sz)}
		count := kernel.ID3{X: clamp(cx), Y: clamp(cy), Z: clamp(cz)}

		total := size.X * size.Y * size.Z * count.X * count.Y * count.Z
		seen := make(map[uint32]bool, total)

		for gz := range count.Z {
			for gy := range count.Y {
				for gx := range count.X {
					for lz := range size.Z {
						for ly := range size.Y {
							for lx := range size.X {
								th := kernel.NewThread(
									kernel.ID3{X: gx*size.X + lx, Y: gy*size.Y + ly, Z: gz*size.Z + lz},
									kernel.ID3{X: lx, Y: ly, Z: lz},
									kernel.ID3{X: gx, Y: gy, Z: gz},
									size, count,
								)
								i := th.GlobalIndex()
								if seen[i] {
									t.Fatalf("GlobalIndex %d appears twice at size %+v count %+v: "+
										"an invocation's result would be silently dropped", i, size, count)
								}
								seen[i] = true

								if l := th.LocalIndex(); l >= size.X*size.Y*size.Z {
									t.Fatalf("LocalIndex %d is outside the %+v workgroup", l, size)
								}
								if g := th.GroupIndex(); g >= count.X*count.Y*count.Z {
									t.Fatalf("GroupIndex %d is outside the %+v grid", g, count)
								}
							}
						}
					}
				}
			}
		}

		if uint32(len(seen)) != total {
			t.Fatalf("%d distinct indices over %d invocations", len(seen), total)
		}
		for i := range total {
			if !seen[i] {
				t.Fatalf("no invocation has GlobalIndex %d, so a buffer element goes unwritten", i)
			}
		}
	})
}
