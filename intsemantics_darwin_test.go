// Copyright 2026 The golang.design Initiative Authors.
// All rights reserved. Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

//go:build darwin

package accel_test

import (
	"go/token"
	"math"
	"strings"
	"testing"

	"golang.design/x/accel"
	"golang.design/x/accel/internal/kernelc/ir"
	"golang.design/x/accel/kernelabi"
)

// intOpIR builds `out[i] = a[i] op b[i]` over one integer type.
func intOpIR(name string, kind ir.Kind, op token.Token) *ir.Func {
	elem := &ir.Type{Kind: kind}
	k, ps, i := elementwiseIR(name, elem, []string{"a", "b"}, []string{"out"})
	at := func(p *ir.Param) ir.Value { return ir.NewIndex(0, elem, p, i, p.Index) }
	k.Body.List = append(k.Body.List, guarded(i, ps[2],
		ir.NewAssign(0, at(ps[2]), ir.NewBinary(0, elem, op, at(ps[0]), at(ps[1]))),
	))
	return k
}

func intOpKernel[T int32 | uint32](t *testing.T, name string, kind ir.Kind, dt kernelabi.DType, op token.Token, flat func(a, b T) T) *kernelabi.Kernel {
	t.Helper()
	return &kernelabi.Kernel{
		Name: name, WorkgroupSize: accel.ID3{X: 64, Y: 1, Z: 1},
		Bindings: []kernelabi.Binding{
			{Name: "a", DType: dt, Access: kernelabi.Read},
			{Name: "b", DType: dt, Access: kernelabi.Read},
			{Name: "out", DType: dt, Access: kernelabi.Write},
		},
		Digest: "test:" + name, Generator: kernelabi.Version, OrderIndependent: true,
		MSL: emitMSL(t, intOpIR(name, kind, op)),
		Flat: func(th accel.Thread, args kernelabi.Args) {
			a := kernelabi.Slice[T](args, 0)
			b := kernelabi.Slice[T](args, 1)
			out := kernelabi.Slice[T](args, 2)
			i := th.GlobalID().X
			if i < uint32(len(out)) {
				out[i] = flat(a[i], b[i])
			}
		},
	}
}

func runIntOp[T int32 | uint32](t *testing.T, d *accel.Device, k *kernelabi.Kernel, dt accel.DType, a, b []T) ([]T, error) {
	t.Helper()
	p, ba, bb := typedPipeline(t, d, k, dt, len(a))
	bout := typedBuffer(t, d, "out", dt, len(a))
	writeBuffer(t, d, ba, a)
	writeBuffer(t, d, bb, b)
	binds := []accel.Binding{{Index: 0, Buffer: whole(t, ba)}, {Index: 1, Buffer: whole(t, bb)}, {Index: 2, Buffer: whole(t, bout)}}
	r := d.NewRecorder()
	r.Dispatch(p, binds, nil, accel.WorkgroupCount{X: (len(a) + 63) / 64})
	g, err := r.Build()
	if err != nil {
		t.Fatalf("build: %v", err)
	}
	defer g.Close()
	if err := d.Queue().Submit(g).Wait(); err != nil {
		return nil, err
	}
	out := make([]T, len(a))
	readInto(t, d, bout, out)
	return out, nil
}

// A shift by any count gives Go's result on both backends.
//
// specs/008-numerics.md section 3 makes shifts exact for counts in [0, 31].
// Go defines the rest too -- zero, or the sign for a signed right shift -- and
// MSL leaves a count at or above the width undefined, so the emitter spells
// Go's rule and the two agree on every count rather than on the portable ones.
func TestShiftsAgreeOnEveryCount(t *testing.T) {
	var a32, n32 []int32
	var au, nu []uint32
	for _, x := range []int32{1, -1, 0x40000000, math.MinInt32, 12345, -12345} {
		for n := int32(0); n <= 40; n++ {
			a32 = append(a32, x)
			n32 = append(n32, n)
			au = append(au, uint32(x))
			nu = append(nu, uint32(n))
		}
	}
	cases := []struct {
		name string
		run  func(t *testing.T, d *accel.Device) any
	}{
		{"i32 <<", func(t *testing.T, d *accel.Device) any {
			k := intOpKernel[int32](t, "ShlI32", ir.I32, kernelabi.I32, token.SHL, func(a, b int32) int32 { return a << b })
			out, err := runIntOp[int32](t, d, k, accel.I32, a32, n32)
			if err != nil {
				t.Fatal(err)
			}
			return out
		}},
		{"i32 >>", func(t *testing.T, d *accel.Device) any {
			k := intOpKernel[int32](t, "ShrI32", ir.I32, kernelabi.I32, token.SHR, func(a, b int32) int32 { return a >> b })
			out, err := runIntOp[int32](t, d, k, accel.I32, a32, n32)
			if err != nil {
				t.Fatal(err)
			}
			return out
		}},
		{"u32 <<", func(t *testing.T, d *accel.Device) any {
			k := intOpKernel[uint32](t, "ShlU32", ir.U32, kernelabi.U32, token.SHL, func(a, b uint32) uint32 { return a << b })
			out, err := runIntOp[uint32](t, d, k, accel.U32, au, nu)
			if err != nil {
				t.Fatal(err)
			}
			return out
		}},
		{"u32 >>", func(t *testing.T, d *accel.Device) any {
			k := intOpKernel[uint32](t, "ShrU32", ir.U32, kernelabi.U32, token.SHR, func(a, b uint32) uint32 { return a >> b })
			out, err := runIntOp[uint32](t, d, k, accel.U32, au, nu)
			if err != nil {
				t.Fatal(err)
			}
			return out
		}},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			cpu := c.run(t, openDevice(t))
			gpu := c.run(t, openMetal(t))
			switch cpu := cpu.(type) {
			case []int32:
				for i, v := range cpu {
					if g := gpu.([]int32)[i]; g != v {
						t.Errorf("%d %s %d: CPU %d, Metal %d", a32[i], c.name[4:], n32[i], v, g)
					}
				}
			case []uint32:
				for i, v := range cpu {
					if g := gpu.([]uint32)[i]; g != v {
						t.Errorf("%d %s %d: CPU %d, Metal %d", au[i], c.name[4:], nu[i], v, g)
					}
				}
			}
		})
	}
}

// Integer division by zero is an error on both backends, and MinInt32 / -1 is
// Go's wrapped result on both.
//
// specs/008-numerics.md section 3: the excluded cases are execution errors,
// not values. The CPU runs Go and panics, which the backend reports through
// the fence; Metal has no trap, so the kernel records a fault word the backend
// reads after the submission. MinInt32 / -1 is excluded from the exact domain
// and defined in Go, so the device gives Go's answer rather than a fault.
func TestIntegerDivisionByZeroIsAnErrorOnBothBackends(t *testing.T) {
	div := intOpKernel[int32](t, "DivI32", ir.I32, kernelabi.I32, token.QUO, func(a, b int32) int32 { return a / b })
	for _, d := range []struct {
		name string
		dev  *accel.Device
	}{{"CPU", openDevice(t)}, {"Metal", openMetal(t)}} {
		t.Run(d.name, func(t *testing.T) {
			_, err := runIntOp[int32](t, d.dev, div, accel.I32, []int32{7, 9}, []int32{2, 0})
			if err == nil {
				t.Fatal("dividing by zero produced a value rather than an error")
			}
			if d.name == "Metal" && !strings.Contains(err.Error(), "divided an integer by zero") {
				t.Errorf("Metal's error does not name the fault: %v", err)
			}
			// The device is usable afterwards: a fault is the kernel's.
			out, err := runIntOp[int32](t, d.dev, div, accel.I32, []int32{7, math.MinInt32}, []int32{2, -1})
			if err != nil {
				t.Fatalf("after a fault the device refused an ordinary dispatch: %v", err)
			}
			if out[0] != 3 || out[1] != math.MinInt32 {
				t.Errorf("got %v, want [3 %d]", out, math.MinInt32)
			}
		})
	}
}
