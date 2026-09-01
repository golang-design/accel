// Copyright 2026 The golang.design Initiative Authors.
// All rights reserved. Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

//go:build darwin

package accel_test

import (
	"go/constant"
	"go/token"
	"math"
	"testing"

	"golang.design/x/accel"
	"golang.design/x/accel/internal/kernelc/emit"
	"golang.design/x/accel/internal/kernelc/ir"
	"golang.design/x/accel/kernelabi"
	"golang.design/x/accel/kmath"
)

// The two numeric contracts the MSL emitter got wrong on inputs the corpus
// never supplies, each run on the CPU oracle and on Metal from one record.
//
// The record is built here rather than taken from the corpus, because the
// corpus differential scales its inputs to keep every intermediate finite and
// in range -- which is right for what it measures and is exactly why a NaN or
// an overflowing sum never reached the device. The MSL is what the emitter
// produces for the hand-built IR, so the text under test is the emitter's; the
// Flat lowering is written by hand and calls the same kmath functions the Go
// emitter would.

// A minimum or maximum with a NaN operand is NaN on both backends.
//
// kmath.Min and kmath.Max propagate a NaN and call it a contract. MSL's min
// and max on float are fmin and fmax, which return the other operand, so Metal
// answered 5 where the oracle answered NaN.
func TestMinMaxPropagateNaNOnBothBackends(t *testing.T) {
	src := emitMSL(t, nanMinMaxIR())
	k := &kernelabi.Kernel{
		Name: "NaNMinMax", WorkgroupSize: accel.ID3{X: 64, Y: 1, Z: 1},
		Bindings: []kernelabi.Binding{
			{Name: "a", DType: kernelabi.F32, Access: kernelabi.Read},
			{Name: "b", DType: kernelabi.F32, Access: kernelabi.Read},
			{Name: "lo", DType: kernelabi.F32, Access: kernelabi.Write},
			{Name: "hi", DType: kernelabi.F32, Access: kernelabi.Write},
		},
		Digest: "test:NaNMinMax", Generator: kernelabi.Version, OrderIndependent: true,
		MSL: src,
		Flat: func(t accel.Thread, args kernelabi.Args) {
			a := kernelabi.Slice[float32](args, 0)
			b := kernelabi.Slice[float32](args, 1)
			lo := kernelabi.Slice[float32](args, 2)
			hi := kernelabi.Slice[float32](args, 3)
			i := t.GlobalID().X
			if i < uint32(len(lo)) {
				lo[i] = kmath.Min(a[i], b[i])
				hi[i] = kmath.Max(a[i], b[i])
			}
		},
	}

	nan := float32(math.NaN())
	inf := float32(math.Inf(1))
	a := []float32{1, nan, 3, nan, -inf, 2.5, nan, -4}
	b := []float32{2, 5, nan, nan, 7, -2.5, inf, -4}
	run := func(t *testing.T, d *accel.Device) (lo, hi []float32) {
		p, ba, bb := typedPipeline(t, d, k, accel.F32, len(a))
		blo := typedBuffer(t, d, "lo", accel.F32, len(a))
		bhi := typedBuffer(t, d, "hi", accel.F32, len(a))
		writeBuffer(t, d, ba, a)
		writeBuffer(t, d, bb, b)
		dispatch(t, d, p, []*accel.Buffer{ba, bb, blo, bhi}, len(a))
		lo = make([]float32, len(a))
		hi = make([]float32, len(a))
		readInto(t, d, blo, lo)
		readInto(t, d, bhi, hi)
		return lo, hi
	}
	cpuLo, cpuHi := run(t, openDevice(t))
	gpuLo, gpuHi := run(t, openMetal(t))

	for i := range a {
		nanIn := math.IsNaN(float64(a[i])) || math.IsNaN(float64(b[i]))
		for _, r := range []struct {
			what     string
			cpu, gpu float32
		}{{"min", cpuLo[i], gpuLo[i]}, {"max", cpuHi[i], gpuHi[i]}} {
			if math.Float32bits(r.cpu) != math.Float32bits(r.gpu) {
				t.Errorf("%s(%v, %v): the CPU backend produced %v (%#08x) and Metal %v (%#08x)",
					r.what, a[i], b[i], r.cpu, math.Float32bits(r.cpu), r.gpu, math.Float32bits(r.gpu))
			}
			// The contract itself, so a test where both backends dropped the
			// NaN the same way could not pass.
			if got := math.IsNaN(float64(r.gpu)); got != nanIn {
				t.Errorf("%s(%v, %v) is %v on Metal; a NaN operand must give NaN and only then",
					r.what, a[i], b[i], r.gpu)
			}
		}
	}
}

// Signed 32-bit add, subtract and multiply wrap on both backends.
//
// specs/008-numerics.md section 3 makes them exact wrapping operations, and
// Go's int32 wraps. MSL's int overflow is undefined, which the compiler is
// entitled to fold: `a + 1 > a` is true whatever a holds, where the oracle says
// false at MaxInt32. The flag output is that fold made visible.
func TestSignedArithmeticWrapsOnBothBackends(t *testing.T) {
	src := emitMSL(t, wrapI32IR())
	k := &kernelabi.Kernel{
		Name: "WrapI32", WorkgroupSize: accel.ID3{X: 64, Y: 1, Z: 1},
		Bindings: []kernelabi.Binding{
			{Name: "a", DType: kernelabi.I32, Access: kernelabi.Read},
			{Name: "b", DType: kernelabi.I32, Access: kernelabi.Read},
			{Name: "sum", DType: kernelabi.I32, Access: kernelabi.Write},
			{Name: "diff", DType: kernelabi.I32, Access: kernelabi.Write},
			{Name: "prod", DType: kernelabi.I32, Access: kernelabi.Write},
			{Name: "flag", DType: kernelabi.I32, Access: kernelabi.Write},
		},
		Digest: "test:WrapI32", Generator: kernelabi.Version, OrderIndependent: true,
		MSL: src,
		Flat: func(t accel.Thread, args kernelabi.Args) {
			a := kernelabi.Slice[int32](args, 0)
			b := kernelabi.Slice[int32](args, 1)
			sum := kernelabi.Slice[int32](args, 2)
			diff := kernelabi.Slice[int32](args, 3)
			prod := kernelabi.Slice[int32](args, 4)
			flag := kernelabi.Slice[int32](args, 5)
			i := t.GlobalID().X
			if i < uint32(len(sum)) {
				sum[i] = a[i] + b[i]
				diff[i] = a[i] - b[i]
				prod[i] = a[i] * b[i]
				flag[i] = 0
				if a[i]+1 > a[i] {
					flag[i] = 1
				}
			}
		},
	}

	a := []int32{math.MaxInt32, math.MinInt32, 1 << 30, -7, 65536, math.MaxInt32, 0, -1}
	b := []int32{1, -1, 1 << 30, 3, 65536, math.MaxInt32, 0, math.MinInt32}
	run := func(t *testing.T, d *accel.Device) [][]int32 {
		p, ba, bb := typedPipeline(t, d, k, accel.I32, len(a))
		writeBuffer(t, d, ba, a)
		writeBuffer(t, d, bb, b)
		outs := make([]*accel.Buffer, 4)
		for i, name := range []string{"sum", "diff", "prod", "flag"} {
			outs[i] = typedBuffer(t, d, name, accel.I32, len(a))
		}
		dispatch(t, d, p, append([]*accel.Buffer{ba, bb}, outs...), len(a))
		got := make([][]int32, 4)
		for i, o := range outs {
			got[i] = make([]int32, len(a))
			readInto(t, d, o, got[i])
		}
		return got
	}
	cpu := run(t, openDevice(t))
	gpu := run(t, openMetal(t))

	for j, what := range []string{"a + b", "a - b", "a * b", "a + 1 > a"} {
		for i := range a {
			if cpu[j][i] != gpu[j][i] {
				t.Errorf("%s with a=%d b=%d: the CPU backend produced %d and Metal %d",
					what, a[i], b[i], cpu[j][i], gpu[j][i])
			}
		}
	}
	// The inputs overflowed, so a test on which both backends computed the
	// exact sum could not have passed.
	if gpu[0][0] != math.MinInt32 {
		t.Errorf("MaxInt32 + 1 is %d on Metal, want the wrapped MinInt32", gpu[0][0])
	}
	if gpu[3][0] != 0 || gpu[3][3] != 1 {
		t.Errorf("a + 1 > a is %d at MaxInt32 and %d at -7 on Metal, want 0 and 1",
			gpu[3][0], gpu[3][3])
	}
}

// The IR for NaNMinMax:
//
//	i := t.GlobalID().X
//	if i < uint32(len(lo)) { lo[i] = kmath.Min(a[i], b[i]); hi[i] = kmath.Max(a[i], b[i]) }
func nanMinMaxIR() *ir.Func {
	f32 := &ir.Type{Kind: ir.F32}
	k, ps, i := elementwiseIR("NaNMinMax", f32, []string{"a", "b"}, []string{"lo", "hi"})
	at := func(p *ir.Param) ir.Value { return ir.NewIndex(0, f32, p, i, p.Index) }
	k.Body.List = append(k.Body.List, guarded(i, ps[2],
		ir.NewAssign(0, at(ps[2]), ir.NewIntrinsic(0, f32, ir.OpMin, nil, []ir.Value{at(ps[0]), at(ps[1])})),
		ir.NewAssign(0, at(ps[3]), ir.NewIntrinsic(0, f32, ir.OpMax, nil, []ir.Value{at(ps[0]), at(ps[1])})),
	))
	return k
}

// The IR for WrapI32:
//
//	i := t.GlobalID().X
//	if i < uint32(len(sum)) {
//		sum[i] = a[i] + b[i]; diff[i] = a[i] - b[i]; prod[i] = a[i] * b[i]
//		flag[i] = 0
//		if a[i] + 1 > a[i] { flag[i] = 1 }
//	}
func wrapI32IR() *ir.Func {
	i32 := &ir.Type{Kind: ir.I32}
	boolT := &ir.Type{Kind: ir.Bool}
	k, ps, i := elementwiseIR("WrapI32", i32, []string{"a", "b"}, []string{"sum", "diff", "prod", "flag"})
	at := func(p *ir.Param) ir.Value { return ir.NewIndex(0, i32, p, i, p.Index) }
	c := func(v int64) ir.Value { return ir.NewConst(0, i32, constant.MakeInt64(v)) }
	k.Body.List = append(k.Body.List, guarded(i, ps[2],
		ir.NewAssign(0, at(ps[2]), ir.NewBinary(0, i32, token.ADD, at(ps[0]), at(ps[1]))),
		ir.NewAssign(0, at(ps[3]), ir.NewBinary(0, i32, token.SUB, at(ps[0]), at(ps[1]))),
		ir.NewAssign(0, at(ps[4]), ir.NewBinary(0, i32, token.MUL, at(ps[0]), at(ps[1]))),
		ir.NewAssign(0, at(ps[5]), c(0)),
		ir.NewIf(0,
			ir.NewBinary(0, boolT, token.GTR,
				ir.NewBinary(0, i32, token.ADD, at(ps[0]), c(1)), at(ps[0])),
			ir.NewBlock(0, ir.NewAssign(0, at(ps[5]), c(1))), nil),
	))
	return k
}

// elementwiseIR builds a flat kernel over slices of one element type, with the
// global index declared and nothing else: the caller appends the body.
func elementwiseIR(name string, elem *ir.Type, in, out []string) (*ir.Func, []*ir.Param, *ir.Local) {
	sliceT := &ir.Type{Kind: ir.Slice, Elem: elem}
	u32 := &ir.Type{Kind: ir.U32}
	thread := ir.NewParam(0, &ir.Type{Kind: ir.Struct, Name: "Thread"}, 0, "t", nil)
	k := &ir.Func{
		Name: name, Stage: ir.StageCompute, Workgroup: [3]uint32{64, 1, 1}, Thread: 0,
		Params: []*ir.Param{thread},
	}
	var ps []*ir.Param
	for n, names := range [][]string{in, out} {
		for _, s := range names {
			p := ir.NewParam(0, sliceT, len(k.Params), s, nil)
			k.Params = append(k.Params, p)
			k.Bindings = append(k.Bindings, &ir.Binding{
				Name: s, Index: p.Index, Type: sliceT, Read: n == 0, Write: n == 1,
			})
			ps = append(ps, p)
		}
	}
	i := ir.NewLocal(0, u32, 0, "i", nil)
	gid := ir.NewFieldSel(0, u32,
		ir.NewIntrinsic(0, &ir.Type{Kind: ir.ID3Kind}, ir.OpGlobalID, thread, nil), 0, "X")
	k.Body = ir.NewBlock(0, ir.NewDeclare(0, i, gid))
	return k, ps, i
}

// guarded wraps statements in the bounds check every flat kernel writes.
func guarded(i *ir.Local, bound *ir.Param, body ...ir.Stmt) ir.Stmt {
	u32 := &ir.Type{Kind: ir.U32}
	i32 := &ir.Type{Kind: ir.I32}
	return ir.NewIf(0,
		ir.NewBinary(0, &ir.Type{Kind: ir.Bool}, token.LSS, i,
			ir.NewConvert(0, u32, ir.NewLen(0, i32, bound))),
		ir.NewBlock(0, body...), nil)
}

func emitMSL(t *testing.T, k *ir.Func) string {
	t.Helper()
	src, err := emit.MSL(k)
	if err != nil {
		t.Fatalf("emitting %s: %v", k.Name, err)
	}
	t.Logf("%s:\n%s", k.Name, src)
	return src
}

// typedPipeline compiles the record and allocates its two input buffers.
func typedPipeline(t *testing.T, d *accel.Device, k *kernelabi.Kernel, dt accel.DType, n int) (
	*accel.ComputePipeline, *accel.Buffer, *accel.Buffer) {
	t.Helper()
	p, err := d.NewComputePipeline(accel.ComputePipelineDescriptor{Kernel: k, Label: k.Name})
	if err != nil {
		t.Fatalf("pipeline %s: %v", k.Name, err)
	}
	t.Cleanup(func() { _ = p.Close() })
	return p, typedBuffer(t, d, "a", dt, n), typedBuffer(t, d, "b", dt, n)
}

func typedBuffer(t *testing.T, d *accel.Device, label string, dt accel.DType, n int) *accel.Buffer {
	t.Helper()
	b, err := d.NewBuffer(accel.BufferDescriptor{
		DType: dt, Count: n, Label: label,
		Usage: accel.BufferStorage | accel.BufferCopySrc | accel.BufferCopyDst,
	})
	if err != nil {
		t.Fatalf("buffer %q: %v", label, err)
	}
	t.Cleanup(func() { _ = b.Close() })
	return b
}

func writeBuffer(t *testing.T, d *accel.Device, b *accel.Buffer, data any) {
	t.Helper()
	if err := d.Queue().WriteBuffer(b, 0, data); err != nil {
		t.Fatalf("write %v: %v", b, err)
	}
}

func readInto(t *testing.T, d *accel.Device, b *accel.Buffer, into any) {
	t.Helper()
	if err := d.Queue().ReadBuffer(b, 0, into); err != nil {
		t.Fatalf("read %v: %v", b, err)
	}
}

// dispatch runs the pipeline once over n elements, with the buffers bound in
// order.
func dispatch(t *testing.T, d *accel.Device, p *accel.ComputePipeline, bufs []*accel.Buffer, n int) {
	t.Helper()
	binds := make([]accel.Binding, len(bufs))
	for i, b := range bufs {
		binds[i] = accel.Binding{Index: i, Buffer: whole(t, b)}
	}
	r := d.NewRecorder()
	r.Dispatch(p, binds, nil, accel.WorkgroupCount{X: (n + 63) / 64})
	g, err := r.Build()
	if err != nil {
		t.Fatalf("build: %v", err)
	}
	defer g.Close()
	if err := d.Queue().Submit(g).Wait(); err != nil {
		t.Fatalf("submit: %v", err)
	}
}
