// Copyright 2026 The golang.design Initiative Authors.
// All rights reserved. Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

package accel_test

import (
	"fmt"
	"testing"

	"golang.design/x/accel"
	"golang.design/x/accel/internal/testkernels"
)

// randomGraph builds one graph from a seed, twice: once optimized and once
// under the conservative plan. Both recordings are identical by construction,
// because the same function drives both.
type oracleCase struct {
	seed []byte
	d    *accel.Device
	p    *accel.ComputePipeline
	in   *accel.Buffer
	out  *accel.Buffer
}

const oracleElems = 32

func (c oracleCase) record(r *accel.Recorder) {
	at := 0
	next := func(mod int) int {
		if len(c.seed) == 0 || mod <= 0 {
			return 0
		}
		v := int(c.seed[at%len(c.seed)])
		at++
		return v % mod
	}
	storage := accel.UsageStorage | accel.UsageCopySrc | accel.UsageCopyDst

	// Transients of varying size, so packing has something to sort. Shape is
	// randomized too, not only node count: the bug the interference relation
	// exists for needs a diamond, and a generator producing only chains would
	// never see it.
	var views []accel.BufferView
	for i := range 2 + next(5) {
		count := oracleElems / (1 + next(2))
		v := r.Transient(accel.BufferDescriptor{
			DType: accel.F32, Count: count, Usage: storage,
			Label: fmt.Sprintf("t%d", i),
		})
		if v.Buffer != nil {
			views = append(views, v)
		}
	}
	views = append(views, whole2(c.in), whole2(c.out))

	pick := func() accel.BufferView { return views[next(len(views))] }
	for range 2 + next(10) {
		switch next(3) {
		case 0:
			dst := pick()
			r.CopyToBuffer(dst, make([]float32, dst.Count))
		case 1:
			dst, src := pick(), pick()
			if dst.Count == src.Count {
				r.CopyBuffer(dst, src)
			}
		case 2:
			a, b, out := pick(), pick(), pick()
			// A dispatch needs all three the same length, since Add indexes
			// them together and the kernel's own bound is the shortest.
			n := min(min(a.Count, b.Count), out.Count)
			a.Count, b.Count, out.Count = n, n, n
			if n > 0 {
				r.Dispatch(c.p, []accel.Binding{
					{Index: 0, Buffer: a}, {Index: 1, Buffer: b}, {Index: 2, Buffer: out},
				}, nil, accel.WorkgroupCount{X: 1})
			}
		}
	}
	// The output is always written last, so every seed produces something to
	// compare rather than a graph whose result is whatever was there before.
	r.CopyBuffer(whole2(c.out), views[0])
}

func whole2(b *accel.Buffer) accel.BufferView {
	v, err := b.View(0, b.Count())
	if err != nil {
		panic(err)
	}
	return v
}

// FuzzWholePlanOracle is spec 003's whole-plan oracle: execute a graph twice,
// once under the optimized plan and once under the conservative one, and
// compare the bytes.
//
// Any disagreement is a planner or barrier bug, localized to the builder rather
// than to a kernel, because both sides ran the same kernels over the same
// inputs. It is the test that retires spec 009's "graph aliasing is unsound"
// risk row, and it lands with the aliasing rather than after it for exactly
// that reason.
func FuzzWholePlanOracle(f *testing.F) {
	f.Add([]byte{1})
	f.Add([]byte{2, 5, 1, 9, 3})
	f.Add([]byte{7, 7, 2, 0, 4, 6, 1, 1, 8, 3})
	f.Add([]byte{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13})

	d, err := accel.OpenCPU(accel.CPUOptions{})
	if err != nil {
		f.Fatalf("open: %v", err)
	}
	f.Cleanup(func() { _ = d.Close() })
	p, err := d.NewComputePipeline(accel.ComputePipelineDescriptor{
		Kernel: &testkernels.AddKernel, Label: "add",
	})
	if err != nil {
		f.Fatalf("pipeline: %v", err)
	}
	f.Cleanup(func() { _ = p.Close() })

	storage := accel.UsageStorage | accel.UsageCopySrc | accel.UsageCopyDst
	mk := func(label string) *accel.Buffer {
		b, err := d.NewBuffer(accel.BufferDescriptor{
			DType: accel.F32, Count: oracleElems, Usage: storage, Label: label,
		})
		if err != nil {
			f.Fatalf("buffer: %v", err)
		}
		f.Cleanup(func() { _ = b.Close() })
		return b
	}
	in, out := mk("in"), mk("out")

	f.Fuzz(func(t *testing.T, seed []byte) {
		c := oracleCase{seed: seed, d: d, p: p, in: in, out: out}

		run := func(naive bool) ([]float32, error) {
			// The same inputs both times: the comparison is of plans, so
			// anything else varying would make a disagreement uninterpretable.
			vals := make([]float32, oracleElems)
			for i := range vals {
				vals[i] = float32(i%7) + 1
			}
			if err := d.Queue().WriteBuffer(in, 0, vals); err != nil {
				return nil, err
			}
			if err := d.Queue().WriteBuffer(out, 0, make([]float32, oracleElems)); err != nil {
				return nil, err
			}

			r := d.NewRecorder()
			c.record(r)
			var g *accel.Graph
			var err error
			if naive {
				g, err = r.BuildNaive()
			} else {
				g, err = r.Build()
			}
			if err != nil {
				return nil, err
			}
			defer g.Close()
			if err := d.Queue().Submit(g).Wait(); err != nil {
				return nil, err
			}
			got := make([]float32, oracleElems)
			if err := d.Queue().ReadBuffer(out, 0, got); err != nil {
				return nil, err
			}
			return got, nil
		}

		optimized, errOpt := run(false)
		naive, errNaive := run(true)

		// The two plans must agree about whether the graph is valid at all. One
		// accepting what the other rejects is itself a planner bug.
		if (errOpt == nil) != (errNaive == nil) {
			t.Fatalf("the optimized plan returned %v and the conservative plan %v",
				errOpt, errNaive)
		}
		if errOpt != nil {
			return
		}
		if !equalFloats(optimized, naive) {
			t.Fatalf("the plans disagree:\n optimized %v\n naive     %v", optimized, naive)
		}
	})
}

func equalFloats(a, b []float32) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		// Bit comparison, because both sides ran the same kernels over the same
		// inputs in the same order: anything but an exact match is a planning
		// difference, and a tolerance here would hide one.
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

// The oracle's own premise: the two plans really are different. Without this a
// fuzz that passed would prove nothing, because it would be comparing a plan
// against itself.
func TestTheTwoPlansAreActuallyDifferent(t *testing.T) {
	const n = 64
	d := openDevice(t)
	storage := accel.UsageStorage | accel.UsageCopySrc | accel.UsageCopyDst
	p, err := d.NewComputePipeline(accel.ComputePipelineDescriptor{
		Kernel: &testkernels.AddKernel, Label: "op",
	})
	if err != nil {
		t.Fatalf("pipeline: %v", err)
	}
	defer p.Close()

	in := newBuffer(t, d, "in", n, storage)
	out := newBuffer(t, d, "out", n, storage)
	// The graph needs both shapes, because the two plans differ in two ways and
	// each needs its own. Independent nodes are what makes the barrier counts
	// differ: in a strict chain every node needs a barrier anyway, so the plans
	// coincide. An ordered chain is what makes aliasing possible: transients
	// written by unordered nodes cannot share bytes, correctly.
	record := func(r *accel.Recorder) {
		mk := func() accel.BufferView {
			return r.Transient(accel.BufferDescriptor{DType: accel.F32, Count: n, Usage: storage})
		}
		dis := func(a, b, o accel.BufferView) {
			r.Dispatch(p, []accel.Binding{
				{Index: 0, Buffer: a}, {Index: 1, Buffer: b}, {Index: 2, Buffer: o},
			}, nil, accel.WorkgroupCount{X: 1})
		}
		// Two independent dispatches: no hazard between them.
		x, y := mk(), mk()
		dis(whole(t, in), whole(t, in), x)
		dis(whole(t, in), whole(t, in), y)
		// Then a chain through them, whose transients are fully ordered against
		// each other and therefore alias.
		a, b := mk(), mk()
		dis(x, y, a)
		dis(a, whole(t, in), b)
		r.CopyBuffer(whole(t, out), b)
	}

	r1 := d.NewRecorder()
	record(r1)
	opt, err := r1.Build()
	if err != nil {
		t.Fatalf("build: %v", err)
	}
	defer opt.Close()

	r2 := d.NewRecorder()
	record(r2)
	nai, err := r2.BuildNaive()
	if err != nil {
		t.Fatalf("build naive: %v", err)
	}
	defer nai.Close()

	if nai.Barriers() != len(nai.Nodes()) {
		t.Errorf("the conservative plan barriers every node: %d of %d",
			nai.Barriers(), len(nai.Nodes()))
	}
	if opt.Barriers() >= nai.Barriers() {
		t.Errorf("the optimized plan emits %d barriers and the conservative one %d; "+
			"they are supposed to differ", opt.Barriers(), nai.Barriers())
	}
	if m := nai.Memory(); m.TransientBytes != m.UnaliasedBytes {
		t.Errorf("the conservative plan does not alias: pool %d, unaliased %d",
			m.TransientBytes, m.UnaliasedBytes)
	}
	if m := opt.Memory(); m.TransientBytes >= m.UnaliasedBytes {
		t.Errorf("the optimized plan should alias: pool %d, unaliased %d",
			m.TransientBytes, m.UnaliasedBytes)
	}
}

// A byte-level check that the two plans agree on a graph with a diamond, which
// is the shape the interference relation exists for.
func TestTheOracleAgreesOnADiamond(t *testing.T) {
	const n = 32
	d := openDevice(t)
	storage := accel.UsageStorage | accel.UsageCopySrc | accel.UsageCopyDst
	p, err := d.NewComputePipeline(accel.ComputePipelineDescriptor{
		Kernel: &testkernels.AddKernel, Label: "op",
	})
	if err != nil {
		t.Fatalf("pipeline: %v", err)
	}
	defer p.Close()

	in := newBuffer(t, d, "in", n, storage)
	out := newBuffer(t, d, "out", n, storage)
	vals := make([]float32, n)
	for i := range vals {
		vals[i] = float32(i) + 1
	}

	record := func(r *accel.Recorder) {
		mk := func(l string) accel.BufferView {
			return r.Transient(accel.BufferDescriptor{
				DType: accel.F32, Count: n, Usage: storage, Label: l,
			})
		}
		t0, t1, t2, t3 := mk("t0"), mk("t1"), mk("t2"), mk("t3")
		dis := func(a, b, o accel.BufferView) {
			r.Dispatch(p, []accel.Binding{
				{Index: 0, Buffer: a}, {Index: 1, Buffer: b}, {Index: 2, Buffer: o},
			}, nil, accel.WorkgroupCount{X: 1})
		}
		dis(whole(t, in), whole(t, in), t0) // fan out from here
		dis(t0, whole(t, in), t1)           // arm one
		dis(t0, whole(t, in), t2)           // arm two, unordered against arm one
		dis(t1, t2, t3)                     // join
		r.CopyBuffer(whole(t, out), t3)
	}

	result := func(naive bool) []float32 {
		t.Helper()
		if err := d.Queue().WriteBuffer(in, 0, vals); err != nil {
			t.Fatalf("write: %v", err)
		}
		r := d.NewRecorder()
		record(r)
		var g *accel.Graph
		var err error
		if naive {
			g, err = r.BuildNaive()
		} else {
			g, err = r.Build()
		}
		if err != nil {
			t.Fatalf("build: %v", err)
		}
		defer g.Close()
		if err := d.Queue().Submit(g).Wait(); err != nil {
			t.Fatalf("submit: %v", err)
		}
		got := make([]float32, n)
		if err := d.Queue().ReadBuffer(out, 0, got); err != nil {
			t.Fatalf("readback: %v", err)
		}
		return got
	}

	if opt, nai := result(false), result(true); !equalFloats(opt, nai) {
		t.Fatalf("the plans disagree on a diamond:\n optimized %v\n naive     %v", opt, nai)
	}
}
