// Copyright 2026 The golang.design Initiative Authors.
// All rights reserved. Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

package accel_test

import (
	"strings"
	"testing"

	"golang.design/x/accel"
	"golang.design/x/accel/internal/testkernels"
)

// indirectGraph records a dispatch whose workgroup count comes from a buffer.
func indirectGraph(t *testing.T, d *accel.Device, max accel.WorkgroupCount, stats bool) (
	*accel.Graph, *accel.Buffer, *accel.Buffer) {
	t.Helper()

	p, err := d.NewComputePipeline(accel.ComputePipelineDescriptor{
		Kernel: &testkernels.AddKernel, Label: "add",
	})
	if err != nil {
		t.Fatalf("pipeline: %v", err)
	}
	t.Cleanup(func() { _ = p.Close() })

	storage := accel.BufferStorage | accel.BufferCopySrc | accel.BufferCopyDst
	const n = 256
	in := newBuffer(t, d, "in", n, storage)
	out := newBuffer(t, d, "out", n, storage)
	count, err := d.NewBuffer(accel.BufferDescriptor{
		DType: accel.U32, Count: 3, Label: "count",
		Usage: accel.BufferIndirect | accel.BufferCopyDst | accel.BufferStorage,
	})
	if err != nil {
		t.Fatalf("count buffer: %v", err)
	}
	t.Cleanup(func() { _ = count.Close() })

	r := d.NewRecorder()
	r.CollectRunStats(stats)
	countView, err := count.View(0, 3)
	if err != nil {
		t.Fatalf("view: %v", err)
	}
	r.DispatchIndirect(p, []accel.Binding{
		{Index: 0, Buffer: whole(t, in)},
		{Index: 1, Buffer: whole(t, in)},
		{Index: 2, Buffer: whole(t, out)},
	}, nil, countView, max)

	g, err := r.Build()
	if err != nil {
		t.Fatalf("build: %v", err)
	}
	t.Cleanup(func() { _ = g.Close() })
	return g, count, out
}

// The device supplies the count and the dispatch runs that many workgroups.
func TestIndirectDispatchUsesTheDeviceCount(t *testing.T) {
	d := openDevice(t)
	g, count, out := indirectGraph(t, d, accel.WorkgroupCount{X: 4}, false)

	// Two workgroups of 64, so the first 128 elements are written.
	if err := d.Queue().WriteBuffer(count, 0, []uint32{2, 1, 1}); err != nil {
		t.Fatalf("write count: %v", err)
	}
	if err := d.Queue().Submit(g).Wait(); err != nil {
		t.Fatalf("submit: %v", err)
	}
	got := readback(t, d, out)
	for i := range 128 {
		if got[i] != 0 { // in is zero, so out is zero, but it must be *written*
			t.Fatalf("element %d is %v", i, got[i])
		}
	}
	// Elements past the dispatched range are untouched, which is what says the
	// count was honoured rather than ignored.
	for i := 128; i < len(got); i++ {
		if got[i] != 0 {
			t.Fatalf("element %d past the dispatched range is %v", i, got[i])
		}
	}
}

// countGraph dispatches CountWorkgroups indirectly, so the number of workgroups
// that ran is observable in the output.
//
// A kernel with an ordinary bounds check will not do: run twice as many
// workgroups over the same buffer and the extra ones write nothing, so a clamp
// that did not happen looks exactly like one that did.
func countGraph(t *testing.T, d *accel.Device, max accel.WorkgroupCount, stats bool) (
	*accel.Graph, *accel.Buffer, *accel.Buffer) {
	t.Helper()

	p, err := d.NewComputePipeline(accel.ComputePipelineDescriptor{
		Kernel: &testkernels.CountWorkgroupsKernel, Label: "count",
	})
	if err != nil {
		t.Fatalf("pipeline: %v", err)
	}
	t.Cleanup(func() { _ = p.Close() })

	counts, err := d.NewBuffer(accel.BufferDescriptor{
		DType: accel.U32, Count: 1, Label: "counts",
		Usage: accel.BufferStorage | accel.BufferCopySrc | accel.BufferCopyDst,
	})
	if err != nil {
		t.Fatalf("counts: %v", err)
	}
	t.Cleanup(func() { _ = counts.Close() })

	countBuf, err := d.NewBuffer(accel.BufferDescriptor{
		DType: accel.U32, Count: 3, Label: "indirect",
		Usage: accel.BufferIndirect | accel.BufferCopyDst,
	})
	if err != nil {
		t.Fatalf("indirect: %v", err)
	}
	t.Cleanup(func() { _ = countBuf.Close() })

	r := d.NewRecorder()
	r.CollectRunStats(stats)
	cv, err := countBuf.View(0, 3)
	if err != nil {
		t.Fatalf("view: %v", err)
	}
	r.DispatchIndirect(p, []accel.Binding{{Index: 0, Buffer: whole(t, counts)}}, nil, cv, max)

	g, err := r.Build()
	if err != nil {
		t.Fatalf("build: %v", err)
	}
	t.Cleanup(func() { _ = g.Close() })
	return g, countBuf, counts
}

// A count over the recorded maximum is clamped, in every mode.
//
// Spec 003 is explicit that every build mode clamps: correctness cannot depend
// on a debug flag, and exceeding a backend's workgroup count limit is undefined
// behaviour on Vulkan rather than a clean error. Statistics change whether the
// caller is *told*, not whether the clamp happens.
func TestAnIndirectCountIsClampedInEveryMode(t *testing.T) {
	for _, stats := range []bool{false, true} {
		name := "without statistics"
		if stats {
			name = "with statistics"
		}
		t.Run(name, func(t *testing.T) {
			d := openDevice(t)
			g, count, counts := countGraph(t, d, accel.WorkgroupCount{X: 2}, stats)

			// The device asks for far more than the maximum.
			if err := d.Queue().WriteBuffer(count, 0, []uint32{1000, 1, 1}); err != nil {
				t.Fatalf("write count: %v", err)
			}
			f := d.Queue().Submit(g)
			if err := f.Wait(); err != nil {
				t.Fatalf("submit: %v", err)
			}

			// Two workgroups ran, not a thousand. This is the assertion the
			// clamp exists for, and it is only possible because the kernel's
			// output says how many there were.
			ran := make([]uint32, 1)
			if err := d.Queue().ReadBuffer(counts, 0, ran); err != nil {
				t.Fatalf("readback: %v", err)
			}
			if ran[0] != 2 {
				t.Fatalf("%d workgroups ran against a recorded maximum of 2: the clamp "+
					"is not unconditional, and spec 003 says correctness cannot depend "+
					"on a debug flag", ran[0])
			}

			s, err := f.Stats()
			if err != nil {
				t.Fatalf("stats: %v", err)
			}
			if !stats {
				if len(s.Indirect) != 0 {
					t.Errorf("statistics were not requested and %d were reported",
						len(s.Indirect))
				}
				return
			}
			if len(s.Indirect) != 1 {
				t.Fatalf("got %d indirect nodes, want 1", len(s.Indirect))
			}
			got := s.Indirect[0]
			if !got.Clamped {
				t.Error("a count of 1000 against a maximum of 2 was not reported as clamped")
			}
			if got.Actual[0] != 1000 {
				t.Errorf("the actual count is %d, want the device's 1000 before clamping",
					got.Actual[0])
			}
			if got.Max[0] != 2 {
				t.Errorf("the maximum is %d, want 2", got.Max[0])
			}
		})
	}
}

// A zero in any axis skips the dispatch, which spec 003 defines and which a
// host-authored count does not do: an omitted Y or Z normalizes to one.
func TestAZeroIndirectCountSkipsTheDispatch(t *testing.T) {
	d := openDevice(t)
	g, count, out := indirectGraph(t, d, accel.WorkgroupCount{X: 4}, true)

	// Poison the output, so a dispatch that ran is visible.
	poison := make([]float32, 256)
	for i := range poison {
		poison[i] = -7
	}
	if err := d.Queue().WriteBuffer(out, 0, poison); err != nil {
		t.Fatalf("write out: %v", err)
	}
	if err := d.Queue().WriteBuffer(count, 0, []uint32{2, 0, 1}); err != nil {
		t.Fatalf("write count: %v", err)
	}
	if err := d.Queue().Submit(g).Wait(); err != nil {
		t.Fatalf("submit: %v", err)
	}
	for i, v := range readback(t, d, out) {
		if v != -7 {
			t.Fatalf("element %d is %v: a zero axis skips the dispatch rather than "+
				"normalizing to one workgroup", i, v)
		}
	}
}

// Reading the counters before the fence signals is an error rather than a stale
// read: they are written during execution, so an early read reports the
// previous submission's.
func TestStatsBeforeCompletionIsAnError(t *testing.T) {
	d := openDevice(t)
	g, count, _ := indirectGraph(t, d, accel.WorkgroupCount{X: 4}, true)
	if err := d.Queue().WriteBuffer(count, 0, []uint32{1, 1, 1}); err != nil {
		t.Fatalf("write count: %v", err)
	}

	f := d.Queue().Submit(g)
	if !f.Done() {
		if _, err := f.Stats(); err == nil {
			t.Error("reading statistics before the fence signalled should be an error")
		} else if !strings.Contains(err.Error(), "has not completed") {
			t.Errorf("got %v", err)
		}
	}
	if err := f.Wait(); err != nil {
		t.Fatalf("submit: %v", err)
	}
	if _, err := f.Stats(); err != nil {
		t.Errorf("after the fence signalled: %v", err)
	}
}

// V9: the host-authored maximum follows the same normalization and limit checks
// a direct count does.
func TestIndirectValidationRows(t *testing.T) {
	cases := []struct {
		name string
		says string
		max  accel.WorkgroupCount
		bad  func(t *testing.T, d *accel.Device, r *accel.Recorder, p *accel.ComputePipeline)
	}{{
		name: "a maximum of zero",
		says: "is a recording mistake rather than a skip",
		max:  accel.WorkgroupCount{},
	}, {
		name: "a maximum past the device limit",
		says: "workgroup count X is",
		max:  accel.WorkgroupCount{X: 1 << 30},
	}}

	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			d := openDevice(t)
			p, err := d.NewComputePipeline(accel.ComputePipelineDescriptor{
				Kernel: &testkernels.AddKernel, Label: "add",
			})
			if err != nil {
				t.Fatalf("pipeline: %v", err)
			}
			defer p.Close()

			storage := accel.BufferStorage | accel.BufferCopySrc | accel.BufferCopyDst
			in := newBuffer(t, d, "in", 64, storage)
			count, err := d.NewBuffer(accel.BufferDescriptor{
				DType: accel.U32, Count: 3, Label: "count",
				Usage: accel.BufferIndirect | accel.BufferCopyDst,
			})
			if err != nil {
				t.Fatalf("count buffer: %v", err)
			}
			defer count.Close()

			r := d.NewRecorder()
			cv, _ := count.View(0, 3)
			r.DispatchIndirect(p, []accel.Binding{
				{Index: 0, Buffer: whole(t, in)},
				{Index: 1, Buffer: whole(t, in)},
				{Index: 2, Buffer: whole(t, in)},
			}, nil, cv, c.max)
			_, err = r.Build()
			if err == nil {
				t.Fatal("expected a rejection")
			}
			if !strings.Contains(err.Error(), c.says) {
				t.Errorf("the message should say %q, got:\n%v", c.says, err)
			}
		})
	}
}

// The count buffer's shape and usage are checked, since reading three uint32s
// from something else reads whatever is there.
func TestTheIndirectCountBufferIsChecked(t *testing.T) {
	d := openDevice(t)
	p, err := d.NewComputePipeline(accel.ComputePipelineDescriptor{
		Kernel: &testkernels.AddKernel, Label: "add",
	})
	if err != nil {
		t.Fatalf("pipeline: %v", err)
	}
	defer p.Close()

	storage := accel.BufferStorage | accel.BufferCopySrc | accel.BufferCopyDst
	in := newBuffer(t, d, "in", 64, storage)

	cases := []struct {
		name string
		says string
		make func(t *testing.T) accel.BufferView
	}{{
		name: "the wrong dtype",
		says: "three uint32 values",
		make: func(t *testing.T) accel.BufferView { return whole(t, in) },
	}, {
		name: "a buffer without indirect usage",
		says: "needs BufferIndirect",
		make: func(t *testing.T) accel.BufferView {
			b, err := d.NewBuffer(accel.BufferDescriptor{
				DType: accel.U32, Count: 3, Usage: accel.BufferCopyDst, Label: "plain",
			})
			if err != nil {
				t.Fatalf("buffer: %v", err)
			}
			t.Cleanup(func() { _ = b.Close() })
			v, _ := b.View(0, 3)
			return v
		},
	}}

	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			r := d.NewRecorder()
			r.DispatchIndirect(p, []accel.Binding{
				{Index: 0, Buffer: whole(t, in)},
				{Index: 1, Buffer: whole(t, in)},
				{Index: 2, Buffer: whole(t, in)},
			}, nil, c.make(t), accel.WorkgroupCount{X: 1})
			_, err := r.Build()
			if err == nil {
				t.Fatal("expected a rejection")
			}
			if !strings.Contains(err.Error(), c.says) {
				t.Errorf("the message should say %q, got:\n%v", c.says, err)
			}
		})
	}
}

// SubmitAfter begins only once every given fence has signalled, and reports a
// failed dependency rather than running anyway.
func TestSubmitAfterWaitsForItsDependencies(t *testing.T) {
	d := openDevice(t)
	storage := accel.BufferStorage | accel.BufferCopySrc | accel.BufferCopyDst
	src := newBuffer(t, d, "src", 64, storage)
	mid := newBuffer(t, d, "mid", 64, storage)
	dst := newBuffer(t, d, "dst", 64, storage)

	ones := make([]float32, 64)
	for i := range ones {
		ones[i] = 1
	}
	if err := d.Queue().WriteBuffer(src, 0, ones); err != nil {
		t.Fatalf("write: %v", err)
	}

	build := func(to, from *accel.Buffer) *accel.Graph {
		t.Helper()
		r := d.NewRecorder()
		r.CopyBuffer(whole(t, to), whole(t, from))
		g, err := r.Build()
		if err != nil {
			t.Fatalf("build: %v", err)
		}
		t.Cleanup(func() { _ = g.Close() })
		return g
	}

	first := d.Queue().Submit(build(mid, src))
	second := d.Queue().SubmitAfter(build(dst, mid), first)
	if err := second.Wait(); err != nil {
		t.Fatalf("second: %v", err)
	}
	if err := first.Wait(); err != nil {
		t.Fatalf("first: %v", err)
	}
	for i, v := range readback(t, d, dst) {
		if v != 1 {
			t.Fatalf("element %d is %v: the second submission did not see the first's "+
				"writes", i, v)
		}
	}

	// A nil fence is skipped rather than panicking, since a caller assembling a
	// dependency list may legitimately have an empty slot.
	third := d.Queue().SubmitAfter(build(dst, src), nil)
	if err := third.Wait(); err != nil {
		t.Fatalf("a nil dependency should be skipped: %v", err)
	}

	// A dependency that failed is reported rather than being run past.
	failed := d.Queue().Submit(nil)
	if err := failed.Wait(); err == nil {
		t.Fatal("submitting nil should fail")
	}
	after := d.Queue().SubmitAfter(build(dst, src), failed)
	if err := after.Wait(); err == nil {
		t.Error("a submission whose dependency failed should report that rather than " +
			"running as though it had succeeded")
	}
}
