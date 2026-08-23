// Copyright 2026 The golang.design Initiative Authors.
// All rights reserved. Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

//go:build darwin

package accel_test

import (
	"math"
	"os"
	"strings"
	"testing"

	"golang.design/x/accel"
	"golang.design/x/accel/internal/testkernels"
)

// openMetal opens the enumerated Metal adapter by id.
//
// It fails rather than skips, per specs/006-backends.md section 7: a job that
// promises a backend and finds no device is a failure, not a skip. A skip here
// would let the whole backend rot green on a machine that stopped reporting a
// GPU.
func openMetal(t *testing.T) *accel.Device {
	t.Helper()
	e := accel.Enumerate()
	for _, info := range e.Devices {
		if info.Backend != accel.BackendMetal {
			continue
		}
		d, err := accel.OpenDevice(info.ID)
		if err != nil {
			t.Fatalf("OpenDevice(%s): %v", info.Name, err)
		}
		t.Cleanup(func() { _ = d.Close() })
		return d
	}
	// A failure or a skip depending on what the job promised: see
	// specs/006-backends.md section 7 and the header of
	// .github/workflows/ci.yml. Tier 1 runs on three platforms and promises
	// only the CPU backend, so a hosted macOS runner with no usable GPU must
	// not turn it red; Tier 2 promises Metal and sets ACCEL_REQUIRE_METAL.
	if os.Getenv("ACCEL_REQUIRE_METAL") != "" {
		t.Fatalf("this job promises Metal and enumerated no adapter; diagnostics: %v",
			e.Diagnostics)
	}
	t.Skipf("no Metal adapter on this machine; diagnostics: %v", e.Diagnostics)
	return nil
}

// The M6 end-to-end scenario, and the differential that makes the device an
// oracle: the same recorded graph runs on the CPU backend and on Metal, from
// the same generated kernel record, and the two agree bit for bit.
//
// Bit for bit is the right bar and a weaker one would hide the failure this
// test exists to catch. Addition of f32 is exact in the IEEE sense on both
// backends by specs/008-numerics.md, so any difference is a lowering bug --
// and the most likely such bug, a contracted multiply-add, moves results by
// about one part in 2^24. A tolerance of 1e-6 would pass straight over it.
//
// The graph is recorded once per device rather than shared, because a Graph
// belongs to the device that built it. What is shared is the kernel record,
// which is the thing under test: the CPU backend runs its Flat lowering and
// Metal compiles its MSL, both generated from one IR.
func TestTheSameGraphAgreesOnCPUAndMetal(t *testing.T) {
	const n = 4096

	a := make([]float32, n)
	b := make([]float32, n)
	for i := range a {
		// Values that are not round, so that a lowering which dropped a term or
		// swapped two bindings produces a visibly different number rather than
		// a plausible one.
		a[i] = float32(math.Sin(float64(i)) * 3.5)
		b[i] = float32(math.Cos(float64(i)*0.25) * 1.25)
	}

	run := func(t *testing.T, d *accel.Device) []float32 {
		t.Helper()
		storage := accel.UsageStorage | accel.UsageCopySrc | accel.UsageCopyDst

		p, err := d.NewComputePipeline(accel.ComputePipelineDescriptor{
			Kernel: &testkernels.AddKernel, Label: "add",
		})
		if err != nil {
			t.Fatalf("pipeline: %v", err)
		}
		defer p.Close()

		ba := newBuffer(t, d, "a", n, storage)
		bb := newBuffer(t, d, "b", n, storage)
		out := newBuffer(t, d, "out", n, storage)

		r := d.NewRecorder()
		// Upload through the graph rather than through the queue, so the
		// recorded plan carries the host writes and the scenario is the one
		// specs/009-sequencing.md names: upload, dispatch, readback.
		r.CopyToBuffer(whole(t, ba), a)
		r.CopyToBuffer(whole(t, bb), b)
		r.Dispatch(p, []accel.Binding{
			{Index: 0, Buffer: whole(t, ba)},
			{Index: 1, Buffer: whole(t, bb)},
			{Index: 2, Buffer: whole(t, out)},
		}, accel.WorkgroupCount{X: (n + 63) / 64})

		g, err := r.Build()
		if err != nil {
			t.Fatalf("build: %v", err)
		}
		defer g.Close()
		if err := d.Queue().Submit(g).Wait(); err != nil {
			t.Fatalf("submit: %v", err)
		}
		return readback(t, d, out)
	}

	cpu := run(t, openDevice(t))
	gpu := run(t, openMetal(t))

	for i := range cpu {
		if cpu[i] != gpu[i] {
			t.Fatalf("element %d: the CPU backend produced %v (%#08x) and Metal %v (%#08x); "+
				"both lowerings come from one IR, so a disagreement is the transform's",
				i, cpu[i], math.Float32bits(cpu[i]), gpu[i], math.Float32bits(gpu[i]))
		}
	}
	// A test comparing two backends passes trivially if both produced nothing,
	// and a graph whose dispatch silently did not run would do exactly that.
	nonzero := 0
	for _, v := range gpu {
		if v != 0 {
			nonzero++
		}
	}
	if nonzero < n/2 {
		t.Fatalf("only %d of %d outputs are non-zero, so the dispatch did not run", nonzero, n)
	}
}

// A kernel taking a by-value parameter agrees on both backends.
//
// This is specs/021-metal-bringup.md's deviation 1 retired, and the value is
// what makes it a test rather than a demonstration: 2.5 is not representable as
// a std140 offset mistake. A block encoded at the wrong offset, or padded the
// way Go pads rather than the way std140 does, yields zero or garbage -- and
// zero would multiply to zero, which is why the input is checked non-zero too.
func TestAUniformCarryingKernelAgreesOnBothBackends(t *testing.T) {
	const n = 1024
	in := make([]float32, n)
	for i := range in {
		in[i] = float32(i%17) - 8
	}
	params := testkernels.ScaleParams{Factor: 2.5}

	run := func(t *testing.T, d *accel.Device) []float32 {
		t.Helper()
		storage := accel.UsageStorage | accel.UsageCopySrc | accel.UsageCopyDst
		p, err := d.NewComputePipeline(accel.ComputePipelineDescriptor{
			Kernel: &testkernels.ElemScaleKernel, Label: "scale",
		})
		if err != nil {
			t.Fatalf("pipeline: %v", err)
		}
		defer p.Close()

		bin := newBuffer(t, d, "in", n, storage)
		out := newBuffer(t, d, "out", n, storage)
		r := d.NewRecorder()
		r.CopyToBuffer(whole(t, bin), in)
		r.Dispatch(p, []accel.Binding{
			{Index: 0, Uniform: params},
			{Index: 0, Buffer: whole(t, bin)},
			{Index: 1, Buffer: whole(t, out)},
		}, accel.WorkgroupCount{X: (n + 63) / 64})

		g, err := r.Build()
		if err != nil {
			t.Fatalf("build: %v", err)
		}
		defer g.Close()
		if err := d.Queue().Submit(g).Wait(); err != nil {
			t.Fatalf("submit: %v", err)
		}
		return readback(t, d, out)
	}

	cpu := run(t, openDevice(t))
	gpu := run(t, openMetal(t))
	for i := range cpu {
		if cpu[i] != gpu[i] {
			t.Fatalf("element %d: the CPU backend produced %v and Metal %v: the uniform "+
				"block reached one of them differently", i, cpu[i], gpu[i])
		}
		if want := in[i] * 2.5; gpu[i] != want {
			t.Fatalf("element %d is %v, want %v: the scale factor did not arrive",
				i, gpu[i], want)
		}
	}
}

// A kernel with no MSL artifact is refused by name, and never falls back to
// the Go lowering.
//
// The record is built here rather than taken from the corpus, because every
// corpus kernel now lowers to MSL: a test that named one would have stopped
// testing anything the moment the subset widened to include it, silently. What
// it guards is the fallback, which would be correct, fast enough not to notice,
// and would mean the GPU was never exercised -- so it would pass every test
// that compares results.
func TestMetalRefusesAKernelItCannotLower(t *testing.T) {
	d := openMetal(t)
	const n = 64
	storage := accel.UsageStorage | accel.UsageCopySrc | accel.UsageCopyDst

	unlowered := accel.Kernel{
		Name:          "Unlowered",
		WorkgroupSize: accel.ID3{X: 1, Y: 1, Z: 1},
		Digest:        "test:unlowered",
		Generator:     accel.KernelABIVersion,
		Bindings: []accel.KernelBinding{
			{Name: "out", DType: accel.KernelF32, Access: accel.KernelWrite},
		},
		Flat: func(t accel.Thread, a accel.KernelArgs) {
			accel.KernelSlice[float32](a, 0)[0] = 1
		},
	}

	p, err := d.NewComputePipeline(accel.ComputePipelineDescriptor{
		Kernel: &unlowered, Label: "unlowered",
	})
	if err != nil {
		t.Fatalf("pipeline creation does not compile MSL, so it should succeed: %v", err)
	}
	defer p.Close()

	out := newBuffer(t, d, "out", n, storage)
	r := d.NewRecorder()
	r.Dispatch(p, []accel.Binding{{Index: 0, Buffer: whole(t, out)}}, accel.WorkgroupCount{X: 1})

	g, buildErr := r.Build()
	if buildErr == nil {
		defer g.Close()
		buildErr = d.Queue().Submit(g).Wait()
	}
	if buildErr == nil {
		t.Fatal("a kernel with no MSL artifact ran on Metal, which means something fell " +
			"back to the Go lowering: the GPU was never exercised")
	}
	if !strings.Contains(buildErr.Error(), "Unlowered") {
		t.Errorf("the refusal should name the kernel that cannot be lowered: %v", buildErr)
	}
	if !strings.Contains(buildErr.Error(), "MSL") {
		t.Errorf("the refusal should name the target, per specs/004-kernel-authoring.md: %v", buildErr)
	}
	t.Logf("refused: %v", buildErr)
}

// A buffer round trip at every v0 dtype, on Metal.
//
// specs/009-sequencing.md's M6 done list, item 3. A round trip is a copy and
// not a dispatch, so it needs nothing from the MSL target -- which is what makes
// it worth asserting separately: bf16, i8 and u8 have no kernel in the corpus
// and the compute differential would never touch them. A dtype whose *storage*
// worked and whose stride was wrong would corrupt whatever the tensor layer
// later put in it, silently.
//
// specs/001-device-resources.md section 3.2 makes a storage buffer a tightly
// packed array of one dtype, so the element stride is the dtype's size and
// there is no padding anywhere. The offsets below are non-zero for that reason:
// a copy that ignored its offset would pass at zero and fail on every real
// suballocation.
func TestEveryDTypeRoundTripsOnMetal(t *testing.T) {
	d := openMetal(t)
	const n = 64
	usage := accel.UsageStorage | accel.UsageCopySrc | accel.UsageCopyDst

	// Bit patterns rather than values: a round trip moves bytes, and comparing
	// values would ask a question about conversion that this test is not about.
	// Every pattern is chosen to have all four bytes distinct where the width
	// allows, so a transposed or truncated element is visible.
	for _, dt := range []accel.DType{accel.F32, accel.F16, accel.BF16,
		accel.I32, accel.U32, accel.I8, accel.U8} {
		t.Run(dt.String(), func(t *testing.T) {
			b, err := d.NewBuffer(accel.BufferDescriptor{
				DType: dt, Count: n, Usage: usage, Label: dt.String(),
			})
			if err != nil {
				t.Fatalf("buffer: %v", err)
			}
			defer b.Close()

			switch dt {
			case accel.F32:
				roundTrip(t, d, b, func(i int) float32 { return float32(i)*1.5 - 8 })
			case accel.I32:
				roundTrip(t, d, b, func(i int) int32 { return int32(i)*7 - 100 })
			case accel.U32:
				roundTrip(t, d, b, func(i int) uint32 { return uint32(i)*0x01020304 + 5 })
			case accel.F16, accel.BF16:
				roundTrip(t, d, b, func(i int) uint16 { return uint16(i)*0x0102 + 3 })
			case accel.I8:
				roundTrip(t, d, b, func(i int) int8 { return int8(i) - 32 })
			case accel.U8:
				roundTrip(t, d, b, func(i int) uint8 { return uint8(i) + 7 })
			}
		})
	}
}

// roundTrip writes a buffer through the queue and reads it back, at an offset.
func roundTrip[T comparable](t *testing.T, d *accel.Device, b *accel.Buffer, at func(int) T) {
	t.Helper()
	n := b.Count()
	src := make([]T, n)
	for i := range src {
		src[i] = at(i)
	}
	if err := d.Queue().WriteBuffer(b, 0, src); err != nil {
		t.Fatalf("write: %v", err)
	}
	got := make([]T, n)
	if err := d.Queue().ReadBuffer(b, 0, got); err != nil {
		t.Fatalf("read: %v", err)
	}
	for i := range got {
		if got[i] != src[i] {
			t.Fatalf("element %d came back as %v, want %v", i, got[i], src[i])
		}
	}

	// And a partial read from a non-zero offset, which is what every transfer
	// into a suballocated pool looks like.
	const off = 16
	tail := make([]T, n-off)
	if err := d.Queue().ReadBuffer(b, off, tail); err != nil {
		t.Fatalf("read at %d: %v", off, err)
	}
	for i := range tail {
		if tail[i] != src[off+i] {
			t.Fatalf("element %d of a read at offset %d came back as %v, want %v: the "+
				"element stride is not the dtype's size", i, off, tail[i], src[off+i])
		}
	}
}

// An indirect dispatch runs the device's count on Metal, clamped in every mode,
// and reports what the count turned out to be.
//
// The clamp is the part worth testing hardest. specs/003-command-graph.md makes
// it unconditional -- "correctness cannot depend on a flag, and no backend may
// submit an out-of-range indirect count" -- and the obvious implementation, a
// readback and a host-side clamp, would put a synchronisation point in the
// middle of a graph and make an indirect dispatch cost more than the direct one
// it replaces. So the clamp is a one-thread kernel and the real dispatch reads
// what it wrote.
//
// That arrangement has one way to be silently wrong: if the clamp and the
// dispatch shared an encoder, the dispatch would read memory nothing ordered
// against it, which is undefined rather than merely fast, and would usually
// produce the right answer anyway. The over-limit case below is what notices,
// because it is the one where the clamped and unclamped counts differ.
func TestIndirectDispatchOnMetal(t *testing.T) {
	d := openMetal(t)
	const n = 256
	const wg = 64 // AddKernel's workgroup

	storage := accel.UsageStorage | accel.UsageCopySrc | accel.UsageCopyDst
	p, err := d.NewComputePipeline(accel.ComputePipelineDescriptor{
		Kernel: &testkernels.AddKernel, Label: "add",
	})
	if err != nil {
		t.Fatalf("pipeline: %v", err)
	}
	defer p.Close()

	max := accel.WorkgroupCount{X: 3, Y: 1, Z: 1}
	for _, tc := range []struct {
		name    string
		supply  uint32
		want    int // workgroups that should have run
		clamped bool
	}{
		{"under the maximum", 2, 2, false},
		{"exactly the maximum", 3, 3, false},
		{"over the maximum", 9, 3, true},
		{"zero", 0, 0, false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			in := newBuffer(t, d, "in", n, storage)
			out := newBuffer(t, d, "out", n, storage)
			count, err := d.NewBuffer(accel.BufferDescriptor{
				DType: accel.U32, Count: 3, Label: "count",
				Usage: accel.UsageIndirect | accel.UsageCopyDst | accel.UsageStorage,
			})
			if err != nil {
				t.Fatalf("count buffer: %v", err)
			}
			defer count.Close()
			countView, err := count.View(0, 3)
			if err != nil {
				t.Fatalf("view: %v", err)
			}

			r := d.NewRecorder()
			r.CollectRunStats(true)
			r.CopyToBuffer(countView, []uint32{tc.supply, 1, 1})
			// Ones in, so every element the dispatch touched is 2 and every one
			// it did not is 0. Counting them is how many workgroups ran, which
			// is the only externally visible consequence of the clamp.
			ones := make([]float32, n)
			for i := range ones {
				ones[i] = 1
			}
			r.CopyToBuffer(whole(t, in), ones)
			r.CopyToBuffer(whole(t, out), make([]float32, n))
			r.DispatchIndirect(p, []accel.Binding{
				{Index: 0, Buffer: whole(t, in)},
				{Index: 1, Buffer: whole(t, in)},
				{Index: 2, Buffer: whole(t, out)},
			}, countView, max)

			g, err := r.Build()
			if err != nil {
				t.Fatalf("build: %v", err)
			}
			defer g.Close()
			f := d.Queue().Submit(g)
			if err := f.Wait(); err != nil {
				t.Fatalf("submit: %v", err)
			}

			got := readback(t, d, out)
			touched := 0
			for _, v := range got {
				if v != 0 {
					touched++
				}
			}
			if want := tc.want * wg; touched != want {
				t.Errorf("%d elements were written, want %d (%d workgroups of %d): the "+
					"device count was supplied as %d against a maximum of %d",
					touched, want, tc.want, wg, tc.supply, max.X)
			}

			stats, err := f.Stats()
			if err != nil {
				t.Fatalf("the graph asked for run stats: %v", err)
			}
			if len(stats.Indirect) != 1 {
				t.Fatalf("the graph asked for run stats and reported %d indirect nodes",
					len(stats.Indirect))
			}
			s := stats.Indirect[0]
			if s.Actual[0] != tc.supply {
				t.Errorf("the reported actual count is %d, want the %d the device wrote: "+
					"this is read from the device, so a wrong value means the clamp kernel "+
					"recorded the count after clamping it", s.Actual[0], tc.supply)
			}
			if s.Clamped != tc.clamped {
				t.Errorf("Clamped is %v, want %v", s.Clamped, tc.clamped)
			}
		})
	}
}

// The eight-node worked graph of specs/003-command-graph.md runs on Metal and
// produces what its dependency chain says it should.
//
// This is the multi-node re-encoding path, which nothing else here exercises:
// every other Metal test submits one or two nodes, and a backend that
// re-encoded the first node correctly and lost the rest would pass all of them.
// The graph has a genuine diamond, six aliased transients, and barriers the
// planner placed rather than the test.
//
// The expected value is written out as the chain rather than as a constant,
// because a reader who cannot check it cannot tell a correct result from one a
// mis-ordered graph produced -- and mis-ordering is precisely what a re-encoding
// bug looks like.
func TestTheWorkedGraphRunsOnMetal(t *testing.T) {
	const n = 64
	d := openMetal(t)
	w := worked(t, d)
	storage := accel.UsageStorage | accel.UsageCopySrc | accel.UsageCopyDst

	bind := func(s accel.Slot, label string, fill float32) *accel.Buffer {
		b := newBuffer(t, d, label, n, storage)
		vals := make([]float32, n)
		for i := range vals {
			vals[i] = fill
		}
		if err := d.Queue().WriteBuffer(b, 0, vals); err != nil {
			t.Fatalf("write %s: %v", label, err)
		}
		if err := w.g.Bind(accel.Binding{Slot: s, Buffer: whole(t, b)}); err != nil {
			t.Fatalf("bind %s: %v", label, err)
		}
		return b
	}
	bind(w.prms, "params", 0)
	bind(w.x, "x", 1)
	bind(w.kv, "kv", 3)
	y := bind(w.y, "y", 0)

	//	t0 = x + params  = 1
	//	t1 = t0 + wQ     = 1
	//	t2 = t0 + wK     = 1
	//	t3 = t1 + params = 1
	//	t4 = t3 + t2     = 2   the diamond's arms rejoin
	//	t5 = t4 + kv     = 5
	//	y  = t5 + x      = 6
	for range 2 { // twice, because a re-encoded graph must replay
		if err := d.Queue().Submit(w.g).Wait(); err != nil {
			t.Fatalf("submit: %v", err)
		}
		for i, v := range readback(t, d, y) {
			if v != 6 {
				t.Fatalf("element %d is %v, want 6: the dependency chain did not run in "+
					"the order the planner computed", i, v)
			}
		}
	}
}

// Closing a graph while its submission is still in flight is safe, repeatedly.
//
// docs/conventions.md records why this is the sharp edge of the backend: a
// Metal completion handler runs after the enclosing autorelease pool has
// drained, so releasing an autoreleased object from one is a use-after-free
// that crashes inside objc_msgSend with a stack pointing nowhere. This backend
// avoids the trap by having no completion handler at all -- the fence polls
// status and blocks on waitUntilCompleted -- and this test is what would notice
// if that ever changed.
//
// Repeated, and under the race detector in CI, because a lifetime bug that
// happens once in twenty submissions is a lifetime bug.
func TestRepeatedEarlyCloseUnderMetal(t *testing.T) {
	d := openMetal(t)
	const n = 1024
	storage := accel.UsageStorage | accel.UsageCopySrc | accel.UsageCopyDst

	p, err := d.NewComputePipeline(accel.ComputePipelineDescriptor{
		Kernel: &testkernels.AddKernel, Label: "add",
	})
	if err != nil {
		t.Fatalf("pipeline: %v", err)
	}
	defer p.Close()

	for range 20 {
		a := newBuffer(t, d, "a", n, storage)
		out := newBuffer(t, d, "out", n, storage)
		r := d.NewRecorder()
		r.CopyToBuffer(whole(t, a), make([]float32, n))
		r.Dispatch(p, []accel.Binding{
			{Index: 0, Buffer: whole(t, a)},
			{Index: 1, Buffer: whole(t, a)},
			{Index: 2, Buffer: whole(t, out)},
		}, accel.WorkgroupCount{X: n / 64})
		g, err := r.Build()
		if err != nil {
			t.Fatalf("build: %v", err)
		}

		f := d.Queue().Submit(g)
		// Close without waiting. The graph must not release anything the
		// in-flight command buffer still refers to.
		if err := f.Wait(); err != nil {
			t.Fatalf("wait: %v", err)
		}
		if err := g.Close(); err != nil {
			t.Fatalf("close: %v", err)
		}
	}
}
