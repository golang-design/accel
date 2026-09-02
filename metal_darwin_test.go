// Copyright 2026 The golang.design Initiative Authors.
// All rights reserved. Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

//go:build darwin

package accel_test

import (
	"errors"
	"golang.design/x/accel/kernelabi"
	"math"
	"os"
	"runtime"
	"strings"
	"testing"

	"golang.design/x/accel"
	"golang.design/x/accel/internal/conformance/numeq"
	"golang.design/x/accel/internal/kernels"
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
		storage := accel.BufferStorage | accel.BufferCopySrc | accel.BufferCopyDst

		p, err := d.NewComputePipeline(accel.ComputePipelineDescriptor{
			Kernel: &kernels.AddKernel, Label: "add",
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
		r.UploadToBuffer(whole(t, ba), a)
		r.UploadToBuffer(whole(t, bb), b)
		r.Dispatch(p, []accel.Binding{
			{Index: 0, Buffer: whole(t, ba)},
			{Index: 1, Buffer: whole(t, bb)},
			{Index: 2, Buffer: whole(t, out)},
		}, nil, accel.WorkgroupCount{X: (n + 63) / 64})

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
	params := kernels.ScaleParams{Factor: 2.5}

	run := func(t *testing.T, d *accel.Device) []float32 {
		t.Helper()
		storage := accel.BufferStorage | accel.BufferCopySrc | accel.BufferCopyDst
		p, err := d.NewComputePipeline(accel.ComputePipelineDescriptor{
			Kernel: &kernels.ElemScaleKernel, Label: "scale",
		})
		if err != nil {
			t.Fatalf("pipeline: %v", err)
		}
		defer p.Close()

		bin := newBuffer(t, d, "in", n, storage)
		out := newBuffer(t, d, "out", n, storage)
		r := d.NewRecorder()
		r.UploadToBuffer(whole(t, bin), in)
		r.Dispatch(p, []accel.Binding{
			{Index: 0, Buffer: whole(t, bin)},
			{Index: 1, Buffer: whole(t, out)},
		}, []accel.UniformValue{{Index: 0, Value: params}}, accel.WorkgroupCount{X: (n + 63) / 64})

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
	storage := accel.BufferStorage | accel.BufferCopySrc | accel.BufferCopyDst

	unlowered := accel.Kernel{
		Name:          "Unlowered",
		WorkgroupSize: accel.ID3{X: 1, Y: 1, Z: 1},
		Digest:        "test:unlowered",
		Generator:     kernelabi.Version,
		Bindings: []kernelabi.Binding{
			{Name: "out", DType: kernelabi.F32, Access: kernelabi.Write},
		},
		Flat: func(t accel.Thread, a kernelabi.Args) {
			kernelabi.Slice[float32](a, 0)[0] = 1
		},
	}

	// At pipeline creation, which is the first moment the kernel and the device
	// are in the same place. This used to be accepted here and refused at graph
	// build -- correct, and after a caller had uploaded their weights.
	_, err := d.NewComputePipeline(accel.ComputePipelineDescriptor{
		Kernel: &unlowered, Label: "unlowered",
	})
	if err == nil {
		t.Fatal("a pipeline was built for a kernel with no MSL artifact; the refusal " +
			"then has to come later, after the caller has paid for their uploads")
	}
	if !errors.Is(err, accel.ErrUnsupported) {
		t.Errorf("the refusal should be ErrUnsupported, which is what a caller branches "+
			"on to fall back to another device: %v", err)
	}
	if !strings.Contains(err.Error(), "Unlowered") {
		t.Errorf("the refusal should name the kernel that cannot be lowered: %v", err)
	}
	if !strings.Contains(err.Error(), "MSL") {
		t.Errorf("the refusal should name the target, per specs/004-kernel-authoring.md: %v", err)
	}

	// And the CPU backend builds the same pipeline, which is what makes the
	// refusal about this device rather than about the kernel.
	cpu, err := accel.OpenCPU(accel.CPUOptions{})
	if err != nil {
		t.Fatalf("OpenCPU: %v", err)
	}
	defer cpu.Close()
	cp, err := cpu.NewComputePipeline(accel.ComputePipelineDescriptor{
		Kernel: &unlowered, Label: "unlowered",
	})
	if err != nil {
		t.Fatalf("the CPU backend should run a kernel with no MSL: %v", err)
	}
	defer cp.Close()
	_ = n
	_ = storage
}

// The buffer round trip at every dtype moved, and it moved because it was not
// a parity test.
//
// It enumerated all seven dtypes and then checked Metal alone, which reads in a
// summary as coverage of the dtype surface and is not: a backend that stored
// bf16 at the wrong stride would round-trip through its own queue perfectly and
// disagree with the oracle. specs/062-backend-parity.md section 6.9 lists it as
// one of four tests in that shape. It is now parity_transfer_test.go's
// dtypeParityCases, which does the same round trip -- through a recorded copy,
// with an offset read -- on both backends and compares the bytes, and which the
// section 6.1 gate holds to all seven.

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
	metal := openMetal(t)
	// The CPU backend runs the same graph, because the clamp is a portable
	// rule and not Metal's: specs/003-command-graph.md makes it unconditional
	// on every backend. Checking it on one was checking that this backend is
	// self-consistent, which a backend that clamped to the wrong number would
	// also be. specs/062-backend-parity.md section 6.9.
	cpu, err := accel.OpenCPU(accel.CPUOptions{})
	if err != nil {
		t.Fatalf("OpenCPU: %v", err)
	}
	defer cpu.Close()

	const n = 256
	const wg = 64 // AddKernel's workgroup
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
			onCPU := indirectRun(t, cpu, n, tc.supply, max)
			onMetal := indirectRun(t, metal, n, tc.supply, max)

			// Byte for byte: a dispatch of ones over ones is exact addition,
			// so the two backends have no rounding to differ about, and the
			// only thing that can differ is how many workgroups ran.
			if r := numeq.Exact(onMetal.out, onCPU.out); !r.Equal {
				t.Fatalf("the two backends wrote different results: %v\n  a clamp is "+
					"portable, so a difference here is one backend running a different "+
					"number of workgroups", r)
			}
			if onCPU.stats.Clamped != onMetal.stats.Clamped {
				t.Errorf("Clamped is %v on the CPU backend and %v on Metal",
					onCPU.stats.Clamped, onMetal.stats.Clamped)
			}
			if onCPU.stats.Actual[0] != onMetal.stats.Actual[0] {
				t.Errorf("the reported actual count is %d on the CPU backend and %d on "+
					"Metal", onCPU.stats.Actual[0], onMetal.stats.Actual[0])
			}

			got, s := onMetal.out, onMetal.stats
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

// indirectResult is one backend's answer: what the dispatch wrote, and what the
// graph reported about the count it read.
type indirectResult struct {
	out   []float32
	stats accel.IndirectStats
}

// indirectRun records the indirect dispatch and returns both halves.
//
// Ones in, so every element the dispatch touched is 2 and every one it did not
// is 0. Counting them is how many workgroups ran, which is the only externally
// visible consequence of the clamp.
func indirectRun(t *testing.T, d *accel.Device, n int, supply uint32,
	max accel.WorkgroupCount) indirectResult {
	t.Helper()
	storage := accel.BufferStorage | accel.BufferCopySrc | accel.BufferCopyDst
	p, err := d.NewComputePipeline(accel.ComputePipelineDescriptor{
		Kernel: &kernels.AddKernel, Label: "add",
	})
	if err != nil {
		t.Fatalf("%v pipeline: %v", d.Info().Backend, err)
	}
	defer p.Close()

	in := newBuffer(t, d, "in", n, storage)
	out := newBuffer(t, d, "out", n, storage)
	count, err := d.NewBuffer(accel.BufferDescriptor{
		DType: accel.U32, Count: 3, Label: "count",
		Usage: accel.BufferIndirect | accel.BufferCopyDst | accel.BufferStorage,
	})
	if err != nil {
		t.Fatalf("%v count buffer: %v", d.Info().Backend, err)
	}
	defer count.Close()
	countView, err := count.View(0, 3)
	if err != nil {
		t.Fatalf("%v view: %v", d.Info().Backend, err)
	}

	r := d.NewRecorder()
	r.CollectRunStats(true)
	r.UploadToBuffer(countView, []uint32{supply, 1, 1})
	ones := make([]float32, n)
	for i := range ones {
		ones[i] = 1
	}
	r.UploadToBuffer(whole(t, in), ones)
	r.UploadToBuffer(whole(t, out), make([]float32, n))
	r.DispatchIndirect(p, []accel.Binding{
		{Index: 0, Buffer: whole(t, in)},
		{Index: 1, Buffer: whole(t, in)},
		{Index: 2, Buffer: whole(t, out)},
	}, nil, countView, max)

	g, err := r.Build()
	if err != nil {
		t.Fatalf("%v build: %v", d.Info().Backend, err)
	}
	defer g.Close()
	f := d.Queue().Submit(g)
	if err := f.Wait(); err != nil {
		t.Fatalf("%v submit: %v", d.Info().Backend, err)
	}
	stats, err := f.Stats()
	if err != nil {
		t.Fatalf("%v asked for run stats: %v", d.Info().Backend, err)
	}
	if len(stats.Indirect) != 1 {
		t.Fatalf("%v asked for run stats and reported %d indirect nodes",
			d.Info().Backend, len(stats.Indirect))
	}
	return indirectResult{out: readback(t, d, out), stats: stats.Indirect[0]}
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
	storage := accel.BufferStorage | accel.BufferCopySrc | accel.BufferCopyDst

	bind := func(s accel.Slot, label string, fill float32) *accel.Buffer {
		b := newBuffer(t, d, label, n, storage)
		vals := make([]float32, n)
		for i := range vals {
			vals[i] = fill
		}
		if err := d.Queue().WriteBuffer(b, 0, vals); err != nil {
			t.Fatalf("write %s: %v", label, err)
		}
		if err := w.g.Bind(accel.SlotBinding{Slot: s, Buffer: whole(t, b)}); err != nil {
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

// Closing a graph while its submission is in flight is *refused*, repeatedly,
// and closing after waiting succeeds.
//
// The refusal is the design and it is worth stating plainly, because the
// milestone criterion this answers is worded as "survives repeated early
// closes" and could be read as "closes and copes". It does not: a graph with a
// submission outstanding reports a LifetimeError, because a caller who closed
// it would be releasing resources the GPU is still reading and no error
// afterwards could tell them what happened.
//
// docs/conventions.md records why this is the sharp edge of the backend: a
// Metal completion handler runs after the enclosing autorelease pool has
// drained, so releasing an autoreleased object from one is a use-after-free
// that crashes inside objc_msgSend with a stack pointing nowhere. This backend
// avoids the trap by having no completion handler at all -- the fence polls
// -status and blocks on -waitUntilCompleted -- so the strongest form of "a
// handler releases nothing it did not retain" holds vacuously. This test is
// what would notice if that ever changed.
//
// Repeated, and under the race detector in CI, because a lifetime bug that
// happens once in twenty submissions is a lifetime bug.
func TestRepeatedEarlyCloseUnderMetal(t *testing.T) {
	d := openMetal(t)
	// Large enough that the submission is usually still running when Close is
	// called on the next line. "Usually" is why the loop counts how often it
	// caught one rather than requiring every iteration to.
	const n = 1 << 20
	storage := accel.BufferStorage | accel.BufferCopySrc | accel.BufferCopyDst

	p, err := d.NewComputePipeline(accel.ComputePipelineDescriptor{
		Kernel: &kernels.AddKernel, Label: "add",
	})
	if err != nil {
		t.Fatalf("pipeline: %v", err)
	}
	defer p.Close()

	refused := 0
	for range 20 {
		a := newBuffer(t, d, "a", n, storage)
		out := newBuffer(t, d, "out", n, storage)
		r := d.NewRecorder()
		r.UploadToBuffer(whole(t, a), make([]float32, n))
		// Several dispatches, so the submission is long enough that the close
		// on the next line usually lands while it is still running. One was
		// caught roughly once in twenty, which is a path that would rot.
		for range 16 {
			r.Dispatch(p, []accel.Binding{
				{Index: 0, Buffer: whole(t, a)},
				{Index: 1, Buffer: whole(t, out)},
				{Index: 2, Buffer: whole(t, out)},
			}, nil, accel.WorkgroupCount{X: n / 64})
		}
		g, err := r.Build()
		if err != nil {
			t.Fatalf("build: %v", err)
		}

		f := d.Queue().Submit(g)

		// Submit is asynchronous: it hands the work to the queue's serial
		// stream and returns, so the graph is not marked in flight until that
		// worker reaches it. Closing immediately raced the worker's *start*
		// rather than the GPU, and caught it once in twenty. Yielding until the
		// submission is observably running reaches the state this test is about
		// instead of sampling whether the scheduler got there first.
		for !f.Done() && !inFlight(g) {
			runtime.Gosched()
		}

		// Closing now, before waiting, must be refused rather than tolerated.
		// A submission can complete between the Submit above and this call, in
		// which case there is nothing in flight and the close is legitimate --
		// so the assertion is on the pair: either it refused, or the work had
		// already finished.
		early := g.Close()
		if early == nil {
			// The submission finished before Close was reached, so there was
			// nothing in flight and closing was legitimate. Nothing left to
			// wait for: the graph owns the fence and is now shut.
			continue
		}
		refused++
		var lifetime *accel.LifetimeError
		if !errors.As(early, &lifetime) {
			t.Fatalf("closing an in-flight graph failed with %v, want a LifetimeError "+
				"naming the reason", early)
		}
		if !strings.Contains(early.Error(), "in flight") {
			t.Errorf("the refusal should say the submission is in flight: %v", early)
		}

		if err := f.Wait(); err != nil {
			t.Fatalf("wait: %v", err)
		}
		// A refused close must not have left the graph half-shut: after the
		// work completes it closes normally.
		if err := g.Close(); err != nil {
			t.Fatalf("close after waiting: %v", err)
		}
	}

	// If nothing was ever in flight, this ran twenty times and proved nothing
	// about the path it exists for.
	if refused == 0 {
		t.Fatal("no close was refused in twenty attempts, so the in-flight path never ran")
	}
	t.Logf("%d of 20 closes caught a submission in flight", refused)
}

// inFlight reports whether a graph has a submission outstanding, by asking it
// to do something it refuses while one is.
//
// There is no accessor for this and there should not be: it is a race by
// construction for a caller, who can only know by holding the fence. A test
// spinning on it is reaching a state, not depending on one.
func inFlight(g *accel.Graph) bool {
	// Binding an empty batch: the call reaches the in-flight check and
	// changes nothing, so it reports the state without disturbing it.
	err := g.Bind()
	return err != nil && strings.Contains(err.Error(), "in flight")
}
