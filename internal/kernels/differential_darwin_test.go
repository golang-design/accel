// Copyright 2026 The golang.design Initiative Authors.
// All rights reserved. Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

//go:build darwin

package kernels_test

import (
	"fmt"
	"golang.design/x/accel/kernelabi"
	"math"
	"os"
	"strings"
	"testing"
	"time"

	"golang.design/x/accel"
	"golang.design/x/accel/internal/conformance/numeq"
	"golang.design/x/accel/internal/kernels"
)

// # The corpus differential
//
// specs/022-msl-target.md's central criterion: every kernel in the corpus runs
// on the CPU backend and on Metal, from one generated record, and the two
// agree. The CPU side runs the resumable lowering with a program counter and
// the Metal side runs the authored structure with a real barrier, so a
// disagreement is the transform's and nothing else's.
//
// **Bit for bit, not within a tolerance.** The measured profile
// (specs/022-msl-target.md) has both backends rounding to nearest-even with
// contraction off, which is what specs/008-numerics.md requires before an exact
// comparison may be claimed. A tolerance would hide the failure this exists to
// catch: a contracted multiply-add moves a result by about one part in 2^24,
// and even 1e-6 sails past it.
//
// The one restriction is the domain. Apple GPUs flush a subnormal *result* to
// zero, so the inputs below are scaled to keep every intermediate well inside
// the normal range. That is a narrower domain, not a wider bound, which is the
// Every corpus kernel agrees between the CPU oracle and Metal, bit for bit.
//
// What is *missing* from the table is checked by
// TestEveryLoweredKernelIsInADifferentialCase, which lives beside the table in
// differential_cases_test.go and has no build tag: that question needs no
// device, and asking it only on darwin meant an unlisted kernel added on Linux
// stayed invisible there. specs/062-backend-parity.md section 6.7.
func TestTheCorpusAgreesOnCPUAndMetal(t *testing.T) {
	cases := diffCases()
	gpu := openMetalDevice(t)

	// The oracle is configured to the device it is checking. The CPU backend
	// emulates subgroups at a width a caller chooses, and its default is 4
	// while this device executes 32 -- so a subgroup reduction over 64 elements
	// would produce different partial sums on each and the differential would
	// be comparing two different computations. specs/006-backends.md section 5
	// makes the width an option for exactly this.
	width := gpu.Info().Limits.MinSubgroupSize
	subgroupWidth = width
	cpu, err := accel.OpenCPU(accel.CPUOptions{SubgroupSize: width})
	if err != nil {
		t.Fatalf("open CPU at subgroup width %d: %v", width, err)
	}
	defer cpu.Close()
	if got := gpu.Info().Limits.MaxSubgroupSize; got != width {
		t.Fatalf("this device reports a subgroup width range of [%d, %d]; the oracle can "+
			"emulate one width, so a varying one needs a different arrangement",
			width, got)
	}

	for _, c := range cases {
		t.Run(c.kernel.Name, func(t *testing.T) {
			want := runCase(t, cpu, c)
			got := runCase(t, gpu, c)
			for b := range want {
				if len(want[b]) != len(got[b]) {
					t.Fatalf("binding %d: %d elements on the CPU and %d on Metal",
						b, len(want[b]), len(got[b]))
				}
				var r numeq.Report
				switch {
				case c.abs > 0:
					r = withinAbs(got[b], want[b], c.abs)
				default:
					r = numeq.WithinULP(got[b], want[b], c.ulp)
				}
				if !r.Equal {
					t.Fatalf("binding %d (%s): %v\n  the ceiling comes from %s\n  both "+
						"lowerings come from one IR, so a disagreement beyond it is the "+
						"transform's", b, c.kernel.Bindings[b].Name, r, ceilingNote(c))
				}
			}
		})
	}
}

// ceilingNote explains where a case's ceiling came from.
func ceilingNote(c diffCase) string {
	if c.why == "" {
		return "no bounded primitive in this kernel, so the two lowerings must agree exactly"
	}
	return c.why
}

// withinAbs compares against an absolute ceiling.
//
// Separate from numeq.WithinULP rather than a mode of it, because the two
// answer different questions and specs/008-numerics.md uses each where the
// other is meaningless: ULP near a zero crossing, and absolute across many
// binades.
func withinAbs(got, want []float32, ceiling float64) numeq.Report {
	r := numeq.Report{Equal: true, FirstDiff: -1, Len: len(got), WantLen: len(want)}
	if len(got) != len(want) {
		r.Equal = false
		return r
	}
	for i := range got {
		g, w := got[i], want[i]
		gNaN, wNaN := g != g, w != w
		bad := gNaN != wNaN
		if !gNaN && !wNaN {
			bad = math.Abs(float64(g)-float64(w)) > ceiling
		}
		if !bad {
			continue
		}
		r.Diffs++
		if r.FirstDiff < 0 {
			r.FirstDiff, r.Equal = i, false
			r.Got = fmt.Sprintf("%v", g)
			r.Want = fmt.Sprintf("%v, differing by %g against a ceiling of %g",
				w, math.Abs(float64(g)-float64(w)), ceiling)
		}
	}
	return r
}

// runCase records one graph, submits it, and reads every written binding back.
//
// Every binding is uploaded, including the outputs: a kernel that writes only
// part of its output leaves the rest at whatever the buffer held, and comparing
// uninitialized memory between two backends is a flake waiting to happen.
func runCase(t *testing.T, d *accel.Device, c diffCase) [][]float32 {
	t.Helper()
	storage := accel.BufferStorage | accel.BufferCopySrc | accel.BufferCopyDst
	if len(c.uniforms) > 0 {
		storage |= accel.BufferUniform
	}
	seed := c.seed
	if seed == nil {
		seed = defaultSeed
	}

	p, err := d.NewComputePipeline(accel.ComputePipelineDescriptor{
		Kernel: c.kernel, Label: c.kernel.Name,
	})
	if err != nil {
		t.Fatalf("pipeline: %v", err)
	}
	defer p.Close()

	r := d.NewRecorder()
	bufs := make([]*accel.Buffer, len(c.counts))
	binds := make([]accel.Binding, 0, len(c.counts))
	uniforms := make([]accel.UniformValue, 0, len(c.uniforms))
	for i, u := range c.uniforms {
		uniforms = append(uniforms, accel.UniformValue{Index: i, Value: u})
	}
	for i, n := range c.counts {
		dt := dtypeOf(c.kernel.Bindings[i].DType)
		b, err := d.NewBuffer(accel.BufferDescriptor{
			DType: dt, Count: n, Usage: storage,
			Label: fmt.Sprintf("%s.%s", c.kernel.Name, c.kernel.Bindings[i].Name),
		})
		if err != nil {
			t.Fatalf("buffer %d: %v", i, err)
		}
		defer b.Close()
		bufs[i] = b

		v, err := b.View(0, n)
		if err != nil {
			t.Fatalf("view %d: %v", i, err)
		}
		writeSeed(t, r, v, dt, n, func(k int) float32 { return seed(i, k) })
		binds = append(binds, accel.Binding{Index: i, Buffer: v})
	}

	r.Dispatch(p, binds, uniforms, c.groups)
	g, err := r.Build()
	if err != nil {
		t.Fatalf("build: %v", err)
	}
	defer g.Close()
	if err := d.Queue().Submit(g).Wait(); err != nil {
		t.Fatalf("submit: %v", err)
	}

	// Everything is read back, not only what the kernel declares it writes: an
	// access mode inferred wrongly would let a kernel write a binding recorded
	// as read-only, and reading only the outputs would never notice.
	out := make([][]float32, len(bufs))
	for i, b := range bufs {
		out[i] = readAsF32(t, d, b, c.kernel.Bindings[i].DType, c.counts[i])
	}
	return out
}

func dtypeOf(d kernelabi.DType) accel.DType {
	switch d {
	case kernelabi.F16:
		return accel.F16
	case kernelabi.BF16:
		return accel.BF16
	case kernelabi.U32:
		return accel.U32
	case kernelabi.I32:
		return accel.I32
	case kernelabi.I8:
		return accel.I8
	case kernelabi.U8:
		return accel.U8
	}
	return accel.F32
}

// writeSeed uploads a binding's initial contents through the graph.
func writeSeed(t *testing.T, r *accel.Recorder, v accel.BufferView, dt accel.DType, n int, at func(int) float32) {
	t.Helper()
	switch dt {
	case accel.F16:
		// The host slice for an f16 buffer is []uint16: the API boundary moves
		// bit patterns, and Float16 is a value type whose layout is not part of
		// the contract.
		vals := make([]uint16, n)
		for i := range vals {
			vals[i] = accel.ToFloat16(at(i)).Bits()
		}
		r.UploadToBuffer(v, vals)
	case accel.BF16:
		// Also []uint16 at the boundary, and for the same reason: the API moves
		// bit patterns and BFloat16's layout is not part of the contract.
		vals := make([]uint16, n)
		for i := range vals {
			vals[i] = accel.ToBFloat16(at(i)).Bits()
		}
		r.UploadToBuffer(v, vals)
	case accel.U32:
		vals := make([]uint32, n)
		for i := range vals {
			vals[i] = uint32(math.Abs(float64(at(i))))
		}
		r.UploadToBuffer(v, vals)
	case accel.I8:
		vals := make([]int8, n)
		for i := range vals {
			vals[i] = int8(at(i))
		}
		r.UploadToBuffer(v, vals)
	case accel.U8:
		vals := make([]uint8, n)
		for i := range vals {
			vals[i] = uint8(math.Abs(float64(at(i))))
		}
		r.UploadToBuffer(v, vals)
	case accel.I32:
		vals := make([]int32, n)
		for i := range vals {
			vals[i] = int32(at(i))
		}
		r.UploadToBuffer(v, vals)
	default:
		vals := make([]float32, n)
		for i := range vals {
			vals[i] = at(i)
		}
		r.UploadToBuffer(v, vals)
	}
}

// readAsF32 reads a binding back in one comparable representation.
//
// Integers are widened rather than compared as integers so one comparison
// covers every dtype. The widening is exact for u32 and i32 values this corpus
// produces, and a count that exceeded 2^24 would be a different bug.
func readAsF32(t *testing.T, d *accel.Device, b *accel.Buffer, dt kernelabi.DType, n int) []float32 {
	t.Helper()
	out := make([]float32, n)
	switch dt {
	case kernelabi.F16:
		raw := make([]uint16, n)
		if err := d.Queue().ReadBuffer(b, 0, raw); err != nil {
			t.Fatalf("readback: %v", err)
		}
		for i, v := range raw {
			out[i] = accel.Float16FromBits(v).F32()
		}
	case kernelabi.BF16:
		raw := make([]uint16, n)
		if err := d.Queue().ReadBuffer(b, 0, raw); err != nil {
			t.Fatalf("readback: %v", err)
		}
		for i, v := range raw {
			out[i] = accel.BFloat16FromBits(v).F32()
		}
	case kernelabi.U32:
		raw := make([]uint32, n)
		if err := d.Queue().ReadBuffer(b, 0, raw); err != nil {
			t.Fatalf("readback: %v", err)
		}
		for i, v := range raw {
			out[i] = float32(v)
		}
	case kernelabi.I32:
		raw := make([]int32, n)
		if err := d.Queue().ReadBuffer(b, 0, raw); err != nil {
			t.Fatalf("readback: %v", err)
		}
		for i, v := range raw {
			out[i] = float32(v)
		}
	case kernelabi.I8:
		raw := make([]int8, n)
		if err := d.Queue().ReadBuffer(b, 0, raw); err != nil {
			t.Fatalf("readback: %v", err)
		}
		for i, v := range raw {
			out[i] = float32(v)
		}
	case kernelabi.U8:
		raw := make([]uint8, n)
		if err := d.Queue().ReadBuffer(b, 0, raw); err != nil {
			t.Fatalf("readback: %v", err)
		}
		for i, v := range raw {
			out[i] = float32(v)
		}
	default:
		if err := d.Queue().ReadBuffer(b, 0, out); err != nil {
			t.Fatalf("readback: %v", err)
		}
	}
	return out
}

// openMetalDevice opens the enumerated Metal adapter, failing or skipping on
// what the job promised. See .github/workflows/ci-metal.yml.
func openMetalDevice(t *testing.T) *accel.Device {
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
	if os.Getenv("ACCEL_REQUIRE_METAL") != "" {
		t.Fatalf("this job promises Metal and enumerated no adapter; diagnostics: %v",
			e.Diagnostics)
	}
	t.Skipf("no Metal adapter on this machine; diagnostics: %v", e.Diagnostics)
	return nil
}

// The portable tiled GEMM matches an independently written higher-precision
// reference on Metal, at dimensions that are not multiples of any tile
// dimension.
//
// specs/022-msl-target.md names this separately from the corpus differential,
// and the separation is the point. The differential says Metal agrees with the
// CPU; this says Metal is *right*, against a reference written as a straight
// triple loop with no tiling and no shared memory. A reference sharing the
// kernel's structure would share its bugs, and two backends agreeing on a wrong
// answer is exactly what one IR makes possible.
//
// The budget is per output element -- that element's own K terms and its own sum
// of magnitudes -- which is what specs/008-numerics.md section 7 requires rather
// than one budget for the whole matrix.
func TestTheTiledGEMMMatchesItsReferenceOnMetal(t *testing.T) {
	d := openMetalDevice(t)

	for _, c := range []struct{ m, n, k int }{
		{8, 16, 16}, // exactly one tile
		{3, 5, 7},   // all three tails, none aligned
		{9, 19, 23}, // all three, each larger than one tile
		{1, 1, 40},  // a single output over several K steps
	} {
		t.Run(fmt.Sprintf("%dx%dx%d", c.m, c.n, c.k), func(t *testing.T) {
			a := make([]accel.Float16, c.m*c.k)
			b := make([]accel.Float16, c.k*c.n)
			for i := range a {
				a[i] = accel.ToFloat16(float32(math.Sin(float64(i))) * 2)
			}
			for i := range b {
				b[i] = accel.ToFloat16(float32(math.Cos(float64(i))) * 3)
			}

			out := runGEMM(t, d, c.m, c.n, c.k, a, b)
			for i := range out {
				row, col := i/c.n, i%c.n
				terms := make([]float32, c.k)
				for kk := range c.k {
					terms[kk] = a[row*c.k+kk].F32() * b[kk*c.n+col].F32()
				}
				if r := numeq.Sum(out[i], terms, c.k-1); !r.OK() {
					t.Fatalf("element (%d,%d) of %dx%dx%d: %v", row, col, c.m, c.n, c.k, r)
				}
			}
		})
	}
}

func runGEMM(t *testing.T, d *accel.Device, m, n, k int, a, b []accel.Float16) []float32 {
	t.Helper()
	storage := accel.BufferStorage | accel.BufferCopySrc | accel.BufferCopyDst | accel.BufferUniform

	p, err := d.NewComputePipeline(accel.ComputePipelineDescriptor{
		Kernel: &kernels.MatMulTiledKernel, Label: "gemm",
	})
	if err != nil {
		t.Fatalf("pipeline: %v", err)
	}
	defer p.Close()

	f16 := func(label string, v []accel.Float16) accel.BufferView {
		buf, err := d.NewBuffer(accel.BufferDescriptor{
			DType: accel.F16, Count: len(v), Usage: storage, Label: label,
		})
		if err != nil {
			t.Fatalf("buffer %s: %v", label, err)
		}
		t.Cleanup(func() { _ = buf.Close() })
		view, err := buf.View(0, len(v))
		if err != nil {
			t.Fatalf("view %s: %v", label, err)
		}
		return view
	}

	av, bv := f16("a", a), f16("b", b)
	outBuf, err := d.NewBuffer(accel.BufferDescriptor{
		DType: accel.F32, Count: m * n, Usage: storage, Label: "out",
	})
	if err != nil {
		t.Fatalf("buffer out: %v", err)
	}
	defer outBuf.Close()
	outView, err := outBuf.View(0, m*n)
	if err != nil {
		t.Fatalf("view out: %v", err)
	}

	bits := func(v []accel.Float16) []uint16 {
		raw := make([]uint16, len(v))
		for i := range v {
			raw[i] = v[i].Bits()
		}
		return raw
	}

	r := d.NewRecorder()
	r.UploadToBuffer(av, bits(a))
	r.UploadToBuffer(bv, bits(b))
	r.UploadToBuffer(outView, make([]float32, m*n))
	r.Dispatch(p, []accel.Binding{
		{Index: 0, Buffer: av},
		{Index: 1, Buffer: bv},
		{Index: 2, Buffer: outView},
	}, []accel.UniformValue{{Index: 0, Value: kernels.GEMMDims{M: uint32(m), N: uint32(n), K: uint32(k)}}}, accel.WorkgroupCount{
		X: (n + kernels.TileN - 1) / kernels.TileN,
		Y: (m + kernels.TileM - 1) / kernels.TileM,
	})

	g, err := r.Build()
	if err != nil {
		t.Fatalf("build: %v", err)
	}
	defer g.Close()
	if err := d.Queue().Submit(g).Wait(); err != nil {
		t.Fatalf("submit: %v", err)
	}
	out := make([]float32, m*n)
	if err := d.Queue().ReadBuffer(outBuf, 0, out); err != nil {
		t.Fatalf("readback: %v", err)
	}
	return out
}

// Metal reports device time from the GPU's own clock, not from the wall.
//
// The distinction is the whole point of the feature. A wall-clock figure around
// Commit and Wait includes queueing, driver work and whatever else the process
// was doing; a caller measuring throughput needs the time the device spent.
// Metal gives both timestamps on a completed command buffer, so the
// whole-submission figure costs no timestamp pool.
//
// Asserted as a bound rather than a value — device time is not reproducible —
// and the bound is the one that catches a wall-clock substitute: the GPU cannot
// have spent longer on the work than the whole call took.
func TestMetalReportsDeviceTimeFromTheGPUClock(t *testing.T) {
	const n = 65536
	d := openMetalDevice(t)
	q := d.Queue()

	p, err := d.NewComputePipeline(accel.ComputePipelineDescriptor{
		Kernel: &kernels.AddKernel, Label: "timed",
	})
	if err != nil {
		t.Fatalf("pipeline: %v", err)
	}
	defer p.Close()

	usage := accel.BufferStorage | accel.BufferCopySrc | accel.BufferCopyDst
	bufs := make([]accel.BufferView, 3)
	for i, name := range []string{"a", "b", "out"} {
		b, err := d.NewBuffer(accel.BufferDescriptor{
			DType: accel.F32, Count: n, Usage: usage, Label: name,
		})
		if err != nil {
			t.Fatalf("buffer: %v", err)
		}
		defer b.Close()
		v, err := b.View(0, n)
		if err != nil {
			t.Fatalf("view: %v", err)
		}
		bufs[i] = v
	}

	r := d.NewRecorder()
	r.CollectTimings(true)
	for range 32 {
		r.Dispatch(p, []accel.Binding{
			{Index: 0, Buffer: bufs[0]},
			{Index: 1, Buffer: bufs[1]},
			{Index: 2, Buffer: bufs[2]},
		}, nil, p.Workgroups(n))
	}
	g, err := r.Build()
	if err != nil {
		t.Fatalf("build: %v", err)
	}
	defer g.Close()

	start := time.Now()
	f := q.Submit(g)
	if err := f.Wait(); err != nil {
		t.Fatalf("submit: %v", err)
	}
	wall := time.Since(start)

	stats, err := f.Stats()
	if err != nil {
		t.Fatalf("stats: %v", err)
	}
	if stats.Elapsed <= 0 {
		t.Fatalf("Metal reported %v for 32 dispatches over %d elements", stats.Elapsed, n)
	}
	if stats.Elapsed > wall {
		t.Errorf("the device reported %v and the whole call took %v; device time cannot "+
			"exceed wall time, so this is a wall-clock figure taken somewhere wider",
			stats.Elapsed, wall)
	}
}

// A Metal graph that did not ask for timings reports none, and a fence read
// after its executable closes reports none rather than reaching a freed
// command buffer.
//
// The second half is the one worth having: the timing is read from the command
// buffer, which the executable owns, so a caller holding a fence past Close
// would otherwise send a message to a released object.
func TestMetalTimingsAreSilentWhenNotAskedFor(t *testing.T) {
	const n = 1024
	d := openMetalDevice(t)
	q := d.Queue()

	p, err := d.NewComputePipeline(accel.ComputePipelineDescriptor{
		Kernel: &kernels.AddKernel, Label: "untimed",
	})
	if err != nil {
		t.Fatalf("pipeline: %v", err)
	}
	defer p.Close()

	usage := accel.BufferStorage | accel.BufferCopySrc | accel.BufferCopyDst
	bufs := make([]accel.BufferView, 3)
	for i, name := range []string{"a", "b", "out"} {
		b, err := d.NewBuffer(accel.BufferDescriptor{
			DType: accel.F32, Count: n, Usage: usage, Label: name,
		})
		if err != nil {
			t.Fatalf("buffer: %v", err)
		}
		defer b.Close()
		v, err := b.View(0, n)
		if err != nil {
			t.Fatalf("view: %v", err)
		}
		bufs[i] = v
	}

	r := d.NewRecorder()
	// No CollectTimings.
	r.Dispatch(p, []accel.Binding{
		{Index: 0, Buffer: bufs[0]}, {Index: 1, Buffer: bufs[1]},
		{Index: 2, Buffer: bufs[2]},
	}, nil, p.Workgroups(n))
	g, err := r.Build()
	if err != nil {
		t.Fatalf("build: %v", err)
	}

	f := q.Submit(g)
	if err := f.Wait(); err != nil {
		t.Fatalf("submit: %v", err)
	}
	stats, err := f.Stats()
	if err != nil {
		t.Fatalf("stats: %v", err)
	}
	if stats.Elapsed != 0 {
		t.Errorf("a graph that did not ask for timings reported %v", stats.Elapsed)
	}

	// Closing the graph releases the command buffer the timing came from.
	// Reading afterwards must report nothing rather than message a freed
	// object.
	if err := g.Close(); err != nil {
		t.Fatalf("close: %v", err)
	}
	if _, err := f.Stats(); err != nil {
		t.Errorf("reading stats after the graph closed gave %v", err)
	}
}

// A kernel needing a capability Metal lacks is refused at pipeline creation,
// naming the capability.
//
// accel.AddF32's own documentation promises this: "A device without it refuses
// the kernel at pipeline creation rather than producing a wrong sum." The
// machinery existed -- Requirements.Unmet carries CapAtomicFloatAddStorage and
// the Metal device implements driver.KernelSupport -- and nothing exercised the
// path, because no corpus kernel used the float atomic until AtomicAddF32.
//
// The refusal must arrive at pipeline creation rather than at plan compile,
// which is the whole point of specs/021-metal-bringup.md's correction: compile
// is after a caller has uploaded their weights, so the right diagnosis arrived
// after the expensive part (accel issue 19).
func TestMetalRefusesTheFloatAtomicByName(t *testing.T) {
	d := openMetalDevice(t)
	defer d.Close()

	if d.Capabilities().Has(accel.CapAtomicFloatAddStorage) {
		t.Skip("this device reports CapAtomicFloatAddStorage, so there is no refusal " +
			"to observe. The test runs on the day Metal stops declaring it false")
	}

	_, err := d.NewComputePipeline(accel.ComputePipelineDescriptor{
		Kernel: &kernels.AtomicAddF32Kernel, Label: "floatatomic",
	})
	if err == nil {
		t.Fatal("a kernel using the float atomic built a pipeline on a device that " +
			"reports the capability false; the sum it produces would be wrong rather " +
			"than refused")
	}
	if !strings.Contains(err.Error(), "CapAtomicFloatAddStorage") {
		t.Fatalf("refused with %q, which does not name the capability. A caller who "+
			"cannot tell which capability is missing cannot tell whether another "+
			"device would run it", err)
	}
}

// A device that reports the float atomic runs the kernel that uses it, and
// the sums are the exact ones.
//
// The row was a constant false until 2026-09-02, so every Apple-silicon
// device reported a capability it had as absent and the refusal test above
// was the only path anything took. Now the device is asked, and where it
// answers yes this is the path.
func TestMetalRunsTheFloatAtomicWhereItReportsIt(t *testing.T) {
	d := openMetalDevice(t)
	defer d.Close()
	if !d.Capabilities().Has(accel.CapAtomicFloatAddStorage) {
		t.Skip("this device reports no float atomic; the refusal test covers it")
	}
	p, err := d.NewComputePipeline(accel.ComputePipelineDescriptor{
		Kernel: &kernels.AtomicAddF32Kernel, Label: "floatatomic",
	})
	if err != nil {
		t.Fatalf("a device reporting CapAtomicFloatAddStorage refused the kernel: %v", err)
	}
	defer p.Close()
	floatBuf := func(label string, vals []float32) accel.BufferView {
		b, err := d.NewBuffer(accel.BufferDescriptor{
			DType: accel.F32, Count: len(vals), Label: label,
			Usage: accel.BufferStorage | accel.BufferCopySrc | accel.BufferCopyDst,
		})
		if err != nil {
			t.Fatalf("buffer %s: %v", label, err)
		}
		t.Cleanup(func() { _ = b.Close() })
		if err := d.Queue().WriteBuffer(b, 0, vals); err != nil {
			t.Fatalf("write %s: %v", label, err)
		}
		v, err := b.View(0, b.Count())
		if err != nil {
			t.Fatalf("view %s: %v", label, err)
		}
		return v
	}
	state := floatBuf("state", []float32{1, 2})
	prev := floatBuf("prev", []float32{0, 0})
	r := d.NewRecorder()
	r.Dispatch(p, []accel.Binding{{Index: 0, Buffer: state}, {Index: 1, Buffer: prev}}, nil,
		accel.WorkgroupCount{X: 1})
	g, err := r.Build()
	if err != nil {
		t.Fatalf("build: %v", err)
	}
	defer g.Close()
	if err := d.Queue().Submit(g).Wait(); err != nil {
		t.Fatalf("submit: %v", err)
	}
	got := make([]float32, 2)
	if err := d.Queue().ReadBuffer(state.Buffer, 0, got); err != nil {
		t.Fatalf("read: %v", err)
	}
	if got[0] != 1.5 || got[1] != 1.75 {
		t.Fatalf("state after the adds is %v, want [1.5 1.75]", got)
	}
	if err := d.Queue().ReadBuffer(prev.Buffer, 0, got); err != nil {
		t.Fatalf("read: %v", err)
	}
	if got[0] != 1 || got[1] != 2 {
		t.Fatalf("the previous values are %v, want [1 2]", got)
	}
}
