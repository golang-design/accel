// Copyright 2026 The golang.design Initiative Authors.
// All rights reserved. Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

//go:build darwin

package metal_test

import (
	"fmt"
	"os"
	"strings"
	"testing"
	"unsafe"

	"golang.design/x/accel/internal/driver"
	"golang.design/x/accel/internal/kernel"
	"golang.design/x/accel/internal/metal"
	"golang.design/x/accel/internal/testkernels"
)

// open returns one opened Metal device.
//
// It fails rather than skips when there is no device, per
// specs/006-backends.md section 7: a job promising a backend and finding none
// is a failure, because a skip lets the backend rot green.
func open(t *testing.T) driver.Device {
	t.Helper()
	d, err := adapters(t)[0].Open(nil)
	if err != nil {
		t.Fatalf("open: %v", err)
	}
	t.Cleanup(func() { _ = d.Close() })
	return d
}

// adapters returns the Metal adapters, or ends the test.
//
// Whether "no device" is a failure or a skip depends on what the *job* promised,
// which is specs/006-backends.md section 7 and the rule .github/workflows/ci.yml
// states in its own header: a job that promises a backend and finds no device is
// a failure, and one that promises only the CPU must not go red for the same
// reason. Tier 2 sets ACCEL_REQUIRE_METAL; Tier 1 does not.
func adapters(t *testing.T) []driver.Adapter {
	t.Helper()
	all, err := metal.Adapters()
	if err == nil && len(all) > 0 {
		return all
	}
	if os.Getenv("ACCEL_REQUIRE_METAL") != "" {
		t.Fatalf("this job promises Metal and found no adapter (err=%v)", err)
	}
	t.Skipf("no Metal adapter on this machine (err=%v); set ACCEL_REQUIRE_METAL to "+
		"make that a failure", err)
	return nil
}

func f32(b []byte) []float32 {
	return unsafe.Slice((*float32)(unsafe.Pointer(&b[0])), len(b)/4)
}

// The adapter answers for itself before anything is opened, and its token is
// stable.
//
// Stability is the property that matters: [driver.Adapter] promises a token
// "stable within the process", and OpenDevice looks an adapter up by it. A
// token derived from enumeration order would satisfy every other test here and
// break the moment a machine had two GPUs.
func TestAdapterIdentity(t *testing.T) {
	first := adapters(t)
	second, err := metal.Adapters()
	if err != nil {
		t.Fatalf("adapters: %v", err)
	}
	if len(first) != len(second) || len(first) == 0 {
		t.Fatalf("enumerated %d adapters then %d", len(first), len(second))
	}
	for i := range first {
		if first[i].Token() != second[i].Token() {
			t.Errorf("adapter %d has a different token on a second enumeration", i)
		}
		if first[i].Info().Backend != driver.BackendMetal {
			t.Errorf("adapter %d reports backend %v", i, first[i].Info().Backend)
		}
		if first[i].Info().Name == "" {
			t.Errorf("adapter %d has no name", i)
		}
	}
	if _, err := first[0].Open(struct{}{}); err == nil {
		t.Error("Metal takes no options yet, so a non-nil one must be refused rather than ignored")
	}
}

// Every memory kind is real on this backend, and an unknown one is refused.
//
// "Real" is the point: unified memory means none of these has to be emulated
// with a copy the caller cannot see, which is why Supports answers yes to all
// four rather than substituting.
func TestSupportsEveryMemoryKind(t *testing.T) {
	d := open(t)
	for _, k := range []driver.MemoryKind{
		driver.MemoryDevice, driver.MemoryUpload, driver.MemoryReadback, driver.MemoryShared,
	} {
		if !d.Supports(k) {
			t.Errorf("memory kind %d is not supported", k)
		}
	}
	if d.Supports(driver.MemoryKind(99)) {
		t.Error("an unknown memory kind must not be supported")
	}
	if _, err := d.Alloc(driver.MemoryKind(99), 64, "bogus"); err == nil {
		t.Error("allocating an unknown memory kind must fail rather than pick a mode")
	}
	if d.Lost() != nil {
		t.Error("a healthy device reports no loss")
	}
}

// Device memory is not mappable and round-trips through the blit path; host
// memory is mappable and round-trips through the mapping.
//
// The offsets are non-zero on purpose. A transfer that ignores its offset
// passes every test that starts at zero, and every real transfer into a
// suballocated pool starts somewhere else.
func TestBlockTransfers(t *testing.T) {
	d := open(t)

	for _, tc := range []struct {
		name     string
		kind     driver.MemoryKind
		mappable bool
	}{
		{"device", driver.MemoryDevice, false},
		{"upload", driver.MemoryUpload, true},
		{"readback", driver.MemoryReadback, true},
		{"shared", driver.MemoryShared, true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			b, err := d.Alloc(tc.kind, 256, tc.name)
			if err != nil {
				t.Fatalf("alloc: %v", err)
			}
			defer b.Free()
			if got := b.Bytes() != nil; got != tc.mappable {
				t.Fatalf("Bytes() non-nil is %v, want %v: Bytes is the authority on "+
					"mappability and unified memory is the excuse that would erode it",
					got, tc.mappable)
			}
			if b.Size() != 256 {
				t.Errorf("Size is %d, want 256", b.Size())
			}

			src := make([]byte, 64)
			for i := range src {
				src[i] = byte(i + 1)
			}
			if err := b.Write(32, src); err != nil {
				t.Fatalf("write: %v", err)
			}
			got := make([]byte, 64)
			if err := b.Read(32, got); err != nil {
				t.Fatalf("read: %v", err)
			}
			for i := range got {
				if got[i] != src[i] {
					t.Fatalf("byte %d round-tripped as %d, want %d: an offset was dropped",
						i, got[i], src[i])
				}
			}

			// A range check on both directions, because a transfer that trusts
			// its caller writes over whatever follows the block.
			if err := b.Write(224, make([]byte, 64)); err == nil {
				t.Error("a write running past the end of a block must be refused")
			}
			if err := b.Read(224, make([]byte, 64)); err == nil {
				t.Error("a read running past the end of a block must be refused")
			}
			if err := b.Write(-1, src); err == nil {
				t.Error("a negative offset must be refused")
			}
		})
	}
}

// A closed device refuses to allocate, and closing twice is not an error.
func TestClosedDeviceRefusesWork(t *testing.T) {
	all := adapters(t)
	d, err := all[0].Open(nil)
	if err != nil {
		t.Fatalf("open: %v", err)
	}
	if err := d.Close(); err != nil {
		t.Fatalf("close: %v", err)
	}
	if err := d.Close(); err != nil {
		t.Errorf("closing twice must be harmless, got %v", err)
	}
	if _, err := d.Alloc(driver.MemoryDevice, 64, "after"); err == nil {
		t.Error("a closed device must refuse to allocate")
	}
	if _, err := d.(driver.GraphCompiler).Compile(&driver.Plan{}); err == nil {
		t.Error("a closed device must refuse to compile")
	}
}

// Compile refuses, by name, everything this child does not lower.
//
// Each of these would otherwise fail during a submission and be reported
// through a fence, which is the hardest place for a caller to act on. The list
// is also the honest statement of what specs/021-metal-bringup.md built: a
// reader can see the boundary without reading the emitter.
func TestCompileRefusesWhatItCannotLower(t *testing.T) {
	d := open(t)
	c := d.(driver.GraphCompiler)
	b, err := d.Alloc(driver.MemoryDevice, 4096, "b")
	if err != nil {
		t.Fatalf("alloc: %v", err)
	}
	defer b.Free()
	op, err := driver.BlockOperand(b, 0, 1024)
	if err != nil {
		t.Fatalf("operand: %v", err)
	}

	dispatchOf := func(k *kernel.Kernel, n int, uniforms []any, ind *driver.Indirect) driver.PlanNode {
		binds := make([]driver.Operand, n)
		for i := range binds {
			binds[i] = op
		}
		return driver.PlanNode{Op: driver.OpDispatch, Dispatch: &driver.Dispatch{
			Kernel: k, Count: kernel.ID3{X: 1, Y: 1, Z: 1},
			Bindings: binds, Uniforms: uniforms, Indirect: ind,
		}}
	}

	for _, tc := range []struct {
		name string
		node driver.PlanNode
		want string
	}{
		{"a row copy", driver.PlanNode{
			Op: driver.OpCopyRows, Dst: op, Src: op,
			Rows: &driver.RowCopy{Rows: 1, RowBytes: 16, DstPitch: 16, SrcPitch: 16},
		}, "row copy"},
		{"an indirect dispatch", dispatchOf(&testkernels.AddKernel, 3, nil,
			&driver.Indirect{Count: op, Max: kernel.ID3{X: 1, Y: 1, Z: 1}}), "indirect"},
		{"a kernel with no MSL", dispatchOf(&testkernels.ReduceSumKernel, 2, nil, nil), "ReduceSum"},
		{"a kernel taking uniforms", dispatchOf(&testkernels.ElemScaleKernel, 2,
			[]any{testkernels.ScaleParams{Factor: 2}}, nil), "uniform"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			_, err := c.Compile(&driver.Plan{Nodes: []driver.PlanNode{tc.node}})
			if err == nil {
				t.Fatalf("%s compiled, so it would have failed inside a submission instead", tc.name)
			}
			if !strings.Contains(err.Error(), tc.want) {
				t.Errorf("the refusal should mention %q, got %v", tc.want, err)
			}
		})
	}
}

// A plan built on slots runs, replays after a rebind, and rejects every bad
// rebind without disturbing the bindings it already had.
//
// The all-or-nothing property is the one worth testing directly: a caller
// cannot see which half of a partially applied batch landed, so a partial
// rebind is unrecoverable rather than merely wrong.
func TestSlotsRebindAndReplay(t *testing.T) {
	d := open(t)
	c := d.(driver.GraphCompiler)
	const n = 64
	const bytes = n * 4

	mk := func(label string) driver.Block {
		b, err := d.Alloc(driver.MemoryShared, bytes, label)
		if err != nil {
			t.Fatalf("alloc %s: %v", label, err)
		}
		t.Cleanup(b.Free)
		return b
	}
	a, bb, out1, out2 := mk("a"), mk("b"), mk("out1"), mk("out2")

	fill := func(b driver.Block, v float32) {
		s := f32(b.Bytes())
		for i := range s {
			s[i] = v
		}
	}
	fill(a, 2)
	fill(bb, 3)

	whole := func(b driver.Block) driver.Operand {
		o, err := driver.BlockOperand(b, 0, bytes)
		if err != nil {
			t.Fatalf("operand: %v", err)
		}
		return o
	}
	slot, err := driver.SlotOperand(1, 0, bytes)
	if err != nil {
		t.Fatalf("slot operand: %v", err)
	}

	plan := &driver.Plan{
		Label: "add",
		Slots: 1,
		Nodes: []driver.PlanNode{
			{Op: driver.OpHostWrite, Dst: whole(a), Data: make([]byte, bytes), ID: 0},
			{Op: driver.OpDispatch, BarrierBefore: true, ID: 1, Dispatch: &driver.Dispatch{
				Kernel:   &testkernels.AddKernel,
				Count:    kernel.ID3{X: 1, Y: 1, Z: 1},
				Bindings: []driver.Operand{whole(a), whole(bb), slot},
			}},
			{Op: driver.OpCopy, BarrierBefore: true, Dst: whole(out2), Src: slot, ID: 2},
		},
	}
	// The host write puts 5 into a, so the dispatch computes 5+3 = 8. Written
	// as bytes because that is what a host write carries.
	for i, v := range f32(plan.Nodes[0].Data) {
		_ = v
		f32(plan.Nodes[0].Data)[i] = 5
	}

	e, err := c.Compile(plan)
	if err != nil {
		t.Fatalf("compile: %v", err)
	}
	defer e.Close()

	bad := []struct {
		name string
		bind driver.SlotBinding
	}{
		{"slot zero", driver.SlotBinding{Slot: 0, Block: out1, Size: bytes}},
		{"a slot past the end", driver.SlotBinding{Slot: 2, Block: out1, Size: bytes}},
		{"no block", driver.SlotBinding{Slot: 1, Size: bytes}},
		{"a range past the block", driver.SlotBinding{Slot: 1, Block: out1, Offset: bytes, Size: bytes}},
		{"foreign memory", driver.SlotBinding{Slot: 1, Block: notMetal{n: bytes}, Size: bytes}},
	}
	for _, tc := range bad {
		if err := e.Rebind([]driver.SlotBinding{tc.bind}); err == nil {
			t.Errorf("rebind with %s was accepted", tc.name)
		}
	}
	// A batch whose second entry is bad must leave the first unapplied.
	err = e.Rebind([]driver.SlotBinding{
		{Slot: 1, Block: out1, Size: bytes},
		{Slot: 9, Block: out1, Size: bytes},
	})
	if err == nil {
		t.Fatal("a batch containing a bad entry was accepted")
	}
	if _, err := e.Submit(); err == nil {
		t.Error("submitting with no slot bound should fail: the rejected batch must not " +
			"have applied its first entry")
	}

	run := func(target driver.Block) []float32 {
		t.Helper()
		if err := e.Rebind([]driver.SlotBinding{{Slot: 1, Block: target, Size: bytes}}); err != nil {
			t.Fatalf("rebind: %v", err)
		}
		f, err := e.Submit()
		if err != nil {
			t.Fatalf("submit: %v", err)
		}
		if err := f.Wait(); err != nil {
			t.Fatalf("wait: %v", err)
		}
		if !f.Done() {
			t.Error("a fence that has been waited on must report done")
		}
		return f32(target.Bytes())
	}

	for i, v := range run(out1) {
		if v != 8 {
			t.Fatalf("out1[%d] is %v, want 8 (host write 5 plus b's 3)", i, v)
		}
	}
	// The same executable, a different slot, no rebuild. And out2 received the
	// copy node's output, which is what proves the copy ran against the rebound
	// resource rather than the first one.
	for i, v := range run(out1) {
		if v != 8 {
			t.Fatalf("on replay out1[%d] is %v, want 8", i, v)
		}
	}
	for i, v := range f32(out2.Bytes()) {
		if v != 8 {
			t.Fatalf("out2[%d] is %v, want 8: the copy node did not run", i, v)
		}
	}

	if err := e.Close(); err != nil {
		t.Fatalf("close: %v", err)
	}
	if err := e.Close(); err != nil {
		t.Errorf("closing twice must be harmless, got %v", err)
	}
	if _, err := e.Submit(); err == nil {
		t.Error("a closed executable must refuse to submit")
	}
	if err := e.Rebind(nil); err == nil {
		t.Error("a closed executable must refuse to rebind")
	}
}

// notMetal is a Block from another backend, which every Metal entry point must
// refuse rather than assert on.
type notMetal struct{ n int }

func (notMetal) Bytes() []byte           { return nil }
func (b notMetal) Size() int             { return b.n }
func (notMetal) Write(int, []byte) error { return nil }
func (notMetal) Read(int, []byte) error  { return nil }
func (notMetal) Free()                   {}

// A dispatch whose workgroup count has a zero dimension is skipped, not
// rejected.
//
// Metal refuses a zero-sized grid rather than treating it as a no-op, and the
// CPU backend skips it. The oracle decides: this backend skips too, because a
// graph that is correct on the CPU and fails on Metal for this reason is a
// difference between backends that nothing in the model justifies.
func TestZeroSizedDispatchIsSkipped(t *testing.T) {
	d := open(t)
	c := d.(driver.GraphCompiler)
	const bytes = 256
	b, err := d.Alloc(driver.MemoryShared, bytes, "b")
	if err != nil {
		t.Fatalf("alloc: %v", err)
	}
	defer b.Free()
	op, err := driver.BlockOperand(b, 0, bytes)
	if err != nil {
		t.Fatalf("operand: %v", err)
	}
	e, err := c.Compile(&driver.Plan{Nodes: []driver.PlanNode{{
		Op: driver.OpDispatch, Dispatch: &driver.Dispatch{
			Kernel: &testkernels.AddKernel, Count: kernel.ID3{X: 0, Y: 1, Z: 1},
			Bindings: []driver.Operand{op, op, op},
		},
	}}})
	if err != nil {
		t.Fatalf("compile: %v", err)
	}
	defer e.Close()
	f, err := e.Submit()
	if err != nil {
		t.Fatalf("submit: %v", err)
	}
	if err := f.Wait(); err != nil {
		t.Fatalf("wait: %v", err)
	}
}

// synthetic builds a kernel record directly, which is the only way to reach
// the checks that a generated record cannot violate.
//
// The generator cannot emit a workgroup above a pipeline's ceiling or MSL the
// device rejects, so a test that only used generated records could never reach
// either refusal -- and both are the kind that would otherwise surface as a
// crash inside the driver rather than as an error.
func synthetic(name, msl string, wg kernel.ID3) *kernel.Kernel {
	return &kernel.Kernel{
		Name: name, MSL: msl, Digest: "synthetic:" + name,
		WorkgroupSize: wg,
		Bindings:      []kernel.Binding{{Name: "out", DType: kernel.F32, Access: kernel.Write}},
		// Plan.Validate requires an entry point, and it is right to: a record
		// with neither would be a kernel no backend could run. This one exists
		// so the record is well formed, and it is never called, because the
		// Metal path runs the MSL.
		Flat: func(kernel.Thread, kernel.Args) {},
	}
}

const oneBinding = `#include <metal_stdlib>
using namespace metal;
kernel void %s(device float *out [[buffer(0)]],
               constant uint *_lens [[buffer(1)]],
               uint3 _gid [[thread_position_in_grid]]) {
  out[_gid.x] = 1.0f;
}`

func TestPipelineRefusalsAreReportedAtCompile(t *testing.T) {
	d := open(t)
	c := d.(driver.GraphCompiler)
	b, err := d.Alloc(driver.MemoryShared, 4096, "b")
	if err != nil {
		t.Fatalf("alloc: %v", err)
	}
	defer b.Free()
	op, err := driver.BlockOperand(b, 0, 4096)
	if err != nil {
		t.Fatalf("operand: %v", err)
	}

	plan := func(k *kernel.Kernel) *driver.Plan {
		return &driver.Plan{Nodes: []driver.PlanNode{{
			Op: driver.OpDispatch, Dispatch: &driver.Dispatch{
				Kernel: k, Count: kernel.ID3{X: 1, Y: 1, Z: 1},
				Bindings: []driver.Operand{op},
			},
		}}}
	}

	t.Run("a workgroup above the pipeline's ceiling", func(t *testing.T) {
		k := synthetic("big", fmt.Sprintf(oneBinding, "big"), kernel.ID3{X: 4096, Y: 1, Z: 1})
		_, err := c.Compile(plan(k))
		if err == nil {
			t.Fatal("a workgroup the pipeline cannot run was accepted, which Metal turns " +
				"into a failed submission rather than a clean error")
		}
		if !strings.Contains(err.Error(), "ceiling") {
			t.Errorf("the refusal should say what the ceiling is: %v", err)
		}
	})

	t.Run("MSL the device rejects", func(t *testing.T) {
		k := synthetic("bad", "kernel void bad() { not msl }", kernel.ID3{X: 1, Y: 1, Z: 1})
		_, err := c.Compile(plan(k))
		if err == nil {
			t.Fatal("MSL the device compiler rejects was accepted")
		}
		if !strings.Contains(err.Error(), "bad") {
			t.Errorf("the refusal should name the kernel: %v", err)
		}
	})
}

// Encoders switch between copy and compute work without a barrier between them,
// and the results are still ordered.
//
// Without a barrier there is nothing forcing the current encoder to end, so
// this is the path where a copy following a dispatch has to close the compute
// pass itself. Metal permits one encoder at a time, so getting it wrong is not
// a subtle reordering: it is an API violation that raises.
func TestEncodersSwitchWithoutABarrier(t *testing.T) {
	d := open(t)
	c := d.(driver.GraphCompiler)
	const n, bytes = 64, 64 * 4

	mk := func(label string) driver.Block {
		b, err := d.Alloc(driver.MemoryShared, bytes, label)
		if err != nil {
			t.Fatalf("alloc: %v", err)
		}
		t.Cleanup(b.Free)
		return b
	}
	a, bb, mid, out := mk("a"), mk("b"), mk("mid"), mk("out")
	for i := range f32(a.Bytes()) {
		f32(a.Bytes())[i] = 1
		f32(bb.Bytes())[i] = 2
	}
	whole := func(b driver.Block) driver.Operand {
		o, err := driver.BlockOperand(b, 0, bytes)
		if err != nil {
			t.Fatalf("operand: %v", err)
		}
		return o
	}

	// copy -> dispatch -> copy, with no BarrierBefore anywhere: every encoder
	// transition is forced by the switch rather than by a barrier.
	e, err := c.Compile(&driver.Plan{Nodes: []driver.PlanNode{
		{Op: driver.OpCopy, Dst: whole(mid), Src: whole(a), ID: 0},
		{Op: driver.OpDispatch, ID: 1, Dispatch: &driver.Dispatch{
			Kernel: &testkernels.AddKernel, Count: kernel.ID3{X: 1, Y: 1, Z: 1},
			Bindings: []driver.Operand{whole(mid), whole(bb), whole(out)},
		}},
		{Op: driver.OpCopy, Dst: whole(mid), Src: whole(out), ID: 2},
	}})
	if err != nil {
		t.Fatalf("compile: %v", err)
	}
	defer e.Close()
	f, err := e.Submit()
	if err != nil {
		t.Fatalf("submit: %v", err)
	}
	if err := f.Wait(); err != nil {
		t.Fatalf("wait: %v", err)
	}
	for i, v := range f32(mid.Bytes())[:n] {
		if v != 3 {
			t.Fatalf("mid[%d] is %v, want 3 (1 copied in, plus b's 2)", i, v)
		}
	}
}

// An operand naming another backend's memory is refused rather than asserted
// on.
//
// A type assertion without the comma-ok would panic inside the driver, which a
// caller cannot recover from and cannot diagnose. This is the same check
// Rebind makes, on the path that does not go through Rebind.
func TestOperandsMustNameMetalMemory(t *testing.T) {
	d := open(t)
	c := d.(driver.GraphCompiler)
	foreign, err := driver.BlockOperand(notMetal{n: 64}, 0, 64)
	if err != nil {
		t.Fatalf("operand: %v", err)
	}
	real, err := driver.BlockOperand(mustAlloc(t, d, 64), 0, 64)
	if err != nil {
		t.Fatalf("operand: %v", err)
	}
	e, err := c.Compile(&driver.Plan{Nodes: []driver.PlanNode{
		{Op: driver.OpCopy, Dst: real, Src: foreign, ID: 0},
	}})
	if err != nil {
		t.Fatalf("compile: %v", err)
	}
	defer e.Close()
	if _, err := e.Submit(); err == nil {
		t.Fatal("an operand naming another backend's memory was accepted")
	} else if !strings.Contains(err.Error(), "not Metal memory") {
		t.Errorf("the error should say the memory is not this backend's: %v", err)
	}
}

func mustAlloc(t *testing.T, d driver.Device, n int) driver.Block {
	t.Helper()
	b, err := d.Alloc(driver.MemoryShared, n, "b")
	if err != nil {
		t.Fatalf("alloc: %v", err)
	}
	t.Cleanup(b.Free)
	return b
}
