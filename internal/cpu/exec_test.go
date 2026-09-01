// Copyright 2026 The golang.design Initiative Authors.
// All rights reserved. Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

package cpu_test

import (
	"bytes"
	"errors"
	"fmt"
	"runtime"
	"strings"
	"sync"
	"testing"

	"golang.design/x/accel/internal/cpu"
	"golang.design/x/accel/internal/driver"
	"golang.design/x/accel/internal/kernel"
)

func open(t *testing.T, o cpu.Options) (driver.Device, driver.GraphCompiler) {
	t.Helper()
	dev, err := cpu.Adapter{}.Open(o)
	if err != nil {
		t.Fatalf("open: %v", err)
	}
	t.Cleanup(func() { _ = dev.Close() })
	c, ok := dev.(driver.GraphCompiler)
	if !ok {
		t.Fatal("the CPU device should compile graphs")
	}
	return dev, c
}

func block(t *testing.T, dev driver.Device, kind driver.MemoryKind, n int) driver.Block {
	t.Helper()
	b, err := dev.Alloc(kind, n, "test")
	if err != nil {
		t.Fatalf("alloc: %v", err)
	}
	t.Cleanup(b.Free)
	return b
}

func blockOperand(t *testing.T, b driver.Block, off, size int) driver.Operand {
	t.Helper()
	o, err := driver.BlockOperand(b, off, size)
	if err != nil {
		t.Fatalf("operand: %v", err)
	}
	return o
}

func slotOperand(t *testing.T, slot, off, size int) driver.Operand {
	t.Helper()
	o, err := driver.SlotOperand(slot, off, size)
	if err != nil {
		t.Fatalf("operand: %v", err)
	}
	return o
}

// The smallest thing that is genuinely a plan: write bytes in, copy them
// across, read them back out of the destination block.
func TestExecutableRunsAHostWriteAndACopy(t *testing.T) {
	dev, c := open(t, cpu.Options{})
	src := block(t, dev, driver.MemoryShared, 32)
	dst := block(t, dev, driver.MemoryShared, 32)

	want := []byte{9, 8, 7, 6, 5, 4, 3, 2}
	exe, err := c.Compile(&driver.Plan{Nodes: []driver.PlanNode{
		{ID: 0, Op: driver.OpHostWrite, Dst: blockOperand(t, src, 0, 8), Data: want},
		{ID: 1, Op: driver.OpCopy, Dst: blockOperand(t, dst, 16, 8), Src: blockOperand(t, src, 0, 8), BarrierBefore: true},
	}})
	if err != nil {
		t.Fatalf("compile: %v", err)
	}
	defer exe.Close()

	f, err := exe.Submit()
	if err != nil {
		t.Fatalf("submit: %v", err)
	}
	if err := f.Wait(); err != nil {
		t.Fatalf("wait: %v", err)
	}
	if !f.Done() {
		t.Error("a waited fence should report done")
	}
	if got := dst.Bytes()[16:24]; !bytes.Equal(got, want) {
		t.Errorf("got %v, want %v", got, want)
	}
}

// The same plan submitted twice produces the same bytes, which is the property
// that makes a graph worth building once.
func TestAPlanIsResubmittable(t *testing.T) {
	dev, c := open(t, cpu.Options{})
	dst := block(t, dev, driver.MemoryShared, 8)
	exe, err := c.Compile(&driver.Plan{Nodes: []driver.PlanNode{
		{Op: driver.OpHostWrite, Dst: blockOperand(t, dst, 0, 4), Data: []byte{1, 2, 3, 4}},
	}})
	if err != nil {
		t.Fatalf("compile: %v", err)
	}
	defer exe.Close()

	for i := range 3 {
		copy(dst.Bytes(), make([]byte, 8)) // clear between runs
		f, err := exe.Submit()
		if err != nil {
			t.Fatalf("submit %d: %v", i, err)
		}
		if err := f.Wait(); err != nil {
			t.Fatalf("wait %d: %v", i, err)
		}
		if !bytes.Equal(dst.Bytes()[:4], []byte{1, 2, 3, 4}) {
			t.Fatalf("submission %d wrote %v", i, dst.Bytes()[:4])
		}
	}
}

func TestSlotsResolveAtRebind(t *testing.T) {
	dev, c := open(t, cpu.Options{})
	a := block(t, dev, driver.MemoryShared, 16)
	b := block(t, dev, driver.MemoryShared, 16)
	dst := block(t, dev, driver.MemoryShared, 16)
	copy(a.Bytes(), []byte{1, 1, 1, 1})
	copy(b.Bytes(), []byte{2, 2, 2, 2})

	exe, err := c.Compile(&driver.Plan{
		Slots: 1,
		Nodes: []driver.PlanNode{
			{Op: driver.OpCopy, Dst: blockOperand(t, dst, 0, 4), Src: slotOperand(t, 1, 0, 4)},
		},
	})
	if err != nil {
		t.Fatalf("compile: %v", err)
	}
	defer exe.Close()

	// Submitting with nothing bound must fail rather than copying zeroes.
	if _, err := exe.Submit(); err == nil {
		t.Fatal("submitting with an unbound slot should fail")
	}

	for _, c := range []struct {
		blk  driver.Block
		want byte
	}{{a, 1}, {b, 2}} {
		if err := exe.Rebind([]driver.SlotBinding{{Slot: 1, Block: c.blk, Size: 16}}); err != nil {
			t.Fatalf("rebind: %v", err)
		}
		f, err := exe.Submit()
		if err != nil {
			t.Fatalf("submit: %v", err)
		}
		if err := f.Wait(); err != nil {
			t.Fatalf("wait: %v", err)
		}
		if dst.Bytes()[0] != c.want {
			t.Fatalf("got %d, want %d", dst.Bytes()[0], c.want)
		}
	}
}

// A slot operand's offset is relative to the bound window, so one plan can run
// over different slices of a larger allocation without knowing where they are.
func TestSlotOffsetsAreRelativeToTheBoundWindow(t *testing.T) {
	dev, c := open(t, cpu.Options{})
	big := block(t, dev, driver.MemoryShared, 64)
	dst := block(t, dev, driver.MemoryShared, 8)
	for i := range big.Bytes() {
		big.Bytes()[i] = byte(i)
	}

	exe, err := c.Compile(&driver.Plan{
		Slots: 1,
		Nodes: []driver.PlanNode{
			{Op: driver.OpCopy, Dst: blockOperand(t, dst, 0, 4), Src: slotOperand(t, 1, 0, 4)},
		},
	})
	if err != nil {
		t.Fatalf("compile: %v", err)
	}
	defer exe.Close()

	if err := exe.Rebind([]driver.SlotBinding{{Slot: 1, Block: big, Offset: 32, Size: 16}}); err != nil {
		t.Fatalf("rebind: %v", err)
	}
	f, _ := exe.Submit()
	if err := f.Wait(); err != nil {
		t.Fatalf("wait: %v", err)
	}
	if got := dst.Bytes()[:4]; !bytes.Equal(got, []byte{32, 33, 34, 35}) {
		t.Errorf("got %v, want bytes 32..35", got)
	}
}

// An operand reaching past the window a slot was bound to is rejected at
// submit, which is the earliest point the sizes are both known.
func TestSlotOperandsAreCheckedAgainstTheBoundWindow(t *testing.T) {
	dev, c := open(t, cpu.Options{})
	big := block(t, dev, driver.MemoryShared, 64)
	dst := block(t, dev, driver.MemoryShared, 32)

	exe, err := c.Compile(&driver.Plan{
		Slots: 1,
		Nodes: []driver.PlanNode{
			{Op: driver.OpCopy, Dst: blockOperand(t, dst, 0, 32), Src: slotOperand(t, 1, 0, 32)},
		},
	})
	if err != nil {
		t.Fatalf("compile: %v", err)
	}
	defer exe.Close()

	if err := exe.Rebind([]driver.SlotBinding{{Slot: 1, Block: big, Offset: 0, Size: 16}}); err != nil {
		t.Fatalf("rebind: %v", err)
	}
	_, err = exe.Submit()
	if err == nil || !strings.Contains(err.Error(), "outside the 16 bytes") {
		t.Fatalf("expected a window rejection, got %v", err)
	}
}

func TestRebindIsRejectedAsABatch(t *testing.T) {
	dev, c := open(t, cpu.Options{})
	good := block(t, dev, driver.MemoryShared, 16)
	dst := block(t, dev, driver.MemoryShared, 16)

	exe, err := c.Compile(&driver.Plan{
		Slots: 2,
		Nodes: []driver.PlanNode{
			{Op: driver.OpCopy, Dst: blockOperand(t, dst, 0, 4), Src: slotOperand(t, 1, 0, 4)},
		},
	})
	if err != nil {
		t.Fatalf("compile: %v", err)
	}
	defer exe.Close()

	// Slot 1 is valid and slot 3 does not exist. Neither may land: a caller
	// cannot see which half of a partial rebind applied.
	err = exe.Rebind([]driver.SlotBinding{
		{Slot: 1, Block: good, Size: 16},
		{Slot: 3, Block: good, Size: 16},
	})
	if err == nil || !strings.Contains(err.Error(), "slot 3 of 2") {
		t.Fatalf("expected slot 3 to be rejected, got %v", err)
	}
	if _, err := exe.Submit(); err == nil {
		t.Fatal("slot 1 should still be unbound: the batch was rejected whole")
	}
}

func TestRebindRejectsMalformedBindings(t *testing.T) {
	dev, c := open(t, cpu.Options{})
	b := block(t, dev, driver.MemoryShared, 16)
	dst := block(t, dev, driver.MemoryShared, 16)
	exe, err := c.Compile(&driver.Plan{
		Slots: 1,
		Nodes: []driver.PlanNode{{Op: driver.OpCopy, Dst: blockOperand(t, dst, 0, 4), Src: slotOperand(t, 1, 0, 4)}},
	})
	if err != nil {
		t.Fatalf("compile: %v", err)
	}
	defer exe.Close()

	cases := []struct {
		name string
		bind driver.SlotBinding
		says string
	}{
		{"slot zero", driver.SlotBinding{Slot: 0, Block: b, Size: 16}, "slot 0 of 1"},
		{"no block", driver.SlotBinding{Slot: 1, Size: 16}, "names no block"},
		{"negative offset", driver.SlotBinding{Slot: 1, Block: b, Offset: -1, Size: 4}, "of a 16-byte block"},
		{"window past the block", driver.SlotBinding{Slot: 1, Block: b, Offset: 8, Size: 16}, "of a 16-byte block"},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			err := exe.Rebind([]driver.SlotBinding{c.bind})
			if err == nil || !strings.Contains(err.Error(), c.says) {
				t.Fatalf("error should say %q, got %v", c.says, err)
			}
		})
	}
}

// Device-local memory is unmappable by contract and a transient pool is
// device-local, so a plan must still be able to touch it: a backend copying
// between its own allocations is not mapping them. If this ever regresses to
// using the host mapping, every graph with a transient breaks.
func TestPlansReachDeviceLocalMemory(t *testing.T) {
	dev, c := open(t, cpu.Options{})
	priv := block(t, dev, driver.MemoryDevice, 16)
	if priv.Bytes() != nil {
		t.Fatal("device-local memory must have no host mapping")
	}
	out := block(t, dev, driver.MemoryShared, 16)

	exe, err := c.Compile(&driver.Plan{Nodes: []driver.PlanNode{
		{ID: 0, Op: driver.OpHostWrite, Dst: blockOperand(t, priv, 0, 4), Data: []byte{1, 2, 3, 4}},
		{ID: 1, Op: driver.OpCopy, Dst: blockOperand(t, out, 0, 4), Src: blockOperand(t, priv, 0, 4)},
	}})
	if err != nil {
		t.Fatalf("compile: %v", err)
	}
	defer exe.Close()
	f, err := exe.Submit()
	if err != nil {
		t.Fatalf("submit: %v", err)
	}
	if err := f.Wait(); err != nil {
		t.Fatalf("wait: %v", err)
	}
	if got := out.Bytes()[:4]; !bytes.Equal(got, []byte{1, 2, 3, 4}) {
		t.Errorf("got %v through device-local memory, want 1..4", got)
	}
}

// A block freed while a plan still names it must be reported, not read as a
// nil slice.
func TestAFreedBlockIsReported(t *testing.T) {
	dev, c := open(t, cpu.Options{})
	b, err := dev.Alloc(driver.MemoryShared, 16, "freed")
	if err != nil {
		t.Fatalf("alloc: %v", err)
	}
	exe, err := c.Compile(&driver.Plan{Nodes: []driver.PlanNode{
		{Op: driver.OpHostWrite, Dst: blockOperand(t, b, 0, 4), Data: []byte{1, 2, 3, 4}},
	}})
	if err != nil {
		t.Fatalf("compile: %v", err)
	}
	defer exe.Close()
	b.Free()
	if _, err := exe.Submit(); err == nil || !strings.Contains(err.Error(), "freed") {
		t.Fatalf("expected a freed-block rejection, got %v", err)
	}
}

func TestOneSubmissionAtATime(t *testing.T) {
	dev, c := open(t, cpu.Options{})
	dst := block(t, dev, driver.MemoryShared, 1<<16)
	nodes := make([]driver.PlanNode, 256)
	payload := make([]byte, 256)
	for i := range nodes {
		nodes[i] = driver.PlanNode{ID: i, Op: driver.OpHostWrite,
			Dst: blockOperand(t, dst, i*256, 256), Data: payload}
	}
	exe, err := c.Compile(&driver.Plan{Nodes: nodes})
	if err != nil {
		t.Fatalf("compile: %v", err)
	}
	defer exe.Close()

	// Hammer submit and rebind concurrently. Whatever the interleaving, no
	// submission may start while another is running and nothing may race.
	var wg sync.WaitGroup
	for range 8 {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for range 20 {
				if f, err := exe.Submit(); err == nil {
					_ = f.Wait()
				}
			}
		}()
	}
	wg.Wait()

	f, err := exe.Submit()
	if err != nil {
		t.Fatalf("a submission should be possible once the others finished: %v", err)
	}
	if err := f.Wait(); err != nil {
		t.Fatalf("wait: %v", err)
	}
}

func TestClosedExecutableRefusesEverything(t *testing.T) {
	dev, c := open(t, cpu.Options{})
	dst := block(t, dev, driver.MemoryShared, 16)
	exe, err := c.Compile(&driver.Plan{
		Slots: 1,
		Nodes: []driver.PlanNode{{Op: driver.OpHostWrite, Dst: blockOperand(t, dst, 0, 4), Data: []byte{1, 2, 3, 4}}},
	})
	if err != nil {
		t.Fatalf("compile: %v", err)
	}
	if err := exe.Close(); err != nil {
		t.Fatalf("close: %v", err)
	}
	if err := exe.Close(); err != nil {
		t.Errorf("close should be idempotent: %v", err)
	}
	if _, err := exe.Submit(); err == nil {
		t.Error("submit after close should fail")
	}
	if err := exe.Rebind([]driver.SlotBinding{{Slot: 1, Block: dst, Size: 16}}); err == nil {
		t.Error("rebind after close should fail")
	}
}

// A refused submission does not count towards the injected loss.
//
// LoseAtSubmission counts submissions the device accepted: a caller asking to
// lose the second submission means the second one that ran. A submit refused
// for a closed executable, a busy one, or an unbound slot never reached the
// device, and counting it would lose a different submission from the one the
// test named.
func TestARefusedSubmissionDoesNotAdvanceTheLossCounter(t *testing.T) {
	dev, c := open(t, cpu.Options{LoseAtSubmission: 2})
	dst := block(t, dev, driver.MemoryShared, 16)
	plan := &driver.Plan{Nodes: []driver.PlanNode{{
		Op: driver.OpHostWrite, Dst: blockOperand(t, dst, 0, 4), Data: []byte{1, 2, 3, 4},
	}}}

	closed, err := c.Compile(plan)
	if err != nil {
		t.Fatalf("compile: %v", err)
	}
	if err := closed.Close(); err != nil {
		t.Fatalf("close: %v", err)
	}
	if _, err := closed.Submit(); err == nil {
		t.Fatal("submit after close should be refused")
	}

	exe, err := c.Compile(plan)
	if err != nil {
		t.Fatalf("compile: %v", err)
	}
	defer exe.Close()
	f, err := exe.Submit()
	if err != nil {
		t.Fatalf("the first accepted submission should run, got %v: the refused one "+
			"was counted", err)
	}
	if err := f.Wait(); err != nil {
		t.Fatalf("wait: %v", err)
	}
	if _, err := exe.Submit(); !errors.Is(err, driver.ErrDeviceLost) {
		t.Fatalf("the second accepted submission should report loss, got %v", err)
	}
}

func TestCompileOnAClosedDeviceFails(t *testing.T) {
	dev, err := cpu.Adapter{}.Open(cpu.Options{})
	if err != nil {
		t.Fatalf("open: %v", err)
	}
	if err := dev.Close(); err != nil {
		t.Fatalf("close: %v", err)
	}
	if _, err := dev.(driver.GraphCompiler).Compile(&driver.Plan{}); err == nil {
		t.Fatal("compile on a closed device should fail")
	}
}

func BenchmarkSubmit(b *testing.B) {
	dev, err := cpu.Adapter{}.Open(cpu.Options{})
	if err != nil {
		b.Fatalf("open: %v", err)
	}
	defer dev.Close()
	dst, err := dev.Alloc(driver.MemoryShared, 1<<16, "bench")
	if err != nil {
		b.Fatalf("alloc: %v", err)
	}
	defer dst.Free()

	nodes := make([]driver.PlanNode, 64)
	payload := make([]byte, 256)
	for i := range nodes {
		o, err := driver.BlockOperand(dst, i*256, 256)
		if err != nil {
			b.Fatalf("operand: %v", err)
		}
		nodes[i] = driver.PlanNode{ID: i, Op: driver.OpHostWrite, Dst: o, Data: payload}
	}
	exe, err := dev.(driver.GraphCompiler).Compile(&driver.Plan{Nodes: nodes})
	if err != nil {
		b.Fatalf("compile: %v", err)
	}
	defer exe.Close()

	b.ReportAllocs()
	for b.Loop() {
		f, err := exe.Submit()
		if err != nil {
			b.Fatal(err)
		}
		if err := f.Wait(); err != nil {
			b.Fatal(err)
		}
	}
}

// A fence signalling means the submission finished, so resubmitting the instant
// it signals must work.
//
// This was a race between two pieces of state that had to agree: a bool cleared
// by the completion goroutine and the fence it signalled. It signalled first,
// so a caller who waited and resubmitted was refused by a flag nobody had got
// round to clearing.
//
// Worth being precise about what this test is and is not. The window between
// those two statements is a few nanoseconds, and no version of this test
// reproduced the old behaviour reliably -- it took a hundred concurrent
// submissions through the graph layer to see it once. So this is a smoke check
// on the invariant, not the regression guard. The guard is that "in flight" is
// now derived from the fence rather than tracked beside it, which leaves no two
// pieces of state to disagree.
func TestResubmittingTheInstantAFenceSignalsWorks(t *testing.T) {
	dev, c := open(t, cpu.Options{})
	dst := block(t, dev, driver.MemoryShared, 1<<16)
	nodes := make([]driver.PlanNode, 64)
	payload := make([]byte, 1024)
	for i := range nodes {
		nodes[i] = driver.PlanNode{ID: i, Op: driver.OpHostWrite,
			Dst: blockOperand(t, dst, i*1024, 1024), Data: payload}
	}
	exe, err := c.Compile(&driver.Plan{Nodes: nodes})
	if err != nil {
		t.Fatalf("compile: %v", err)
	}
	defer exe.Close()

	for i := range 500 {
		f, err := exe.Submit()
		if err != nil {
			t.Fatalf("submission %d was refused although the previous fence had already "+
				"reported done: %v", i, err)
		}
		// Spin on Done rather than blocking on Wait, then submit on the next
		// statement: blocking hands the scheduler an opportunity that hides the
		// window, and the point is that there is no window to hide.
		for !f.Done() {
		}
		if err := f.Wait(); err != nil {
			t.Fatal(err)
		}
	}
}

// Closing while a submission runs is refused rather than freeing state the
// executing goroutine still reads.
func TestCloseDuringASubmissionIsRefused(t *testing.T) {
	dev, c := open(t, cpu.Options{})
	dst := block(t, dev, driver.MemoryShared, 1<<20)
	nodes := make([]driver.PlanNode, 1024)
	payload := make([]byte, 1024)
	for i := range nodes {
		nodes[i] = driver.PlanNode{ID: i, Op: driver.OpHostWrite,
			Dst: blockOperand(t, dst, i*1024, 1024), Data: payload}
	}
	exe, err := c.Compile(&driver.Plan{Nodes: nodes})
	if err != nil {
		t.Fatalf("compile: %v", err)
	}
	f, err := exe.Submit()
	if err != nil {
		t.Fatalf("submit: %v", err)
	}
	if err := exe.Close(); err != nil && !strings.Contains(err.Error(), "in flight") {
		t.Errorf("close during a submission should be refused as in-flight, got %v", err)
	}
	if err := f.Wait(); err != nil {
		t.Fatalf("wait: %v", err)
	}
	if err := exe.Close(); err != nil {
		t.Errorf("close after the fence signalled: %v", err)
	}
}

// Every binding element type reaches its kernel as the right Go slice. The
// bytes are the device's representation already, so this is a reinterpretation,
// and getting the element width wrong silently gives a kernel a slice of the
// wrong length over the right memory.
func TestEveryBindingTypeReachesTheKernel(t *testing.T) {
	dev, c := open(t, cpu.Options{})

	cases := []struct {
		dtype kernel.DType
		elems int
		check func(t *testing.T, args kernel.Args)
	}{
		{kernel.F32, 4, func(t *testing.T, a kernel.Args) {
			s, ok := a.Slices[0].([]float32)
			if !ok || len(s) != 4 {
				t.Fatalf("got %T of len %d", a.Slices[0], reflectLen(a.Slices[0]))
			}
			s[0] = 1.5
		}},
		{kernel.U32, 4, func(t *testing.T, a kernel.Args) {
			s, ok := a.Slices[0].([]uint32)
			if !ok || len(s) != 4 {
				t.Fatalf("got %T of len %d", a.Slices[0], reflectLen(a.Slices[0]))
			}
		}},
		{kernel.I32, 4, func(t *testing.T, a kernel.Args) {
			if s, ok := a.Slices[0].([]int32); !ok || len(s) != 4 {
				t.Fatalf("got %T", a.Slices[0])
			}
		}},
		{kernel.U8, 16, func(t *testing.T, a kernel.Args) {
			if s, ok := a.Slices[0].([]byte); !ok || len(s) != 16 {
				t.Fatalf("got %T", a.Slices[0])
			}
		}},
		{kernel.I8, 16, func(t *testing.T, a kernel.Args) {
			if s, ok := a.Slices[0].([]int8); !ok || len(s) != 16 {
				t.Fatalf("got %T", a.Slices[0])
			}
		}},
		{kernel.F16, 8, func(t *testing.T, a kernel.Args) {
			if s, ok := a.Slices[0].([]kernel.Float16); !ok || len(s) != 8 {
				t.Fatalf("got %T", a.Slices[0])
			}
		}},
		{kernel.BF16, 8, func(t *testing.T, a kernel.Args) {
			if s, ok := a.Slices[0].([]kernel.BFloat16); !ok || len(s) != 8 {
				t.Fatalf("got %T", a.Slices[0])
			}
		}},
	}

	for _, tc := range cases {
		t.Run(tc.dtype.String(), func(t *testing.T) {
			b := block(t, dev, driver.MemoryShared, 16)
			ran := false
			k := &kernel.Kernel{
				Name: "K", WorkgroupSize: kernel.ID3{X: 1, Y: 1, Z: 1},
				Generator: kernel.ABIVersion,
				Bindings:  []kernel.Binding{{Name: "b", DType: tc.dtype, Access: kernel.Write}},
				Flat: func(_ kernel.Thread, a kernel.Args) {
					ran = true
					tc.check(t, a)
				},
			}
			exe, err := c.Compile(&driver.Plan{Nodes: []driver.PlanNode{{
				Op: driver.OpDispatch,
				Dispatch: &driver.Dispatch{
					Kernel: k, Count: kernel.ID3{X: 1},
					Bindings: []driver.Operand{blockOperand(t, b, 0, 16)},
				},
			}}})
			if err != nil {
				t.Fatalf("compile: %v", err)
			}
			defer exe.Close()

			f, err := exe.Submit()
			if err != nil {
				t.Fatalf("submit: %v", err)
			}
			if err := f.Wait(); err != nil {
				t.Fatalf("wait: %v", err)
			}
			if !ran {
				t.Fatal("the kernel did not run")
			}
		})
	}
}

func reflectLen(v any) int {
	switch s := v.(type) {
	case []float32:
		return len(s)
	case []uint32:
		return len(s)
	case []int32:
		return len(s)
	case []byte:
		return len(s)
	case []int8:
		return len(s)
	}
	return -1
}

// A binding whose byte range is not a whole number of elements is refused
// rather than truncated: truncation hides a range computed with the wrong
// element size, and the kernel then runs over one element fewer than the caller
// believes it bound.
func TestAPartialElementIsRefused(t *testing.T) {
	dev, c := open(t, cpu.Options{})
	b := block(t, dev, driver.MemoryShared, 16)
	k := &kernel.Kernel{
		Name: "K", WorkgroupSize: kernel.ID3{X: 1, Y: 1, Z: 1}, Generator: kernel.ABIVersion,
		Bindings: []kernel.Binding{{Name: "b", DType: kernel.F32, Access: kernel.Read}},
		Flat:     func(kernel.Thread, kernel.Args) {},
	}
	exe, err := c.Compile(&driver.Plan{Nodes: []driver.PlanNode{{
		Op: driver.OpDispatch,
		Dispatch: &driver.Dispatch{
			Kernel: k, Count: kernel.ID3{X: 1},
			// Six bytes is one and a half float32s.
			Bindings: []driver.Operand{blockOperand(t, b, 0, 6)},
		},
	}}})
	if err != nil {
		t.Fatalf("compile: %v", err)
	}
	defer exe.Close()
	if _, err := exe.Submit(); err == nil ||
		!strings.Contains(err.Error(), "not a whole number of 4-byte elements") {
		t.Fatalf("expected a partial-element rejection, got %v", err)
	}
}

// A dispatch through a slot resolves like any other operand.
func TestADispatchBindingCanComeFromASlot(t *testing.T) {
	dev, c := open(t, cpu.Options{})
	b := block(t, dev, driver.MemoryShared, 16)
	var saw []float32
	k := &kernel.Kernel{
		Name: "K", WorkgroupSize: kernel.ID3{X: 1, Y: 1, Z: 1}, Generator: kernel.ABIVersion,
		Bindings: []kernel.Binding{{Name: "b", DType: kernel.F32, Access: kernel.Write}},
		Flat: func(_ kernel.Thread, a kernel.Args) {
			saw = a.Slices[0].([]float32)
			saw[0] = 7
		},
	}
	exe, err := c.Compile(&driver.Plan{
		Slots: 1,
		Nodes: []driver.PlanNode{{
			Op: driver.OpDispatch,
			Dispatch: &driver.Dispatch{
				Kernel: k, Count: kernel.ID3{X: 1},
				Bindings: []driver.Operand{slotOperand(t, 1, 0, 16)},
			},
		}},
	})
	if err != nil {
		t.Fatalf("compile: %v", err)
	}
	defer exe.Close()

	if _, err := exe.Submit(); err == nil {
		t.Fatal("dispatching through an unbound slot should fail")
	}
	if err := exe.Rebind([]driver.SlotBinding{{Slot: 1, Block: b, Size: 16}}); err != nil {
		t.Fatalf("rebind: %v", err)
	}
	f, err := exe.Submit()
	if err != nil {
		t.Fatalf("submit: %v", err)
	}
	if err := f.Wait(); err != nil {
		t.Fatalf("wait: %v", err)
	}
	if b.Bytes()[0] == 0 && b.Bytes()[1] == 0 && b.Bytes()[2] == 0 && b.Bytes()[3] == 0 {
		t.Error("the kernel wrote nothing through the slot")
	}
}

// A kernel whose dispatch fails reports through the fence rather than
// panicking, so a caller learns rather than the process dying.
func TestADispatchFailureReachesTheFence(t *testing.T) {
	dev, c := open(t, cpu.Options{})
	b := block(t, dev, driver.MemoryShared, 16)
	k := &kernel.Kernel{
		Name: "K", WorkgroupSize: kernel.ID3{X: 1, Y: 1, Z: 1}, Generator: kernel.ABIVersion,
		Bindings: []kernel.Binding{{Name: "b", DType: kernel.F32, Access: kernel.Read}},
		Flat:     func(kernel.Thread, kernel.Args) {},
	}
	// Compile with a valid record, then make the record stale, which Bind
	// refuses inside the dispatch loop.
	exe, err := c.Compile(&driver.Plan{Nodes: []driver.PlanNode{{
		Op: driver.OpDispatch,
		Dispatch: &driver.Dispatch{
			Kernel: k, Count: kernel.ID3{X: 1},
			Bindings: []driver.Operand{blockOperand(t, b, 0, 16)},
		},
	}}})
	if err != nil {
		t.Fatalf("compile: %v", err)
	}
	defer exe.Close()
	k.Generator = kernel.ABIVersion + 1

	f, err := exe.Submit()
	if err != nil {
		t.Fatalf("submit: %v", err)
	}
	if err := f.Wait(); err == nil || !strings.Contains(err.Error(), "re-run go generate") {
		t.Fatalf("the fence should report the stale record, got %v", err)
	}
}

// earlyReturn is a cooperative kernel in which one invocation returns without
// reaching the barrier its peers wait at, and undefinedRead one that reads
// shared memory nothing wrote. Each is a diagnostic in developer mode.
func earlyReturn() *kernel.Kernel {
	return &kernel.Kernel{
		Name: "Early", WorkgroupSize: kernel.ID3{X: 4, Y: 1, Z: 1},
		Generator: kernel.ABIVersion, Suspensions: 1,
		Cooperative: func(th kernel.Thread, a kernel.Args, f *kernel.Frame) bool {
			if th.LocalID().X == 1 {
				return false
			}
			if f.Pass == 0 {
				f.Pass = 1
				f.Barrier = kernel.BarrierID{Index: 0, Pos: "k.go:3:2"}
				return true
			}
			return false
		},
	}
}

func undefinedRead() *kernel.Kernel {
	return &kernel.Kernel{
		Name: "Undefined", WorkgroupSize: kernel.ID3{X: 1, Y: 1, Z: 1},
		Generator: kernel.ABIVersion, SharedSizes: []int{4},
		NewShared: func() []any {
			var sh [4]float32
			return []any{&sh}
		},
		Cooperative: func(th kernel.Thread, a kernel.Args, f *kernel.Frame) bool {
			sh := kernel.SharedSlice[[4]float32](a, 0)
			_ = sh[f.Shared.ReadAt(0, 2)]
			return false
		},
	}
}

// dispatchOnce submits one dispatch of k and returns what the fence reports.
func dispatchOnce(t *testing.T, o cpu.Options, k *kernel.Kernel) error {
	t.Helper()
	_, c := open(t, o)
	exe, err := c.Compile(&driver.Plan{Nodes: []driver.PlanNode{{
		Op: driver.OpDispatch, Dispatch: &driver.Dispatch{Kernel: k, Count: kernel.ID3{X: 1}},
	}}})
	if err != nil {
		t.Fatalf("compile: %v", err)
	}
	defer exe.Close()
	f, err := exe.Submit()
	if err != nil {
		t.Fatalf("submit: %v", err)
	}
	return f.Wait()
}

// Strict mode keeps the cooperative diagnostics on.
//
// specs/006-backends.md section 5 defines Strict as a capability and limits
// profile: the intersection of a target set. It says nothing about
// instrumentation, and a kernel that is portable to Vulkan and Metal is one
// whose barriers every invocation reaches, so the oracle's checks are the
// point of running under Strict rather than a cost of it. Turning them off is
// a separate request, made by NoDiagnostics.
func TestStrictModeStillReportsDiagnostics(t *testing.T) {
	strict := cpu.Options{Mode: cpu.Strict, StrictTargets: []driver.Backend{driver.BackendVulkan}}
	for _, c := range []struct {
		name string
		k    *kernel.Kernel
		says string
	}{
		{"an early return", earlyReturn(), "barrier arrival mismatch"},
		{"an undefined read", undefinedRead(), "undefined shared memory"},
	} {
		t.Run(c.name, func(t *testing.T) {
			err := dispatchOnce(t, strict, c.k)
			if err == nil || !strings.Contains(err.Error(), c.says) {
				t.Fatalf("under Strict %s should be reported as %q, got %v: Strict is a "+
					"capability profile, not an instrumentation switch", c.name, c.says, err)
			}
			off := strict
			off.NoDiagnostics = true
			if err := dispatchOnce(t, off, c.k); err != nil {
				t.Fatalf("with NoDiagnostics %s should run unchecked, got %v", c.name, err)
			}
		})
	}
}

// A kernel that panics reports through the fence rather than taking the
// process down. On a GPU an out-of-bounds access is clamped or undefined; here
// it is a Go panic, and it would otherwise abort from inside a goroutine the
// caller did not start and cannot recover in.
func TestAPanickingKernelReachesTheFence(t *testing.T) {
	dev, c := open(t, cpu.Options{})
	b := block(t, dev, driver.MemoryShared, 16)
	// The line that overruns, recorded from inside the kernel so the
	// assertion below does not depend on where this test sits in the file.
	var site string
	k := &kernel.Kernel{
		Name: "Overrun", WorkgroupSize: kernel.ID3{X: 1, Y: 1, Z: 1}, Generator: kernel.ABIVersion,
		Bindings: []kernel.Binding{{Name: "b", DType: kernel.F32, Access: kernel.Write}},
		Flat: func(_ kernel.Thread, a kernel.Args) {
			s := a.Slices[0].([]float32)
			_, file, line, _ := runtime.Caller(0)
			site = fmt.Sprintf("%s:%d", file, line+2)
			s[len(s)+1] = 1 // out of bounds, as a real kernel bug would be
		},
	}
	exe, err := c.Compile(&driver.Plan{Nodes: []driver.PlanNode{{
		ID: 3, Op: driver.OpDispatch,
		Dispatch: &driver.Dispatch{
			Kernel: k, Count: kernel.ID3{X: 1},
			Bindings: []driver.Operand{blockOperand(t, b, 0, 16)},
		},
	}}})
	if err != nil {
		t.Fatalf("compile: %v", err)
	}
	defer exe.Close()

	f, err := exe.Submit()
	if err != nil {
		t.Fatalf("submit: %v", err)
	}
	err = f.Wait()
	if err == nil {
		t.Fatal("a panicking kernel should report through the fence")
	}
	// The kernel's name and node, and the source line that failed: a panic
	// recovered in a goroutine the caller did not start has no other way of
	// telling them which statement it was.
	for _, want := range []string{"Overrun", "node 3", "out-of-bounds", site} {
		if !strings.Contains(err.Error(), want) {
			t.Errorf("the message should say %q, got:\n%v", want, err)
		}
	}

	// And the executable is usable afterwards: the panic did not leave it
	// believing a submission is still in flight.
	if _, err := exe.Submit(); err != nil {
		t.Errorf("the executable should still be usable: %v", err)
	}
}
