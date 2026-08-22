// Copyright 2026 The golang.design Initiative Authors.
// All rights reserved. Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

package cpu_test

import (
	"bytes"
	"strings"
	"sync"
	"testing"

	"golang.design/x/accel/internal/cpu"
	"golang.design/x/accel/internal/driver"
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

// Device memory has no host mapping, so a plan touching it must fail with a
// reason rather than by indexing a nil slice.
func TestDeviceMemoryIsNotReachableFromAPlanYet(t *testing.T) {
	dev, c := open(t, cpu.Options{})
	priv := block(t, dev, driver.MemoryDevice, 16)
	exe, err := c.Compile(&driver.Plan{Nodes: []driver.PlanNode{
		{Op: driver.OpHostWrite, Dst: blockOperand(t, priv, 0, 4), Data: []byte{1, 2, 3, 4}},
	}})
	if err != nil {
		t.Fatalf("compile: %v", err)
	}
	defer exe.Close()
	if _, err := exe.Submit(); err == nil || !strings.Contains(err.Error(), "not host-visible") {
		t.Fatalf("expected a host-visibility rejection, got %v", err)
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
