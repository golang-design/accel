// Copyright 2026 The golang.design Initiative Authors.
// All rights reserved. Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

package driver_test

import (
	"errors"
	"strings"
	"testing"

	"golang.design/x/accel/internal/cpu"
	"golang.design/x/accel/internal/driver"
	"golang.design/x/accel/internal/kernel"
)

func openCPU(t *testing.T, o cpu.Options) driver.Device {
	t.Helper()
	dev, err := cpu.Adapter{}.Open(o)
	if err != nil {
		t.Fatalf("open: %v", err)
	}
	t.Cleanup(func() { _ = dev.Close() })
	return dev
}

func alloc(t *testing.T, dev driver.Device, n int) driver.Block {
	t.Helper()
	b, err := dev.Alloc(driver.MemoryShared, n, "test")
	if err != nil {
		t.Fatalf("alloc: %v", err)
	}
	t.Cleanup(b.Free)
	return b
}

// An operand cannot be half-built. The alternative shape, a struct with a block
// field and a slot field and nothing enforcing that exactly one is set, admits
// a node that copies nothing and reports success.
func TestOperandsCannotBeHalfBuilt(t *testing.T) {
	var zero driver.Operand
	if zero.Kind() != driver.OperandUnset {
		t.Fatalf("the zero operand should be unset, got %v", zero.Kind())
	}

	if _, err := driver.BlockOperand(nil, 0, 4); err == nil {
		t.Fatal("a block operand with no block should be rejected")
	}
	if _, err := driver.SlotOperand(0, 0, 4); err == nil {
		t.Fatal("slot 0 should be rejected: indices start at one")
	}
	if _, err := driver.SlotOperand(-1, 0, 4); err == nil {
		t.Fatal("a negative slot should be rejected")
	}
}

func TestBlockOperandChecksItsRange(t *testing.T) {
	dev := openCPU(t, cpu.Options{})
	b := alloc(t, dev, 64)

	for _, c := range []struct{ off, size int }{
		{-1, 4}, {0, -1}, {65, 0}, {60, 8}, {0, 65},
	} {
		if _, err := driver.BlockOperand(b, c.off, c.size); err == nil {
			t.Errorf("[%d, %d) of 64 bytes should be rejected", c.off, c.off+c.size)
		}
	}
	if _, err := driver.BlockOperand(b, 60, 4); err != nil {
		t.Errorf("the last four bytes should be accepted: %v", err)
	}
}

// A range that wraps when offset and size are added must be rejected rather
// than admitted by an int overflow, which is the bug the buffer views had.
func TestOperandRangesCannotWrap(t *testing.T) {
	dev := openCPU(t, cpu.Options{})
	b := alloc(t, dev, 64)
	const huge = int(^uint(0) >> 1)
	if _, err := driver.BlockOperand(b, huge, huge); err == nil {
		t.Fatal("an operand whose offset plus size overflows should be rejected")
	}
}

func TestValidateRejectsMalformedPlans(t *testing.T) {
	dev := openCPU(t, cpu.Options{})
	b := alloc(t, dev, 64)
	op := func(off, size int) driver.Operand {
		o, err := driver.BlockOperand(b, off, size)
		if err != nil {
			t.Fatalf("operand: %v", err)
		}
		return o
	}
	slot := func(i, off, size int) driver.Operand {
		o, err := driver.SlotOperand(i, off, size)
		if err != nil {
			t.Fatalf("operand: %v", err)
		}
		return o
	}

	cases := []struct {
		name string
		plan driver.Plan
		says string
	}{
		{"no op", driver.Plan{Nodes: []driver.PlanNode{{Dst: op(0, 4)}}}, "no operation"},
		{"no destination", driver.Plan{Nodes: []driver.PlanNode{{Op: driver.OpCopy, Src: op(0, 4)}}}, "no destination operand"},
		{"no source", driver.Plan{Nodes: []driver.PlanNode{{Op: driver.OpCopy, Dst: op(0, 4)}}}, "no source operand"},
		{"size mismatch", driver.Plan{Nodes: []driver.PlanNode{{Op: driver.OpCopy, Dst: op(0, 4), Src: op(8, 8)}}}, "copies 8 bytes into 4"},
		{"host write length", driver.Plan{Nodes: []driver.PlanNode{{Op: driver.OpHostWrite, Dst: op(0, 4), Data: []byte{1}}}}, "writes 1 bytes into a 4-byte"},
		{"host write with source", driver.Plan{Nodes: []driver.PlanNode{{Op: driver.OpHostWrite, Dst: op(0, 4), Src: op(8, 4), Data: []byte{1, 2, 3, 4}}}}, "has a source operand"},
		{"slot out of range", driver.Plan{Slots: 1, Nodes: []driver.PlanNode{{Op: driver.OpHostWrite, Dst: slot(2, 0, 4), Data: []byte{1, 2, 3, 4}}}}, "names slot 2 of 1"},
		{"negative slot count", driver.Plan{Slots: -1}, "declares -1 slots"},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			err := c.plan.Validate()
			if err == nil {
				t.Fatalf("expected a rejection")
			}
			if !strings.Contains(err.Error(), c.says) {
				t.Fatalf("error should say %q, got %v", c.says, err)
			}
		})
	}

	var nilPlan *driver.Plan
	if err := nilPlan.Validate(); err == nil {
		t.Fatal("a nil plan should be rejected")
	}
}

// Compile is where a malformed plan must be caught, not execution: a backend
// that discovers a nil block while replaying has already started work.
func TestCompileRejectsAnInvalidPlan(t *testing.T) {
	dev := openCPU(t, cpu.Options{})
	c, ok := dev.(driver.GraphCompiler)
	if !ok {
		t.Fatal("the CPU device should compile graphs")
	}
	if _, err := c.Compile(&driver.Plan{Nodes: []driver.PlanNode{{}}}); err == nil {
		t.Fatal("a node with no operation should be rejected at compile")
	}
}

func TestOperandStringNamesItsCase(t *testing.T) {
	dev := openCPU(t, cpu.Options{})
	b := alloc(t, dev, 64)
	blk, _ := driver.BlockOperand(b, 8, 16)
	slt, _ := driver.SlotOperand(2, 4, 8)
	var unset driver.Operand
	for _, c := range []struct{ got, want string }{
		{blk.String(), "block[8:24]"},
		{slt.String(), "slot 2[4:12]"},
		{unset.String(), "unset operand"},
	} {
		if c.got != c.want {
			t.Errorf("got %q, want %q", c.got, c.want)
		}
	}
	if blk.Block() == nil || blk.Slot() != 0 || slt.Block() != nil || slt.Slot() != 2 {
		t.Error("accessors should report the case the operand was built as")
	}
	if driver.OpCopy.String() != "copy" || driver.OpHostWrite.String() != "host write" ||
		driver.OpInvalid.String() != "invalid" {
		t.Error("ops should name themselves")
	}
}

func TestErrDeviceLostIsSticky(t *testing.T) {
	dev := openCPU(t, cpu.Options{LoseAtSubmission: 1})
	if err := dev.Lost(); err != nil {
		t.Fatalf("a fresh device is not lost: %v", err)
	}
	b := alloc(t, dev, 16)
	c := dev.(driver.GraphCompiler)
	dst, _ := driver.BlockOperand(b, 0, 4)
	exe, err := c.Compile(&driver.Plan{Nodes: []driver.PlanNode{
		{Op: driver.OpHostWrite, Dst: dst, Data: []byte{1, 2, 3, 4}},
	}})
	if err != nil {
		t.Fatalf("compile: %v", err)
	}
	defer exe.Close()

	if _, err := exe.Submit(); !errors.Is(err, driver.ErrDeviceLost) {
		t.Fatalf("the first submission should report loss, got %v", err)
	}
	if err := dev.Lost(); !errors.Is(err, driver.ErrDeviceLost) {
		t.Fatalf("loss should stick, got %v", err)
	}
	// Every subsequent call reports it. A device that recovered would leave a
	// caller running on resources whose contents are undefined.
	if _, err := exe.Submit(); !errors.Is(err, driver.ErrDeviceLost) {
		t.Fatalf("a later submission should report loss, got %v", err)
	}
	if _, err := dev.Alloc(driver.MemoryShared, 16, "after"); !errors.Is(err, driver.ErrDeviceLost) {
		t.Fatalf("allocation after loss should report it, got %v", err)
	}
	if _, err := c.Compile(&driver.Plan{}); !errors.Is(err, driver.ErrDeviceLost) {
		t.Fatalf("compile after loss should report it, got %v", err)
	}
}

// Loss at a later submission is what makes the path testable mid-graph: the
// first submission succeeds, the second reports the loss.
func TestLossCanBeInjectedAtALaterSubmission(t *testing.T) {
	dev := openCPU(t, cpu.Options{LoseAtSubmission: 2})
	b := alloc(t, dev, 16)
	dst, _ := driver.BlockOperand(b, 0, 4)
	exe, err := dev.(driver.GraphCompiler).Compile(&driver.Plan{Nodes: []driver.PlanNode{
		{Op: driver.OpHostWrite, Dst: dst, Data: []byte{1, 2, 3, 4}},
	}})
	if err != nil {
		t.Fatalf("compile: %v", err)
	}
	defer exe.Close()

	f, err := exe.Submit()
	if err != nil {
		t.Fatalf("the first submission should succeed: %v", err)
	}
	if err := f.Wait(); err != nil {
		t.Fatalf("wait: %v", err)
	}
	if _, err := exe.Submit(); !errors.Is(err, driver.ErrDeviceLost) {
		t.Fatalf("the second submission should report loss, got %v", err)
	}
}

func TestValidateRejectsMalformedDispatches(t *testing.T) {
	dev := openCPU(t, cpu.Options{})
	b := alloc(t, dev, 64)
	op, err := driver.BlockOperand(b, 0, 16)
	if err != nil {
		t.Fatalf("operand: %v", err)
	}
	k := &kernel.Kernel{
		Name: "K", WorkgroupSize: kernel.ID3{X: 1, Y: 1, Z: 1},
		Bindings: []kernel.Binding{{Name: "in", DType: kernel.F32, Access: kernel.Read}},
		Flat:     func(kernel.Thread, kernel.Args) {},
	}
	coop := *k
	coop.Flat = nil

	cases := []struct {
		name string
		node driver.PlanNode
		says string
	}{
		{"no payload", driver.PlanNode{Op: driver.OpDispatch}, "no payload"},
		{"no kernel", driver.PlanNode{Op: driver.OpDispatch, Dispatch: &driver.Dispatch{}}, "dispatches no kernel"},
		{"cooperative", driver.PlanNode{Op: driver.OpDispatch,
			Dispatch: &driver.Dispatch{Kernel: &coop}}, "cooperative"},
		{"wrong binding count", driver.PlanNode{Op: driver.OpDispatch,
			Dispatch: &driver.Dispatch{Kernel: k}}, "binds 0 resources"},
		{"unset binding operand", driver.PlanNode{Op: driver.OpDispatch,
			Dispatch: &driver.Dispatch{Kernel: k, Bindings: []driver.Operand{{}}}}, `no "in" operand`},
		{"slot out of range", driver.PlanNode{Op: driver.OpDispatch,
			Dispatch: &driver.Dispatch{Kernel: k, Bindings: []driver.Operand{mustSlot(t, 3, 0, 16)}}},
			"names slot 3 of 0"},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			p := driver.Plan{Nodes: []driver.PlanNode{c.node}}
			err := p.Validate()
			if err == nil || !strings.Contains(err.Error(), c.says) {
				t.Fatalf("the message should say %q, got %v", c.says, err)
			}
		})
	}

	// A well-formed one is accepted, so the rows above are rejections rather
	// than a validator that refuses every dispatch.
	ok := driver.Plan{Nodes: []driver.PlanNode{{Op: driver.OpDispatch,
		Dispatch: &driver.Dispatch{Kernel: k, Bindings: []driver.Operand{op}}}}}
	if err := ok.Validate(); err != nil {
		t.Fatalf("a well-formed dispatch should validate: %v", err)
	}
	if driver.OpDispatch.String() != "dispatch" {
		t.Errorf("OpDispatch should name itself, got %q", driver.OpDispatch.String())
	}
}

func mustSlot(t *testing.T, slot, off, size int) driver.Operand {
	t.Helper()
	o, err := driver.SlotOperand(slot, off, size)
	if err != nil {
		t.Fatalf("slot operand: %v", err)
	}
	return o
}

// SlotOperand checks the shape it can check without a resource: the range must
// be non-negative, and the rest waits for Rebind, which is the earliest point
// the bound size is known.
func TestSlotOperandChecksWhatItCan(t *testing.T) {
	for _, c := range []struct{ off, size int }{{-1, 4}, {0, -1}} {
		if _, err := driver.SlotOperand(1, c.off, c.size); err == nil {
			t.Errorf("slot range [%d, %d) should be rejected", c.off, c.off+c.size)
		}
	}
	if _, err := driver.SlotOperand(1, 1<<40, 8); err != nil {
		t.Errorf("a large offset is legal until a resource is bound: %v", err)
	}
}

func TestValidateRejectsMalformedRowCopies(t *testing.T) {
	dev := openCPU(t, cpu.Options{})
	b := alloc(t, dev, 1024)
	op := func(off, size int) driver.Operand {
		o, err := driver.BlockOperand(b, off, size)
		if err != nil {
			t.Fatalf("operand: %v", err)
		}
		return o
	}

	cases := []struct {
		name string
		node driver.PlanNode
		says string
	}{
		{"no payload", driver.PlanNode{Op: driver.OpCopyRows,
			Dst: op(0, 64), Src: op(64, 64)}, "no payload"},
		{"no rows", driver.PlanNode{Op: driver.OpCopyRows, Dst: op(0, 64), Src: op(64, 64),
			Rows: &driver.RowCopy{Rows: 0, RowBytes: 8, DstPitch: 8, SrcPitch: 8}},
			"copies 0 rows"},
		{"a pitch shorter than a row", driver.PlanNode{Op: driver.OpCopyRows,
			Dst: op(0, 64), Src: op(64, 64),
			Rows: &driver.RowCopy{Rows: 2, RowBytes: 16, DstPitch: 8, SrcPitch: 16}},
			"cannot be shorter than one"},
		{"a destination too small", driver.PlanNode{Op: driver.OpCopyRows,
			Dst: op(0, 8), Src: op(64, 64),
			Rows: &driver.RowCopy{Rows: 4, RowBytes: 8, DstPitch: 8, SrcPitch: 8}},
			"destination is 8 bytes"},
		{"a source too small", driver.PlanNode{Op: driver.OpCopyRows,
			Dst: op(0, 64), Src: op(64, 8),
			Rows: &driver.RowCopy{Rows: 4, RowBytes: 8, DstPitch: 8, SrcPitch: 8}},
			"source is 8 bytes"},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			p := driver.Plan{Nodes: []driver.PlanNode{c.node}}
			err := p.Validate()
			if err == nil || !strings.Contains(err.Error(), c.says) {
				t.Fatalf("the message should say %q, got %v", c.says, err)
			}
		})
	}

	// The last row needs only its own bytes, not a full pitch: a tightly packed
	// image's final row has no padding after it, and requiring one would refuse
	// a correctly sized buffer.
	exact := driver.Plan{Nodes: []driver.PlanNode{{
		Op: driver.OpCopyRows, Dst: op(0, 4*8), Src: op(64, 3*16+8),
		Rows: &driver.RowCopy{Rows: 4, RowBytes: 8, DstPitch: 8, SrcPitch: 16},
	}}}
	if err := exact.Validate(); err != nil {
		t.Fatalf("a buffer sized for its last row without trailing padding was refused: %v", err)
	}
	if driver.OpCopyRows.String() != "row copy" {
		t.Errorf("OpCopyRows names itself %q", driver.OpCopyRows.String())
	}
}

// A pitched copy moves each row by its own stride, which is what a
// texture-buffer copy is.
func TestARowCopyStepsByEachSidesPitch(t *testing.T) {
	dev, c := openCompiler(t, cpu.Options{})
	src := alloc(t, dev, 64)
	dst := alloc(t, dev, 64)

	// Three rows of four bytes: the source strides by eight and the
	// destination by four, so the padding between source rows is dropped.
	for i := range src.Bytes() {
		src.Bytes()[i] = byte(i)
	}
	srcOp, _ := driver.BlockOperand(src, 0, 3*8)
	dstOp, _ := driver.BlockOperand(dst, 0, 3*4)

	exe, err := c.Compile(&driver.Plan{Nodes: []driver.PlanNode{{
		Op: driver.OpCopyRows, Dst: dstOp, Src: srcOp,
		Rows: &driver.RowCopy{Rows: 3, RowBytes: 4, DstPitch: 4, SrcPitch: 8},
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

	for row := range 3 {
		for i := range 4 {
			want := byte(row*8 + i)
			if got := dst.Bytes()[row*4+i]; got != want {
				t.Fatalf("row %d byte %d is %d, want %d: each side steps by its own pitch",
					row, i, got, want)
			}
		}
	}
}

func openCompiler(t *testing.T, o cpu.Options) (driver.Device, driver.GraphCompiler) {
	t.Helper()
	dev := openCPU(t, o)
	c, ok := dev.(driver.GraphCompiler)
	if !ok {
		t.Fatal("the CPU device should compile graphs")
	}
	return dev, c
}
