// Copyright 2026 The golang.design Initiative Authors.
// All rights reserved. Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

package accel_test

import (
	"errors"
	"strings"
	"testing"

	"golang.design/x/accel"
)

// Spec 003's validation table, restricted to the rows this milestone owns.
// Spec 015 section 4 maps all 24 rows to a child or to a written deferral, so a
// row absent here is scoped out on the record rather than forgotten.
func TestValidationRows(t *testing.T) {
	cases := []struct {
		row  string
		what string
		says string
		run  func(t *testing.T, d *accel.Device) error
	}{{
		row:  "V2",
		what: "a buffer bound to a slot of another kind",
		says: "bound to a",
		run: func(t *testing.T, d *accel.Device) error {
			g, s := graphWithSlot(t, d, accel.SlotDescriptor{
				Name: "tex", Kind: accel.BindingSampledTexture,
				DType: accel.F32, Access: accel.AccessRead, MinCount: 4,
			})
			b := newBuffer(t, d, "b", 4, accel.BufferStorage|accel.BufferCopySrc)
			return g.Bind(accel.SlotBinding{Slot: s, Buffer: whole(t, b)})
		},
	}, {
		row:  "V3",
		what: "a view whose dtype differs from the slot's",
		says: "the slot is f32 and the view is u32",
		run: func(t *testing.T, d *accel.Device) error {
			g, s := graphWithSlot(t, d, readSlot(4))
			b := newBuffer(t, d, "b", 4, accel.BufferStorage|accel.BufferCopySrc)
			return g.Bind(accel.SlotBinding{Slot: s, Buffer: mustViewAs(t, b, accel.U32)})
		},
	}, {
		row:  "V5",
		what: "a view smaller than the recorded nodes need",
		says: "need 4 elements and the view has 2",
		run: func(t *testing.T, d *accel.Device) error {
			g, s := graphWithSlot(t, d, readSlot(4))
			b := newBuffer(t, d, "b", 4, accel.BufferStorage|accel.BufferCopySrc)
			v, err := b.View(0, 2)
			if err != nil {
				t.Fatalf("view: %v", err)
			}
			return g.Bind(accel.SlotBinding{Slot: s, Buffer: v})
		},
	}, {
		row:  "V6",
		what: "a buffer created without the usage the slot needs",
		says: "needs BufferStorage",
		run: func(t *testing.T, d *accel.Device) error {
			g, s := graphWithSlot(t, d, readSlot(4))
			b := newBuffer(t, d, "b", 4, accel.BufferCopySrc)
			return g.Bind(accel.SlotBinding{Slot: s, Buffer: whole(t, b)})
		},
	}, {
		row:  "V1",
		what: "a copy from a slot this recorder did not declare",
		says: "CopyFromSlot: slot 99 was not declared by this recorder",
		run: func(t *testing.T, d *accel.Device) error {
			dst := newBuffer(t, d, "dst", 4, accel.BufferStorage|accel.BufferCopyDst)
			r := d.NewRecorder()
			r.CopyFromSlot(whole(t, dst), accel.Slot(99), 0, 4)
			_, err := r.Build()
			return err
		},
	}, {
		row:  "V1",
		what: "a copy to a slot this recorder did not declare",
		says: "CopyToSlot: slot 99 was not declared by this recorder",
		run: func(t *testing.T, d *accel.Device) error {
			src := newBuffer(t, d, "src", 4, accel.BufferStorage|accel.BufferCopySrc)
			r := d.NewRecorder()
			r.CopyToSlot(accel.Slot(99), 0, 4, whole(t, src))
			_, err := r.Build()
			return err
		},
	}, {
		row:  "V18",
		what: "a copy whose ends are different sizes",
		says: "the destination holds 16 bytes and the source 8",
		run: func(t *testing.T, d *accel.Device) error {
			dst := newBuffer(t, d, "dst", 4, accel.BufferStorage|accel.BufferCopyDst)
			src := newBuffer(t, d, "src", 2, accel.BufferStorage|accel.BufferCopySrc)
			r := d.NewRecorder()
			r.CopyBuffer(whole(t, dst), whole(t, src))
			_, err := r.Build()
			return err
		},
	}, {
		row:  "V18",
		what: "a copy reaching past its buffer",
		says: "outside the buffer",
		run: func(t *testing.T, d *accel.Device) error {
			dst := newBuffer(t, d, "dst", 4, accel.BufferStorage|accel.BufferCopyDst)
			r := d.NewRecorder()
			r.UploadToBuffer(accel.BufferView{Buffer: dst, DType: accel.F32, Offset: 2, Count: 4},
				[]float32{1, 2, 3, 4})
			_, err := r.Build()
			return err
		},
	}, {
		row:  "V19",
		what: "a resource closed before the graph is built",
		says: "closed",
		run: func(t *testing.T, d *accel.Device) error {
			b, err := d.NewBuffer(accel.BufferDescriptor{
				DType: accel.F32, Count: 4, Usage: accel.BufferStorage | accel.BufferCopyDst, Label: "gone",
			})
			if err != nil {
				t.Fatalf("buffer: %v", err)
			}
			v := whole(t, b)
			if err := b.Close(); err != nil {
				t.Fatalf("close: %v", err)
			}
			r := d.NewRecorder()
			r.UploadToBuffer(v, []float32{1, 2, 3, 4})
			_, err = r.Build()
			return err
		},
	}, {
		row:  "V19",
		what: "a resource closed after binding but before submitting",
		says: "closed",
		run: func(t *testing.T, d *accel.Device) error {
			g, s := graphWithSlot(t, d, readSlot(4))
			b, err := d.NewBuffer(accel.BufferDescriptor{
				DType: accel.F32, Count: 4, Usage: accel.BufferStorage | accel.BufferCopySrc, Label: "gone",
			})
			if err != nil {
				t.Fatalf("buffer: %v", err)
			}
			if err := g.Bind(accel.SlotBinding{Slot: s, Buffer: whole(t, b)}); err != nil {
				t.Fatalf("bind: %v", err)
			}
			if err := b.Close(); err != nil {
				t.Fatalf("close: %v", err)
			}
			return d.Queue().Submit(g).Wait()
		},
	}, {
		row:  "V19",
		what: "a resource from another device",
		says: "different device",
		run: func(t *testing.T, d *accel.Device) error {
			other := openDevice(t)
			b := newBuffer(t, other, "elsewhere", 4, accel.BufferStorage|accel.BufferCopyDst)
			r := d.NewRecorder()
			r.UploadToBuffer(whole(t, b), []float32{1, 2, 3, 4})
			_, err := r.Build()
			return err
		},
	}, {
		row:  "V21",
		what: "a use reaching past what the slot descriptor promised",
		says: "at least 16 bytes and this use is",
		run: func(t *testing.T, d *accel.Device) error {
			dst := newBuffer(t, d, "dst", 8, accel.BufferStorage|accel.BufferCopyDst)
			r := d.NewRecorder()
			s := r.Slot(readSlot(4))
			r.CopyFromSlot(whole(t, dst), s, 0, 8)
			_, err := r.Build()
			return err
		},
	}, {
		row:  "V21",
		what: "a use whose mode the slot descriptor does not permit",
		says: "declared read and this use is write",
		run: func(t *testing.T, d *accel.Device) error {
			src := newBuffer(t, d, "src", 4, accel.BufferStorage|accel.BufferCopySrc)
			r := d.NewRecorder()
			s := r.Slot(readSlot(4))
			r.CopyToSlot(s, 0, 4, whole(t, src))
			_, err := r.Build()
			return err
		},
	}, {
		row:  "V1 (slots)",
		what: "a submission with a slot left unbound",
		says: "has no resource bound",
		run: func(t *testing.T, d *accel.Device) error {
			g, _ := graphWithSlot(t, d, readSlot(4))
			return d.Queue().Submit(g).Wait()
		},
	}, {
		row:  "V24",
		what: "two slots bound to overlapping bytes where one writes",
		says: "and at least one of them writes",
		run: func(t *testing.T, d *accel.Device) error {
			r := d.NewRecorder()
			in := r.Slot(readSlot(4))
			out := r.Slot(accel.SlotDescriptor{
				Name: "out", Kind: accel.BindingStorageBuffer,
				DType: accel.F32, Access: accel.AccessWrite, MinCount: 4,
			})
			mid := r.Transient(accel.BufferDescriptor{
				DType: accel.F32, Count: 4,
				Usage: accel.BufferStorage | accel.BufferCopySrc | accel.BufferCopyDst,
			})
			r.CopyFromSlot(mid, in, 0, 4)
			r.CopyToSlot(out, 0, 4, mid)
			g, err := r.Build()
			if err != nil {
				t.Fatalf("build: %v", err)
			}
			t.Cleanup(func() { _ = g.Close() })

			b := newBuffer(t, d, "shared", 4,
				accel.BufferStorage|accel.BufferCopySrc|accel.BufferCopyDst)
			return g.Bind([]accel.SlotBinding{
				{Slot: in, Buffer: whole(t, b)},
				{Slot: out, Buffer: whole(t, b)},
			}...)
		},
	}, {
		row:  "V24",
		what: "a slot bound over a concrete resource the graph writes",
		says: "and at least one of them writes",
		run: func(t *testing.T, d *accel.Device) error {
			b := newBuffer(t, d, "shared", 4,
				accel.BufferStorage|accel.BufferCopySrc|accel.BufferCopyDst)
			r := d.NewRecorder()
			in := r.Slot(readSlot(4))
			r.UploadToBuffer(whole(t, b), []float32{1, 2, 3, 4}) // a concrete write
			r.CopyFromSlot(whole(t, b), in, 0, 4)
			g, err := r.Build()
			if err != nil {
				t.Fatalf("build: %v", err)
			}
			t.Cleanup(func() { _ = g.Close() })
			return g.Bind(accel.SlotBinding{Slot: in, Buffer: whole(t, b)})
		},
	}}

	for _, c := range cases {
		t.Run(c.row+"/"+c.what, func(t *testing.T) {
			d := openDevice(t)
			err := c.run(t, d)
			if err == nil {
				t.Fatalf("%s: expected a rejection", c.row)
			}
			if !strings.Contains(err.Error(), c.says) {
				t.Fatalf("%s: the message should say %q, got:\n%v", c.row, c.says, err)
			}
		})
	}
}

// Two slots bound to the same read-only bytes is legal: nothing writes, so no
// hazard was missed. Without this the V24 rows above would pass against a check
// that simply rejects every overlap.
func TestOverlappingReadOnlySlotsAreAccepted(t *testing.T) {
	d := openDevice(t)
	dst := newBuffer(t, d, "dst", 8, accel.BufferStorage|accel.BufferCopyDst)

	r := d.NewRecorder()
	a := r.Slot(readSlot(4))
	b := r.Slot(readSlot(4))
	lo, err := dst.View(0, 4)
	if err != nil {
		t.Fatalf("view: %v", err)
	}
	hi, err := dst.View(4, 4)
	if err != nil {
		t.Fatalf("view: %v", err)
	}
	r.CopyFromSlot(lo, a, 0, 4)
	r.CopyFromSlot(hi, b, 0, 4)
	g, err := r.Build()
	if err != nil {
		t.Fatalf("build: %v", err)
	}
	defer g.Close()

	src := newBuffer(t, d, "src", 4, accel.BufferStorage|accel.BufferCopySrc)
	if err := g.Bind([]accel.SlotBinding{
		{Slot: a, Buffer: whole(t, src)},
		{Slot: b, Buffer: whole(t, src)},
	}...); err != nil {
		t.Fatalf("two read-only slots over one buffer should be accepted: %v", err)
	}
}

// Disjoint ranges of one buffer do not overlap, even when one writes. A check
// comparing resource identity rather than bytes would reject this.
func TestDisjointRangesOfOneBufferAreAccepted(t *testing.T) {
	d := openDevice(t)
	shared := newBuffer(t, d, "shared", 8,
		accel.BufferStorage|accel.BufferCopySrc|accel.BufferCopyDst)

	r := d.NewRecorder()
	in := r.Slot(readSlot(4))
	out := r.Slot(accel.SlotDescriptor{
		Name: "out", Kind: accel.BindingStorageBuffer,
		DType: accel.F32, Access: accel.AccessWrite, MinCount: 4,
	})
	mid := r.Transient(accel.BufferDescriptor{
		DType: accel.F32, Count: 4,
		Usage: accel.BufferStorage | accel.BufferCopySrc | accel.BufferCopyDst,
	})
	r.CopyFromSlot(mid, in, 0, 4)
	r.CopyToSlot(out, 0, 4, mid)
	g, err := r.Build()
	if err != nil {
		t.Fatalf("build: %v", err)
	}
	defer g.Close()

	lo, err := shared.View(0, 4)
	if err != nil {
		t.Fatalf("view: %v", err)
	}
	hi, err := shared.View(4, 4)
	if err != nil {
		t.Fatalf("view: %v", err)
	}
	if err := g.Bind([]accel.SlotBinding{
		{Slot: in, Buffer: lo},
		{Slot: out, Buffer: hi},
	}...); err != nil {
		t.Fatalf("disjoint halves of one buffer should be accepted: %v", err)
	}
}

// A rejected batch leaves the previous bindings intact: a caller cannot see
// which half of a partially applied update landed.
func TestARejectedRebindChangesNothing(t *testing.T) {
	d := openDevice(t)
	g, s := graphWithSlot(t, d, readSlot(4))
	good := newBuffer(t, d, "good", 4, accel.BufferStorage|accel.BufferCopySrc)
	small := newBuffer(t, d, "small", 2, accel.BufferStorage|accel.BufferCopySrc)

	if err := g.Bind(accel.SlotBinding{Slot: s, Buffer: whole(t, good)}); err != nil {
		t.Fatalf("bind: %v", err)
	}
	if err := d.Queue().WriteBuffer(good, 0, []float32{1, 2, 3, 4}); err != nil {
		t.Fatalf("write: %v", err)
	}
	if err := g.Bind([]accel.SlotBinding{{Slot: s, Buffer: whole(t, small)}}...); err == nil {
		t.Fatal("binding a too-small view should fail")
	}
	// The good binding survives, so the graph still runs.
	if err := d.Queue().Submit(g).Wait(); err != nil {
		t.Fatalf("the earlier binding should still be in place: %v", err)
	}
}

func TestBindRejectsMalformedBindings(t *testing.T) {
	d := openDevice(t)
	g, s := graphWithSlot(t, d, readSlot(4))
	b := newBuffer(t, d, "b", 4, accel.BufferStorage|accel.BufferCopySrc)

	for _, c := range []struct {
		name string
		bind accel.SlotBinding
		says string
	}{
		{"slot zero", accel.SlotBinding{Buffer: whole(t, b)}, "slot 0 is not one of"},
		{"unknown slot", accel.SlotBinding{Slot: s + 7, Buffer: whole(t, b)}, "is not one of"},
		{"no resource", accel.SlotBinding{Slot: s}, "no resource bound"},
		// A texture through a slot used to be a runtime rejection here. It is a
		// compile error now: SlotBinding has no Texture field, because the bind
		// path and the dispatch path stopped sharing a type. A refusal the
		// compiler makes beats one a test has to find.
	} {
		t.Run(c.name, func(t *testing.T) {
			err := g.Bind(c.bind)
			if err == nil || !strings.Contains(err.Error(), c.says) {
				t.Fatalf("the message should say %q, got %v", c.says, err)
			}
		})
	}
}

// A transient is the builder's, and a caller reaching into one directly would
// be reading memory the builder may hand to another node.
func TestATransientIsNotACallerResource(t *testing.T) {
	d := openDevice(t)
	r := d.NewRecorder()
	v := r.Transient(accel.BufferDescriptor{
		DType: accel.F32, Count: 4, Usage: accel.BufferStorage | accel.BufferCopyDst,
	})
	g, err := r.Build()
	if err != nil {
		t.Fatalf("build: %v", err)
	}
	defer g.Close()

	err = d.Queue().WriteBuffer(v.Buffer, 0, []float32{1, 2, 3, 4})
	if err == nil || !strings.Contains(err.Error(), "graph transient") {
		t.Fatalf("writing a transient directly should be refused, got %v", err)
	}

	// And one recorder's transient is not another's.
	other := d.NewRecorder()
	other.UploadToBuffer(v, []float32{1, 2, 3, 4})
	if _, err := other.Build(); err == nil ||
		!strings.Contains(err.Error(), "another graph's transient") {
		t.Fatalf("expected a foreign-transient rejection, got %v", err)
	}
}

// Device loss is terminal. Spec 001 section 7.4 calls this path close to
// untestable at v0, which is why the CPU backend has an injection mode.
func TestDeviceLossIsTerminal(t *testing.T) {
	d, err := accel.OpenCPU(accel.CPUOptions{LoseAtSubmission: 1})
	if err != nil {
		t.Fatalf("open: %v", err)
	}
	defer d.Close()

	dst, err := d.NewBuffer(accel.BufferDescriptor{
		DType: accel.F32, Count: 4, Usage: accel.BufferStorage | accel.BufferCopyDst, Label: "dst",
	})
	if err != nil {
		t.Fatalf("buffer: %v", err)
	}
	defer dst.Close()
	v, err := dst.View(0, 4)
	if err != nil {
		t.Fatalf("view: %v", err)
	}

	r := d.NewRecorder()
	r.UploadToBuffer(v, []float32{1, 2, 3, 4})
	g, err := r.Build()
	if err != nil {
		t.Fatalf("build: %v", err)
	}
	defer g.Close()

	// The fence reports the loss rather than never signalling, which is the
	// whole reason loss is on the fence at all.
	if err := d.Queue().Submit(g).Wait(); !errors.Is(err, accel.ErrDeviceLost) {
		t.Fatalf("the submission should report loss, got %v", err)
	}
	// And it sticks: every subsequent call reports it.
	if err := d.Queue().Submit(g).Wait(); !errors.Is(err, accel.ErrDeviceLost) {
		t.Fatalf("a later submission should report loss, got %v", err)
	}
	if _, err := d.NewRecorder().Build(); !errors.Is(err, accel.ErrDeviceLost) {
		t.Fatalf("Build after loss should report it, got %v", err)
	}
	if err := g.Bind(accel.SlotBinding{}); !errors.Is(err, accel.ErrDeviceLost) {
		t.Fatalf("Bind after loss should report it, got %v", err)
	}
}

func readSlot(n int) accel.SlotDescriptor {
	return accel.SlotDescriptor{
		Name: "in", Kind: accel.BindingStorageBuffer,
		DType: accel.F32, Access: accel.AccessRead, MinCount: n,
	}
}

// graphWithSlot builds the smallest graph that reads one slot.
func graphWithSlot(t *testing.T, d *accel.Device, desc accel.SlotDescriptor) (*accel.Graph, accel.Slot) {
	t.Helper()
	dst := newBuffer(t, d, "sink", desc.MinCount, accel.BufferStorage|accel.BufferCopyDst)
	r := d.NewRecorder()
	s := r.Slot(desc)
	r.CopyFromSlot(whole(t, dst), s, 0, desc.MinCount)
	g, err := r.Build()
	if err != nil {
		t.Fatalf("build: %v", err)
	}
	t.Cleanup(func() { _ = g.Close() })
	return g, s
}

// A transient cannot be bound through a slot, which is what makes V24's
// transient term unreachable rather than missing.
//
// specs/015-graph-recording.md §4 expected V24 to gain a term once transients
// had placements: a slot-supplied resource could then overlap one. It cannot,
// because BufferView.check refuses a transient at bind time — and that refusal
// is stricter than an overlap test, since it rejects every transient offered
// through a slot rather than only the overlapping ones.
//
// This test pins the upstream refusal, not the overlap. If someone relaxes
// BufferView.check, V24 becomes incomplete without V24 being touched, and this
// is the test that says so. See specs/017-graph-aliasing.md §5.
func TestATransientCannotReachASlot(t *testing.T) {
	d := openDevice(t)
	r := d.NewRecorder()
	in := r.Slot(accel.SlotDescriptor{
		Name: "in", Kind: accel.BindingStorageBuffer,
		DType: accel.F32, Access: accel.AccessRead, MinCount: 4,
	})
	mid := r.Transient(accel.BufferDescriptor{
		DType: accel.F32, Count: 4,
		Usage: accel.BufferStorage | accel.BufferCopySrc | accel.BufferCopyDst, Label: "mid",
	})
	r.CopyFromSlot(mid, in, 0, 4)

	g, err := r.Build()
	if err != nil {
		t.Fatalf("build: %v", err)
	}
	t.Cleanup(func() { _ = g.Close() })

	err = g.Bind([]accel.SlotBinding{{Slot: in, Buffer: mid}}...)
	if err == nil {
		t.Fatal("a graph transient was bound through a slot; if this is now legal, V24 " +
			"needs the transient term specs/017-graph-aliasing.md §5 records as unreachable")
	}
	if !strings.Contains(err.Error(), "graph transient") {
		t.Errorf("the refusal should name what the resource is: %v", err)
	}
}

// Each binding refusal is matchable with errors.Is.
//
// specs/003-command-graph.md's error taxonomy, the half of it that shipped on
// 2026-08-27. The spec's own argument for these is a caller writing a fallback
// path — try an f16 layout, fall back to f32 — who has to tell a dtype
// disagreement from a size one. Before this they could only match a string,
// which is a contract nobody wrote down and every message change breaks.
//
// The messages are asserted elsewhere; what this asserts is the sentinel, since
// those are two different promises and only one of them is machine-readable.
func TestEveryBindingRefusalIsMatchable(t *testing.T) {
	d := openDevice(t)

	for _, tc := range []struct {
		name string
		is   error
		run  func(t *testing.T) error
	}{{
		name: "a slot index the graph does not have",
		is:   accel.ErrSlotUnbound,
		run: func(t *testing.T) error {
			g, _ := graphWithSlot(t, d, readSlot(4))
			src := newBuffer(t, d, "src", 4, accel.BufferStorage)
			return g.Bind(accel.SlotBinding{Slot: accel.Slot(99), Buffer: whole(t, src)})
		},
	}, {
		name: "a slot left unbound at submit",
		is:   accel.ErrSlotUnbound,
		run: func(t *testing.T) error {
			g, _ := graphWithSlot(t, d, readSlot(4))
			return d.Queue().Submit(g).Wait()
		},
	}, {
		name: "a view whose dtype is not the slot's",
		is:   accel.ErrDTypeMismatch,
		run: func(t *testing.T) error {
			g, s := graphWithSlot(t, d, readSlot(4))
			src := newBuffer(t, d, "src", 4, accel.BufferStorage)
			v := whole(t, src)
			v.DType = accel.U32
			return g.Bind(accel.SlotBinding{Slot: s, Buffer: v})
		},
	}, {
		name: "a view smaller than the recorded nodes need",
		is:   accel.ErrTooSmall,
		run: func(t *testing.T) error {
			g, s := graphWithSlot(t, d, readSlot(8))
			src := newBuffer(t, d, "src", 2, accel.BufferStorage)
			return g.Bind(accel.SlotBinding{Slot: s, Buffer: whole(t, src)})
		},
	}, {
		name: "a buffer created without the usage the slot needs",
		is:   accel.ErrUsageMissing,
		run: func(t *testing.T) error {
			g, s := graphWithSlot(t, d, readSlot(4))
			src := newBuffer(t, d, "src", 4, accel.BufferCopyDst)
			return g.Bind(accel.SlotBinding{Slot: s, Buffer: whole(t, src)})
		},
	}, {
		name: "a buffer from another device",
		is:   accel.ErrForeignResource,
		run: func(t *testing.T) error {
			g, s := graphWithSlot(t, d, readSlot(4))
			other := openDevice(t)
			src := newBuffer(t, other, "src", 4, accel.BufferStorage)
			return g.Bind(accel.SlotBinding{Slot: s, Buffer: whole(t, src)})
		},
	}} {
		t.Run(tc.name, func(t *testing.T) {
			err := tc.run(t)
			if err == nil {
				t.Fatalf("%s was accepted", tc.name)
			}
			if !errors.Is(err, tc.is) {
				t.Errorf("errors.Is(_, %v) is false for:\n%v", tc.is, err)
			}
		})
	}
}
