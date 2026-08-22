// Copyright 2026 The golang.design Initiative Authors.
// All rights reserved. Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

package accel

import (
	"errors"
	"strings"
	"testing"
)

// TestViewCheckAtUse covers spec 001 section 7.3's rule that every *use* of a
// view re-validates it against the live buffer.
//
// It is a white-box test because M1 has no public use site: the immediate
// transfer path addresses buffers, and recorded copies, which are what address
// views, arrive with the graph at M3. The rule is implemented and proven here
// so that wiring it up at M3 is wiring rather than design, and so a
// hand-constructed view is never the thing that discovers it.
func TestViewCheckAtUse(t *testing.T) {
	d, err := OpenCPU(CPUOptions{})
	if err != nil {
		t.Fatal(err)
	}
	defer d.Close()
	p, err := d.NewPool(MemoryDevice, 1<<20)
	if err != nil {
		t.Fatal(err)
	}
	defer p.Close()

	b, err := p.Alloc(BufferDescriptor{DType: F32, Count: 16, Label: "small"})
	if err != nil {
		t.Fatal(err)
	}

	good, err := b.View(4, 8)
	if err != nil {
		t.Fatal(err)
	}
	if err := good.check("CopyBuffer"); err != nil {
		t.Fatalf("a valid view was rejected: %v", err)
	}
	if off, length := good.byteRange(); off != 16 || length != 32 {
		t.Errorf("byteRange = (%d, %d), want (16, 32)", off, length)
	}

	// A BufferView is a plain value with an exported *Buffer field, so it can be
	// built by hand with nonsense in it. Every field is re-validated against the
	// live buffer, so the worst outcome is a rejection rather than a read past
	// the allocation.
	for _, tc := range []struct {
		name string
		view BufferView
		want string
	}{
		{"count past the allocation", BufferView{Buffer: b, DType: F32, Count: 1 << 20}, "outside the buffer"},
		{"offset past the allocation", BufferView{Buffer: b, DType: F32, Offset: 1 << 20, Count: 1}, "outside the buffer"},
		{"negative offset", BufferView{Buffer: b, DType: F32, Offset: -8, Count: 1}, "outside the buffer"},
		{"negative count", BufferView{Buffer: b, DType: F32, Count: -1}, "outside the buffer"},
		{"not a dtype", BufferView{Buffer: b, DType: DType(99), Count: 1}, "not a dtype"},
		{"no buffer", BufferView{DType: F32, Count: 1}, "names no buffer"},
	} {
		err := tc.view.check("CopyBuffer")
		if err == nil {
			t.Errorf("a view with %s was accepted", tc.name)
			continue
		}
		if !strings.Contains(err.Error(), tc.want) {
			t.Errorf("%s: error %q does not explain %q", tc.name, err, tc.want)
		}
	}

	// A view of a closed buffer stays an error forever, even once the same
	// offsets have been handed to something else. The closed flag is monotonic
	// and a Buffer object is never reused for a different allocation.
	if err := b.Close(); err != nil {
		t.Fatal(err)
	}
	next, err := p.Alloc(BufferDescriptor{DType: F32, Count: 16, Label: "next"})
	if err != nil {
		t.Fatal(err)
	}
	defer next.Close()

	err = good.check("CopyBuffer")
	var le *LifetimeError
	if !errors.As(err, &le) {
		t.Fatalf("a view of a closed buffer returned %v, want *LifetimeError", err)
	}
	if le.Resource != "small" {
		t.Errorf("the error names %q, want the closed buffer", le.Resource)
	}
}
