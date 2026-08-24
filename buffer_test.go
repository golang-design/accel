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

// TestBufferAccessors checks that a buffer reports the descriptor it was made
// from, since every later validation error quotes those numbers back.
func TestBufferAccessors(t *testing.T) {
	d := openCPU(t, accel.CPUOptions{})
	p := newPool(t, d, accel.MemoryDevice, 1<<20)
	defer p.Close()

	b := alloc(t, p, accel.BufferDescriptor{
		DType: accel.F16, Count: 300, Usage: accel.BufferStorage | accel.BufferCopyDst, Label: "kv",
	})
	defer b.Close()

	if got := b.DType(); got != accel.F16 {
		t.Errorf("DType = %v, want f16", got)
	}
	if got := b.Count(); got != 300 {
		t.Errorf("Count = %d, want 300", got)
	}
	if got, want := b.Bytes(), 600; got != want {
		t.Errorf("Bytes = %d, want %d: the buffer's size is the caller's number and is never rounded", got, want)
	}
	if got := b.Usage(); got != accel.BufferStorage|accel.BufferCopyDst {
		t.Errorf("Usage = %v", got)
	}
}

// TestViewCountsElementsNotBytes is spec 001 section 6.1: Offset and Count are
// in elements of the *view's* dtype, never in bytes. The dtype here is not four
// bytes wide on purpose, so an elements-versus-bytes confusion cannot pass.
func TestViewCountsElementsNotBytes(t *testing.T) {
	d := openCPU(t, accel.CPUOptions{})
	p := newPool(t, d, accel.MemoryDevice, 1<<20)
	defer p.Close()

	b := alloc(t, p, accel.BufferDescriptor{DType: accel.F16, Count: 64, Label: "heads"})
	defer b.Close()

	v, err := b.View(8, 16)
	if err != nil {
		t.Fatalf("View: %v", err)
	}
	if v.Buffer != b {
		t.Error("the view names a different buffer")
	}
	if v.DType != accel.F16 || v.Offset != 8 || v.Count != 16 {
		t.Errorf("View(8, 16) = %+v, want f16 elements [8, 24)", v)
	}

	// ViewAs takes elements of the new dtype, not of the buffer's. Anything else
	// would make ViewAs require the caller to do the arithmetic it exists to do.
	u, err := b.ViewAs(accel.U32, 4, 8)
	if err != nil {
		t.Fatalf("ViewAs: %v", err)
	}
	if u.Offset != 4 || u.Count != 8 || u.DType != accel.U32 {
		t.Errorf("ViewAs(U32, 4, 8) = %+v", u)
	}
	// Those u32 elements are bytes [16, 48), which is f16 elements [8, 24): the
	// same range the View above describes.
	if u.Offset*u.DType.Size() != v.Offset*v.DType.Size() {
		t.Errorf("ViewAs(U32, 4, ...) starts at byte %d but View(8, ...) starts at byte %d",
			u.Offset*u.DType.Size(), v.Offset*v.DType.Size())
	}
}

// TestViewAsRejections is spec 001 section 6.1's legality rules, each with the
// error naming both dtypes and the byte range.
func TestViewAsRejections(t *testing.T) {
	d := openCPU(t, accel.CPUOptions{})
	p := newPool(t, d, accel.MemoryDevice, 1<<20)
	defer p.Close()

	b := alloc(t, p, accel.BufferDescriptor{DType: accel.U8, Count: 1025, Label: "quants"})
	defer b.Close()

	for _, tc := range []struct {
		name          string
		d             accel.DType
		offset, count int
		want          string
	}{
		{"negative offset", accel.U8, -1, 4, "negative"},
		{"negative count", accel.U8, 0, -1, "negative"},
		{"past the end", accel.F32, 250, 20, "past the buffer"},
		{"unknown dtype", accel.DType(99), 0, 1, "not a dtype"},
		{"whole buffer at an indivisible width", accel.F32, 0, 1025 / 4, "not a multiple of 4"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			_, err := b.ViewAs(tc.d, tc.offset, tc.count)
			if err == nil {
				t.Fatalf("ViewAs(%v, %d, %d) was accepted", tc.d, tc.offset, tc.count)
			}
			if !strings.Contains(err.Error(), tc.want) {
				t.Errorf("error %q does not explain %q", err, tc.want)
			}
			if !strings.Contains(err.Error(), "quants") {
				t.Errorf("error %q does not name the buffer", err)
			}
		})
	}

	// A partial view at a wider dtype is legal even when the buffer's byte
	// length does not divide: only a view covering the whole buffer has to.
	if _, err := b.ViewAs(accel.F32, 0, 100); err != nil {
		t.Errorf("a partial f32 view of an indivisible buffer was rejected: %v", err)
	}
}

// TestViewOfClosedBufferIsRejectedAtCreation is the creation-time half of spec
// 001 section 7.3. The per-use half needs a use, and M1's immediate transfers
// address buffers rather than views, so it is checked white-box in
// view_internal_test.go until the recorded copy path lands at M3.
func TestViewOfClosedBufferIsRejectedAtCreation(t *testing.T) {
	d := openCPU(t, accel.CPUOptions{})
	p := newPool(t, d, accel.MemoryDevice, 1<<20)
	defer p.Close()

	b := alloc(t, p, accel.BufferDescriptor{DType: accel.F32, Count: 64, Label: "logits"})
	if err := b.Close(); err != nil {
		t.Fatal(err)
	}

	_, err := b.View(0, 16)
	if err == nil {
		t.Fatal("a closed buffer produced a view")
	}
	var le *accel.LifetimeError
	if !errors.As(err, &le) {
		t.Fatalf("error is %T, want *LifetimeError", err)
	}
	if le.Resource != "logits" {
		t.Errorf("the error names %q, want the closed buffer", le.Resource)
	}
}
