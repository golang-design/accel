// Copyright 2026 The golang.design Initiative Authors.
// All rights reserved. Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

package accel_test

import (
	"testing"

	"golang.design/x/accel"
)

// FuzzViewOffsets is the target the offset-overflow bug earned.
//
// Every offset in this design is in elements and every device address is in
// bytes, so the scaling between them is a validation boundary. Scaling before
// bounding let a large offset wrap and land back inside the buffer, and
// WriteBuffer at element 1<<62 on an f32 buffer wrote element 0. Spec 001
// section 7.3 promises the worst outcome of a hand-constructed view is a
// rejection, and a silent write at the wrong address is not that.
//
// The property is: for any offset and count, a view either is refused or names
// a byte range inside the buffer. There is no third outcome.
func FuzzViewOffsets(f *testing.F) {
	f.Add(0, 4, 0)
	f.Add(-1, 4, 1)
	f.Add(1<<62, 1, 0)
	f.Add(0, 1<<62, 2)
	f.Add(1<<62, 1<<62, 3)
	f.Add(4, 4, 6)

	dtypes := []accel.DType{accel.F32, accel.F16, accel.BF16, accel.I32, accel.U32, accel.I8, accel.U8}

	f.Fuzz(func(t *testing.T, offset, count, dtypeIndex int) {
		d := openCPU(t, accel.CPUOptions{})
		p, err := d.NewPool(accel.MemoryDevice, 1<<16)
		if err != nil {
			t.Fatal(err)
		}
		defer p.Close()

		b, err := p.Alloc(accel.BufferDescriptor{
			DType: accel.F32, Count: 64,
			Usage: accel.UsageCopyDst | accel.UsageCopySrc, Label: "fuzz",
		})
		if err != nil {
			t.Fatal(err)
		}
		defer b.Close()

		if dtypeIndex < 0 {
			dtypeIndex = -dtypeIndex
		}
		view, err := b.ViewAs(dtypes[dtypeIndex%len(dtypes)], offset, count)
		if err != nil {
			return // a refusal is always a legal outcome
		}

		// It was accepted, so it must name a range inside the buffer, computed
		// the way every consumer computes it.
		elem := view.DType.Size()
		if elem == 0 {
			t.Fatalf("an accepted view has dtype %v, whose size is zero", view.DType)
		}
		if view.Offset < 0 || view.Count < 0 {
			t.Fatalf("an accepted view has offset %d and count %d", view.Offset, view.Count)
		}
		end := view.Offset + view.Count
		if end < view.Offset {
			t.Fatalf("an accepted view's element range overflows: %d + %d", view.Offset, view.Count)
		}
		bytes := end * elem
		if bytes < 0 || bytes > b.Bytes() {
			t.Fatalf("an accepted view covers bytes [0, %d) of a %d-byte buffer, from offset %d "+
				"count %d at %v", bytes, b.Bytes(), view.Offset, view.Count, view.DType)
		}
	})
}

// FuzzTransferOffsets is the same property on the transfer path, which is where
// the wrapped offset actually wrote.
//
// The buffer is filled with a known pattern and checked afterwards: a rejected
// transfer must leave it untouched, and an accepted one must change only the
// range it named. That is stronger than checking the error, because the bug
// this replaces returned no error at all.
func FuzzTransferOffsets(f *testing.F) {
	f.Add(0, 4)
	f.Add(-1, 1)
	f.Add(1<<62, 1)
	f.Add(60, 8)
	f.Add(63, 1)

	f.Fuzz(func(t *testing.T, offset, count int) {
		if count < 0 || count > 1024 {
			t.Skip() // the caller's slice length, not part of the property
		}
		d := openCPU(t, accel.CPUOptions{})
		q := d.Queue()
		p, err := d.NewPool(accel.MemoryDevice, 1<<16)
		if err != nil {
			t.Fatal(err)
		}
		defer p.Close()

		const n = 64
		b, err := p.Alloc(accel.BufferDescriptor{
			DType: accel.F32, Count: n,
			Usage: accel.UsageCopyDst | accel.UsageCopySrc, Label: "fuzz",
		})
		if err != nil {
			t.Fatal(err)
		}
		defer b.Close()

		base := make([]float32, n)
		for i := range base {
			base[i] = float32(i) + 1000
		}
		if err := q.WriteBuffer(b, 0, base); err != nil {
			t.Fatal(err)
		}

		payload := make([]float32, count)
		for i := range payload {
			payload[i] = -1
		}
		writeErr := q.WriteBuffer(b, offset, payload)

		got := make([]float32, n)
		if err := q.ReadBuffer(b, 0, got); err != nil {
			t.Fatal(err)
		}

		for i := range got {
			inRange := writeErr == nil && offset >= 0 && i >= offset && i < offset+count
			switch {
			case inRange && got[i] != -1:
				t.Fatalf("element %d is %v, want the written -1", i, got[i])
			case !inRange && got[i] != base[i]:
				t.Fatalf("element %d is %v, want the untouched %v: a transfer at offset %d "+
					"count %d reached outside the range it named (err=%v)",
					i, got[i], base[i], offset, count, writeErr)
			}
		}
	})
}

// FuzzPoolAllocation checks that a pool's accounting stays consistent under any
// sequence of sizes, which is the invariant a caller reads PoolStats for.
func FuzzPoolAllocation(f *testing.F) {
	f.Add(uint(1), uint(0), uint(7))
	f.Add(uint(4096), uint(3), uint(1))
	f.Add(uint(0), uint(0), uint(0))

	f.Fuzz(func(t *testing.T, rawCount, rawDType, rawUsage uint) {
		d := openCPU(t, accel.CPUOptions{})
		p, err := d.NewPool(accel.MemoryDevice, 1<<20)
		if err != nil {
			t.Fatal(err)
		}
		defer p.Close()

		dtypes := []accel.DType{accel.F32, accel.F16, accel.BF16, accel.I32, accel.U32, accel.I8, accel.U8}
		desc := accel.BufferDescriptor{
			DType: dtypes[rawDType%uint(len(dtypes))],
			Count: int(rawCount % 4096),
			Usage: accel.BufferUsage(rawUsage),
			Label: "fuzz",
		}

		before := p.Stats()
		b, err := p.Alloc(desc)
		if err != nil {
			if after := p.Stats(); after != before {
				t.Fatalf("a refused allocation changed the pool: %+v then %+v", before, after)
			}
			return
		}

		after := p.Stats()
		if after.Allocations != before.Allocations+1 {
			t.Fatalf("Allocations went from %d to %d", before.Allocations, after.Allocations)
		}
		if after.Used < before.Used+b.Bytes() {
			t.Fatalf("Used went from %d to %d for a %d-byte buffer: the allocation size "+
				"includes padding and is never below the request", before.Used, after.Used, b.Bytes())
		}
		if after.Free != after.Size-after.Used {
			t.Fatalf("Free is %d, want Size-Used = %d", after.Free, after.Size-after.Used)
		}
		if after.LargestFree > after.Free {
			t.Fatalf("LargestFree %d exceeds Free %d", after.LargestFree, after.Free)
		}

		if err := b.Close(); err != nil {
			t.Fatal(err)
		}
		if final := p.Stats(); final.Used != before.Used || final.Allocations != before.Allocations {
			t.Fatalf("closing did not restore the pool: %+v then %+v", before, final)
		}
	})
}
