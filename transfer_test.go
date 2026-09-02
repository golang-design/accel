// Copyright 2026 The golang.design Initiative Authors.
// All rights reserved. Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

package accel_test

import (
	"errors"
	"fmt"
	"math"
	"runtime"
	"strings"
	"sync"
	"testing"

	"golang.design/x/accel"
)

// allKinds is every memory kind the CPU backend reports, which is all four.
var allKinds = []accel.MemoryKind{
	accel.MemoryDevice, accel.MemoryUpload, accel.MemoryReadback, accel.MemoryShared,
}

// TestRoundTripEveryDTypeAndKind is spec 001 section 11.1: every dtype round
// trips host to device to host unchanged, at every memory kind the device
// reports, for buffer sizes that are not multiples of anything interesting.
func TestRoundTripEveryDTypeAndKind(t *testing.T) {
	d := openCPU(t, accel.CPUOptions{})
	q := d.Queue()

	counts := []int{1, 3, 65537}
	for _, kind := range allKinds {
		p := newPool(t, d, kind, 4<<20)
		for _, count := range counts {
			for _, dt := range []accel.DType{accel.F32, accel.F16, accel.BF16, accel.I32, accel.U32, accel.I8, accel.U8} {
				name := fmt.Sprintf("%v/%v/%d", kind, dt, count)
				t.Run(name, func(t *testing.T) {
					b := alloc(t, p, accel.BufferDescriptor{
						DType: dt, Count: count,
						Usage: accel.BufferCopySrc | accel.BufferCopyDst, Label: name,
					})
					defer b.Close()

					want := pattern(dt, count)
					if err := q.WriteBuffer(b, 0, want); err != nil {
						t.Fatalf("WriteBuffer: %v", err)
					}
					got := blank(dt, count)
					if err := q.ReadBuffer(b, 0, got); err != nil {
						t.Fatalf("ReadBuffer: %v", err)
					}
					if !sameSlice(want, got) {
						t.Errorf("round trip changed the data")
					}
				})
			}
		}
		if err := p.Close(); err != nil {
			t.Fatalf("Close %v pool: %v", kind, err)
		}
	}
}

// TestWriteCopiesOutOfTheCallersSlice is spec 001 section 8.2's promise: the
// caller may reuse or modify their slice the moment WriteBuffer returns.
func TestWriteCopiesOutOfTheCallersSlice(t *testing.T) {
	d := openCPU(t, accel.CPUOptions{})
	q := d.Queue()

	for _, kind := range allKinds {
		p := newPool(t, d, kind, 1<<20)
		b := alloc(t, p, accel.BufferDescriptor{
			DType: accel.F32, Count: 8, Usage: accel.BufferCopyDst | accel.BufferCopySrc,
			Label: fmt.Sprintf("%v", kind),
		})

		data := []float32{1, 2, 3, 4, 5, 6, 7, 8}
		if err := q.WriteBuffer(b, 0, data); err != nil {
			t.Fatal(err)
		}
		for i := range data {
			data[i] = -1 // the caller's slice must no longer matter
		}

		got := make([]float32, 8)
		if err := q.ReadBuffer(b, 0, got); err != nil {
			t.Fatal(err)
		}
		for i, v := range got {
			if v != float32(i+1) {
				t.Fatalf("%v: element %d is %v, want %v: the write aliased the caller's slice",
					kind, i, v, i+1)
			}
		}
		b.Close()
		p.Close()
	}
}

// TestPartialWriteTouchesOnlyItsRange is spec 001 sections 8.2 and 11.1:
// partial updates are first-class in both directions, address elements of the
// buffer's dtype, and touch only the range they name. One case deliberately
// starts at a byte offset that is not copy-aligned.
func TestPartialWriteTouchesOnlyItsRange(t *testing.T) {
	d := openCPU(t, accel.CPUOptions{})
	q := d.Queue()

	for _, kind := range allKinds {
		p := newPool(t, d, kind, 1<<20)
		b := alloc(t, p, accel.BufferDescriptor{
			DType: accel.F32, Count: 16, Usage: accel.BufferCopyDst | accel.BufferCopySrc,
			Label: "partial",
		})

		base := make([]float32, 16)
		for i := range base {
			base[i] = float32(i)
		}
		if err := q.WriteBuffer(b, 0, base); err != nil {
			t.Fatal(err)
		}

		// Element 7 is byte 28, which is not a multiple of the 16-byte copy
		// alignment, so this is the read-merge-write case rather than a plain
		// aligned copy.
		if err := q.WriteBuffer(b, 7, []float32{99}); err != nil {
			t.Fatal(err)
		}

		got := make([]float32, 16)
		if err := q.ReadBuffer(b, 0, got); err != nil {
			t.Fatal(err)
		}
		for i, v := range got {
			want := float32(i)
			if i == 7 {
				want = 99
			}
			if v != want {
				t.Errorf("%v: element %d is %v, want %v: a partial write touched more than its range",
					kind, i, v, want)
			}
		}

		// A partial read addresses elements too.
		tail := make([]float32, 4)
		if err := q.ReadBuffer(b, 12, tail); err != nil {
			t.Fatal(err)
		}
		for i, v := range tail {
			if v != float32(12+i) {
				t.Errorf("%v: partial read element %d is %v, want %v", kind, i, v, 12+i)
			}
		}

		b.Close()
		p.Close()
	}
}

// TestReadFlushesPendingWrites is spec 001 sections 8.2 and 11.1: WriteBuffer
// then ReadBuffer with no submission in between returns the written data.
func TestReadFlushesPendingWrites(t *testing.T) {
	d := openCPU(t, accel.CPUOptions{})
	q := d.Queue()

	// MemoryDevice is the kind that actually stages, because the CPU backend
	// reports it unmappable even though its memory physically could be mapped.
	p := newPool(t, d, accel.MemoryDevice, 1<<20)
	defer p.Close()

	b := alloc(t, p, accel.BufferDescriptor{
		DType: accel.U32, Count: 4, Usage: accel.BufferCopyDst | accel.BufferCopySrc, Label: "staged",
	})
	defer b.Close()

	if err := q.WriteBuffer(b, 0, []uint32{10, 20, 30, 40}); err != nil {
		t.Fatal(err)
	}
	got := make([]uint32, 4)
	if err := q.ReadBuffer(b, 0, got); err != nil {
		t.Fatal(err)
	}
	for i, want := range []uint32{10, 20, 30, 40} {
		if got[i] != want {
			t.Fatalf("element %d is %d, want %d: the read did not flush pending writes", i, got[i], want)
		}
	}
}

// TestFlushWithoutRead covers the explicit flush path and its fence, which
// exists for the caller who wants the batch out without reading it back.
func TestFlushWithoutRead(t *testing.T) {
	d := openCPU(t, accel.CPUOptions{})
	q := d.Queue()
	p := newPool(t, d, accel.MemoryDevice, 1<<20)
	defer p.Close()

	b := alloc(t, p, accel.BufferDescriptor{
		DType: accel.I32, Count: 2, Usage: accel.BufferCopyDst | accel.BufferCopySrc, Label: "flushed",
	})
	defer b.Close()

	// A flush with nothing pending returns an already-signalled fence.
	empty := q.Flush()
	if !empty.Done() {
		t.Error("a flush with nothing pending returned an unsignalled fence")
	}
	if err := empty.Wait(); err != nil {
		t.Errorf("Wait on an empty flush: %v", err)
	}

	if err := q.WriteBuffer(b, 0, []int32{-1, -2}); err != nil {
		t.Fatal(err)
	}
	f := q.Flush()
	if err := f.Wait(); err != nil {
		t.Fatalf("Wait: %v", err)
	}
	if !f.Done() {
		t.Error("a fence that returned from Wait reports not done")
	}
	select {
	case <-f.C():
	default:
		t.Error("the fence channel is not closed after Wait returned")
	}

	got := make([]int32, 2)
	if err := q.ReadBuffer(b, 0, got); err != nil {
		t.Fatal(err)
	}
	if got[0] != -1 || got[1] != -2 {
		t.Errorf("after an explicit flush the buffer holds %v", got)
	}
}

// TestClosingAPendingDestinationIsReported is spec 001 sections 8.2 and 11.1:
// closing a WriteBuffer destination before flush reports pending transfer, and
// the flush still completes without a use after free.
func TestClosingAPendingDestinationIsReported(t *testing.T) {
	d, err := accel.OpenCPU(accel.CPUOptions{})
	if err != nil {
		t.Fatal(err)
	}
	q := d.Queue()
	p, err := d.NewPool(accel.PoolDescriptor{Kind: accel.MemoryDevice, Bytes: 1 << 20})
	if err != nil {
		t.Fatal(err)
	}

	b, err := p.AllocBuffer(accel.BufferDescriptor{
		DType: accel.U8, Count: 64, Usage: accel.BufferCopyDst, Label: "staging",
	})
	if err != nil {
		t.Fatal(err)
	}
	if err := q.WriteBuffer(b, 0, make([]byte, 64)); err != nil {
		t.Fatal(err)
	}

	// Device close does not hide the unflushed batch.
	if err := d.Close(); err == nil {
		t.Error("a device with an unflushed batch closed")
	}

	err = b.Close()
	if err == nil {
		t.Fatal("closing a pending destination was silent")
	}
	var le *accel.LifetimeError
	if !errors.As(err, &le) {
		t.Fatalf("error is %T, want *LifetimeError", err)
	}
	if le.Reason != "pending transfer" {
		t.Errorf("reason is %q, want \"pending transfer\"", le.Reason)
	}
	if !strings.Contains(err.Error(), "Queue.Flush().Wait()") {
		t.Errorf("message %q does not say how to avoid it", err)
	}

	// The queued write still lands, and the memory comes back afterwards.
	if err := q.Flush().Wait(); err != nil {
		t.Fatalf("Flush after the handle was retired: %v", err)
	}
	if s := p.Stats(); s.Allocations != 0 {
		t.Errorf("the buffer's memory did not return after the batch completed: %+v", s)
	}
	if err := p.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}
	if err := d.Close(); err != nil {
		t.Fatalf("Close after flushing: %v", err)
	}
}

// TestTransferTypeChecking rejects a host slice whose element type does not
// match the buffer's dtype, naming both.
func TestTransferTypeChecking(t *testing.T) {
	d := openCPU(t, accel.CPUOptions{})
	q := d.Queue()
	p := newPool(t, d, accel.MemoryUpload, 1<<20)
	defer p.Close()

	b := alloc(t, p, accel.BufferDescriptor{
		DType: accel.F32, Count: 8, Usage: accel.BufferCopyDst | accel.BufferCopySrc, Label: "typed",
	})
	defer b.Close()

	for _, bad := range []any{
		[]byte{1, 2, 3, 4},
		[]uint32{1, 2},
		[]int32{1, 2},
		[]uint16{1, 2},
		[]int8{1, 2},
		[]float64{1, 2},
		"not a slice",
		nil,
	} {
		if err := q.WriteBuffer(b, 0, bad); err == nil {
			t.Errorf("WriteBuffer accepted %T into an f32 buffer", bad)
		} else if !strings.Contains(err.Error(), "[]float32") {
			t.Errorf("error for %T does not name the expected slice type: %v", bad, err)
		}
		if err := q.ReadBuffer(b, 0, bad); err == nil {
			t.Errorf("ReadBuffer accepted %T from an f32 buffer", bad)
		}
	}
}

// TestEveryDTypeNamesItsHostSlice checks that the rejection tells a caller which
// Go slice each dtype expects, since guessing is how a caller writes the right
// number of bytes with the wrong meaning.
func TestEveryDTypeNamesItsHostSlice(t *testing.T) {
	d := openCPU(t, accel.CPUOptions{})
	q := d.Queue()
	p := newPool(t, d, accel.MemoryUpload, 1<<20)
	defer p.Close()

	for _, tc := range []struct {
		dt   accel.DType
		want string
	}{
		{accel.F32, "[]float32"},
		{accel.F16, "[]uint16"},
		{accel.BF16, "[]uint16"},
		{accel.I32, "[]int32"},
		{accel.U32, "[]uint32"},
		{accel.I8, "[]int8"},
		{accel.U8, "[]byte"},
	} {
		b := alloc(t, p, accel.BufferDescriptor{
			DType: tc.dt, Count: 4, Usage: accel.BufferCopyDst, Label: tc.dt.String(),
		})
		// A []float64 matches no dtype, so every case takes the rejection path.
		err := q.WriteBuffer(b, 0, []float64{1, 2, 3, 4})
		if err == nil {
			t.Errorf("%v accepted a []float64", tc.dt)
		} else if !strings.Contains(err.Error(), tc.want) {
			t.Errorf("%v: error %q does not name %s", tc.dt, err, tc.want)
		}
		b.Close()
	}
}

// TestTransferRangeChecking rejects a transfer that would run past the buffer.
func TestTransferRangeChecking(t *testing.T) {
	d := openCPU(t, accel.CPUOptions{})
	q := d.Queue()
	p := newPool(t, d, accel.MemoryDevice, 1<<20)
	defer p.Close()

	b := alloc(t, p, accel.BufferDescriptor{
		DType: accel.F32, Count: 8, Usage: accel.BufferCopyDst | accel.BufferCopySrc, Label: "small",
	})
	defer b.Close()

	for _, tc := range []struct {
		name          string
		offset, count int
	}{
		{"past the end", 6, 4},
		{"offset past the end", 100, 1},
		{"negative offset", -1, 1},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if err := q.WriteBuffer(b, tc.offset, make([]float32, tc.count)); err == nil {
				t.Error("WriteBuffer was accepted")
			}
			if err := q.ReadBuffer(b, tc.offset, make([]float32, tc.count)); err == nil {
				t.Error("ReadBuffer was accepted")
			}
		})
	}

	// No buffer at all is an error at both entry points, not a fault: both
	// read the buffer's label and dtype before anything checked the pointer.
	for _, tc := range []struct {
		name string
		call func() error
	}{
		{"WriteBuffer", func() error { return q.WriteBuffer(nil, 0, []float32{1}) }},
		{"ReadBuffer", func() error { return q.ReadBuffer(nil, 0, make([]float32, 1)) }},
	} {
		err := tc.call()
		if err == nil {
			t.Errorf("%s on a nil buffer was accepted", tc.name)
		} else if !strings.Contains(err.Error(), "no buffer") {
			t.Errorf("%s on a nil buffer should say so: %v", tc.name, err)
		}
	}

	// A closed buffer is rejected at both entry points.
	if err := b.Close(); err != nil {
		t.Fatal(err)
	}
	for _, tc := range []struct {
		name string
		call func() error
	}{
		{"WriteBuffer", func() error { return q.WriteBuffer(b, 0, []float32{1}) }},
		{"ReadBuffer", func() error { return q.ReadBuffer(b, 0, make([]float32, 1)) }},
	} {
		var le *accel.LifetimeError
		if err := tc.call(); !errors.As(err, &le) {
			t.Errorf("%s on a closed buffer returned %v, want *LifetimeError", tc.name, err)
		} else if le.Reason != "closed" {
			t.Errorf("%s reported reason %q", tc.name, le.Reason)
		}
	}
}

// TestBitPatternsSurvive is spec 001 section 11.3's bit-pattern guarantee, seen
// through the transfer path: an f32 buffer read back as f32 preserves negative
// zero, infinities, and NaNs, because accel copies bytes and never converts.
func TestBitPatternsSurvive(t *testing.T) {
	d := openCPU(t, accel.CPUOptions{})
	q := d.Queue()
	p := newPool(t, d, accel.MemoryDevice, 1<<20)
	defer p.Close()

	want := []float32{
		float32(math.Copysign(0, -1)),
		float32(math.Inf(1)),
		float32(math.Inf(-1)),
		float32(math.NaN()),
		math.SmallestNonzeroFloat32,
		math.MaxFloat32,
	}
	b := alloc(t, p, accel.BufferDescriptor{
		DType: accel.F32, Count: len(want), Usage: accel.BufferCopyDst | accel.BufferCopySrc, Label: "specials",
	})
	defer b.Close()

	if err := q.WriteBuffer(b, 0, want); err != nil {
		t.Fatal(err)
	}
	got := make([]float32, len(want))
	if err := q.ReadBuffer(b, 0, got); err != nil {
		t.Fatal(err)
	}
	for i := range want {
		if math.Float32bits(got[i]) != math.Float32bits(want[i]) {
			t.Errorf("element %d: bits %#08x, want %#08x",
				i, math.Float32bits(got[i]), math.Float32bits(want[i]))
		}
	}

	// The same bytes read through a u32 buffer at the same offsets are the IEEE
	// encodings, which is the guarantee a backend could break by converting.
	raw := alloc(t, p, accel.BufferDescriptor{
		DType: accel.U32, Count: len(want), Usage: accel.BufferCopyDst | accel.BufferCopySrc, Label: "raw",
	})
	defer raw.Close()

	bits := make([]uint32, len(want))
	for i, v := range want {
		bits[i] = math.Float32bits(v)
	}
	if err := q.WriteBuffer(raw, 0, bits); err != nil {
		t.Fatal(err)
	}
	asFloats := make([]uint32, len(want))
	if err := q.ReadBuffer(raw, 0, asFloats); err != nil {
		t.Fatal(err)
	}
	for i := range bits {
		if asFloats[i] != bits[i] {
			t.Errorf("u32 element %d is %#08x, want %#08x", i, asFloats[i], bits[i])
		}
	}
}

// TestQueueStats checks the counters a caller uses to see staging pressure.
func TestQueueStats(t *testing.T) {
	d := openCPU(t, accel.CPUOptions{})
	q := d.Queue()
	p := newPool(t, d, accel.MemoryDevice, 1<<20)
	defer p.Close()

	b := alloc(t, p, accel.BufferDescriptor{
		DType: accel.U8, Count: 1024, Usage: accel.BufferCopyDst | accel.BufferCopySrc, Label: "counted",
	})
	defer b.Close()

	before := q.Stats()
	if err := q.WriteBuffer(b, 0, make([]byte, 512)); err != nil {
		t.Fatal(err)
	}
	if err := q.Flush().Wait(); err != nil {
		t.Fatal(err)
	}
	if err := q.ReadBuffer(b, 0, make([]byte, 512)); err != nil {
		t.Fatal(err)
	}

	after := q.Stats()
	if got := after.BytesStaged - before.BytesStaged; got != 512 {
		t.Errorf("BytesStaged rose by %d, want 512", got)
	}
	if got := after.Submissions - before.Submissions; got != 1 {
		t.Errorf("Submissions rose by %d, want 1", got)
	}
	if got := after.ImmediateReads - before.ImmediateReads; got != 1 {
		t.Errorf("ImmediateReads rose by %d, want 1", got)
	}
	// Nothing blocked, because M1 has no fixed staging ring to fill; see spec
	// 009's M1 deviation.
	if after.StagingWaits != 0 {
		t.Errorf("StagingWaits = %d, want 0 at M1", after.StagingWaits)
	}
}

// TestHostVisibleWritesDoNotStage is spec 001 section 8.3's table as an
// assertion rather than a claim. That table says a write to a Device pool moves
// twice the payload and a write to a Shared pool moves it once, and the whole
// argument for MemoryShared being a kind rather than a hint rests on it.
//
// The property is measured as **whether allocation scales with the payload**,
// not as a count. A staging copy is payload-sized, so staging makes the bytes
// allocated per write grow with the data; writing into a mapping does not. An
// allocation count cannot see the difference, because flushing allocates a
// fence either way and a first version of this test passed with the second copy
// reinstated.
func TestHostVisibleWritesDoNotStage(t *testing.T) {
	d := openCPU(t, accel.CPUOptions{})
	q := d.Queue()

	const (
		small = 1 << 8
		large = 1 << 16
		runs  = 200
	)

	bytesPerWrite := func(kind accel.MemoryKind, n int) uint64 {
		t.Helper()
		p := newPool(t, d, kind, 4<<20)
		defer p.Close()
		b := alloc(t, p, accel.BufferDescriptor{
			DType: accel.F32, Count: n, Usage: accel.BufferCopyDst, Label: "measured",
		})
		defer b.Close()
		data := make([]float32, n)

		write := func() {
			if err := q.WriteBuffer(b, 0, data); err != nil {
				t.Fatal(err)
			}
			if err := q.Flush().Wait(); err != nil {
				t.Fatal(err)
			}
		}
		write() // warm any one-time allocation out of the measurement

		// The fewest bytes per write over several rounds: TotalAlloc is
		// process-wide and anything else allocating during a round can only
		// add to it, so the minimum is the write's own cost.
		const rounds = 5
		var best uint64
		for round := range rounds {
			var before, after runtime.MemStats
			runtime.GC()
			runtime.ReadMemStats(&before)
			for range runs {
				write()
			}
			runtime.ReadMemStats(&after)
			if per := (after.TotalAlloc - before.TotalAlloc) / runs; round == 0 || per < best {
				best = per
			}
		}
		return best
	}

	// The payload difference, in bytes, between the two measurements.
	payloadDelta := uint64((large - small) * 4)

	deviceGrowth := bytesPerWrite(accel.MemoryDevice, large) - bytesPerWrite(accel.MemoryDevice, small)
	if deviceGrowth < payloadDelta/2 {
		t.Fatalf("a Device write's allocation grew by %d bytes for a %d-byte larger payload; "+
			"it has to stage, and staging is payload-sized", deviceGrowth, payloadDelta)
	}

	for _, kind := range []accel.MemoryKind{accel.MemoryShared, accel.MemoryUpload, accel.MemoryReadback} {
		growth := bytesPerWrite(kind, large) - bytesPerWrite(kind, small)
		if growth > payloadDelta/8 {
			t.Errorf("a %v write's allocation grew by %d bytes for a %d-byte larger payload, "+
				"against %d for Device: a kind the device maps takes the bytes straight from "+
				"the caller's slice, and an allocation that scales with the payload is the "+
				"second copy MemoryShared exists to remove", kind, growth, payloadDelta, deviceGrowth)
		}
	}
}

// TestConcurrentTransfers is spec 001 sections 1.2 and 11.7: immediate
// transfers on disjoint ranges through one queue are safe from any goroutine.
func TestConcurrentTransfers(t *testing.T) {
	d := openCPU(t, accel.CPUOptions{})
	q := d.Queue()
	p := newPool(t, d, accel.MemoryDevice, 4<<20)
	defer p.Close()

	const goroutines, each = 8, 32
	b := alloc(t, p, accel.BufferDescriptor{
		DType: accel.U32, Count: goroutines * each,
		Usage: accel.BufferCopyDst | accel.BufferCopySrc, Label: "shared",
	})
	defer b.Close()

	var wg sync.WaitGroup
	for g := range goroutines {
		wg.Add(1)
		go func() {
			defer wg.Done()
			chunk := make([]uint32, each)
			for i := range chunk {
				chunk[i] = uint32(g*each + i)
			}
			if err := q.WriteBuffer(b, g*each, chunk); err != nil {
				t.Errorf("goroutine %d: %v", g, err)
			}
		}()
	}
	wg.Wait()

	got := make([]uint32, goroutines*each)
	if err := q.ReadBuffer(b, 0, got); err != nil {
		t.Fatal(err)
	}
	for i, v := range got {
		if v != uint32(i) {
			t.Fatalf("element %d is %d, want %d: a concurrent write was lost", i, v, i)
		}
	}
}

// TestM1EndToEnd is the milestone's named scenario from spec 009 and spec 011
// section 8: open the CPU device, allocate, write through the queue, read back,
// close. It uses public APIs only and owns its resources with deterministic
// cleanup, because a milestone whose acceptance runs through internal shortcuts
// has not demonstrated the thing it claims.
func TestM1EndToEnd(t *testing.T) {
	dev, err := accel.OpenCPU(accel.CPUOptions{})
	if err != nil {
		t.Fatalf("OpenCPU: %v", err)
	}

	weights, err := dev.NewPool(accel.PoolDescriptor{Kind: accel.MemoryDevice, Bytes: 1 << 20})
	if err != nil {
		t.Fatalf("NewPool: %v", err)
	}

	buf, err := weights.AllocBuffer(accel.BufferDescriptor{
		DType: accel.F32,
		Count: 1024,
		Usage: accel.BufferStorage | accel.BufferCopySrc | accel.BufferCopyDst,
		Label: "blk.0.attn_q.weight",
	})
	if err != nil {
		t.Fatalf("Alloc: %v", err)
	}

	want := make([]float32, 1024)
	for i := range want {
		want[i] = float32(i) * 0.5
	}
	if err := dev.Queue().WriteBuffer(buf, 0, want); err != nil {
		t.Fatalf("WriteBuffer: %v", err)
	}

	got := make([]float32, 1024)
	if err := dev.Queue().ReadBuffer(buf, 0, got); err != nil {
		t.Fatalf("ReadBuffer: %v", err)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("element %d is %v, want %v", i, got[i], want[i])
		}
	}

	if err := buf.Close(); err != nil {
		t.Fatalf("Buffer.Close: %v", err)
	}
	if s := weights.Stats(); s.Allocations != 0 || s.Used != 0 {
		t.Errorf("the pool still holds %+v after its only buffer closed", s)
	}
	if err := weights.Close(); err != nil {
		t.Fatalf("Pool.Close: %v", err)
	}
	if err := dev.Close(); err != nil {
		t.Fatalf("Device.Close: %v", err)
	}
}

// pattern builds a distinct value per element for the dtype under test.
func pattern(d accel.DType, n int) any {
	switch d {
	case accel.F32:
		s := make([]float32, n)
		for i := range s {
			s[i] = float32(i)*1.5 - 3
		}
		return s
	case accel.F16, accel.BF16:
		s := make([]uint16, n)
		for i := range s {
			s[i] = uint16(i*7 + 1)
		}
		return s
	case accel.I32:
		s := make([]int32, n)
		for i := range s {
			s[i] = int32(i) - 5
		}
		return s
	case accel.U32:
		s := make([]uint32, n)
		for i := range s {
			s[i] = uint32(i) * 3
		}
		return s
	case accel.I8:
		s := make([]int8, n)
		for i := range s {
			s[i] = int8(i%251) - 100
		}
		return s
	default:
		s := make([]byte, n)
		for i := range s {
			s[i] = byte(i % 251)
		}
		return s
	}
}

func blank(d accel.DType, n int) any {
	switch d {
	case accel.F32:
		return make([]float32, n)
	case accel.F16, accel.BF16:
		return make([]uint16, n)
	case accel.I32:
		return make([]int32, n)
	case accel.U32:
		return make([]uint32, n)
	case accel.I8:
		return make([]int8, n)
	default:
		return make([]byte, n)
	}
}

func sameSlice(a, b any) bool {
	switch x := a.(type) {
	case []float32:
		return equal(x, b.([]float32))
	case []uint16:
		return equal(x, b.([]uint16))
	case []int32:
		return equal(x, b.([]int32))
	case []uint32:
		return equal(x, b.([]uint32))
	case []int8:
		return equal(x, b.([]int8))
	case []byte:
		return equal(x, b.([]byte))
	}
	return false
}

func equal[T comparable](a, b []T) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}
