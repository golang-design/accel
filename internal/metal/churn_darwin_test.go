// Copyright 2026 The golang.design Initiative Authors.
// All rights reserved. Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

//go:build darwin

package metal_test

import (
	"bytes"
	"testing"

	"golang.design/x/accel/internal/driver"
	"golang.design/x/accel/internal/mtl"
)

// A rows copy is one dispatch, and it is right at every alignment.
//
// It used to be one blit per row. The kernel copies four bytes per thread
// when every offset, pitch and row length allows and one byte otherwise;
// both paths are exercised here, with pitches that differ between the sides
// and a source offset, against the bytes the copy must produce.
func TestARowCopyIsExactAtEitherAlignment(t *testing.T) {
	d := open(t)
	c := d.(driver.GraphCompiler)
	for _, tc := range []struct {
		name               string
		rows, rowBytes     int
		srcPitch, dstPitch int
		srcOff, dstOff     int
	}{
		{"word aligned", 5, 16, 32, 20, 64, 4},
		{"byte aligned", 5, 13, 21, 15, 3, 1},
	} {
		t.Run(tc.name, func(t *testing.T) {
			src, err := d.Alloc(driver.MemoryDevice, 1024, "src")
			if err != nil {
				t.Fatal(err)
			}
			defer src.Free()
			dst, err := d.Alloc(driver.MemoryDevice, 1024, "dst")
			if err != nil {
				t.Fatal(err)
			}
			defer dst.Free()
			fill := make([]byte, 1024)
			for i := range fill {
				fill[i] = byte(i*7 + 3)
			}
			if err := src.Write(0, fill); err != nil {
				t.Fatal(err)
			}
			if err := dst.Write(0, make([]byte, 1024)); err != nil {
				t.Fatal(err)
			}
			so, err := driver.BlockOperand(src, tc.srcOff, tc.srcPitch*tc.rows)
			if err != nil {
				t.Fatal(err)
			}
			do, err := driver.BlockOperand(dst, tc.dstOff, tc.dstPitch*tc.rows)
			if err != nil {
				t.Fatal(err)
			}
			ex, err := c.Compile(&driver.Plan{Nodes: []driver.PlanNode{{
				Op: driver.OpCopyRows, Dst: do, Src: so,
				Rows: &driver.RowCopy{Rows: tc.rows, RowBytes: tc.rowBytes, DstPitch: tc.dstPitch, SrcPitch: tc.srcPitch},
			}}})
			if err != nil {
				t.Fatal(err)
			}
			defer ex.Close()
			f, err := ex.Submit()
			if err != nil {
				t.Fatal(err)
			}
			if err := f.Wait(); err != nil {
				t.Fatal(err)
			}
			got := make([]byte, 1024)
			if err := dst.Read(0, got); err != nil {
				t.Fatal(err)
			}
			want := make([]byte, 1024)
			for r := 0; r < tc.rows; r++ {
				copy(want[tc.dstOff+r*tc.dstPitch:tc.dstOff+r*tc.dstPitch+tc.rowBytes],
					fill[tc.srcOff+r*tc.srcPitch:tc.srcOff+r*tc.srcPitch+tc.rowBytes])
			}
			if !bytes.Equal(got, want) {
				for i := range got {
					if got[i] != want[i] {
						t.Fatalf("byte %d is %d, want %d", i, got[i], want[i])
					}
				}
			}
		})
	}
}

// Device memory is not mappable, and on a unified-memory device its transfers
// are a memcpy rather than a staged blit.
//
// specs/006-backends.md section 1 makes Block.Bytes the authority on
// mappability and this backend answers nil for MemoryDevice whatever storage
// it chose, so a caller cannot depend on a mapping one device happens to
// have. What shared storage buys is the immediate path: Write and Read used to
// allocate a staging buffer, encode a blit and wait on it per call, which on
// every Apple-silicon device was a command buffer per transfer for nothing.
func TestDeviceMemoryIsUnmappableAndTransfersWithoutAStagedBlit(t *testing.T) {
	d := open(t)
	b, err := d.Alloc(driver.MemoryDevice, 4096, "device")
	if err != nil {
		t.Fatal(err)
	}
	defer b.Free()
	if b.Bytes() != nil {
		t.Fatal("device memory answered a mapping, which the contract forbids")
	}
	data := make([]byte, 4096)
	for i := range data {
		data[i] = byte(i)
	}
	before := mtl.LiveCommandBuffers()
	if err := b.Write(0, data); err != nil {
		t.Fatal(err)
	}
	got := make([]byte, 4096)
	if err := b.Read(0, got); err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(got, data) {
		t.Fatal("the bytes written are not the bytes read")
	}
	if unifiedMemory(t) && mtl.LiveCommandBuffers() != before {
		t.Fatalf("a transfer of device memory used a command buffer on a unified-memory device")
	}
}

// unifiedMemory reports whether the first Metal device shares memory with the
// host, which is what makes the memcpy path available.
func unifiedMemory(t *testing.T) bool {
	t.Helper()
	devs, err := mtl.Devices()
	if err != nil || len(devs) == 0 {
		return false
	}
	defer func() {
		for _, dv := range devs {
			dv.Close()
		}
	}()
	return devs[0].UnifiedMemory
}
