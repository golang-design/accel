// Copyright 2026 The golang.design Initiative Authors.
// All rights reserved. Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

//go:build darwin

package metal

import (
	"bytes"
	"os"
	"strings"
	"testing"

	"golang.design/x/accel/internal/driver"
	"golang.design/x/accel/internal/mtl"
)

func openInternal(t *testing.T) *device {
	t.Helper()
	as, err := Adapters()
	if err != nil || len(as) == 0 {
		if os.Getenv("ACCEL_REQUIRE_METAL") != "" {
			t.Fatalf("this job promises Metal and found no adapter (err=%v)", err)
		}
		t.Skipf("no Metal adapter (err=%v)", err)
	}
	dv, err := as[0].Open(nil)
	if err != nil {
		t.Fatalf("open: %v", err)
	}
	d, ok := dv.(*device)
	if !ok {
		t.Fatalf("Open returned %T", dv)
	}
	t.Cleanup(func() { _ = d.Close() })
	return d
}

// A private block transfers through a staged blit, and answers no mapping.
//
// On a unified-memory device MemoryDevice is shared storage and this path is
// not taken by Alloc, so it is exercised on a private buffer directly: it is
// what a discrete device runs, and it has to stay right there.
func TestAPrivateBlockTransfersThroughAStagedBlit(t *testing.T) {
	d := openInternal(t)
	buf, err := d.dev.NewBuffer(4096, mtl.StoragePrivate)
	if err != nil {
		t.Fatal(err)
	}
	defer buf.Close()
	b := &block{dev: d, buf: buf, label: "private"}
	if b.Bytes() != nil {
		t.Fatal("a private block answered a mapping")
	}
	if got := b.Size(); got != 4096 {
		t.Fatalf("size %d", got)
	}
	data := make([]byte, 1000)
	for i := range data {
		data[i] = byte(i * 3)
	}
	if err := b.Write(100, data); err != nil {
		t.Fatal(err)
	}
	got := make([]byte, 1000)
	if err := b.Read(100, got); err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(got, data) {
		t.Fatal("the staged round trip lost bytes")
	}
	if err := b.Write(4000, data); err == nil {
		t.Fatal("a write past the end was accepted")
	}
	if err := b.Read(4000, got); err == nil {
		t.Fatal("a read past the end was accepted")
	}

	if m, err := storageFor(driver.MemoryDevice, false); err != nil || m != mtl.StoragePrivate {
		t.Fatalf("device memory without unified memory: %v, %v", m, err)
	}
	if m, err := storageFor(driver.MemoryDevice, true); err != nil || m != mtl.StorageShared {
		t.Fatalf("device memory with unified memory: %v, %v", m, err)
	}
	if _, err := storageFor(driver.MemoryKind(99), true); err == nil {
		t.Fatal("an unknown memory kind got a storage mode")
	}
}

// A fault word with a code the backend does not name is still reported, with
// its number, and a node the plan does not hold has index zero.
func TestAnUnknownFaultCodeIsReportedByNumber(t *testing.T) {
	d := openInternal(t)
	blk, err := d.Alloc(driver.MemoryDevice, 256, "fault")
	if err != nil {
		t.Fatal(err)
	}
	defer blk.Free()
	op, err := driver.BlockOperand(blk, 0, 128)
	if err != nil {
		t.Fatal(err)
	}
	ex, err := d.Compile(&driver.Plan{Nodes: []driver.PlanNode{{Op: driver.OpCopy, Dst: op, Src: op}}})
	if err != nil {
		t.Fatal(err)
	}
	defer ex.Close()
	e := ex.(*executable)
	if e.faultReport() != nil {
		t.Fatal("a clean fault word reported a fault")
	}
	words := e.faults.Bytes()
	words[0], words[1], words[2], words[3] = 7, 0, 0, 0
	err = e.faultReport()
	if err == nil || !strings.Contains(err.Error(), "fault 7") {
		t.Fatalf("an unknown code was not reported by number: %v", err)
	}
	if got := e.nodeIndex(&driver.PlanNode{}); got != 0 {
		t.Fatalf("a foreign node has index %d, want 0", got)
	}
}
