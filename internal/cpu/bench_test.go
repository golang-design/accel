// Copyright 2026 The golang.design Initiative Authors.
// All rights reserved. Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

package cpu_test

import (
	"fmt"
	"testing"

	"golang.design/x/accel/internal/cpu"
	"golang.design/x/accel/internal/driver"
)

func benchDevice(b *testing.B) driver.Device {
	b.Helper()
	d, err := cpu.Adapter{}.Open(nil)
	if err != nil {
		b.Fatal(err)
	}
	b.Cleanup(func() { d.Close() })
	return d
}

// BenchmarkAlloc measures one device allocation, which is a pool. Spec 001
// makes pooling mandatory partly because this is expensive on a real driver;
// on the CPU backend it is a make, and the number is here so the comparison at
// M6 has a baseline.
func BenchmarkAlloc(b *testing.B) {
	d := benchDevice(b)
	b.ReportAllocs()
	for b.Loop() {
		blk, err := d.Alloc(driver.MemoryDevice, 1<<20, "bench")
		if err != nil {
			b.Fatal(err)
		}
		blk.Free()
	}
}

// BenchmarkBlockWrite measures the device-side transfer, which is the path a
// write to unmappable memory takes.
func BenchmarkBlockWrite(b *testing.B) {
	for _, n := range []int{1 << 10, 1 << 20} {
		b.Run(fmt.Sprintf("%dKiB", n>>10), func(b *testing.B) {
			d := benchDevice(b)
			blk, err := d.Alloc(driver.MemoryDevice, n, "bench")
			if err != nil {
				b.Fatal(err)
			}
			defer blk.Free()

			src := make([]byte, n)
			b.SetBytes(int64(n))
			b.ReportAllocs()
			for b.Loop() {
				if err := blk.Write(0, src); err != nil {
					b.Fatal(err)
				}
			}
		})
	}
}

// BenchmarkBlockRead is the other direction.
func BenchmarkBlockRead(b *testing.B) {
	const n = 1 << 20
	d := benchDevice(b)
	blk, err := d.Alloc(driver.MemoryDevice, n, "bench")
	if err != nil {
		b.Fatal(err)
	}
	defer blk.Free()

	dst := make([]byte, n)
	b.SetBytes(int64(n))
	b.ReportAllocs()
	for b.Loop() {
		if err := blk.Read(0, dst); err != nil {
			b.Fatal(err)
		}
	}
}

// BenchmarkResolveProfile measures device open's profile resolution, including
// the strict intersection, which every test in the suite pays once.
func BenchmarkResolveProfile(b *testing.B) {
	for _, mode := range []struct {
		name string
		opts cpu.Options
	}{
		{"developer", cpu.Options{}},
		{"strict", cpu.Options{Mode: cpu.Strict, StrictTargets: []driver.Backend{driver.BackendMetal, driver.BackendVulkan}}},
	} {
		b.Run(mode.name, func(b *testing.B) {
			b.ReportAllocs()
			for b.Loop() {
				d, err := cpu.Adapter{}.Open(&mode.opts)
				if err != nil {
					b.Fatal(err)
				}
				if err := d.Close(); err != nil {
					b.Fatal(err)
				}
			}
		})
	}
}
