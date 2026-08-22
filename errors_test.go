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

// TestErrorMessages pins the message surface of spec 001 section 9.
//
// The realistic messages there are called "the actual specification of these
// types", so they are tested as one. Each has to carry the resource by its
// label and the numbers a caller needs to act, because an error naming only its
// class is one the caller cannot do anything with, which spec 001 calls a defect
// rather than a terse style.
func TestErrorMessages(t *testing.T) {
	tests := []struct {
		name     string
		err      error
		sentinel error
		want     []string
	}{
		{
			name: "fragmented allocation",
			err: &accel.AllocError{
				Label: "blk.31.ffn_up.weight", Pool: "weights", Kind: accel.MemoryDevice,
				Requested: 176 << 20, Alignment: 256,
				Free: 220594176, LargestFree: 88 << 20, PoolSize: 6 << 30,
			},
			sentinel: accel.ErrFragmented,
			want: []string{
				`"blk.31.ffn_up.weight"`, "176.0 MiB", "align 256", `"weights"`, "Device",
				"6.0 GiB", "210.4 MiB", "88.0 MiB", "not contiguous space", "5.3",
			},
		},
		{
			name: "request beyond the pool",
			err: &accel.AllocError{
				Label: "kv", Pool: "session", Kind: accel.MemoryDevice,
				Requested: 5 << 29, Alignment: 256, Free: 2 << 30, PoolSize: 2 << 30,
			},
			sentinel: accel.ErrOutOfDeviceMemory,
			want:     []string{`"kv"`, "2.5 GiB", `"session"`, "2.0 GiB", "do not grow", "5.5"},
		},
		{
			name: "exhausted, not fragmented",
			err: &accel.AllocError{
				Label: "scratch", Pool: "transients", Kind: accel.MemoryUpload,
				Requested: 4096, Alignment: 256, Free: 512, LargestFree: 512, PoolSize: 1 << 20,
			},
			sentinel: accel.ErrOutOfDeviceMemory,
			want:     []string{`"scratch"`, "Upload", "Exhausted, not fragmented"},
		},
		{
			name: "misaligned view offset",
			err: &accel.AlignmentError{
				What: "view offset", Resource: "kv_cache.k", Offset: 8,
				Required: 256, Source: "MinStorageBufferOffsetAlignment",
			},
			sentinel: accel.ErrAlignment,
			want: []string{
				"view offset", `"kv_cache.k"`, "byte 8", "256",
				"Limits.MinStorageBufferOffsetAlignment",
			},
		},
		{
			name: "undeclared usage",
			err: &accel.UsageError{
				Resource: "logits", Node: 41, Slot: 2,
				Declared: accel.UsageStorage | accel.UsageCopySrc,
				Needed:   accel.UsageCopyDst,
				Site:     "model/attention.go:118",
			},
			sentinel: accel.ErrUsage,
			want: []string{
				"node 41 slot 2", `"logits"`, "UsageStorage|UsageCopySrc", "UsageCopyDst",
				"model/attention.go:118",
			},
		},
		{
			name:     "format not usable",
			err:      &accel.FormatError{Format: accel.Depth24PlusStencil8, Want: "host copyable", Device: "Metal"},
			sentinel: accel.ErrFormat,
			want:     []string{"host copyable", "Metal"},
		},
		{
			name:     "closed resource",
			err:      &accel.LifetimeError{Op: "WriteBuffer", Resource: "logits", Reason: "closed"},
			sentinel: accel.ErrLifetime,
			want:     []string{"WriteBuffer", `"logits"`, "closed"},
		},
		{
			name:     "in flight",
			err:      &accel.LifetimeError{Op: "Close", Resource: "hdr_target", Reason: "in flight", InFlight: 2},
			sentinel: accel.ErrLifetime,
			want:     []string{`"hdr_target"`, "2 submissions", "Wait on the fence"},
		},
		{
			name:     "pending transfer",
			err:      &accel.LifetimeError{Op: "Close", Resource: "staging", Reason: "pending transfer"},
			sentinel: accel.ErrLifetime,
			want:     []string{`"staging"`, "not been flushed", "Queue.Flush().Wait()"},
		},
		{
			name:     "live children",
			err:      &accel.LifetimeError{Op: "Close", Resource: "weights", Reason: "has live children", Children: 1284},
			sentinel: accel.ErrLifetime,
			want:     []string{`"weights"`, "1284 live children", "rather than recursive", "7.2"},
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			msg := tc.err.Error()
			for _, want := range tc.want {
				if !strings.Contains(msg, want) {
					t.Errorf("message does not carry %q:\n  %s", want, msg)
				}
			}
			if !errors.Is(tc.err, tc.sentinel) {
				t.Errorf("%v does not unwrap to %v, so a caller cannot branch on the class", tc.err, tc.sentinel)
			}
		})
	}
}

// TestUsageString covers the set formatting every usage error is written
// against.
func TestUsageString(t *testing.T) {
	for _, tc := range []struct {
		u    accel.BufferUsage
		want string
	}{
		{0, "no usage"},
		{accel.UsageStorage, "UsageStorage"},
		{accel.UsageCopySrc | accel.UsageCopyDst, "UsageCopySrc|UsageCopyDst"},
		{accel.UsageStorage | 1<<20, "UsageStorage|BufferUsage(1048576)"},
	} {
		if got := tc.u.String(); got != tc.want {
			t.Errorf("BufferUsage(%d).String() = %q, want %q", tc.u, got, tc.want)
		}
	}
}

// TestMemoryKindString covers the name every pool error quotes.
func TestMemoryKindString(t *testing.T) {
	for _, tc := range []struct {
		k    accel.MemoryKind
		want string
	}{
		{accel.MemoryDevice, "Device"},
		{accel.MemoryUpload, "Upload"},
		{accel.MemoryReadback, "Readback"},
		{accel.MemoryShared, "Shared"},
		{accel.MemoryKind(9), "MemoryKind(9)"},
	} {
		if got := tc.k.String(); got != tc.want {
			t.Errorf("MemoryKind(%d).String() = %q, want %q", int(tc.k), got, tc.want)
		}
	}
}
