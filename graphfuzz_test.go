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

// A recorded graph either builds or fails with a validation error. It never
// panics and it never reports success on a graph it could not lower.
//
// This is the property that has to hold before specs/017-graph-aliasing.md can
// compare an optimized plan against this one: an oracle that crashes on a third
// of the corpus proves nothing about the two thirds it survives.
func FuzzRecordAndBuild(f *testing.F) {
	f.Add([]byte{0})
	f.Add([]byte{1, 2, 3, 4, 5, 6, 7, 8})
	f.Add([]byte{9, 3, 0, 255, 128, 64, 1, 1, 2, 2, 7, 7})

	d, err := accel.OpenCPU(accel.CPUOptions{})
	if err != nil {
		f.Fatalf("open: %v", err)
	}
	f.Cleanup(func() { _ = d.Close() })

	const elems = 64
	bufs := make([]*accel.Buffer, 4)
	for i := range bufs {
		b, err := d.NewBuffer(accel.BufferDescriptor{
			DType: accel.F32, Count: elems, Label: "fuzz",
			Usage: accel.UsageStorage | accel.UsageCopySrc | accel.UsageCopyDst,
		})
		if err != nil {
			f.Fatalf("buffer: %v", err)
		}
		bufs[i] = b
	}
	f.Cleanup(func() {
		for _, b := range bufs {
			_ = b.Close()
		}
	})

	f.Fuzz(func(t *testing.T, seed []byte) {
		if len(seed) == 0 {
			return
		}
		at := 0
		next := func(mod int) int {
			v := int(seed[at%len(seed)])
			at++
			if mod <= 0 {
				return 0
			}
			return v % mod
		}

		r := d.NewRecorder()
		var views []accel.BufferView
		for _, b := range bufs {
			// A whole view and a sub-range each, so the corpus reaches overlapping
			// and disjoint declarations rather than only whole resources.
			if v, err := b.View(0, elems); err == nil {
				views = append(views, v)
			}
			if v, err := b.View(next(elems), next(elems)); err == nil {
				views = append(views, v)
			}
		}
		for range next(6) {
			v := r.Transient(accel.BufferDescriptor{
				DType: accel.F32, Count: 1 + next(elems),
				Usage: accel.UsageStorage | accel.UsageCopySrc | accel.UsageCopyDst,
			})
			if v.Buffer != nil {
				views = append(views, v)
			}
		}
		var slots []accel.Slot
		for range next(3) {
			slots = append(slots, r.Slot(accel.SlotDescriptor{
				Kind: accel.BindingStorageBuffer, DType: accel.F32,
				Access: accel.AccessReadWrite, MinCount: 1 + next(elems),
			}))
		}

		for range 1 + next(12) {
			switch next(4) {
			case 0:
				dst := views[next(len(views))]
				r.CopyToBuffer(dst, make([]float32, dst.Count))
			case 1:
				r.CopyBuffer(views[next(len(views))], views[next(len(views))])
			case 2:
				if len(slots) > 0 {
					r.CopyFromSlot(views[next(len(views))], slots[next(len(slots))], next(elems), next(elems))
				}
			case 3:
				if len(slots) > 0 {
					r.CopyToSlot(slots[next(len(slots))], next(elems), next(elems), views[next(len(views))])
				}
			}
		}

		g, err := r.Build()
		if err != nil {
			// A rejection must name itself. "invalid" tells a caller nothing and
			// is the failure mode spec 001 section 9 calls a defect.
			if !strings.HasPrefix(err.Error(), "accel:") {
				t.Fatalf("a rejection should be an accel error, got %v", err)
			}
			return
		}
		defer g.Close()

		// A built graph's reported plan has to be self-consistent, whatever the
		// input was: a plan with more barriers than nodes, or a pool smaller than
		// its peak, is a builder defect rather than a bad recording.
		m := g.Memory()
		if m.TransientBytes != m.UnaliasedBytes {
			t.Fatalf("this milestone does not alias: pool %d, unaliased %d",
				m.TransientBytes, m.UnaliasedBytes)
		}
		if m.PeakBytes > m.TransientBytes {
			t.Fatalf("peak %d exceeds the pool %d", m.PeakBytes, m.TransientBytes)
		}
		nodes := g.Nodes()
		if g.Barriers() > len(nodes) {
			t.Fatalf("%d barriers for %d nodes: at most one is emitted per node",
				g.Barriers(), len(nodes))
		}
		// Record order must be a topological order of the inferred DAG. That is
		// what makes the conservative plan of specs/015-graph-recording.md
		// correct, and it is what specs/017-graph-aliasing.md's oracle rests on.
		for from, succ := range g.Edges() {
			for _, to := range succ {
				if int(to) <= from {
					t.Fatalf("edge %d -> %d runs backwards in record order", from, int(to))
				}
			}
		}

		// It must also run, or refuse for a reason. Unbound slots are the
		// expected refusal and are not a defect.
		if err := d.Queue().Submit(g).Wait(); err != nil {
			if !strings.Contains(err.Error(), "no resource bound") &&
				!errors.Is(err, accel.ErrGraphInFlight) {
				t.Fatalf("submit: %v", err)
			}
		}
	})
}
