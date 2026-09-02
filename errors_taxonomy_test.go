// Copyright 2026 The golang.design Initiative Authors.
// All rights reserved. Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

package accel_test

import (
	"errors"
	"runtime"
	"strconv"
	"strings"
	"testing"

	"golang.design/x/accel"
	"golang.design/x/accel/internal/kernels"
)

// A build error is a collection of node errors, each naming its node, its
// kind, the caller's line, and the sentinel a caller can switch on.
//
// specs/003-command-graph.md's error taxonomy. The sentinels shipped on
// 2026-08-27 and the structure did not: build errors were an errors.Join of
// formatted strings, so a caller could match a dtype mismatch and not learn
// which of their lines recorded it. Now errors.As reaches *BuildError and each
// *NodeError, and the site is the caller's file and line with a zero column,
// the format the kernel compiler's diagnostics share.
func TestABuildErrorNamesEachNodeAndItsSite(t *testing.T) {
	d := openDevice(t)
	p, err := d.NewComputePipeline(accel.ComputePipelineDescriptor{Kernel: &kernels.ScaleKernel, Label: "scale"})
	if err != nil {
		t.Fatalf("pipeline: %v", err)
	}
	defer p.Close()
	// Scale reads f32; a u32 view is the dtype mismatch V3 refuses.
	wrong := newBuffer(t, d, "wrong", 64, accel.BufferStorage)
	right := newBuffer(t, d, "right", 64, accel.BufferStorage)
	wv, err := wrong.ViewAs(accel.U32, 0, 16)
	if err != nil {
		t.Fatal(err)
	}

	r := d.NewRecorder()
	_, _, line, _ := runtime.Caller(0)
	r.Dispatch(p, []accel.Binding{{Index: 0, Buffer: wv}, {Index: 1, Buffer: whole(t, right)}},
		nil, accel.WorkgroupCount{X: 1})
	r.CopyBuffer(accel.BufferView{}, whole(t, right))
	_, err = r.Build()
	if err == nil {
		t.Fatal("a dtype mismatch and a copy from nothing built")
	}

	var be *accel.BuildError
	if !errors.As(err, &be) {
		t.Fatalf("Build returned %T, not a *BuildError: %v", err, err)
	}
	// The dispatch raises the mismatch and, because the binding then went
	// unfilled, the missing-resource error too; the copy raises one.
	if len(be.Errs) < 2 {
		t.Fatalf("%d node errors, want at least 2:\n%v", len(be.Errs), err)
	}
	if !strings.HasPrefix(err.Error(), "accel: graph build failed: "+strconv.Itoa(len(be.Errs))+" errors") {
		t.Errorf("the message does not lead with the count: %q", err.Error())
	}

	first := be.Errs[0]
	if first.Kind != accel.NodeDispatch || first.Label != "scale" || first.Node != 0 {
		t.Errorf("the first error is node %d %q (%s), want node 0 \"scale\" (dispatch)",
			int(first.Node), first.Label, first.Kind)
	}
	wantSite := "errors_taxonomy_test.go:" + strconv.Itoa(line+1) + ":0"
	if !strings.HasSuffix(first.Site, wantSite) {
		t.Errorf("the first error's site is %q, want a suffix of %q", first.Site, wantSite)
	}
	if !strings.Contains(first.Detail, "u32") {
		t.Errorf("the detail does not carry the numbers: %q", first.Detail)
	}
	last := be.Errs[len(be.Errs)-1]
	if last.Kind != accel.NodeCopyBuffer || last.Node != 1 {
		t.Errorf("the last error is node %d (%s), want node 1 (buffer copy)", int(last.Node), last.Kind)
	}

	// The sentinels still reach through: errors.Is on the collection, and
	// errors.As on the first node.
	var ne *accel.NodeError
	if !errors.As(err, &ne) || ne != first {
		t.Errorf("errors.As did not find the first node error")
	}
	// The format: one entry per line, site first.
	lines := strings.Split(err.Error(), "\n")
	if len(lines) != 1+len(be.Errs) || !strings.HasPrefix(strings.TrimSpace(lines[1]), first.Site) {
		t.Errorf("the message is not one entry per line, site first:\n%s", err.Error())
	}
}

// A sentinel wrapped by a check is reachable through the collection.
func TestASentinelReachesThroughABuildError(t *testing.T) {
	d := openDevice(t)
	r := d.NewRecorder()
	s := r.Slot(accel.SlotDescriptor{Name: "in", DType: accel.F32, MinCount: 4, Access: accel.AccessRead})
	p, err := d.NewComputePipeline(accel.ComputePipelineDescriptor{Kernel: &kernels.ScaleKernel, Label: "scale"})
	if err != nil {
		t.Fatal(err)
	}
	defer p.Close()
	out := newBuffer(t, d, "out", 64, accel.BufferStorage)
	r.Dispatch(p, []accel.Binding{{Index: 0, Slot: s}, {Index: 1, Buffer: whole(t, out)}},
		nil, accel.WorkgroupCount{X: 1})
	g, err := r.Build()
	if err != nil {
		t.Fatalf("build: %v", err)
	}
	defer g.Close()
	wrong := newBuffer(t, d, "wrong", 64, accel.BufferStorage)
	wv, _ := wrong.ViewAs(accel.U32, 0, 16)
	err = g.Bind(accel.SlotBinding{Slot: s, Buffer: wv})
	if !errors.Is(err, accel.ErrDTypeMismatch) {
		t.Fatalf("a u32 buffer on an f32 slot is not ErrDTypeMismatch: %v", err)
	}
}
