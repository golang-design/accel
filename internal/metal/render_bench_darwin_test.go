// Copyright 2026 The golang.design Initiative Authors.
// All rights reserved. Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

//go:build darwin

package metal_test

import (
	"math"
	"testing"

	"golang.design/x/accel/internal/driver"
	"golang.design/x/accel/internal/kernel"
	"golang.design/x/accel/internal/kernels"
	"golang.design/x/accel/internal/metal"
)

// BenchmarkRenderSubmit measures one render pass of many draws, each with a
// by-value parameter on both stages and a depth rule, resubmitted.
//
// What it is for is the allocation count. A replayed pass compiles nothing,
// so everything it allocates per submission is bookkeeping: the pipeline and
// depth-state cache keys, the per-draw map that tracks which uniforms a buffer
// covered, and the encoded uniform bytes. Run with -benchmem.
func BenchmarkRenderSubmit(b *testing.B) {
	const draws = 32
	e := renderExecutable(b, draws)
	b.ReportAllocs()
	benchSubmit(b, e, draws)
}

// BenchmarkIndirectSubmit measures a plan of indirect dispatches, where each
// node encodes a clamp before its dispatch.
func BenchmarkIndirectSubmit(b *testing.B) {
	const nodes = 64
	e := indirectExecutable(b, nodes)
	b.ReportAllocs()
	benchSubmit(b, e, nodes)
}

func openBench(tb testing.TB) driver.Device {
	tb.Helper()
	ads, err := metal.Adapters()
	if err != nil || len(ads) == 0 {
		tb.Skipf("no Metal adapter on this machine (err=%v)", err)
	}
	d, err := ads[0].Open(nil)
	if err != nil {
		tb.Fatalf("open: %v", err)
	}
	tb.Cleanup(func() { _ = d.Close() })
	return d
}

func benchBlock(tb testing.TB, d driver.Device, n int, label string) (driver.Block, driver.Operand) {
	tb.Helper()
	blk, err := d.Alloc(driver.MemoryShared, n, label)
	if err != nil {
		tb.Fatalf("alloc %s: %v", label, err)
	}
	tb.Cleanup(blk.Free)
	op, err := driver.BlockOperand(blk, 0, n)
	if err != nil {
		tb.Fatalf("operand: %v", err)
	}
	return blk, op
}

// renderExecutable compiles one pass of draws over an 8x8 target with depth.
func renderExecutable(tb testing.TB, draws int) driver.Executable {
	tb.Helper()
	d := openBench(tb)
	const w, h, pitch = 8, 8, 256
	_, color := benchBlock(tb, d, pitch*h, "color")
	_, depth := benchBlock(tb, d, pitch*h, "depth")
	verts, vop := benchBlock(tb, d, 3*12, "verts")
	tri := []float32{-1, -1, 0, 3, -1, 0, -1, 3, 0}
	for i, v := range tri {
		bits := math.Float32bits(v)
		copy(verts.Bytes()[i*4:], []byte{byte(bits), byte(bits >> 8), byte(bits >> 16), byte(bits >> 24)})
	}

	rp := &driver.RenderPass{
		Width: w, Height: h,
		Color:       []driver.Operand{color},
		ColorFormat: []driver.Format{driver.RGBA32Float},
		ColorPitch:  []int{pitch},
		ColorLoad:   []driver.LoadOp{driver.LoadClear},
		ColorStore:  []driver.StoreOp{driver.StoreKeep},
		ColorClear:  [][4]float32{{0, 0, 0, 1}},
		Depth:       &depth,
		DepthFormat: driver.Depth32Float,
		DepthPitch:  pitch,
		DepthLoad:   driver.LoadClear,
		DepthStore:  driver.StoreKeep,
		DepthClear:  1,
	}
	for i := range draws {
		rp.Draws = append(rp.Draws, driver.RenderDraw{
			Vertex: &kernels.ScaledVSStage, Fragment: &kernels.TintedFSStage,
			VertexCount: 3, InstanceCount: 1,
			Masks:  []uint8{0xf},
			Blends: []driver.Blend{{}},
			// Alternating rules, so the depth-state cache is consulted with
			// more than one key per submission.
			DepthTest: true, DepthWrite: i%2 == 0, DepthCompare: 3,
			VertexBuffers: []driver.Operand{vop},
			VertexLayouts: []driver.VertexLayout{{
				Stride:     12,
				Attributes: []driver.VertexAttribute{{Location: 0, Offset: 0, Components: 3}},
			}},
			VertexUniforms:   []any{kernels.StageTransform{Scale: 1}},
			FragmentUniforms: []any{kernels.StageTint{Colour: [4]float32{1, 0, 0, 1}}},
		})
	}
	e, err := d.(driver.GraphCompiler).Compile(&driver.Plan{Label: "render", Nodes: []driver.PlanNode{{
		Op: driver.OpRenderPass, Render: rp,
	}}})
	if err != nil {
		tb.Fatalf("compile: %v", err)
	}
	tb.Cleanup(func() { _ = e.Close() })
	return e
}

// indirectExecutable compiles n indirect dispatches of the add kernel, each
// reading its count from a device buffer.
func indirectExecutable(tb testing.TB, n int) driver.Executable {
	tb.Helper()
	d := openBench(tb)
	const count = 256
	_, in1 := benchBlock(tb, d, count*4, "in1")
	_, in2 := benchBlock(tb, d, count*4, "in2")
	_, out := benchBlock(tb, d, count*4, "out")
	cnt, cop := benchBlock(tb, d, 12, "count")
	copy(cnt.Bytes(), []byte{count / 64, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0})

	plan := &driver.Plan{Label: "indirect"}
	for i := range n {
		plan.Nodes = append(plan.Nodes, driver.PlanNode{
			Op: driver.OpDispatch, ID: i,
			Dispatch: &driver.Dispatch{
				Kernel:   &kernels.AddKernel,
				Count:    kernel.ID3{X: count / 64, Y: 1, Z: 1},
				Bindings: []driver.Operand{in1, in2, out},
				Indirect: &driver.Indirect{Count: cop, Max: kernel.ID3{X: count / 64, Y: 1, Z: 1}},
			},
		})
	}
	e, err := d.(driver.GraphCompiler).Compile(plan)
	if err != nil {
		tb.Fatalf("compile: %v", err)
	}
	tb.Cleanup(func() { _ = e.Close() })
	return e
}
