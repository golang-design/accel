// Copyright 2026 The golang.design Initiative Authors.
// All rights reserved. Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

package cpu

import (
	"fmt"

	"golang.design/x/accel/internal/driver"
	"golang.design/x/accel/internal/kernel"
	"golang.design/x/accel/internal/raster"
)

// Executing a render pass on the CPU backend.
//
// This file is a translation layer and nothing else: it turns a plan node into
// the calls internal/raster already takes, and every rule it appears to make is
// that package's. That separation is what specs/035-cpu-rasterizer.md's oracle
// argument rests on — the rasterizer is tested directly, and this is checked by
// the pass producing what the rasterizer produced.

// renderPass rasterizes one recorded pass.
//
// A panic inside a stage becomes an error for the same reason a kernel's does:
// on a GPU an out-of-bounds attribute read is undefined rather than a crash, so
// it is a stage bug either way and the oracle's job is to say so loudly rather
// than take the caller's process down.
func renderPass(n *resolvedNode) (err error) {
	defer func() {
		if r := recover(); r != nil {
			err = fmt.Errorf("accel: render pass %q panicked at node %d: %v; on a GPU "+
				"backend this would be undefined rather than a crash, so it is a stage "+
				"bug either way", n.render.Label, n.id, r)
		}
	}()

	rp := n.render
	fb, err := framebufferFor(n)
	if err != nil {
		return err
	}
	applyLoads(rp, fb)

	for i, d := range rp.Draws {
		if err := drawOne(rp, fb, d); err != nil {
			return fmt.Errorf("accel: render pass %q draw %d: %w", rp.Label, i, err)
		}
	}
	return nil
}

// framebufferFor wraps the resolved attachments as raster targets.
//
// The targets alias the attachment memory rather than copying it: the graph
// ordered other nodes against those bytes, so writing anywhere else would make
// every inferred edge describe memory nobody wrote.
func framebufferFor(n *resolvedNode) (*raster.Framebuffer, error) {
	rp := n.render
	fb := &raster.Framebuffer{}
	for i, pix := range n.colorAttach {
		want := rp.Width * rp.Height * 4
		if len(pix) < want {
			return nil, fmt.Errorf("accel: render pass %q colour attachment %d holds %d "+
				"floats and a %dx%d area needs %d", rp.Label, i, len(pix),
				rp.Width, rp.Height, want)
		}
		fb.Color = append(fb.Color, &raster.ColorTarget{
			W: rp.Width, H: rp.Height, Pix: pix[:want],
		})
	}
	if n.depthAttach != nil {
		want := rp.Width * rp.Height
		if len(n.depthAttach) < want {
			return nil, fmt.Errorf("accel: render pass %q depth attachment holds %d floats "+
				"and a %dx%d area needs %d", rp.Label, len(n.depthAttach),
				rp.Width, rp.Height, want)
		}
		fb.Depth = &raster.DepthTarget{
			W: rp.Width, H: rp.Height, Z: n.depthAttach[:want],
			Stencil: make([]uint8, want),
		}
	}
	return fb, nil
}

// applyLoads performs each attachment's load action.
//
// Clear is a fill here because there is no tile memory to make it free. What
// the action buys on this backend is the *graph* consequence rather than the
// execution one: DontCare is what removed the read-after-write edge, and that
// happened at build.
func applyLoads(rp *driver.RenderPass, fb *raster.Framebuffer) {
	for i, t := range fb.Color {
		if i < len(rp.ColorLoad) && rp.ColorLoad[i] == uint8(loadClear) {
			t.Clear(rp.ColorClear[i])
		}
	}
	if fb.Depth != nil && rp.DepthLoad == uint8(loadClear) {
		fb.Depth.Clear(rp.DepthClear, 0)
	}
}

// The load actions, mirroring the public LoadOp. Named here rather than shared,
// because the public package cannot be imported from a backend.
const (
	loadClear uint8 = iota
	loadKeep
	loadDontCare
)

// drawOne rasterizes one draw.
func drawOne(rp *driver.RenderPass, fb *raster.Framebuffer, d driver.RenderDraw) error {
	vs, ok := d.Vertex.(kernel.VertexFn)
	if !ok {
		return fmt.Errorf("the vertex stage is %T, not a compiled stage", d.Vertex)
	}
	fs, ok := d.Fragment.(kernel.FragmentFn)
	if !ok {
		return fmt.Errorf("the fragment stage is %T, not a compiled stage", d.Fragment)
	}

	ps := raster.PassState{
		State: raster.State{
			Viewport: raster.Viewport{W: rp.Width, H: rp.Height, MinDepth: 0, MaxDepth: 1},
			Front:    raster.FrontFace(d.FrontFace),
			Cull:     raster.Cull(d.Cull),
		},
		Depth: raster.DepthState{
			Test: d.DepthTest, Write: d.DepthWrite,
			Compare: raster.Compare(d.DepthCompare),
		},
	}
	for _, m := range d.Masks {
		ps.Mask = append(ps.Mask, raster.WriteMask(m))
	}

	// The vertex function, in the form primitive assembly takes. Attributes are
	// not bound yet — 033 leaves the vertex layout to the pipeline and no
	// buffer reaches here — so a stage that reads one gets an empty slice
	// rather than another vertex's data.
	vertexFn := func(index, instance, base uint32) raster.Vertex {
		pos, vary := vs(kernel.NewVertex(index, instance), d.Uniforms, nil)
		return raster.Vertex{
			Pos:      raster.Clip{X: pos[0], Y: pos[1], Z: pos[2], W: pos[3]},
			Varyings: vary,
		}
	}

	shade := func(f raster.Fragment) raster.Shaded {
		out := fs(kernel.NewFragment(
			kernel.Vec4{float32(f.X) + 0.5, float32(f.Y) + 0.5, f.Depth, f.InvW},
			f.Front), d.Uniforms, f.Varyings)
		return raster.Shaded{Color: out}
	}

	_, err := raster.DrawPrimitives(ps, fb, raster.DrawCall{
		Topology:      raster.Topology(d.Topology),
		Count:         d.VertexCount,
		Instances:     d.InstanceCount,
		First:         d.FirstVertex,
		FirstInstance: d.FirstInstance,
	}, vertexFn, shade)
	return err
}
