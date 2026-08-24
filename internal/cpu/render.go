// Copyright 2026 The golang.design Initiative Authors.
// All rights reserved. Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

package cpu

import (
	"encoding/binary"
	"fmt"
	"math"

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
		if err := drawOne(rp, fb, d, n.vertexBytes[i], n.indexBytes[i], n.indirectArgs[i]); err != nil {
			return fmt.Errorf("accel: render pass %q draw %d: %w", rp.Label, i, err)
		}
	}
	return storeAttachments(n, fb)
}

// framebufferFor decodes the resolved attachments into raster targets.
//
// # Why this decodes rather than aliasing
//
// It used to alias: the attachment bytes were reinterpreted as []float32 and
// handed to the rasterizer, which is correct for exactly one format and wrong
// in silence for every other. The rasterizer works in float32 components and
// an attachment holds whatever its format holds, so the conversion has to
// happen somewhere, and the only place it can happen without teaching the
// rasterizer about formats is here.
//
// What that costs is a copy in and a copy out per pass. That is the same cost
// specs/045-texture-attachments.md section 1 charges the Metal backend for, and
// it is not the same mistake: Metal pays it to move bytes between two
// resources of the same format, and this pays it to convert. A backend with a
// fixed-function output stage does the conversion in hardware and pays
// nothing.
//
// Every attachment is decoded, including one loaded Clear whose contents the
// clear is about to overwrite. Per-attachment special cases here would be a
// second definition of what a load action means, and the one in applyLoads is
// the definition.
func framebufferFor(n *resolvedNode) (*raster.Framebuffer, error) {
	rp := n.render
	fb := &raster.Framebuffer{}
	for i, raw := range n.colorAttach {
		c := n.colorCodec[i]
		pix := make([]float32, rp.Width*rp.Height*c.components)
		if err := c.decodeImage(pix, raw, rp.Width, rp.Height, rp.ColorPitch[i]); err != nil {
			return nil, fmt.Errorf("accel: render pass %q colour attachment %d: %w",
				rp.Label, i, err)
		}
		fb.Color = append(fb.Color, &raster.ColorTarget{W: rp.Width, H: rp.Height, Pix: pix})
	}
	if n.depthAttach != nil {
		c := n.depthCodec
		z := make([]float32, rp.Width*rp.Height*c.components)
		if err := c.decodeImage(z, n.depthAttach, rp.Width, rp.Height, rp.DepthPitch); err != nil {
			return nil, fmt.Errorf("accel: render pass %q depth attachment: %w", rp.Label, err)
		}
		fb.Depth = &raster.DepthTarget{
			W: rp.Width, H: rp.Height, Z: z,
			Stencil: make([]uint8, rp.Width*rp.Height),
		}
	}
	return fb, nil
}

// storeAttachments encodes the framebuffer back into the attachment memory.
//
// Unconditionally, including for an attachment stored Discard. Discard leaves
// the contents undefined, and undefined permits writing them; skipping the
// write would leave the bytes the pass started with, which is a *defined*
// result that happens to look wrong, and a test asserting it would be
// asserting an accident. What StoreDiscard buys on this backend is the graph
// consequence, the same way LoadDontCare does.
func storeAttachments(n *resolvedNode, fb *raster.Framebuffer) error {
	rp := n.render
	for i, raw := range n.colorAttach {
		if err := n.colorCodec[i].encodeImage(raw, fb.Color[i].Pix,
			rp.Width, rp.Height, rp.ColorPitch[i]); err != nil {
			return fmt.Errorf("accel: render pass %q colour attachment %d: %w",
				rp.Label, i, err)
		}
	}
	if n.depthAttach != nil {
		if err := n.depthCodec.encodeImage(n.depthAttach, fb.Depth.Z,
			rp.Width, rp.Height, rp.DepthPitch); err != nil {
			return fmt.Errorf("accel: render pass %q depth attachment: %w", rp.Label, err)
		}
	}
	return nil
}

// applyLoads performs each attachment's load action.
//
// Clear is a fill here because there is no tile memory to make it free. What
// the action buys on this backend is the *graph* consequence rather than the
// execution one: DontCare is what removed the read-after-write edge, and that
// happened at build.
func applyLoads(rp *driver.RenderPass, fb *raster.Framebuffer) {
	for i, t := range fb.Color {
		if i < len(rp.ColorLoad) && rp.ColorLoad[i] == driver.LoadClear {
			t.Clear(rp.ColorClear[i])
		}
	}
	if fb.Depth != nil && rp.DepthLoad == driver.LoadClear {
		fb.Depth.Clear(rp.DepthClear, 0)
	}
}

// drawOne rasterizes one draw.
func drawOne(rp *driver.RenderPass, fb *raster.Framebuffer, d driver.RenderDraw, bufs [][]byte, indices, args []byte) error {
	vs, fs := d.Vertex.RunVertex, d.Fragment.RunFragment
	if vs == nil || fs == nil {
		return fmt.Errorf("stage %q or %q carries no generated adapter, so this backend "+
			"has nothing to run", d.Vertex.Name, d.Fragment.Name)
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
	for _, b := range d.Blends {
		ps.Blend = append(ps.Blend, rasterBlend(b))
	}

	fetch := newFetcher(d, bufs)

	// The vertex function, in the form primitive assembly takes. The two
	// uniform slices are separate because each stage indexes its own from
	// zero -- one shared slice would give a fragment stage the vertex stage's
	// parameter 0, and the adapter would assert on the wrong type.
	vertexFn := func(index, instance, base uint32) raster.Vertex {
		// base is added to the fetch and not to the index the stage sees.
		// specs/032-stage-abi.md section 2.1 declines to expose a base-vertex
		// built-in because backends disagree about whether theirs reports the
		// pre-offset or post-offset value; the ABI exposes only the one a
		// caller can act on, and that is the pre-offset index.
		pos, vary := vs(kernel.NewVertex(index, instance), d.VertexUniforms,
			fetch(index+base, instance))
		return raster.Vertex{
			Pos:      raster.Clip{X: pos[0], Y: pos[1], Z: pos[2], W: pos[3]},
			Varyings: vary,
		}
	}

	shade := func(f raster.Fragment) raster.Shaded {
		out := fs(kernel.NewFragment(
			kernel.Vec4{float32(f.X) + 0.5, float32(f.Y) + 0.5, f.Depth, f.InvW},
			f.Front), d.FragmentUniforms, f.Varyings)
		return raster.Shaded{Color: out}
	}

	dc := raster.DrawCall{
		Topology:      raster.Topology(d.Topology),
		Count:         d.VertexCount,
		Instances:     d.InstanceCount,
		First:         d.FirstVertex,
		FirstInstance: d.FirstInstance,
		BaseVertex:    d.BaseVertex,
	}
	if d.Indirect {
		dc.Count, dc.Instances, dc.First, dc.FirstInstance = readIndirectDraw(args, d)
	}
	if d.Indexed {
		dc.Index = decodeIndices(indices, d.IndexWidth)
		if err := checkIndexRange(d, dc.Index, bufs); err != nil {
			return err
		}
	}
	_, err := raster.DrawPrimitives(ps, fb, dc, vertexFn, shade)
	return err
}

// newFetcher builds the per-vertex attribute fetch.
//
// One closure for the whole draw rather than a lookup per vertex: the layout is
// fixed for the draw, so the slice shapes and the byte strides are computed once
// and the inner call is an index and a decode. A rasterizer calls this per
// vertex of every primitive, and for a strip that is per primitive again.
//
// The returned slice is reused across calls. A generated adapter copies out of
// it immediately -- it converts to [N]float32 by value -- so nothing outlives
// the call, and allocating a fresh slice per vertex would be the largest
// allocation in the draw.
func newFetcher(d driver.RenderDraw, bufs [][]byte) func(index, instance uint32) [][]float32 {
	// The stage indexes its attributes densely, and pipeline creation checked
	// that the layout declares each exactly once, so the widest location plus
	// one is the count.
	n := 0
	for _, l := range d.VertexLayouts {
		for _, a := range l.Attributes {
			if a.Location+1 > n {
				n = a.Location + 1
			}
		}
	}
	if n == 0 {
		return func(uint32, uint32) [][]float32 { return nil }
	}

	type source struct {
		bytes      []byte
		stride     int
		offset     int
		components int
		perInst    bool
	}
	src := make([]source, n)
	out := make([][]float32, n)
	for i, l := range d.VertexLayouts {
		raw := bufs[i]
		for _, a := range l.Attributes {
			src[a.Location] = source{
				bytes: raw, stride: l.Stride, offset: a.Offset,
				components: a.Components, perInst: l.PerInstance,
			}
			out[a.Location] = make([]float32, a.Components)
		}
	}

	return func(index, instance uint32) [][]float32 {
		for i := range src {
			s := &src[i]
			at := int(index)
			if s.perInst {
				at = int(instance)
			}
			base := at*s.stride + s.offset
			for c := range s.components {
				out[i][c] = math.Float32frombits(
					binary.LittleEndian.Uint32(s.bytes[base+c*4:]))
			}
		}
		return out
	}
}

// rasterBlend maps a plan's blend state onto the rasterizer's.
//
// Written out rather than converted numerically. The two enumerations agree
// today, and a conversion that relies on that agreement is a silent
// mistranslation the day one of them gains a value in the middle -- every blend
// after the insertion would shift, and the result is a plausible image rather
// than an error. TestBlendFactorsMapOneToOne checks every value.
func rasterBlend(b driver.Blend) raster.Blend {
	return raster.Blend{
		Enabled:  b.Enabled,
		SrcColor: rasterFactor(b.SrcColor), DstColor: rasterFactor(b.DstColor),
		ColorOp:  rasterOp(b.ColorOp),
		SrcAlpha: rasterFactor(b.SrcAlpha), DstAlpha: rasterFactor(b.DstAlpha),
		AlphaOp: rasterOp(b.AlphaOp),
	}
}

func rasterFactor(f driver.BlendFactor) raster.BlendFactor {
	switch f {
	case driver.FactorZero:
		return raster.FactorZero
	case driver.FactorOne:
		return raster.FactorOne
	case driver.FactorSrcColor:
		return raster.FactorSrcColor
	case driver.FactorOneMinusSrcColor:
		return raster.FactorOneMinusSrcColor
	case driver.FactorSrcAlpha:
		return raster.FactorSrcAlpha
	case driver.FactorOneMinusSrcAlpha:
		return raster.FactorOneMinusSrcAlpha
	case driver.FactorDstColor:
		return raster.FactorDstColor
	case driver.FactorOneMinusDstColor:
		return raster.FactorOneMinusDstColor
	case driver.FactorDstAlpha:
		return raster.FactorDstAlpha
	case driver.FactorOneMinusDstAlpha:
		return raster.FactorOneMinusDstAlpha
	}
	return raster.FactorZero
}

func rasterOp(o driver.BlendOp) raster.BlendOp {
	switch o {
	case driver.BlendAdd:
		return raster.BlendAdd
	case driver.BlendSubtract:
		return raster.BlendSubtract
	case driver.BlendReverseSubtract:
		return raster.BlendReverseSubtract
	case driver.BlendMin:
		return raster.BlendMin
	case driver.BlendMax:
		return raster.BlendMax
	}
	return raster.BlendAdd
}

// decodeIndices widens an index buffer to what the rasterizer takes.
//
// Widened rather than read in place: the rasterizer works in uint32 so a
// 16-bit buffer would otherwise need a width test in the innermost loop of
// primitive assembly, and this is once per draw.
func decodeIndices(raw []byte, width int) []uint32 {
	out := make([]uint32, len(raw)/width)
	for i := range out {
		if width == 4 {
			out[i] = binary.LittleEndian.Uint32(raw[i*4:])
		} else {
			out[i] = uint32(binary.LittleEndian.Uint16(raw[i*2:]))
		}
	}
	return out
}

// checkIndexRange reports an index that reaches past a vertex buffer.
//
// Once per draw rather than per vertex: the indices are already decoded, so the
// largest is one pass over a slice, and the alternative is a bounds test in the
// innermost loop of primitive assembly. Build cannot do this -- an indexed
// draw's vertex range is decided by the index values, which are data.
func checkIndexRange(d driver.RenderDraw, index []uint32, bufs [][]byte) error {
	if len(index) == 0 {
		return nil
	}
	var high uint32
	for _, i := range index {
		if i > high {
			high = i
		}
	}
	last := int(high) + d.BaseVertex + 1
	for i, l := range d.VertexLayouts {
		if l.PerInstance {
			continue
		}
		if need := last * d.VertexStrides[i]; need > len(bufs[i]) {
			return fmt.Errorf("index %d with base vertex %d reaches element %d of vertex "+
				"buffer %d, which holds %d bytes at a stride of %d and has %d elements",
				high, d.BaseVertex, last-1, i, len(bufs[i]), d.VertexStrides[i],
				len(bufs[i])/d.VertexStrides[i])
		}
	}
	return nil
}

// readIndirectDraw reads the four device-supplied draw arguments and clamps
// each to the maximum the node recorded.
//
// # Why the clamp is unconditional
//
// The same reason the indirect dispatch clamp is: specs/033-render-api.md
// section 4.2 says a count exceeding the maximum is clamped rather than
// silently truncating, and correctness cannot depend on a debug flag. What a
// caller gives up by not collecting statistics is being told that a clamp
// happened, not being protected from one -- which makes the maximum a caller
// obligation in release mode and a diagnostic in debug.
//
// The layout is the one Vulkan, D3D12 and Metal share: vertex count, instance
// count, first vertex, first instance, as four little-endian uint32.
func readIndirectDraw(args []byte, d driver.RenderDraw) (count, instances, first, firstInstance int) {
	read := func(i, limit int) int {
		v := int(binary.LittleEndian.Uint32(args[i*4:]))
		if v > limit {
			return limit
		}
		return v
	}
	return read(0, d.VertexCount), read(1, d.InstanceCount),
		read(2, d.FirstVertex), read(3, d.FirstInstance)
}
