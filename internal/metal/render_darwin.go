// Copyright 2026 The golang.design Initiative Authors.
// All rights reserved. Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

//go:build darwin

package metal

import (
	"fmt"

	"golang.design/x/accel/internal/driver"
	"golang.design/x/accel/internal/kernel"
	"golang.design/x/accel/internal/mslabi"
	"golang.design/x/accel/internal/mtl"
)

// Executing a render pass on Metal.
//
// # The shape of it
//
// An attachment is a buffer view, and a MTLRenderPassDescriptor takes textures,
// so this file's job is to give Metal a texture that *is* the caller's bytes.
// It does that by aliasing: MTLBuffer's newTextureWithDescriptor:offset:bytesPerRow:
// returns a texture over a region of a buffer, so a pass renders straight into
// the allocation the graph ordered every other node against.
//
// # What that deleted
//
// It used to allocate a private texture per attachment per pass, blit the
// caller's bytes in when the load action was Keep, render, and blit the result
// back out. specs/042-surface-completion.md's review measured what that cost --
// several full-frame round trips through system memory per frame at 1080p --
// and noted that none of it was visible as a failing test, because the images
// are identical either way. Aliasing removes the copies rather than making them
// faster.
//
// # When it cannot alias
//
// Metal requires a linear texture's offset and row pitch to be multiples of
// minimumLinearTextureAlignmentForPixelFormat. accel aligns every texture row
// to MinBufferCopyRowPitchAlignment, which is 256 and a multiple of the 16 that
// requires, so a texture attachment always qualifies. A *slot* attachment is a
// buffer view at whatever offset the caller bound, so it may not, and the
// staged path stays for that case -- counted, so a frame that is quietly paying
// for it can be seen rather than guessed at.

// renderPass encodes one pass and the blits around it.
func (e *executable) renderPass(p *pass, n *driver.PlanNode) error {
	rp := n.Render
	if rp == nil {
		return fmt.Errorf("accel: node %d is a render pass with no payload", n.ID)
	}
	// A render pass is its own encoder, so whatever is open ends here. Metal
	// orders one encoder against the next, which is the barrier this backend
	// has between passes.
	p.end()

	targets, err := e.renderTargets(p, rp)
	if err != nil {
		return err
	}
	defer func() {
		for _, t := range targets.textures {
			t.Close()
		}
	}()

	// The textures the stages fetch, materialized and filled before the render
	// encoder opens. A blit cannot be encoded inside a render encoder, so this
	// is the only place it can happen, and it has to be here rather than at
	// plan time: a pass that fetches what an earlier pass drew would otherwise
	// copy the buffer as it was before that pass ran.
	bound, owned, err := e.stageTextures(p, rp)
	defer func() {
		for _, t := range owned {
			t.Close()
		}
	}()
	if err != nil {
		return fmt.Errorf("accel: node %d: %w", n.ID, err)
	}

	// An aliased attachment needs no copy in: the texture already *is* the
	// buffer, so LoadKeep keeps what the buffer holds. A staged one still does.
	for i, a := range targets.color {
		if !targets.colorStaged[i] || rp.ColorLoad[i] != driver.LoadKeep {
			continue
		}
		op, err := e.operand(rp.Color[i])
		if err != nil {
			return fmt.Errorf("accel: node %d colour attachment %d: %w", n.ID, i, err)
		}
		p.blit().CopyBufferToTexture(op.buf, op.off, a.Texture, rp.ColorPitch[i])
	}
	if targets.depth != nil && targets.depthStaged && rp.DepthLoad == driver.LoadKeep {
		op, err := e.operand(*rp.Depth)
		if err != nil {
			return fmt.Errorf("accel: node %d depth attachment: %w", n.ID, err)
		}
		e.copyDepthPlanes(p, rp, op, targets.depth.Texture, true)
	}
	p.end()

	enc, err := p.cb.Render(targets.color, targets.depth)
	if err != nil {
		return fmt.Errorf("accel: node %d: %w", n.ID, err)
	}
	for i, d := range rp.Draws {
		if err := e.encodeDraw(enc, rp, d, bound[i]); err != nil {
			enc.End()
			return fmt.Errorf("accel: node %d draw %d: %w", n.ID, i, err)
		}
	}
	enc.End()

	// And back, for the staged ones only. An aliased attachment was written
	// where the graph ordered every other node against it, so there is nowhere
	// else for it to go.
	//
	// A discarded attachment is not written back either way. Its contents are
	// undefined after the pass, so the copy would be moving bytes nobody may
	// read -- and on a depth attachment that is the whole buffer every frame,
	// which is the saving specs/033-render-api.md names. An *aliased* discarded
	// attachment does not get that saving, because the pass wrote the bytes in
	// place; what it gets instead is the copy never happening at all.
	for i, a := range targets.color {
		if !targets.colorStaged[i] || rp.ColorStore[i] == driver.StoreDiscard {
			continue
		}
		op, err := e.operand(rp.Color[i])
		if err != nil {
			return fmt.Errorf("accel: node %d colour attachment %d: %w", n.ID, i, err)
		}
		// The destination's pitch, which the plan states and the CPU backend
		// already honours. The texture's rows are tight and the buffer's are
		// padded to the device's alignment, so writing at the texture's pitch
		// puts every row after the first in the wrong place.
		p.blit().CopyTextureToBuffer(a.Texture, op.buf, op.off, rp.ColorPitch[i])
	}
	if targets.depth != nil && targets.depthStaged && rp.DepthStore != driver.StoreDiscard {
		op, err := e.operand(*rp.Depth)
		if err != nil {
			return fmt.Errorf("accel: node %d depth attachment: %w", n.ID, err)
		}
		e.copyDepthPlanes(p, rp, op, targets.depth.Texture, false)
	}
	p.end()
	return nil
}

// renderAttachments is a pass's textures and the attachment records naming
// them.
type renderAttachments struct {
	color    []mtl.RenderAttachment
	depth    *mtl.RenderAttachment
	textures []*mtl.Texture

	// colorStaged and depthStaged mark the attachments that could not alias the
	// caller's bytes and therefore need copies around the pass.
	colorStaged []bool
	depthStaged bool
}

// renderTargets gives each attachment a texture over the caller's own bytes.
//
// Aliased where the offset and pitch allow it, which for a texture attachment
// is always: see the file comment. staged records the attachments that had to
// fall back, so the pass knows which ones need copies around it.
func (e *executable) renderTargets(p *pass, rp *driver.RenderPass) (renderAttachments, error) {
	var out renderAttachments
	fail := func(err error) (renderAttachments, error) {
		for _, done := range out.textures {
			done.Close()
		}
		return renderAttachments{}, err
	}
	for i := range rp.Color {
		t, staged, err := e.attachmentTexture(rp.Color[i], rp.ColorFormat[i],
			rp.Width, rp.Height, rp.ColorPitch[i])
		if err != nil {
			return fail(fmt.Errorf("colour attachment %d: %w", i, err))
		}
		out.textures = append(out.textures, t)
		out.colorStaged = append(out.colorStaged, staged)
		out.color = append(out.color, mtl.RenderAttachment{
			Texture: t,
			Load:    metalLoadAction(rp.ColorLoad[i]),
			Store:   metalStoreAction(rp.ColorStore[i]),
			ClearColor: [4]float64{
				float64(rp.ColorClear[i][0]), float64(rp.ColorClear[i][1]),
				float64(rp.ColorClear[i][2]), float64(rp.ColorClear[i][3]),
			},
		})
	}
	if rp.Depth != nil {
		t, staged, err := e.attachmentTexture(*rp.Depth, depthFormat(rp),
			rp.Width, rp.Height, rp.DepthPitch)
		if err != nil {
			return fail(fmt.Errorf("depth attachment: %w", err))
		}
		out.textures = append(out.textures, t)
		out.depthStaged = staged
		out.depth = &mtl.RenderAttachment{
			Texture:    t,
			Load:       metalLoadAction(rp.DepthLoad),
			Store:      metalStoreAction(rp.DepthStore),
			ClearDepth: float64(rp.DepthClear),
			// A combined format takes the same texture on both attachments of
			// the pass descriptor. A pass that set only the depth one would run
			// with no stencil buffer, which is a stencil test that always
			// passes rather than an error.
			Stencil: rp.StencilPitch > 0,
		}
	}
	return out, nil
}

// depthFormat is the one place a pass's depth attachment format is decided,
// for the texture and for the pipeline alike.
//
// It was decided twice: the texture from StencilPitch, and the pipeline as
// Depth32Float unconditionally with no stencil format at all. A stencil pass
// therefore drew into a Depth32FloatStencil8 texture through a pipeline that
// declared neither that depth format nor any stencil format. Metal's
// validation layer aborts on the mismatch; without it the draw ran and the
// pictures happened to agree, which is why nothing noticed.
func depthFormat(rp *driver.RenderPass) driver.Format {
	if rp.StencilPitch > 0 {
		return driver.Depth32FloatStencil8
	}
	return driver.Depth32Float
}

// attachmentTexture aliases the operand's bytes, or stages them when Metal's
// linear-texture alignment rules out aliasing.
//
// The second result says which happened, because the difference is a frame's
// worth of copies and a caller paying for it should be able to find out.
func (e *executable) attachmentTexture(o driver.Operand, f driver.Format, w, h, pitch int) (
	*mtl.Texture, bool, error) {
	pf, err := metalPixelFormat(f)
	if err != nil {
		return nil, false, err
	}
	op, err := e.operand(o)
	if err != nil {
		return nil, false, err
	}
	// A depth or stencil attachment can never alias, and asking is not a
	// question with a safe answer: -minimumLinearTextureAlignmentForPixelFormat:
	// *asserts* on a depth format -- "Linear textures do not support
	// depth/stencil pixel formats" -- which aborts the process rather than
	// returning zero. Measured, by asking it once.
	align := 0
	if !isDepthFormat(f) {
		align = e.dev.dev.LinearTextureAlignment(pf)
	}
	if align > 0 && op.off%align == 0 && pitch%align == 0 {
		t, err := op.buf.NewLinearTexture(e.dev.dev, pf, w, h, op.off, pitch,
			mtl.TextureUsageRenderTarget|mtl.TextureUsageShaderRead)
		if err == nil {
			return t, false, nil
		}
		// Metal accepted the alignment and refused the texture anyway, which
		// is a device saying something this code does not model. Stage rather
		// than fail: the picture is the same either way and the count says the
		// fast path was not taken.
	}
	t, err := e.dev.dev.NewRenderTarget(pf, w, h)
	if err != nil {
		return nil, false, err
	}
	e.stagedAttachments.Add(1)
	return t, true, nil
}

// isDepthFormat reports whether a plan format carries a depth or stencil
// aspect, which is what rules out aliasing it onto a buffer.
func isDepthFormat(f driver.Format) bool {
	switch f {
	case driver.Depth32Float, driver.Depth24PlusStencil8, driver.Depth32FloatStencil8:
		return true
	}
	return false
}

// metalPixelFormat maps a plan's attachment format onto Metal's.
//
// It refuses what it cannot spell rather than defaulting, and that is the whole
// point of it existing: this path used to hardcode RGBA32Float for every colour
// attachment, so a caller who declared RGBA8Unorm got a pipeline that compiled,
// rendered, and read back sixteen bytes per pixel where they had asked for
// four. A silently wrong image is the failure specs/042-surface-completion.md
// section 5.2 found eight of on this surface.
//
// Every colour format the format table marks renderable is here as of
// 2026-08-30. The five float formats were not, and nothing noticed: the table
// marks them renderable on every backend, FormatInfo said so on this device,
// and the refusal arrived at submit. specs/062-backend-parity.md section 6.2's
// enumeration is what found it, which is the argument for enumerating a
// surface rather than testing the members somebody thought of.
//
// The one remaining absence is Depth24PlusStencil8, whose layout is
// device-defined and which the oracle refuses for that reason.
// The refusal names the format so the next caller to want one is told what is
// missing rather than left with a wrong picture.
func metalPixelFormat(f driver.Format) (int, error) {
	switch f {
	case driver.RGBA32Float:
		return mtl.PixelFormatRGBA32Float, nil
	case driver.RGBA8Unorm:
		return mtl.PixelFormatRGBA8Unorm, nil
	case driver.RGBA8UnormSRGB:
		return mtl.PixelFormatRGBA8UnormSRGB, nil
	case driver.BGRA8Unorm:
		return mtl.PixelFormatBGRA8Unorm, nil
	case driver.R16Float:
		return mtl.PixelFormatR16Float, nil
	case driver.RG16Float:
		return mtl.PixelFormatRG16Float, nil
	case driver.RGBA16Float:
		return mtl.PixelFormatRGBA16Float, nil
	case driver.R32Float:
		return mtl.PixelFormatR32Float, nil
	case driver.RG32Float:
		return mtl.PixelFormatRG32Float, nil
	case driver.Depth32Float:
		return mtl.PixelFormatDepth32Float, nil
	case driver.Depth32FloatStencil8:
		return mtl.PixelFormatDepth32FloatStencil8, nil
	}
	return 0, fmt.Errorf("accel: this Metal backend has no pixel format for %v; "+
		"specs/045-texture-attachments.md section 4 owns the mapping", f)
}

// metalWriteMask maps a plan's colour write mask onto Metal's.
//
// Written out by name, and this one is the reason the rule exists. accel
// numbers its channels red-first from bit 0, as Vulkan and D3D12 do;
// MTLColorWriteMask numbers them **alpha-first**, red at bit 3. A numeric
// conversion is therefore the mirror image of what the caller asked for, and it
// is right for exactly the masks that are symmetric about the middle -- which
// includes WriteAll, the default, and the only mask any test used before this
// one. A caller who masked red got alpha, on the GPU only, with no error.
func metalWriteMask(m uint8) int {
	out := mtl.ColorWriteMaskNone
	for _, c := range []struct {
		accel uint8
		metal int
	}{
		{1 << 0, mtl.ColorWriteMaskRed},
		{1 << 1, mtl.ColorWriteMaskGreen},
		{1 << 2, mtl.ColorWriteMaskBlue},
		{1 << 3, mtl.ColorWriteMaskAlpha},
	} {
		if m&c.accel != 0 {
			out |= c.metal
		}
	}
	return out
}

// metalStoreAction maps a plan's store action onto Metal's.
func metalStoreAction(s driver.StoreOp) int {
	if s == driver.StoreDiscard {
		return mtl.StoreActionDontCare
	}
	return mtl.StoreActionStore
}

// metalLoadAction maps a plan's load action onto Metal's.
//
// Written out by name rather than converted numerically, for the reason
// internal/cpu maps its blend factors that way: the two enumerations agree
// today, and a numeric conversion mistranslates silently the day either gains a
// value in the middle.
func metalLoadAction(l driver.LoadOp) int {
	switch l {
	case driver.LoadClear:
		return mtl.LoadActionClear
	case driver.LoadKeep:
		return mtl.LoadActionLoad
	}
	return mtl.LoadActionDontCare
}

// copyDepthPlanes moves a depth attachment between the caller's bytes and the
// staged texture, one aspect at a time when the format has two.
//
// Metal copies a combined depth/stencil texture with
// MTLBlitOptionDepthFromDepthStencil or MTLBlitOptionStencilFromDepthStencil
// and never both, and each copy is a tightly packed plane -- measured in
// internal/mtl's TestOnlyOneAspectAtATimeRoundTrips. That is the whole reason
// specs/045-texture-attachments.md section 12 stores the format as two planes:
// an interleaved layout has nowhere for these two copies to land.
//
// The stencil plane follows the depth plane, at the pitch the plan carries.
func (e *executable) copyDepthPlanes(p *pass, rp *driver.RenderPass, op resolved,
	tex *mtl.Texture, in bool) {
	if rp.StencilPitch == 0 {
		// One aspect, so no option and no second plane.
		if in {
			p.blit().CopyBufferToTexture(op.buf, op.off, tex, rp.DepthPitch)
		} else {
			p.blit().CopyTextureToBuffer(tex, op.buf, op.off, rp.DepthPitch)
		}
		return
	}
	stencilAt := op.off + rp.DepthPitch*rp.Height
	for _, a := range []struct {
		offset, pitch, bpp, option int
	}{
		{op.off, rp.DepthPitch, 4, mtl.BlitOptionDepthFromDepthStencil},
		{stencilAt, rp.StencilPitch, 1, mtl.BlitOptionStencilFromDepthStencil},
	} {
		if in {
			p.blit().CopyBufferToAspect(op.buf, a.offset, tex, a.pitch, a.bpp, a.option)
			continue
		}
		p.blit().CopyAspectToBuffer(tex, op.buf, a.offset, a.pitch, a.bpp, a.option)
	}
}

// drawTextures is one draw's stage textures, in each stage's own dense texture
// order.
//
// Indexed the same as driver.RenderDraw's slices, holes included: a stage that
// declares a texture it does not read leaves a gap that nothing is bound at,
// and preserving the gap is what keeps a later slot at the index the emitter
// gave it. Binding by position in a compacted list would put every texture
// after a hole one argument too low, which is a wrong picture rather than an
// error.
type drawTextures struct {
	vertex   []*mtl.Texture
	fragment []*mtl.Texture
}

// stageTextures materializes every texture the pass's draws fetch.
//
// One Metal texture per binding per draw, allocated and filled per pass. Two
// draws fetching the same view pay for it twice, which is renderTargets'
// arrangement and its reason: a cache keyed by extent and format would be an
// optimisation with no test behind it, and the lifetime would become this
// backend's rather than the graph's.
//
// The second return is every texture allocated, including those from a failed
// call, so the caller closes them whether this succeeds or not.
func (e *executable) stageTextures(p *pass, rp *driver.RenderPass) ([]drawTextures, []*mtl.Texture, error) {
	out := make([]drawTextures, len(rp.Draws))
	var owned []*mtl.Texture
	for i, d := range rp.Draws {
		for _, s := range []struct {
			in   []driver.RenderTexture
			out  *[]*mtl.Texture
			what string
		}{
			{d.VertexTextures, &out[i].vertex, "vertex"},
			{d.FragmentTextures, &out[i].fragment, "fragment"},
		} {
			*s.out = make([]*mtl.Texture, len(s.in))
			for j, rt := range s.in {
				// A stage's declared texture that no draw reads. build.go
				// leaves the slot zero rather than refusing it, because a
				// stage may declare a texture a branch never fetches.
				if rt.Width == 0 || rt.Height == 0 {
					continue
				}
				t, err := e.stageTexture(p, rt)
				if err != nil {
					return out, owned, fmt.Errorf("draw %d %s texture %d: %w", i, s.what, j, err)
				}
				owned = append(owned, t)
				(*s.out)[j] = t
			}
		}
	}
	return out, owned, nil
}

// stageTexture allocates one fetched texture and copies the caller's bytes in.
func (e *executable) stageTexture(p *pass, rt driver.RenderTexture) (*mtl.Texture, error) {
	pf, err := metalPixelFormat(rt.Format)
	if err != nil {
		return nil, err
	}
	t, err := e.dev.dev.NewSampledTexture(pf, rt.Width, rt.Height)
	if err != nil {
		return nil, err
	}
	op, err := e.operand(rt.Operand)
	if err != nil {
		t.Close()
		return nil, err
	}
	// The *source's* pitch, which the plan computed from the device's copy
	// alignment. The texture's rows are tight and the buffer's are padded, so
	// copying at the texture's pitch reads every row after the first from the
	// wrong offset -- the same trap the attachment write-back names, in the
	// other direction.
	p.blit().CopyBufferToTexture(op.buf, op.off, t, rt.Pitch)
	return t, nil
}

// encodeDraw encodes one draw.
func (e *executable) encodeDraw(enc *mtl.RenderEncoder, rp *driver.RenderPass, d driver.RenderDraw, tx drawTextures) error {
	pipe, err := e.renderPipeline(rp, d)
	if err != nil {
		return err
	}
	enc.SetPipeline(pipe)
	enc.SetCull(metalCull(d.Cull))
	enc.SetWinding(metalWinding(d.FrontFace))
	if rp.Depth != nil {
		ds, err := e.depthState(d)
		if err != nil {
			return err
		}
		enc.SetDepthState(ds)
		if d.Stencil.Enabled {
			enc.SetStencilReference(uint32(d.StencilReference))
		}
	}

	for i := range d.VertexBuffers {
		op, err := e.operand(d.VertexBuffers[i])
		if err != nil {
			return fmt.Errorf("vertex buffer %d: %w", i, err)
		}
		enc.SetVertexBuffer(op.buf, op.off, i)
	}
	if err := e.bindStageUniforms(enc, d); err != nil {
		return err
	}
	for i, t := range tx.vertex {
		if t != nil {
			enc.SetVertexTexture(t, mslabi.StageTextureIndex(i))
		}
	}
	for i, t := range tx.fragment {
		if t != nil {
			enc.SetFragmentTexture(t, mslabi.StageTextureIndex(i))
		}
	}

	prim := metalPrimitive(d.Topology)
	if d.Indexed {
		idx, err := e.operand(d.Index)
		if err != nil {
			return fmt.Errorf("index buffer: %w", err)
		}
		indexType := 0 // MTLIndexTypeUInt16
		if d.IndexWidth == 4 {
			indexType = 1
		}
		enc.DrawIndexed(prim, d.VertexCount, indexType, idx.buf,
			idx.off+d.FirstVertex*d.IndexWidth, d.InstanceCount, d.BaseVertex,
			d.FirstInstance)
		return nil
	}
	enc.Draw(prim, d.FirstVertex, d.VertexCount, d.InstanceCount, d.FirstInstance)
	return nil
}

func metalPrimitive(topology uint8) int {
	// MTLPrimitiveTypeTriangle is 3 and TriangleStrip is 4;
	// specs/033-render-api.md admits only those two.
	if topology == 1 {
		return 4
	}
	return 3
}

func metalCull(c uint8) int {
	// MTLCullMode: none 0, front 1, back 2. accel's CullMode is the same order.
	return int(c)
}

func metalWinding(f uint8) int {
	// MTLWinding: clockwise 0, counterclockwise 1. accel's FrontFace has
	// CounterClockwise first, so the two are inverted.
	if f == 0 {
		return 1
	}
	return 0
}

// bindStageUniforms encodes a draw's by-value parameters as buffer bytes.
//
// std140 rather than Metal's own layout, because the generated codec is what
// produced the bytes and the MSL emitter spells the struct with std140's
// padding. One layout, written once, and this is a caller of it rather than a
// second implementation.
//
// The indices come from internal/mslabi, which is also where the emitter gets
// them. A vertex stage's uniforms sit above every vertex buffer because the two
// share one Metal index space, and getting that wrong is not a compile error on
// either side: the uniform lands on vertex buffer zero and the stage reads
// geometry as a transform.
func (e *executable) bindStageUniforms(enc *mtl.RenderEncoder, d driver.RenderDraw) error {
	// A buffer-bound parameter is a buffer binding at an offset, which is what
	// specs/033-render-api.md section 4.1 says this mechanism has a native
	// expression as on every target. It is bound *first* so a parameter that
	// also has a pass-state value is overwritten by neither -- the by-value
	// loop below skips an index the buffer covered.
	bound := map[int]bool{}
	for i, o := range d.VertexUniformBuffers {
		if o.Kind() == driver.OperandUnset {
			continue
		}
		op, err := e.operand(o)
		if err != nil {
			return fmt.Errorf("vertex uniform buffer %d: %w", i, err)
		}
		enc.SetVertexBuffer(op.buf, op.off, mslabi.StageUniformIndex(i))
		bound[i] = true
	}
	for i, v := range d.VertexUniforms {
		if bound[i] || v == nil {
			continue
		}
		b, err := e.uniformBytes(d.Vertex, i, v)
		if err != nil {
			return fmt.Errorf("vertex uniform %d: %w", i, err)
		}
		enc.SetVertexBytes(b, mslabi.StageUniformIndex(i))
	}
	clear(bound)
	for i, o := range d.FragmentUniformBuffers {
		if o.Kind() == driver.OperandUnset {
			continue
		}
		op, err := e.operand(o)
		if err != nil {
			return fmt.Errorf("fragment uniform buffer %d: %w", i, err)
		}
		enc.SetFragmentBuffer(op.buf, op.off, mslabi.StageFragmentUniformIndex(i))
		bound[i] = true
	}
	for i, v := range d.FragmentUniforms {
		if bound[i] || v == nil {
			continue
		}
		b, err := e.uniformBytes(d.Fragment, i, v)
		if err != nil {
			return fmt.Errorf("fragment uniform %d: %w", i, err)
		}
		enc.SetFragmentBytes(b, mslabi.StageFragmentUniformIndex(i))
	}
	return nil
}

// uniformBytes encodes one by-value parameter through the generated codec.
func (e *executable) uniformBytes(s *kernel.Stage, i int, v any) ([]byte, error) {
	if s == nil || i >= len(s.Uniforms) {
		return nil, fmt.Errorf("the stage declares no parameter at index %d", i)
	}
	u := s.Uniforms[i]
	if u.Encode == nil {
		return nil, fmt.Errorf("parameter %q carries no encoder", u.Name)
	}
	b := make([]byte, u.Size)
	if err := u.Encode(b, v); err != nil {
		return nil, err
	}
	return b, nil
}

// renderPipeline compiles a draw's stage pair into a Metal pipeline state, or
// returns the one already compiled for it.
//
// Cached on the executable rather than on the device, because the cache key is
// the whole pipeline: the two stages plus the attachment formats, blend and
// vertex layout. Those are fixed for a plan, so one entry per draw is the
// smallest key that is correct, and a plan replayed compiles nothing.
func (e *executable) renderPipeline(rp *driver.RenderPass, d driver.RenderDraw) (*mtl.RenderPipeline, error) {
	key := renderKey(rp, d)
	if p, ok := e.pipelines[key]; ok {
		return p, nil
	}
	if e.pipelines == nil {
		e.pipelines = map[string]*mtl.RenderPipeline{}
	}

	vs, err := e.stageFunction(d.Vertex)
	if err != nil {
		return nil, err
	}
	fs, err := e.stageFunction(d.Fragment)
	if err != nil {
		return nil, err
	}

	spec := mtl.RenderPipelineSpec{Vertex: vs, Fragment: fs}
	// The attachment's format, not a fixed one. Metal validates a pipeline's
	// colour formats against the pass's attachments at draw time, so a
	// hardcoded value here and a real format there is a pipeline the device
	// rejects -- and until attachments carried a format there was nothing else
	// to write.
	for i := range rp.Color {
		pf, err := metalPixelFormat(rp.ColorFormat[i])
		if err != nil {
			return nil, fmt.Errorf("colour attachment %d: %w", i, err)
		}
		spec.ColorFormats = append(spec.ColorFormats, pf)
	}
	for _, m := range d.Masks {
		spec.WriteMasks = append(spec.WriteMasks, metalWriteMask(m))
	}
	for _, b := range d.Blends {
		spec.Blends = append(spec.Blends, metalBlend(b))
	}
	if rp.Depth != nil {
		pf, err := metalPixelFormat(depthFormat(rp))
		if err != nil {
			return nil, fmt.Errorf("depth attachment: %w", err)
		}
		spec.DepthFormat = pf
		if rp.StencilPitch > 0 {
			spec.StencilFormat = pf
		}
	}
	for _, l := range d.VertexLayouts {
		vl := mtl.VertexLayoutSpec{Stride: l.Stride, PerInstance: l.PerInstance}
		for _, a := range l.Attributes {
			vl.Attributes = append(vl.Attributes, mtl.VertexAttributeSpec{
				Location: a.Location, Offset: a.Offset,
				Format: metalVertexFormat(a),
			})
		}
		spec.VertexLayouts = append(spec.VertexLayouts, vl)
	}

	p, err := e.dev.dev.NewRenderPipeline(spec)
	if err != nil {
		return nil, fmt.Errorf("compiling %s and %s: %w", d.Vertex.Name, d.Fragment.Name, err)
	}
	e.pipelines[key] = p
	return p, nil
}

// stageFunction compiles one stage's MSL, once per executable.
func (e *executable) stageFunction(s *kernel.Stage) (*mtl.Function, error) {
	if s.MSL == "" {
		// Not a fallback. specs/032-stage-abi.md section 12.1: a stage outside
		// the MSL subset has no Metal lowering, and running something else
		// would make the two backends disagree about what a program means.
		return nil, fmt.Errorf("stage %s carries no MSL, so this backend cannot run it",
			s.Name)
	}
	if f, ok := e.functions[s.Name]; ok {
		return f, nil
	}
	if e.functions == nil {
		e.functions = map[string]*mtl.Function{}
	}
	f, err := e.dev.dev.CompileFunction(s.MSL, s.Name)
	if err != nil {
		return nil, fmt.Errorf("compiling stage %s: %w", s.Name, err)
	}
	e.functions[s.Name] = f
	return f, nil
}

// depthState compiles a draw's depth rule, once per distinct rule.
// metalStencilFace maps one face onto Metal's own numbering, by name.
//
// MTLStencilOperation is Keep, Zero, Replace, IncrementClamp, DecrementClamp,
// Invert, IncrementWrap, DecrementWrap -- which happens to be the plan's order
// today. It is written out anyway, for the reason the colour write mask taught
// this file: two enumerations agreeing now is not a property either maintains.
func metalStencilFace(f driver.StencilFace) mtl.StencilFace {
	return mtl.StencilFace{
		Compare:   metalCompare(f.Compare),
		ReadMask:  uint32(f.ReadMask),
		WriteMask: uint32(f.WriteMask),
		Fail:      metalStencilOp(f.Fail),
		DepthFail: metalStencilOp(f.DepthFail),
		Pass:      metalStencilOp(f.Pass),
	}
}

func metalStencilOp(op driver.StencilOp) int {
	switch op {
	case driver.StencilZero:
		return 1
	case driver.StencilReplace:
		return 2
	case driver.StencilIncrementClamp:
		return 3
	case driver.StencilDecrementClamp:
		return 4
	case driver.StencilInvert:
		return 5
	case driver.StencilIncrementWrap:
		return 6
	case driver.StencilDecrementWrap:
		return 7
	}
	return 0 // MTLStencilOperationKeep
}

func (e *executable) depthState(d driver.RenderDraw) (*mtl.DepthState, error) {
	compare := metalCompare(d.DepthCompare)
	if !d.DepthTest {
		// Not testing is testing with Always, which is what Metal's default is
		// and what the CPU backend does. Spelling it rather than leaving the
		// state unset keeps the two backends agreeing when a pass has a depth
		// attachment and a draw does not use it.
		compare = 7 // MTLCompareFunctionAlways
	}
	var spec *mtl.StencilSpec
	if d.Stencil.Enabled {
		spec = &mtl.StencilSpec{
			Front: metalStencilFace(d.Stencil.Front),
			Back:  metalStencilFace(d.Stencil.Back),
		}
	}
	key := fmt.Sprintf("%d/%v/%+v", compare, d.DepthWrite, spec)
	if s, ok := e.depthStates[key]; ok {
		return s, nil
	}
	if e.depthStates == nil {
		e.depthStates = map[string]*mtl.DepthState{}
	}
	s, err := e.dev.dev.NewDepthStencilState(compare, d.DepthWrite, spec)
	if err != nil {
		return nil, err
	}
	e.depthStates[key] = s
	return s, nil
}

// renderKey identifies a pipeline state within one plan.
//
// The attachment formats are part of it, and were not when every attachment was
// RGBA32Float. Metal validates a pipeline's colour formats against the pass's
// attachments at draw time, so two passes differing only in format would share
// a cached pipeline and the second would be rejected by the device -- a failure
// that appears only when a plan happens to hold both, which is the kind a cache
// key omission produces and a test rarely finds.
func renderKey(rp *driver.RenderPass, d driver.RenderDraw) string {
	return fmt.Sprintf("%s|%s|%d|%t|%v|%v|%v|%v|%v", d.Vertex.Name, d.Fragment.Name,
		len(rp.Color), rp.Depth != nil, d.Masks, d.Blends, d.VertexLayouts,
		rp.ColorFormat, rp.DepthFormat)
}

// metalVertexFormat maps a component count onto MTLVertexFormat.
//
// float is 28 and the vector widths follow it, which is the one place this
// backend relies on an enumeration being contiguous -- stated here so a reader
// can check it against the header rather than infer it from arithmetic.
// metalVertexFormat maps a plan's attribute shape onto MTLVertexFormat.
//
// Written out by name for the normalized forms rather than computed. The float
// formats are contiguous from MTLVertexFormatFloat and stay arithmetic; the
// normalized ones are not -- MTLVertexFormat interleaves the plain and
// normalized integer families and the two- and three- and four-wide members of
// each, so any expression over them is a coincidence waiting to stop holding.
// The colour write mask is what this file learned that from.
func metalVertexFormat(a driver.VertexAttribute) int {
	if !a.Normalized {
		if a.Components < 1 || a.Components > 4 {
			return 0
		}
		// MTLVertexFormatFloat is 28 and Float2..4 follow it.
		return 27 + a.Components
	}
	switch {
	case a.Bytes == 1 && !a.Signed && a.Components == 2:
		return 7 // UChar2Normalized
	case a.Bytes == 1 && !a.Signed && a.Components == 4:
		return 9 // UChar4Normalized
	case a.Bytes == 1 && a.Signed && a.Components == 2:
		return 10 // Char2Normalized
	case a.Bytes == 1 && a.Signed && a.Components == 4:
		return 12 // Char4Normalized
	case a.Bytes == 2 && !a.Signed && a.Components == 2:
		return 19 // UShort2Normalized
	case a.Bytes == 2 && !a.Signed && a.Components == 4:
		return 21 // UShort4Normalized
	case a.Bytes == 2 && a.Signed && a.Components == 2:
		return 22 // Short2Normalized
	case a.Bytes == 2 && a.Signed && a.Components == 4:
		return 24 // Short4Normalized
	}
	return 0
}

// metalCompare maps a compare function onto MTLCompareFunction.
//
// The two enumerations agree in order, and this is written out anyway for the
// reason the blend factors are: agreement today is not agreement after either
// list gains a value in the middle.
func metalCompare(c uint8) int {
	switch c {
	case 0:
		return 0 // Never
	case 1:
		return 1 // Less
	case 2:
		return 2 // Equal
	case 3:
		return 3 // LessEqual
	case 4:
		return 4 // Greater
	case 5:
		return 5 // NotEqual
	case 6:
		return 6 // GreaterEqual
	}
	return 7 // Always
}

// metalBlend maps a plan's blend state onto Metal's.
func metalBlend(b driver.Blend) mtl.BlendSpec {
	return mtl.BlendSpec{
		Enabled: b.Enabled,
		SrcRGB:  metalFactor(b.SrcColor), DstRGB: metalFactor(b.DstColor),
		OpRGB: metalBlendOp(b.ColorOp),
		SrcA:  metalFactor(b.SrcAlpha), DstA: metalFactor(b.DstAlpha),
		OpA: metalBlendOp(b.AlphaOp),
	}
}

func metalFactor(f driver.BlendFactor) int {
	switch f {
	case driver.FactorZero:
		return 0
	case driver.FactorOne:
		return 1
	case driver.FactorSrcColor:
		return 2
	case driver.FactorOneMinusSrcColor:
		return 3
	case driver.FactorSrcAlpha:
		return 4
	case driver.FactorOneMinusSrcAlpha:
		return 5
	case driver.FactorDstColor:
		return 6
	case driver.FactorOneMinusDstColor:
		return 7
	case driver.FactorDstAlpha:
		return 8
	case driver.FactorOneMinusDstAlpha:
		return 9
	}
	return 0
}

func metalBlendOp(o driver.BlendOp) int {
	switch o {
	case driver.BlendAdd:
		return 0
	case driver.BlendSubtract:
		return 1
	case driver.BlendReverseSubtract:
		return 2
	case driver.BlendMin:
		return 3
	case driver.BlendMax:
		return 4
	}
	return 0
}

// StagedAttachments reports how many attachments this executable copied through
// a private texture rather than rendering into the caller's bytes.
//
// specs/045-texture-attachments.md section 11. Zero is the expected answer for
// a colour-only pass; a depth attachment is always one, because Metal's linear
// textures do not support depth or stencil formats.
func (e *executable) StagedAttachments() int64 { return e.stagedAttachments.Load() }
