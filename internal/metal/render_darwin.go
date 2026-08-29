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
// A pass renders into private textures and blits them into the caller's
// buffers, because MTLRenderPassDescriptor takes textures and
// specs/033-render-api.md makes an attachment a buffer view. Everything about
// the texture is inside this file: the plan does not mention one and neither
// does a caller.
//
// # What LoadKeep costs here
//
// Keeping prior contents means the texture starts as what the buffer already
// holds, which is a blit *in* before the pass as well as one out after it.
// Clear and DontCare skip it. That is the one place the load action costs
// something on this backend and it is the honest cost of a buffer-shaped
// attachment: a texture-shaped one would keep its own contents for free.

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

	targets, err := e.renderTargets(rp)
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

	// LoadKeep is the only action that needs the buffer's current contents in
	// the texture before the pass runs.
	for i, a := range targets.color {
		if rp.ColorLoad[i] != driver.LoadKeep {
			continue
		}
		op, err := e.operand(rp.Color[i])
		if err != nil {
			return fmt.Errorf("accel: node %d colour attachment %d: %w", n.ID, i, err)
		}
		p.blit().CopyBufferToTexture(op.buf, op.off, a.Texture, rp.ColorPitch[i])
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

	// And back into the buffers the graph ordered other nodes against. Writing
	// anywhere else would make every inferred edge describe memory nobody
	// wrote.
	// A discarded attachment is not written back. Its contents are undefined
	// after the pass, so the blit would be copying bytes nobody may read -- and
	// on a depth attachment that is the whole buffer every frame, which is the
	// saving specs/033-render-api.md names.
	blit := p.blit()
	for i, a := range targets.color {
		if rp.ColorStore[i] == driver.StoreDiscard {
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
		blit.CopyTextureToBuffer(a.Texture, op.buf, op.off, rp.ColorPitch[i])
	}
	if targets.depth != nil && rp.DepthStore != driver.StoreDiscard {
		op, err := e.operand(*rp.Depth)
		if err != nil {
			return fmt.Errorf("accel: node %d depth attachment: %w", n.ID, err)
		}
		blit.CopyTextureToBuffer(targets.depth.Texture, op.buf, op.off, rp.DepthPitch)
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
}

// renderTargets allocates one texture per attachment.
//
// Per pass rather than cached, which is a cost this backend pays until 033's
// texture attachments arrive: a cache keyed by extent and format would be an
// optimisation with no test behind it, and the lifetime rule would then be
// this backend's rather than the graph's.
func (e *executable) renderTargets(rp *driver.RenderPass) (renderAttachments, error) {
	var out renderAttachments
	for i := range rp.Color {
		pf, err := metalPixelFormat(rp.ColorFormat[i])
		if err != nil {
			for _, done := range out.textures {
				done.Close()
			}
			return renderAttachments{}, fmt.Errorf("colour attachment %d: %w", i, err)
		}
		t, err := e.dev.dev.NewRenderTarget(pf, rp.Width, rp.Height)
		if err != nil {
			for _, done := range out.textures {
				done.Close()
			}
			return renderAttachments{}, fmt.Errorf("colour attachment %d: %w", i, err)
		}
		out.textures = append(out.textures, t)
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
		t, err := e.dev.dev.NewRenderTarget(mtl.PixelFormatDepth32Float, rp.Width, rp.Height)
		if err != nil {
			for _, done := range out.textures {
				done.Close()
			}
			return renderAttachments{}, fmt.Errorf("depth attachment: %w", err)
		}
		out.textures = append(out.textures, t)
		out.depth = &mtl.RenderAttachment{
			Texture:    t,
			Load:       metalLoadAction(rp.DepthLoad),
			Store:      metalStoreAction(rp.DepthStore),
			ClearDepth: float64(rp.DepthClear),
		}
	}
	return out, nil
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
// The formats absent here are absent because Metal's constants for them are not
// in internal/mtl yet, not because Metal lacks them. Adding one is a constant
// and a bytesPerPixel row; the refusal names the format so the next caller to
// want one is told what is missing rather than left with a wrong picture.
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
	case driver.Depth32Float:
		return mtl.PixelFormatDepth32Float, nil
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
	// Stencil is not lowered here yet. Refused rather than dropped, per
	// specs/006-backends.md decision 6: a draw whose stencil state was ignored
	// would produce a picture, and the picture would be the one the caller gets
	// when the technique they are building does nothing.
	if d.Stencil.Enabled {
		return fmt.Errorf("accel: this backend does not lower stencil state at " +
			"specs/033-render-api.md section 2.1; the CPU backend does, and the " +
			"comparison resumes when it lands here")
	}
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
	for i, v := range d.VertexUniforms {
		b, err := e.uniformBytes(d.Vertex, i, v)
		if err != nil {
			return fmt.Errorf("vertex uniform %d: %w", i, err)
		}
		enc.SetVertexBytes(b, mslabi.StageUniformIndex(i))
	}
	for i, v := range d.FragmentUniforms {
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
		spec.DepthFormat = mtl.PixelFormatDepth32Float
	}
	for _, l := range d.VertexLayouts {
		vl := mtl.VertexLayoutSpec{Stride: l.Stride, PerInstance: l.PerInstance}
		for _, a := range l.Attributes {
			vl.Attributes = append(vl.Attributes, mtl.VertexAttributeSpec{
				Location: a.Location, Offset: a.Offset,
				Format: metalVertexFormat(a.Components),
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
func (e *executable) depthState(d driver.RenderDraw) (*mtl.DepthState, error) {
	compare := metalCompare(d.DepthCompare)
	if !d.DepthTest {
		// Not testing is testing with Always, which is what Metal's default is
		// and what the CPU backend does. Spelling it rather than leaving the
		// state unset keeps the two backends agreeing when a pass has a depth
		// attachment and a draw does not use it.
		compare = 7 // MTLCompareFunctionAlways
	}
	key := fmt.Sprintf("%d/%v", compare, d.DepthWrite)
	if s, ok := e.depthStates[key]; ok {
		return s, nil
	}
	if e.depthStates == nil {
		e.depthStates = map[string]*mtl.DepthState{}
	}
	s, err := e.dev.dev.NewDepthState(compare, d.DepthWrite)
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
func metalVertexFormat(components int) int {
	if components < 1 || components > 4 {
		return 0
	}
	return 27 + components
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
