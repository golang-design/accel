// Copyright 2026 The golang.design Initiative Authors.
// All rights reserved. Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

//go:build darwin

package mtl

import (
	"fmt"
	"sync"
	"unsafe"

	"github.com/ebitengine/purego/objc"
)

// The Metal render path: textures, render pipeline states, and the render
// command encoder.
//
// # Why a texture and not the caller's buffer
//
// MTLRenderPassDescriptor takes MTLTexture attachments and nothing else, while
// specs/033-render-api.md makes an attachment a buffer view. So a pass renders
// into a private texture and blits the result back. The texture is entirely
// inside this backend: 033 says the shape a caller writes does not change when
// the texture path lands, and this is that path arriving early and hidden.
//
// # The trap this file is built around
//
// -newRenderPipelineStateWithDescriptor:error: **aborts the process** on an
// invalid descriptor rather than returning nil with an error: Metal's
// validation layer calls assert. A missing vertex function is not a bad error
// message, it is the caller's process gone. So every field the validator checks
// is checked here first, and the error says which one.

const (
	// PixelFormatRGBA8Unorm and its sRGB pair. The two differ only in whether
	// the hardware converts on write and on read, which is what makes them a
	// view's choice rather than a texture's -- see
	// specs/045-texture-attachments.md section 2.1.
	PixelFormatRGBA8Unorm     = 70
	PixelFormatRGBA8UnormSRGB = 71

	// PixelFormatRGBA32Float was the only colour format this path rendered, and
	// it is the one specs/035-cpu-rasterizer.md's reference rasterizer writes:
	// four float32 per pixel, so the blit back into a caller's F32 buffer is a
	// copy rather than a conversion.
	PixelFormatRGBA32Float = 125

	// PixelFormatDepth32Float, for the same reason.
	PixelFormatDepth32Float = 252

	textureUsageShaderRead   = 1
	textureUsageRenderTarget = 4
	storageModePrivate       = 2

	// The load and store actions, exported because a backend maps its own onto
	// them and the mapping is written out by name rather than by number.
	LoadActionDontCare = 0
	LoadActionLoad     = 1
	LoadActionClear    = 2

	StoreActionDontCare = 0
	StoreActionStore    = 1

	// MTLColorWriteMask, exported because the mapping onto it belongs beside
	// the other named mappings in internal/metal rather than as a cast.
	//
	// **Metal numbers its channels from alpha.** Red is bit 3 and alpha is bit
	// 0, which is the reverse of the natural RGBA order that accel, Vulkan and
	// D3D12 all use. Writing the constants out is the only way that fact is
	// visible anywhere; a numeric conversion is correct for `All` and for any
	// mask symmetric about the middle, and wrong for every other.
	ColorWriteMaskNone  = 0
	ColorWriteMaskAlpha = 1 << 0
	ColorWriteMaskBlue  = 1 << 1
	ColorWriteMaskGreen = 1 << 2
	ColorWriteMaskRed   = 1 << 3
	ColorWriteMaskAll   = 0xf
)

// The Metal classes, looked up on first use rather than at package
// initialization.
//
// A package-level objc.GetClass runs before Devices() has dlopened the Metal
// framework, so every one of them is zero and the first Send is a message to
// nil -- which Objective-C answers with zero rather than a crash, so the
// symptom is "no render pipeline descriptor" from a call that never reached
// Metal at all.
var (
	clsOnce                 sync.Once
	clsTextureDescriptor    objc.Class
	clsRenderPassDescriptor objc.Class
	clsRenderPipelineDesc   objc.Class
	clsVertexDescriptor     objc.Class
	clsDepthStencilDesc     objc.Class
)

func classes() {
	clsOnce.Do(func() {
		clsTextureDescriptor = objc.GetClass("MTLTextureDescriptor")
		clsRenderPassDescriptor = objc.GetClass("MTLRenderPassDescriptor")
		clsRenderPipelineDesc = objc.GetClass("MTLRenderPipelineDescriptor")
		clsVertexDescriptor = objc.GetClass("MTLVertexDescriptor")
		clsDepthStencilDesc = objc.GetClass("MTLDepthStencilDescriptor")
	})
}

var (
	selTexture2D      = objc.RegisterName("texture2DDescriptorWithPixelFormat:width:height:mipmapped:")
	selSetTexUsage    = objc.RegisterName("setUsage:")
	selSetTexStorage  = objc.RegisterName("setStorageMode:")
	selNewTexture     = objc.RegisterName("newTextureWithDescriptor:")
	selRenderPassDesc = objc.RegisterName("renderPassDescriptor")

	selColorAttachments = objc.RegisterName("colorAttachments")
	selDepthAttachment  = objc.RegisterName("depthAttachment")
	selObjectAtIndexed  = objc.RegisterName("objectAtIndexedSubscript:")
	selSetTexture       = objc.RegisterName("setTexture:")
	selSetLoadAction    = objc.RegisterName("setLoadAction:")
	selSetStoreAction   = objc.RegisterName("setStoreAction:")
	selSetClearColor    = objc.RegisterName("setClearColor:")
	selSetClearDepth    = objc.RegisterName("setClearDepth:")

	selSetVertexFunction   = objc.RegisterName("setVertexFunction:")
	selSetFragmentFunction = objc.RegisterName("setFragmentFunction:")
	selSetPixelFormat      = objc.RegisterName("setPixelFormat:")
	selSetDepthPixelFormat = objc.RegisterName("setDepthAttachmentPixelFormat:")
	selSetVertexDescriptor = objc.RegisterName("setVertexDescriptor:")
	selNewRenderPipeline   = objc.RegisterName("newRenderPipelineStateWithDescriptor:error:")

	selSetBlendingEnabled  = objc.RegisterName("setBlendingEnabled:")
	selSetSourceRGB        = objc.RegisterName("setSourceRGBBlendFactor:")
	selSetDestRGB          = objc.RegisterName("setDestinationRGBBlendFactor:")
	selSetRGBBlendOp       = objc.RegisterName("setRgbBlendOperation:")
	selSetSourceAlpha      = objc.RegisterName("setSourceAlphaBlendFactor:")
	selSetDestAlpha        = objc.RegisterName("setDestinationAlphaBlendFactor:")
	selSetAlphaBlendOp     = objc.RegisterName("setAlphaBlendOperation:")
	selSetWriteMask        = objc.RegisterName("setWriteMask:")
	selVertexAttributes    = objc.RegisterName("attributes")
	selVertexLayouts       = objc.RegisterName("layouts")
	selSetFormat           = objc.RegisterName("setFormat:")
	selSetOffset           = objc.RegisterName("setOffset:")
	selSetBufferIndex      = objc.RegisterName("setBufferIndex:")
	selSetStride           = objc.RegisterName("setStride:")
	selSetStepFunction     = objc.RegisterName("setStepFunction:")
	selSetDepthCompare     = objc.RegisterName("setDepthCompareFunction:")
	selSetDepthWrite       = objc.RegisterName("setDepthWriteEnabled:")
	selNewDepthStencil     = objc.RegisterName("newDepthStencilStateWithDescriptor:")
	selRenderEncoderWith   = objc.RegisterName("renderCommandEncoderWithDescriptor:")
	selSetRenderPipeline   = objc.RegisterName("setRenderPipelineState:")
	selSetDepthStencil     = objc.RegisterName("setDepthStencilState:")
	selSetVertexBuffer     = objc.RegisterName("setVertexBuffer:offset:atIndex:")
	selSetFragmentBuffer   = objc.RegisterName("setFragmentBuffer:offset:atIndex:")
	selSetVertexTexture    = objc.RegisterName("setVertexTexture:atIndex:")
	selSetFragmentTexture  = objc.RegisterName("setFragmentTexture:atIndex:")
	selSetCullMode         = objc.RegisterName("setCullMode:")
	selSetWinding          = objc.RegisterName("setFrontFacingWinding:")
	selDrawPrimitives      = objc.RegisterName("drawPrimitives:vertexStart:vertexCount:instanceCount:baseInstance:")
	selDrawIndexed         = objc.RegisterName("drawIndexedPrimitives:indexCount:indexType:indexBuffer:indexBufferOffset:instanceCount:baseVertex:baseInstance:")
	selSetVertexBytes      = objc.RegisterName("setVertexBytes:length:atIndex:")
	selSetFragmentBytes    = objc.RegisterName("setFragmentBytes:length:atIndex:")
	selCopyBufferToTexture = objc.RegisterName("copyFromBuffer:sourceOffset:sourceBytesPerRow:sourceBytesPerImage:sourceSize:toTexture:destinationSlice:destinationLevel:destinationOrigin:")
	selCopyTextureToBuffer = objc.RegisterName("copyFromTexture:sourceSlice:sourceLevel:sourceOrigin:sourceSize:toBuffer:destinationOffset:destinationBytesPerRow:destinationBytesPerImage:")
)

// bytesPerPixel is a render target format's size.
//
// A table rather than "sixteen unless it is depth", which is what this was: a
// BGRA8 target then took a stride four times too large, and the blit that reads
// it back writes past the end of the destination. Every format this package
// creates is listed, and an unlisted one is refused rather than guessed.
func bytesPerPixel(format int) int {
	switch format {
	case PixelFormatRGBA32Float:
		return 16
	case PixelFormatDepth32Float:
		return 4
	case PixelFormatBGRA8Unorm:
		return 4
	case PixelFormatRGBA8Unorm, PixelFormatRGBA8UnormSRGB:
		return 4
	}
	return 0
}

// Texture is a render target.
type Texture struct {
	id            objc.ID
	width, height int
	bpp           int
}

// NewRenderTarget allocates a private texture to render into.
//
// Private storage because nothing on the host reads it: the result reaches a
// caller through a blit into their own buffer, which is where the bytes have to
// end up anyway.
func (d *Device) NewRenderTarget(format, w, h int) (*Texture, error) {
	return d.newTexture2D(format, w, h, textureUsageRenderTarget, "render target")
}

// NewSampledTexture allocates a private texture a stage fetches from.
//
// A separate constructor rather than a usage argument on NewRenderTarget,
// because the two usages are not interchangeable and neither is a superset of
// the other in intent: an attachment Metal may never read from a shader, and a
// fetched texture Metal may never render into. Declaring both on every texture
// would compile and would tell the driver less than it is told now.
//
// Private storage for the same reason a render target has it. The bytes arrive
// by blit from the caller's buffer, which is where specs/033-render-api.md
// keeps them.
func (d *Device) NewSampledTexture(format, w, h int) (*Texture, error) {
	return d.newTexture2D(format, w, h, textureUsageShaderRead, "sampled texture")
}

// newTexture2D allocates one private 2D texture with the usage named.
func (d *Device) newTexture2D(format, w, h, usage int, what string) (*Texture, error) {
	classes()
	if w <= 0 || h <= 0 {
		return nil, fmt.Errorf("accel/mtl: a %dx%d %s", w, h, what)
	}
	bpp := bytesPerPixel(format)
	if bpp == 0 {
		return nil, fmt.Errorf("accel/mtl: pixel format %d has no known size, and a blit "+
			"needs one: a stride computed from a guess reads or writes the wrong bytes",
			format)
	}
	t := &Texture{width: w, height: h, bpp: bpp}
	withPool(func() {
		desc := objc.ID(clsTextureDescriptor).Send(selTexture2D,
			uintptr(format), uintptr(w), uintptr(h), uintptr(0))
		if desc == 0 {
			return
		}
		desc.Send(selSetTexUsage, uintptr(usage))
		desc.Send(selSetTexStorage, uintptr(storageModePrivate))
		t.id = d.id.Send(selNewTexture, desc)
	})
	if t.id == 0 {
		return nil, fmt.Errorf("accel/mtl: the device refused a %dx%d %s in "+
			"pixel format %d", w, h, what, format)
	}
	return t, nil
}

// Close releases the texture.
func (t *Texture) Close() {
	withPool(func() { release(t.id) })
	t.id = 0
}

// RenderPipelineSpec is everything a render pipeline state is compiled from.
//
// A struct rather than a builder because Metal validates the whole descriptor
// at once and aborts if any of it is wrong. Collecting the fields first means
// this package can check them all before the abort is reachable.
type RenderPipelineSpec struct {
	Vertex, Fragment *Function

	// ColorFormats is one per attachment, and Blends and WriteMasks are the
	// same length or empty.
	ColorFormats []int
	Blends       []BlendSpec
	WriteMasks   []int

	// DepthFormat is 0 for a pipeline with no depth attachment.
	DepthFormat int

	// VertexLayouts describes the buffers the vertex stage fetches from.
	VertexLayouts []VertexLayoutSpec
}

// BlendSpec is one attachment's blend state, in Metal's own enumeration.
type BlendSpec struct {
	Enabled               bool
	SrcRGB, DstRGB, OpRGB int
	SrcA, DstA, OpA       int
}

// VertexLayoutSpec is one bound vertex buffer.
type VertexLayoutSpec struct {
	Stride      int
	PerInstance bool
	Attributes  []VertexAttributeSpec
}

// VertexAttributeSpec is one attribute inside a bound buffer.
type VertexAttributeSpec struct {
	Location int
	Offset   int

	// Format is Metal's MTLVertexFormat: 28..31 are float, float2, float3,
	// float4.
	Format int
}

// RenderPipeline is a compiled render pipeline state.
type RenderPipeline struct{ id objc.ID }

// Close releases the pipeline state.
func (p *RenderPipeline) Close() {
	withPool(func() { release(p.id) })
	p.id = 0
}

// NewRenderPipeline compiles a render pipeline state.
//
// Every check here exists because its absence is an abort rather than an error:
// Metal's descriptor validation calls assert, so a nil vertex function takes
// the caller's process down with a message on stderr and no Go error anywhere.
func (d *Device) NewRenderPipeline(s RenderPipelineSpec) (*RenderPipeline, error) {
	classes()
	if s.Vertex == nil || s.Vertex.fn == 0 {
		return nil, fmt.Errorf("accel/mtl: a render pipeline with no vertex function; " +
			"Metal aborts the process on this rather than reporting it")
	}
	if s.Fragment == nil || s.Fragment.fn == 0 {
		return nil, fmt.Errorf("accel/mtl: a render pipeline with no fragment function; " +
			"Metal aborts the process on this rather than reporting it")
	}
	if len(s.ColorFormats) == 0 {
		return nil, fmt.Errorf("accel/mtl: a render pipeline with no colour attachments")
	}
	for i, f := range s.ColorFormats {
		if f == 0 {
			return nil, fmt.Errorf("accel/mtl: colour attachment %d has no pixel format", i)
		}
	}
	for i, l := range s.VertexLayouts {
		if l.Stride <= 0 {
			return nil, fmt.Errorf("accel/mtl: vertex buffer %d has a stride of %d",
				i, l.Stride)
		}
		for _, a := range l.Attributes {
			if a.Format == 0 {
				return nil, fmt.Errorf("accel/mtl: vertex buffer %d attribute at location "+
					"%d has no format", i, a.Location)
			}
		}
	}

	p := &RenderPipeline{}
	var err error
	withPool(func() {
		desc := objc.ID(clsRenderPipelineDesc).Send(selAlloc).Send(selInit)
		if desc == 0 {
			err = fmt.Errorf("accel/mtl: no render pipeline descriptor")
			return
		}
		defer release(desc)

		desc.Send(selSetVertexFunction, s.Vertex.fn)
		desc.Send(selSetFragmentFunction, s.Fragment.fn)
		if s.DepthFormat != 0 {
			desc.Send(selSetDepthPixelFormat, uintptr(s.DepthFormat))
		}

		atts := desc.Send(selColorAttachments)
		for i, f := range s.ColorFormats {
			a := atts.Send(selObjectAtIndexed, uintptr(i))
			a.Send(selSetPixelFormat, uintptr(f))
			if i < len(s.WriteMasks) {
				a.Send(selSetWriteMask, uintptr(s.WriteMasks[i]))
			}
			if i < len(s.Blends) && s.Blends[i].Enabled {
				b := s.Blends[i]
				a.Send(selSetBlendingEnabled, uintptr(1))
				a.Send(selSetSourceRGB, uintptr(b.SrcRGB))
				a.Send(selSetDestRGB, uintptr(b.DstRGB))
				a.Send(selSetRGBBlendOp, uintptr(b.OpRGB))
				a.Send(selSetSourceAlpha, uintptr(b.SrcA))
				a.Send(selSetDestAlpha, uintptr(b.DstA))
				a.Send(selSetAlphaBlendOp, uintptr(b.OpA))
			}
		}

		if len(s.VertexLayouts) > 0 {
			vd := objc.ID(clsVertexDescriptor).Send(selAlloc).Send(selInit)
			if vd == 0 {
				err = fmt.Errorf("accel/mtl: no vertex descriptor")
				return
			}
			defer release(vd)
			attrs := vd.Send(selVertexAttributes)
			layouts := vd.Send(selVertexLayouts)
			for i, l := range s.VertexLayouts {
				lay := layouts.Send(selObjectAtIndexed, uintptr(i))
				lay.Send(selSetStride, uintptr(l.Stride))
				step := uintptr(1) // MTLVertexStepFunctionPerVertex
				if l.PerInstance {
					step = 2 // MTLVertexStepFunctionPerInstance
				}
				lay.Send(selSetStepFunction, step)
				for _, at := range l.Attributes {
					x := attrs.Send(selObjectAtIndexed, uintptr(at.Location))
					x.Send(selSetFormat, uintptr(at.Format))
					x.Send(selSetOffset, uintptr(at.Offset))
					x.Send(selSetBufferIndex, uintptr(i))
				}
			}
			desc.Send(selSetVertexDescriptor, vd)
		}

		var nsErr objc.ID
		p.id = d.id.Send(selNewRenderPipeline, desc, unsafe.Pointer(&nsErr))
		if p.id == 0 {
			err = describe("creating a render pipeline state", nsErr)
		}
	})
	if err != nil {
		return nil, err
	}
	return p, nil
}

// DepthState is a compiled depth-stencil state.
type DepthState struct{ id objc.ID }

// Close releases it.
func (s *DepthState) Close() {
	withPool(func() { release(s.id) })
	s.id = 0
}

// NewDepthState compiles a depth test and write rule.
//
// compare is MTLCompareFunction. A pipeline that does not test depth still
// needs one of these when the pass has a depth attachment, with Always and
// writes off — Metal's default is Always with writes disabled, so a nil state
// is that, and this exists for everything else.
func (d *Device) NewDepthState(compare int, write bool) (*DepthState, error) {
	classes()
	s := &DepthState{}
	withPool(func() {
		desc := objc.ID(clsDepthStencilDesc).Send(selAlloc).Send(selInit)
		if desc == 0 {
			return
		}
		defer release(desc)
		desc.Send(selSetDepthCompare, uintptr(compare))
		w := uintptr(0)
		if write {
			w = 1
		}
		desc.Send(selSetDepthWrite, w)
		s.id = d.id.Send(selNewDepthStencil, desc)
	})
	if s.id == 0 {
		return nil, fmt.Errorf("accel/mtl: the device refused a depth-stencil state")
	}
	return s, nil
}

// RenderAttachment is one colour or depth attachment of a pass.
type RenderAttachment struct {
	Texture    *Texture
	Load       int
	Store      int
	ClearColor [4]float64
	ClearDepth float64
}

// RenderEncoder encodes draws into a command buffer.
type RenderEncoder struct{ id objc.ID }

// Render begins a render pass.
//
// Neither the descriptor nor the encoder comes from a new* selector, so neither
// is owned: the descriptor is left to the pool and the encoder is retained,
// because it outlives this call. That is the ownership rule objc_darwin.go
// states, and getting it backwards here is an abort rather than a leak.
func (cb *CommandBuffer) Render(color []RenderAttachment, depth *RenderAttachment) (*RenderEncoder, error) {
	classes()
	if len(color) == 0 {
		return nil, fmt.Errorf("accel/mtl: a render pass with no colour attachments")
	}
	e := &RenderEncoder{}
	var err error
	withPool(func() {
		rp := objc.ID(clsRenderPassDescriptor).Send(selRenderPassDesc)
		if rp == 0 {
			err = fmt.Errorf("accel/mtl: no render pass descriptor")
			return
		}
		atts := rp.Send(selColorAttachments)
		for i, a := range color {
			if a.Texture == nil || a.Texture.id == 0 {
				err = fmt.Errorf("accel/mtl: colour attachment %d has no texture", i)
				return
			}
			x := atts.Send(selObjectAtIndexed, uintptr(i))
			x.Send(selSetTexture, a.Texture.id)
			x.Send(selSetLoadAction, uintptr(a.Load))
			x.Send(selSetStoreAction, uintptr(a.Store))
			setClearColor(x, a.ClearColor)
		}
		if depth != nil {
			if depth.Texture == nil || depth.Texture.id == 0 {
				err = fmt.Errorf("accel/mtl: the depth attachment has no texture")
				return
			}
			x := rp.Send(selDepthAttachment)
			x.Send(selSetTexture, depth.Texture.id)
			x.Send(selSetLoadAction, uintptr(depth.Load))
			x.Send(selSetStoreAction, uintptr(depth.Store))
			setClearDepth(x, depth.ClearDepth)
		}
		// Retained for the reason the compute encoder is: the encoder is
		// autoreleased, and this pool drains before End is called. Without the
		// retain it is deallocated while still held, and Metal asserts
		// "released without endEncoding" -- an abort, not an error.
		e.id = retain(cb.id.Send(selRenderEncoderWith, rp))
		if e.id == 0 {
			err = fmt.Errorf("accel/mtl: the command buffer refused a render encoder")
		}
	})
	if err != nil {
		return nil, err
	}
	return e, nil
}

// SetPipeline selects the pipeline state subsequent draws use.
func (e *RenderEncoder) SetPipeline(p *RenderPipeline) {
	e.id.Send(selSetRenderPipeline, p.id)
}

// SetDepthState selects the depth test and write rule.
func (e *RenderEncoder) SetDepthState(s *DepthState) {
	e.id.Send(selSetDepthStencil, s.id)
}

// SetVertexBuffer binds a buffer the vertex stage fetches from.
func (e *RenderEncoder) SetVertexBuffer(b *Buffer, offset, index int) {
	e.id.Send(selSetVertexBuffer, b.id, uintptr(offset), uintptr(index))
}

// SetFragmentBuffer binds a buffer the fragment stage reads.
func (e *RenderEncoder) SetFragmentBuffer(b *Buffer, offset, index int) {
	withPool(func() {
		e.id.Send(selSetFragmentBuffer, b.id, uintptr(offset), uintptr(index))
	})
}

// SetVertexTexture binds a texture the vertex stage fetches from.
func (e *RenderEncoder) SetVertexTexture(t *Texture, index int) {
	e.id.Send(selSetVertexTexture, t.id, uintptr(index))
}

// SetFragmentTexture binds a texture the fragment stage fetches from.
func (e *RenderEncoder) SetFragmentTexture(t *Texture, index int) {
	e.id.Send(selSetFragmentTexture, t.id, uintptr(index))
}

// SetCull selects the cull mode: 0 none, 1 front, 2 back.
func (e *RenderEncoder) SetCull(mode int) { e.id.Send(selSetCullMode, uintptr(mode)) }

// SetWinding selects which winding faces front: 0 clockwise, 1 counterclockwise.
func (e *RenderEncoder) SetWinding(w int) { e.id.Send(selSetWinding, uintptr(w)) }

// Draw records a non-indexed draw. primitive is MTLPrimitiveType.
func (e *RenderEncoder) Draw(primitive, first, count, instances, firstInstance int) {
	e.id.Send(selDrawPrimitives, uintptr(primitive), uintptr(first), uintptr(count),
		uintptr(instances), uintptr(firstInstance))
}

// DrawIndexed records an indexed draw. indexType is 0 for uint16 and 1 for
// uint32.
func (e *RenderEncoder) DrawIndexed(primitive, count, indexType int, idx *Buffer,
	offset, instances, baseVertex, firstInstance int) {
	e.id.Send(selDrawIndexed, uintptr(primitive), uintptr(count), uintptr(indexType),
		idx.id, uintptr(offset), uintptr(instances), uintptr(baseVertex),
		uintptr(firstInstance))
}

// End finishes the pass and releases the encoder. Nothing else may be encoded
// into it afterwards.
func (e *RenderEncoder) End() {
	withPool(func() { e.id.Send(selEndEncoding) })
	release(e.id)
	e.id = 0
}

// ClearColor is MTLClearColor: four doubles passed by value.
//
// A struct and not four arguments, because that is what the selector takes.
// This is the same hazard specs/021-metal-bringup.md section 2 singles out for
// MTLSize: passing the components separately compiles, runs, and clears to a
// colour nobody asked for.
type ClearColor struct{ R, G, B, A float64 }

func setClearColor(att objc.ID, c [4]float64) {
	att.Send(selSetClearColor, ClearColor{R: c[0], G: c[1], B: c[2], A: c[3]})
}

func setClearDepth(att objc.ID, d float64) {
	att.Send(selSetClearDepth, d)
}

// copyTextureToBuffer encodes the blit, with the origin and size structs the
// selector takes by value.
func copyTextureToBuffer(enc objc.ID, src *Texture, dst *Buffer, offset, rowBytes int) {
	type origin struct{ X, Y, Z uint64 }
	type size struct{ W, H, D uint64 }
	if rowBytes <= 0 {
		rowBytes = src.width * src.bpp
	}
	withPool(func() {
		enc.Send(selCopyTextureToBuffer,
			src.id, uintptr(0), uintptr(0),
			origin{}, size{W: uint64(src.width), H: uint64(src.height), D: 1},
			dst.id, uintptr(offset), uintptr(rowBytes),
			uintptr(rowBytes*src.height))
	})
}

// SetVertexBytes and SetFragmentBytes put a small value directly in the command
// stream, which is what Metal offers for a uniform smaller than its 4 KiB
// limit. Larger than that needs a buffer, which the stage uniform path does not
// reach: a std140 block that big is a design mistake rather than a case to
// support.
func (e *RenderEncoder) SetVertexBytes(b []byte, index int) {
	withPool(func() {
		e.id.Send(selSetVertexBytes, unsafe.Pointer(&b[0]), uintptr(len(b)), uintptr(index))
	})
}

// SetFragmentBytes is the fragment stage's half. See [RenderEncoder.SetVertexBytes].
func (e *RenderEncoder) SetFragmentBytes(b []byte, index int) {
	withPool(func() {
		e.id.Send(selSetFragmentBytes, unsafe.Pointer(&b[0]), uintptr(len(b)), uintptr(index))
	})
}

// CopyBufferToTexture blits a tightly packed buffer into a whole texture.
//
// This is what LoadKeep costs on this backend: keeping prior contents means the
// texture must start as what the buffer holds. Clear and DontCare skip it.
// CopyBufferToTexture blits a buffer into a whole texture at a row pitch.
//
// Zero means the texture's own tight pitch. See [BlitEncoder.CopyTextureToBuffer]
// for why the caller's pitch and the texture's are not the same number.
func (b *BlitEncoder) CopyBufferToTexture(src *Buffer, offset int, dst *Texture, rowBytes int) {
	copyBufferToTexture(b.id, src, offset, dst, rowBytes)
}

func copyBufferToTexture(enc objc.ID, src *Buffer, offset int, dst *Texture, rowBytes int) {
	type origin struct{ X, Y, Z uint64 }
	type size struct{ W, H, D uint64 }
	if rowBytes <= 0 {
		rowBytes = dst.width * dst.bpp
	}
	withPool(func() {
		enc.Send(selCopyBufferToTexture,
			src.id, uintptr(offset), uintptr(rowBytes), uintptr(rowBytes*dst.height),
			size{W: uint64(dst.width), H: uint64(dst.height), D: 1},
			dst.id, uintptr(0), uintptr(0), origin{})
	})
}

// CopyTextureToBuffer blits a whole texture into a buffer at a row pitch.
//
// This is how a render result reaches the caller's buffer, which is where
// specs/033-render-api.md says an attachment lives.
//
// The pitch is the destination's, not the texture's, and passing it is the
// whole point: a caller's attachment buffer is sized to the pitch its device
// reports, which is padded to an alignment, while a texture's rows are tight.
// Writing at the tight pitch into a buffer laid out at the padded one puts
// every row after the first in the wrong place -- which read back as an image
// whose lower rows were blank, at every extent whose row was narrower than the
// alignment.
//
// Zero means the texture's own tight pitch, which is what a copy between a
// tightly packed buffer and a texture wants.
func (b *BlitEncoder) CopyTextureToBuffer(src *Texture, dst *Buffer, offset, rowBytes int) {
	copyTextureToBuffer(b.id, src, dst, offset, rowBytes)
}
