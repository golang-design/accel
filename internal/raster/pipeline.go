// Copyright 2026 The golang.design Initiative Authors.
// All rights reserved. Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

package raster

// The per-fragment chain of specs/035-cpu-rasterizer.md section 5, in the order
// that section states:
//
//	1. fragment stage        -- a discard ends the invocation here
//	2. stencil test          -- per face, then its three operations
//	3. depth test and write
//	4. blend                 -- per attachment
//	5. write mask            -- per attachment, per channel
//
// The stage comes first and that is the point. Early depth testing is an
// optimization a backend performs only where it can prove the stage does not
// change the answer, and a stage that can discard cannot be proven that. Running
// the tests first would make specs/032-stage-abi.md section 4.2's guarantee --
// a discard writes no attachment and no depth -- unimplementable.
//
// Attachments here hold float32 components and no format. Format encoding,
// including the sRGB conversion that happens on write and on read rather than in
// the stage, belongs to the layer that owns texture formats; this package would
// otherwise need the format table to rasterize a triangle.

// Compare is a depth or stencil comparison.
type Compare uint8

const (
	CompareNever Compare = iota
	CompareLess
	CompareEqual
	CompareLessEqual
	CompareGreater
	CompareNotEqual
	CompareGreaterEqual
	CompareAlways
)

// test applies the comparison. Ordered so that reversing a compare function is
// one table lookup rather than a chain a reader has to verify.
func (c Compare) test(a, b float32) bool {
	switch c {
	case CompareNever:
		return false
	case CompareLess:
		return a < b
	case CompareEqual:
		return a == b
	case CompareLessEqual:
		return a <= b
	case CompareGreater:
		return a > b
	case CompareNotEqual:
		return a != b
	case CompareGreaterEqual:
		return a >= b
	default:
		return true
	}
}

// StencilOp is what happens to a stencil value after a test.
type StencilOp uint8

const (
	StencilKeep StencilOp = iota
	StencilZero
	StencilReplace
	StencilIncrementClamp
	StencilDecrementClamp
	StencilInvert
	StencilIncrementWrap
	StencilDecrementWrap
)

func (op StencilOp) apply(cur, ref uint8) uint8 {
	switch op {
	case StencilZero:
		return 0
	case StencilReplace:
		return ref
	case StencilIncrementClamp:
		if cur == 0xFF {
			return 0xFF
		}
		return cur + 1
	case StencilDecrementClamp:
		if cur == 0 {
			return 0
		}
		return cur - 1
	case StencilInvert:
		return ^cur
	case StencilIncrementWrap:
		return cur + 1
	case StencilDecrementWrap:
		return cur - 1
	default:
		return cur
	}
}

// StencilFace is one face's stencil configuration.
//
// Per face because that is what every target backend offers and what two-sided
// stencil techniques need. specs/005-graphics.md specifies stencil now rather
// than adding it later because adding it later changes the shape of the pipeline
// descriptor, which is a breaking change for every caller.
type StencilFace struct {
	Compare               Compare
	ReadMask, WriteMask   uint8
	Fail, DepthFail, Pass StencilOp
}

// DepthState is the depth test and write configuration.
//
// Test and Write are separate fields because read-only depth -- test on, write
// off -- is a real and useful configuration: it is how a second pass shades
// exactly the surfaces the geometry pass kept.
type DepthState struct {
	Test    bool
	Write   bool
	Compare Compare
}

// StencilState is both faces plus the one dynamic value.
//
// The reference is dynamic on every backend and everything else about stencil is
// compiled in, which is why it lives beside the faces rather than inside them.
type StencilState struct {
	Enabled     bool
	Front, Back StencilFace
	Reference   uint8
}

// BlendFactor scales one side of a blend.
type BlendFactor uint8

const (
	FactorZero BlendFactor = iota
	FactorOne
	FactorSrcColor
	FactorOneMinusSrcColor
	FactorSrcAlpha
	FactorOneMinusSrcAlpha
	FactorDstColor
	FactorOneMinusDstColor
	FactorDstAlpha
	FactorOneMinusDstAlpha
)

func (f BlendFactor) value(src, dst [4]float32, c int) float32 {
	switch f {
	case FactorZero:
		return 0
	case FactorOne:
		return 1
	case FactorSrcColor:
		return src[c]
	case FactorOneMinusSrcColor:
		return 1 - src[c]
	case FactorSrcAlpha:
		return src[3]
	case FactorOneMinusSrcAlpha:
		return 1 - src[3]
	case FactorDstColor:
		return dst[c]
	case FactorOneMinusDstColor:
		return 1 - dst[c]
	case FactorDstAlpha:
		return dst[3]
	default:
		return 1 - dst[3]
	}
}

// BlendOp combines the two scaled sides.
type BlendOp uint8

const (
	BlendAdd BlendOp = iota
	BlendSubtract
	BlendReverseSubtract
	BlendMin
	BlendMax
)

func (op BlendOp) apply(s, d float32) float32 {
	switch op {
	case BlendSubtract:
		return s - d
	case BlendReverseSubtract:
		return d - s
	case BlendMin:
		return min(s, d)
	case BlendMax:
		return max(s, d)
	default:
		return s + d
	}
}

// Blend is one attachment's blend state.
//
// Colour and alpha carry separate factors and operations, and the whole thing is
// per attachment rather than per pipeline: a G-buffer's albedo target and its
// accumulation target want different answers in the same pass.
type Blend struct {
	Enabled            bool
	SrcColor, DstColor BlendFactor
	ColorOp            BlendOp
	SrcAlpha, DstAlpha BlendFactor
	AlphaOp            BlendOp
}

// WriteMask is which channels of an attachment a fragment may write.
type WriteMask uint8

const (
	WriteR WriteMask = 1 << iota
	WriteG
	WriteB
	WriteA

	WriteAll  = WriteR | WriteG | WriteB | WriteA
	WriteNone = WriteMask(0)
)

func (m WriteMask) has(c int) bool { return m&(1<<uint(c)) != 0 }

// ColorTarget is one colour attachment: four float32 components per pixel, in
// row-major order with row 0 the top row.
//
// It is storage and nothing else. Blend state and the write mask live on
// [PassState] rather than here, because specs/033-render-api.md fixes both at
// *pipeline* creation and every backend agrees: Vulkan puts them in an array on
// the pipeline, Metal on the render pipeline's colour attachment descriptor,
// D3D12 in the blend description's render-target array. Putting them on the
// attachment would mean rewriting the attachment object before every draw, since
// a pass holds one set of attachments and many draws with different pipelines --
// per-submission state on a per-graph object, which is what 003's immutability
// model exists to prevent.
//
// Top-origin here and nowhere else is what makes specs/005-graphics.md's
// three-way origin guarantee one correction rather than three: the y flip
// already happened in the viewport transform, so this buffer's layout is the
// same one a host readback sees.
type ColorTarget struct {
	W, H int
	Pix  []float32
}

// NewColorTarget allocates a target cleared to the given colour.
func NewColorTarget(w, h int, clear [4]float32) *ColorTarget {
	t := &ColorTarget{W: w, H: h, Pix: make([]float32, w*h*4)}
	t.Clear(clear)
	return t
}

// Clear fills every pixel.
func (t *ColorTarget) Clear(c [4]float32) {
	for i := range t.Pix {
		t.Pix[i] = c[i%4]
	}
}

// At reads one pixel.
func (t *ColorTarget) At(x, y int) [4]float32 {
	i := (y*t.W + x) * 4
	return [4]float32{t.Pix[i], t.Pix[i+1], t.Pix[i+2], t.Pix[i+3]}
}

// DepthTarget is a depth and stencil attachment.
//
// One object rather than two because every backend that offers stencil offers it
// packed with depth, and because the stencil operations depend on the depth
// test's outcome: separating them would put the depth-fail case in a type that
// cannot see the depth test.
type DepthTarget struct {
	W, H    int
	Z       []float32
	Stencil []uint8
}

// NewDepthTarget allocates a depth-stencil target cleared to the given values.
func NewDepthTarget(w, h int, z float32, s uint8) *DepthTarget {
	t := &DepthTarget{W: w, H: h, Z: make([]float32, w*h), Stencil: make([]uint8, w*h)}
	t.Clear(z, s)
	return t
}

// Clear fills depth and stencil.
func (t *DepthTarget) Clear(z float32, s uint8) {
	for i := range t.Z {
		t.Z[i] = z
	}
	// A target whose format has no stencil aspect carries no stencil buffer,
	// so there is nothing to clear. Nil rather than a zeroed array, because an
	// array nothing can address is what made the stencil pipeline look
	// reachable for as long as it did.
	for i := range t.Stencil {
		t.Stencil[i] = s
	}
}

// Framebuffer is what one pass writes.
type Framebuffer struct {
	Color []*ColorTarget
	Depth *DepthTarget
}

// Shaded is what a fragment stage returned.
type Shaded struct {
	// Discard ends the invocation: no attachment write and no depth write.
	Discard bool

	// Color is one value per colour attachment, in attachment order.
	// specs/032-stage-abi.md maps a fragment stage's output struct fields onto
	// attachments in declaration order, and this is that mapping already
	// applied.
	Color [][4]float32
}

// PassState is the fixed-function state one draw runs under.
//
// Everything a pipeline compiles in, plus the two dynamic values. Blend and
// Mask are indexed by attachment slot; a slot past the end of either takes the
// default, which is no blending and every channel written.
type PassState struct {
	State
	Depth   DepthState
	Stencil StencilState
	Blend   []Blend
	Mask    []WriteMask
}

// blendAt is the blend state for one attachment slot.
func (ps PassState) blendAt(i int) Blend {
	if i < len(ps.Blend) {
		return ps.Blend[i]
	}
	return Blend{}
}

// maskAt is the write mask for one attachment slot. An absent entry writes every
// channel, so a caller who configures nothing gets the unmasked behaviour rather
// than a target nothing reaches.
func (ps PassState) maskAt(i int) WriteMask {
	if i < len(ps.Mask) {
		return ps.Mask[i]
	}
	return WriteAll
}

// Draw rasterizes one triangle through the full per-fragment chain.
//
// It reports how many fragments reached an attachment write, which is what
// distinguishes "covered nothing" from "covered and every fragment was rejected
// by a test" -- two failures that look identical in the resulting image.
func Draw(ps PassState, fb *Framebuffer, tri [3]Vertex, shade func(Fragment) Shaded) int {
	written := 0
	Rasterize(ps.State, tri, func(f Fragment) {
		// 1. The stage, first. See the file comment.
		out := shade(f)
		if out.Discard {
			return
		}

		idx := 0
		if fb.Depth != nil {
			idx = f.Y*fb.Depth.W + f.X
		}

		// 2. Stencil, whose three operations depend on the depth test's
		// outcome, so the depth comparison is evaluated before the stencil
		// operation is chosen even though the depth *write* happens after.
		face := ps.Stencil.Front
		if !f.Front {
			face = ps.Stencil.Back
		}
		stencilPass := true
		if ps.Stencil.Enabled && fb.Depth != nil && fb.Depth.Stencil != nil {
			cur := fb.Depth.Stencil[idx]
			stencilPass = face.Compare.test(
				float32(ps.Stencil.Reference&face.ReadMask), float32(cur&face.ReadMask))
		}

		depthPass := true
		if ps.Depth.Test && fb.Depth != nil {
			depthPass = ps.Depth.Compare.test(f.Depth, fb.Depth.Z[idx])
		}

		if ps.Stencil.Enabled && fb.Depth != nil && fb.Depth.Stencil != nil {
			op := face.Pass
			switch {
			case !stencilPass:
				op = face.Fail
			case !depthPass:
				op = face.DepthFail
			}
			cur := fb.Depth.Stencil[idx]
			next := op.apply(cur, ps.Stencil.Reference)
			// The write mask selects bits rather than gating the whole write,
			// which is what makes a masked stencil buffer usable for two
			// independent techniques at once.
			fb.Depth.Stencil[idx] = (cur &^ face.WriteMask) | (next & face.WriteMask)
		}
		if !stencilPass || !depthPass {
			return
		}

		// 3. The depth write, after both tests and only if they passed.
		if ps.Depth.Write && fb.Depth != nil {
			fb.Depth.Z[idx] = f.Depth
		}

		// 4 and 5. Blend then mask, per attachment.
		for i, t := range fb.Color {
			if t == nil || i >= len(out.Color) {
				continue
			}
			src := out.Color[i]
			at := (f.Y*t.W + f.X) * 4
			dst := [4]float32{t.Pix[at], t.Pix[at+1], t.Pix[at+2], t.Pix[at+3]}
			res := src
			if b := ps.blendAt(i); b.Enabled {
				res = blend(b, src, dst)
			}
			mask := ps.maskAt(i)
			for c := range 4 {
				if mask.has(c) {
					t.Pix[at+c] = res[c]
				}
			}
		}
		written++
	})
	return written
}

// blend applies specs/035-cpu-rasterizer.md section 5's equation per channel,
// with colour and alpha carrying their own factors and operation.
func blend(b Blend, src, dst [4]float32) [4]float32 {
	var out [4]float32
	for c := range 3 {
		out[c] = b.ColorOp.combine(b.SrcColor, b.DstColor, src, dst, c)
	}
	out[3] = b.AlphaOp.combine(b.SrcAlpha, b.DstAlpha, src, dst, 3)
	return out
}

// combine is one channel of the blend equation:
//
//	dst' = op(F_src * src, F_dst * dst)
//
// except for min and max, which take the unscaled operands:
//
//	dst' = op(src, dst)
//
// **The factors are ignored for min and max on every target backend**, and this
// rasterizer applied them until specs/062-backend-parity.md section 6.4's
// enumeration compared the five operations against Metal one at a time. Vulkan
// states it (VK_BLEND_OP_MIN and MAX "ignore the source and destination blend
// factors"), D3D and Metal do the same, and it is the only reading that makes
// the operation useful: a minimum of two values scaled by different factors is
// not the minimum of anything a caller can name.
//
// The oracle was wrong and Metal was right, which is the case this comparison
// exists for -- the CPU backend being the reference is a claim that has to be
// checked rather than assumed.
func (op BlendOp) combine(fs, fd BlendFactor, src, dst [4]float32, c int) float32 {
	if op == BlendMin || op == BlendMax {
		return op.apply(src[c], dst[c])
	}
	return op.apply(fs.value(src, dst, c)*src[c], fd.value(src, dst, c)*dst[c])
}
