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
)

// The attachment format codec of specs/045-texture-attachments.md section 4.
//
// # Why the CPU backend owns this and Metal does not
//
// internal/raster works in float32 components and an attachment does not, so
// something has to convert. On every hardware backend that something is the
// fixed-function output stage, which is why the conversion is not in the
// rasterizer: 035 section 5 says sRGB converts on write and on read rather
// than in the fragment stage, and the stage sees linear values on every
// backend. Here the output stage is this file.
//
// # Why each conversion has exactly one definition
//
// This backend is the oracle every other backend is compared against, so a
// conversion with two defensible readings would make the oracle assert one
// vendor's answer. Where a format has no single portable definition it is
// refused by name rather than given one -- Depth24PlusStencil8 below is the
// case, and its layout is device-defined by the format table's own account.

// texelCodec is one attachment format's conversion, both directions.
//
// components is how many floats per texel the rasterizer's framebuffer holds,
// and it is a property of the aspect rather than of the format's channel
// count: a colour target is always four floats wide because the write mask and
// the blend factors are defined per channel over four, and a depth target is
// one. A format with fewer channels than four reads its missing ones as the
// (0, 0, 0, 1) every target API defines, and writes them nowhere.
type texelCodec struct {
	format     driver.Format
	bytes      int
	components int
	decode     func(src []byte) [4]float32
	encode     func(dst []byte, v [4]float32)
}

// codecFor is the codec for one format, or the reason there is none.
func codecFor(f driver.Format) (texelCodec, error) {
	switch f {
	case driver.RGBA8Unorm:
		return texelCodec{f, 4, 4, decodeRGBA8, encodeRGBA8}, nil
	case driver.RGBA8UnormSRGB:
		return texelCodec{f, 4, 4, decodeRGBA8SRGB, encodeRGBA8SRGB}, nil
	case driver.BGRA8Unorm:
		return texelCodec{f, 4, 4, decodeBGRA8, encodeBGRA8}, nil
	case driver.R16Float:
		return texelCodec{f, 2, 4, decodeHalf(1), encodeHalf(1)}, nil
	case driver.RG16Float:
		return texelCodec{f, 4, 4, decodeHalf(2), encodeHalf(2)}, nil
	case driver.RGBA16Float:
		return texelCodec{f, 8, 4, decodeHalf(4), encodeHalf(4)}, nil
	case driver.R32Float:
		return texelCodec{f, 4, 4, decodeFloat(1), encodeFloat(1)}, nil
	case driver.RG32Float:
		return texelCodec{f, 8, 4, decodeFloat(2), encodeFloat(2)}, nil
	case driver.RGBA32Float:
		return texelCodec{f, 16, 4, decodeFloat(4), encodeFloat(4)}, nil
	case driver.Depth32Float:
		return texelCodec{f, 4, 1, decodeFloat(1), encodeFloat(1)}, nil
	case driver.Depth32FloatStencil8:
		// The *depth* plane only. The format is planar -- specs/045 section 12
		// -- so its stencil plane is a separate image with its own pitch, read
		// and written beside this rather than through it. A texel codec's unit
		// is one texel of one plane, and pretending otherwise is what the
		// interleaved version did.
		return texelCodec{f, 4, 1, decodeFloat(1), encodeFloat(1)}, nil
	case driver.Depth24PlusStencil8:
		return texelCodec{}, fmt.Errorf("%v has a device-defined layout -- \"24 plus\" "+
			"means at least 24 bits of depth, and a backend may store it as 32 with 8 "+
			"unused or pack it with the stencil. This backend is the oracle every other "+
			"one is compared against, so it refuses a format with two defensible "+
			"encodings rather than asserting one of them", f)
	}
	return texelCodec{}, fmt.Errorf("%v is not an attachment format this backend decodes", f)
}

// decodeImage fills dst with w*h texels read from src at the given row pitch.
//
// The pitch is carried rather than derived from len(src)/h, because a texture's
// rows are padded to the device's alignment and a subresource of one is not the
// whole allocation: deriving it works today and stops working the moment an
// attachment names a mip.
func (c texelCodec) decodeImage(dst []float32, src []byte, w, h, pitch int) error {
	if err := c.checkImage(len(src), len(dst), w, h, pitch); err != nil {
		return err
	}
	for y := range h {
		row := src[y*pitch:]
		out := dst[y*w*c.components:]
		for x := range w {
			v := c.decode(row[x*c.bytes:])
			copy(out[x*c.components:(x+1)*c.components], v[:c.components])
		}
	}
	return nil
}

// encodeImage writes w*h texels from src back into dst at the given row pitch.
func (c texelCodec) encodeImage(dst []byte, src []float32, w, h, pitch int) error {
	if err := c.checkImage(len(dst), len(src), w, h, pitch); err != nil {
		return err
	}
	for y := range h {
		row := dst[y*pitch:]
		in := src[y*w*c.components:]
		for x := range w {
			var v [4]float32
			// The channels the framebuffer does not hold read as the
			// (0, 0, 0, 1) a fetch of a missing channel gives, so a depth
			// target's alpha is not written from uninitialized memory.
			v[3] = 1
			copy(v[:c.components], in[x*c.components:(x+1)*c.components])
			c.encode(row[x*c.bytes:], v)
		}
	}
	return nil
}

// checkImage reports a byte or float side too small for the rectangle.
//
// The last row needs only its own texels and not a full pitch, the same rule a
// row copy follows: a tightly packed image's final row has no padding after it.
func (c texelCodec) checkImage(bytes, floats, w, h, pitch int) error {
	if w <= 0 || h <= 0 {
		return fmt.Errorf("a %dx%d attachment rectangle", w, h)
	}
	if pitch < w*c.bytes {
		return fmt.Errorf("a %d-byte row pitch is shorter than %d %v texels, which are "+
			"%d bytes", pitch, w, c.format, w*c.bytes)
	}
	if need := (h-1)*pitch + w*c.bytes; bytes < need {
		return fmt.Errorf("the attachment holds %d bytes and a %dx%d %v image at a pitch "+
			"of %d needs %d", bytes, w, h, c.format, pitch, need)
	}
	if need := w * h * c.components; floats < need {
		return fmt.Errorf("the framebuffer holds %d floats and a %dx%d %v image needs %d",
			floats, w, h, c.format, need)
	}
	return nil
}

// unorm8ToFloat is the exact definition: v/255, so 0 is 0 and 255 is 1.
func unorm8ToFloat(b byte) float32 { return float32(b) / 255 }

// floatToUnorm8 is the inverse, rounded to nearest and clamped.
//
// Round to nearest rather than truncate, which is what D3D, Vulkan and Metal
// all specify: truncation loses the round trip, since 1/255 decoded and
// re-encoded would land on 0. NaN encodes as zero for the same reason every
// target does it -- the alternative is an unpredictable byte.
func floatToUnorm8(v float32) byte {
	if math.IsNaN(float64(v)) || v <= 0 {
		return 0
	}
	if v >= 1 {
		return 255
	}
	return byte(math.Round(float64(v) * 255))
}

// srgbToLinear is the sRGB electro-optical transfer function of IEC 61966-2-1,
// which is the definition every target API cites:
//
//	linear(c) = c/12.92                       for c <= 0.04045
//	linear(c) = ((c + 0.055)/1.055)^2.4       otherwise
//
// The linear segment near black is not a rounding convenience: it bounds the
// slope at zero, which a pure power function leaves infinite.
func srgbToLinear(c float32) float32 {
	if c <= 0.04045 {
		return c / 12.92
	}
	return float32(math.Pow((float64(c)+0.055)/1.055, 2.4))
}

// linearToSRGB is the inverse, with the threshold at the same point on the
// other side of the curve:
//
//	srgb(c) = 12.92 * c                       for c <= 0.0031308
//	srgb(c) = 1.055 * c^(1/2.4) - 0.055       otherwise
func linearToSRGB(c float32) float32 {
	if c <= 0.0031308 {
		return 12.92 * c
	}
	return float32(1.055*math.Pow(float64(c), 1/2.4) - 0.055)
}

func decodeRGBA8(src []byte) [4]float32 {
	return [4]float32{
		unorm8ToFloat(src[0]), unorm8ToFloat(src[1]),
		unorm8ToFloat(src[2]), unorm8ToFloat(src[3]),
	}
}

func encodeRGBA8(dst []byte, v [4]float32) {
	for i := range 4 {
		dst[i] = floatToUnorm8(v[i])
	}
}

// decodeRGBA8SRGB reads the three colour channels through the transfer
// function and the alpha channel linearly.
//
// Alpha is linear on every target API, and it is not an oversight there: alpha
// is a coverage weight rather than a light intensity, so putting it through a
// display transfer function would make a half-covered pixel composite wrong.
func decodeRGBA8SRGB(src []byte) [4]float32 {
	return [4]float32{
		srgbToLinear(unorm8ToFloat(src[0])),
		srgbToLinear(unorm8ToFloat(src[1])),
		srgbToLinear(unorm8ToFloat(src[2])),
		unorm8ToFloat(src[3]),
	}
}

func encodeRGBA8SRGB(dst []byte, v [4]float32) {
	dst[0] = floatToUnorm8(linearToSRGB(v[0]))
	dst[1] = floatToUnorm8(linearToSRGB(v[1]))
	dst[2] = floatToUnorm8(linearToSRGB(v[2]))
	dst[3] = floatToUnorm8(v[3])
}

// decodeBGRA8 differs from RGBA8 in channel order and in nothing else. The
// order is the format's whole content, which is why it has one definition.
func decodeBGRA8(src []byte) [4]float32 {
	return [4]float32{
		unorm8ToFloat(src[2]), unorm8ToFloat(src[1]),
		unorm8ToFloat(src[0]), unorm8ToFloat(src[3]),
	}
}

func encodeBGRA8(dst []byte, v [4]float32) {
	dst[0] = floatToUnorm8(v[2])
	dst[1] = floatToUnorm8(v[1])
	dst[2] = floatToUnorm8(v[0])
	dst[3] = floatToUnorm8(v[3])
}

// decodeFloat and encodeFloat are the float32 formats, which are a
// reinterpretation rather than a conversion: the stored bytes are already the
// component the rasterizer works in.
func decodeFloat(n int) func([]byte) [4]float32 {
	return func(src []byte) [4]float32 {
		v := [4]float32{0, 0, 0, 1}
		for i := range n {
			v[i] = math.Float32frombits(binary.LittleEndian.Uint32(src[i*4:]))
		}
		return v
	}
}

func encodeFloat(n int) func([]byte, [4]float32) {
	return func(dst []byte, v [4]float32) {
		for i := range n {
			binary.LittleEndian.PutUint32(dst[i*4:], math.Float32bits(v[i]))
		}
	}
}

// decodeHalf and encodeHalf are the float16 formats, through the same
// round-to-nearest-even conversion a narrow kernel binding uses. One
// definition of half, shared, is what keeps an attachment and a storage
// binding from disagreeing about the same sixteen bits.
func decodeHalf(n int) func([]byte) [4]float32 {
	return func(src []byte) [4]float32 {
		v := [4]float32{0, 0, 0, 1}
		for i := range n {
			v[i] = kernel.Float16FromBits(binary.LittleEndian.Uint16(src[i*2:])).F32()
		}
		return v
	}
}

func encodeHalf(n int) func([]byte, [4]float32) {
	return func(dst []byte, v [4]float32) {
		for i := range n {
			binary.LittleEndian.PutUint16(dst[i*2:], kernel.ToFloat16(v[i]).Bits())
		}
	}
}
