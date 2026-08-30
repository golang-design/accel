//go:build darwin

package mtl

import (
	"math"
	"testing"

	"github.com/ebitengine/purego/objc"
)

// Does this device render into a texture aliased onto buffer memory?
//
// specs/045 §8.1 leaves Metal staging every attachment through a buffer: a blit
// in, a render, a blit out, per pass. If MTLBuffer's newTextureWithDescriptor:
// accepts MTLTextureUsageRenderTarget, the attachment and the caller's bytes are
// the same memory and every one of those copies disappears.
//
// Recalled claims say linear textures may not be render targets on Apple GPUs.
// This measures it rather than believing either way, and it stays as a test
// because the whole staging path was deleted on the strength of the answer: a
// driver or family that stopped accepting it would otherwise show up as a
// wrong picture rather than as a named failure.
//
// The alignment it logs is the other half. Metal requires a linear texture's
// row pitch to be a multiple of minimumLinearTextureAlignmentForPixelFormat,
// and accel already aligns every texture row to MinBufferCopyRowPitchAlignment,
// which is 256 -- a multiple of the 16 this reports. The two requirements
// agreeing is what makes the aliasing free rather than a repack.
func TestALinearTextureMayBeARenderTarget(t *testing.T) {
	ds, err := Devices()
	if err != nil || len(ds) == 0 {
		t.Skipf("no Metal device: %v", err)
	}
	d := ds[0]
	classes()

	const w, h = 64, 4
	const bpp = 16 // RGBA32Float

	align := 0
	selAlign := objc.RegisterName("minimumLinearTextureAlignmentForPixelFormat:")
	withPool(func() {
		align = int(d.id.Send(selAlign, uintptr(PixelFormatRGBA32Float)))
	})
	t.Logf("minimumLinearTextureAlignmentForPixelFormat(RGBA32Float) = %d", align)

	pitch := w * bpp
	if align > 0 && pitch%align != 0 {
		pitch = ((pitch + align - 1) / align) * align
	}
	t.Logf("row pitch %d for %d wide", pitch, w)

	buf, err := d.NewBuffer(pitch*h, StorageShared)
	if err != nil {
		t.Fatalf("buffer: %v", err)
	}
	defer buf.Close()

	selNewTexFromBuf := objc.RegisterName("newTextureWithDescriptor:offset:bytesPerRow:")

	for _, c := range []struct {
		name  string
		usage uintptr
	}{
		{"shaderRead only", 1},
		{"renderTarget only", 4},
		{"renderTarget|shaderRead", 5},
	} {
		var id objc.ID
		withPool(func() {
			desc := objc.ID(clsTextureDescriptor).Send(selTexture2D,
				uintptr(PixelFormatRGBA32Float), uintptr(w), uintptr(h), uintptr(0))
			if desc == 0 {
				return
			}
			desc.Send(selSetTexUsage, c.usage)
			// A buffer-backed texture inherits the buffer's storage mode; set
			// it to match so the descriptor is not asking for a different one.
			desc.Send(selSetTexStorage, uintptr(0)) // shared
			id = buf.id.Send(selNewTexFromBuf, desc, uintptr(0), uintptr(pitch))
		})
		if id == 0 {
			t.Errorf("%s: the device refused a buffer-backed texture", c.name)
			continue
		}
		t.Logf("%s: accepted", c.name)
		withPool(func() { release(id) })
	}
}

// And does a clear into one actually land in the buffer's bytes?
//
// Acceptance is not correctness. This renders with LoadActionClear and
// StoreActionStore into a buffer-backed texture and reads the buffer, which is
// the whole claim the staging path would be deleted on.
func TestRenderingIntoALinearTextureLandsInTheBuffer(t *testing.T) {
	ds, err := Devices()
	if err != nil || len(ds) == 0 {
		t.Skipf("no Metal device: %v", err)
	}
	d := ds[0]
	classes()

	const w, h = 64, 4
	pitch := w * 16

	buf, err := d.NewBuffer(pitch*h, StorageShared)
	if err != nil {
		t.Fatalf("buffer: %v", err)
	}
	defer buf.Close()
	for i := range buf.Bytes() {
		buf.Bytes()[i] = 0xAB
	}

	selNewTexFromBuf := objc.RegisterName("newTextureWithDescriptor:offset:bytesPerRow:")
	tex := &Texture{width: w, height: h, bpp: 16}
	withPool(func() {
		desc := objc.ID(clsTextureDescriptor).Send(selTexture2D,
			uintptr(PixelFormatRGBA32Float), uintptr(w), uintptr(h), uintptr(0))
		desc.Send(selSetTexUsage, uintptr(4|1))
		desc.Send(selSetTexStorage, uintptr(0))
		tex.id = buf.id.Send(selNewTexFromBuf, desc, uintptr(0), uintptr(pitch))
	})
	if tex.id == 0 {
		t.Fatal("the device refused a buffer-backed render target")
	}
	defer tex.Close()

	q := d.NewQueue()
	defer q.Close()
	cb := q.Begin()
	enc, err := cb.Render([]RenderAttachment{{
		Texture: tex, Load: LoadActionClear, Store: StoreActionStore,
		ClearColor: [4]float64{0.25, 0.5, 0.75, 1},
	}}, nil)
	if err != nil {
		t.Fatalf("render encoder: %v", err)
	}
	enc.End()
	cb.Commit()
	cb.Wait()

	got := buf.Bytes()
	want := []float32{0.25, 0.5, 0.75, 1}
	for px := range w * h {
		for c := range 4 {
			at := px*16 + c*4
			bits := uint32(got[at]) | uint32(got[at+1])<<8 |
				uint32(got[at+2])<<16 | uint32(got[at+3])<<24
			if f := math.Float32frombits(bits); f != want[c] {
				t.Fatalf("pixel %d channel %d is %v, want %v: the clear did not reach "+
					"the buffer's bytes", px, c, f, want[c])
			}
		}
	}
	t.Logf("a clear into a buffer-backed render target reached all %d pixels", w*h)
}
