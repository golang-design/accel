//go:build darwin

package mtl

import (
	"math"
	"testing"

	"github.com/ebitengine/purego/objc"
)

// What does a combined depth/stencil texture actually allow?
//
// specs/033-render-api.md section 10.5 turns on three recalled claims: that a
// combined format can only be copied one aspect at a time, that the two options
// are mutually exclusive, and that each produces a tightly packed plane. This
// asks the device instead.
//
// Every option *encodes*, which is the point of keeping this beside the round
// trip below: Metal's validation layer is off here, so surviving an encode says
// nothing about what the copy did. The two tests together are the measurement.
func TestACombinedDepthStencilTextureAcceptsEveryBlitOption(t *testing.T) {
	ds, err := Devices()
	if err != nil || len(ds) == 0 {
		t.Skipf("no Metal device: %v", err)
	}
	d := ds[0]
	classes()

	const w, h = 16, 4
	const fmtDepth32FloatStencil8 = 260

	tex := &Texture{width: w, height: h, bpp: 8}
	withPool(func() {
		desc := objc.ID(clsTextureDescriptor).Send(selTexture2D,
			uintptr(fmtDepth32FloatStencil8), uintptr(w), uintptr(h), uintptr(0))
		desc.Send(selSetTexUsage, uintptr(textureUsageRenderTarget))
		desc.Send(selSetTexStorage, uintptr(storageModePrivate))
		tex.id = d.id.Send(selNewTexture, desc)
	})
	if tex.id == 0 {
		t.Fatal("the device refused a Depth32Float_Stencil8 texture")
	}
	defer tex.Close()
	t.Log("Depth32Float_Stencil8 texture: accepted")

	buf, err := d.NewBuffer(w*h*8, StorageShared)
	if err != nil {
		t.Fatalf("buffer: %v", err)
	}
	defer buf.Close()

	sel := objc.RegisterName(
		"copyFromTexture:sourceSlice:sourceLevel:sourceOrigin:sourceSize:" +
			"toBuffer:destinationOffset:destinationBytesPerRow:" +
			"destinationBytesPerImage:options:")

	for _, c := range []struct {
		name string
		opt  uintptr
		row  int
	}{
		{"no option, 8 bytes per row", 0, w * 8},
		{"DepthFromDepthStencil, 4 bytes per row", 1, w * 4},
		{"StencilFromDepthStencil, 1 byte per row", 2, w * 1},
		{"both options", 3, w * 8},
	} {
		q := d.NewQueue()
		cb := q.Begin()
		blit := cb.Blit()
		type origin struct{ X, Y, Z uint64 }
		type size struct{ W, H, D uint64 }
		withPool(func() {
			blit.id.Send(sel,
				tex.id, uintptr(0), uintptr(0),
				origin{}, size{W: w, H: h, D: 1},
				buf.id, uintptr(0), uintptr(c.row), uintptr(c.row*h), c.opt)
		})
		blit.End()
		cb.Commit()
		cb.Wait()
		t.Logf("%s: survived encode and commit", c.name)
		q.Close()
	}
}

// Do the aspects round-trip, and does a no-option copy move the interleaved
// pair?
//
// The encode surviving says nothing: Metal's validation layer is off here, so
// an illegal blit is a no-op or garbage rather than an error. This writes known
// bytes in and reads them back.
func TestOnlyOneAspectAtATimeRoundTrips(t *testing.T) {
	ds, err := Devices()
	if err != nil || len(ds) == 0 {
		t.Skipf("no Metal device: %v", err)
	}
	d := ds[0]
	classes()

	const w, h = 16, 4
	const fmtDS = 260

	tex := &Texture{width: w, height: h, bpp: 8}
	withPool(func() {
		desc := objc.ID(clsTextureDescriptor).Send(selTexture2D,
			uintptr(fmtDS), uintptr(w), uintptr(h), uintptr(0))
		desc.Send(selSetTexUsage, uintptr(textureUsageRenderTarget))
		desc.Send(selSetTexStorage, uintptr(storageModePrivate))
		tex.id = d.id.Send(selNewTexture, desc)
	})
	if tex.id == 0 {
		t.Fatal("the device refused a Depth32Float_Stencil8 texture")
	}
	defer tex.Close()

	toTex := objc.RegisterName(
		"copyFromBuffer:sourceOffset:sourceBytesPerRow:sourceBytesPerImage:sourceSize:" +
			"toTexture:destinationSlice:destinationLevel:destinationOrigin:options:")
	toBuf := objc.RegisterName(
		"copyFromTexture:sourceSlice:sourceLevel:sourceOrigin:sourceSize:" +
			"toBuffer:destinationOffset:destinationBytesPerRow:" +
			"destinationBytesPerImage:options:")
	type origin struct{ X, Y, Z uint64 }
	type size struct{ W, H, D uint64 }

	roundTrip := func(name string, opt uintptr, row int, fill func([]byte)) []byte {
		src, err := d.NewBuffer(row*h, StorageShared)
		if err != nil {
			t.Fatalf("%s src: %v", name, err)
		}
		defer src.Close()
		dst, err := d.NewBuffer(row*h, StorageShared)
		if err != nil {
			t.Fatalf("%s dst: %v", name, err)
		}
		defer dst.Close()
		fill(src.Bytes())

		q := d.NewQueue()
		defer q.Close()
		cb := q.Begin()
		blit := cb.Blit()
		withPool(func() {
			blit.id.Send(toTex,
				src.id, uintptr(0), uintptr(row), uintptr(row*h),
				size{W: w, H: h, D: 1},
				tex.id, uintptr(0), uintptr(0), origin{}, opt)
			blit.id.Send(toBuf,
				tex.id, uintptr(0), uintptr(0), origin{}, size{W: w, H: h, D: 1},
				dst.id, uintptr(0), uintptr(row), uintptr(row*h), opt)
		})
		blit.End()
		cb.Commit()
		cb.Wait()
		out := make([]byte, row*h)
		copy(out, dst.Bytes())
		return out
	}

	t.Run("depth aspect", func(t *testing.T) {
		row := w * 4
		got := roundTrip("depth", 1, row, func(b []byte) {
			for i := range len(b) / 4 {
				binaryPutFloat(b[i*4:], 0.25)
			}
		})
		for i := range len(got) / 4 {
			if f := readFloat(got[i*4:]); f != 0.25 {
				t.Fatalf("texel %d came back %v, want 0.25", i, f)
			}
		}
		t.Log("the depth aspect round-trips at 4 bytes per texel")
	})

	t.Run("stencil aspect", func(t *testing.T) {
		row := w * 1
		got := roundTrip("stencil", 2, row, func(b []byte) {
			for i := range b {
				b[i] = 7
			}
		})
		for i, v := range got {
			if v != 7 {
				t.Fatalf("texel %d came back %d, want 7", i, v)
			}
		}
		t.Log("the stencil aspect round-trips at 1 byte per texel")
	})

	t.Run("no option, interleaved", func(t *testing.T) {
		row := w * 8
		got := roundTrip("both", 0, row, func(b []byte) {
			for i := range len(b) / 8 {
				binaryPutFloat(b[i*8:], 0.5)
				b[i*8+4] = 9
			}
		})
		ok := true
		for i := range len(got) / 8 {
			if readFloat(got[i*8:]) != 0.5 || got[i*8+4] != 9 {
				ok = false
				t.Logf("texel %d came back depth %v stencil %d, want 0.5 and 9",
					i, readFloat(got[i*8:]), got[i*8+4])
				break
			}
		}
		if ok {
			t.Error("an interleaved no-option copy round-tripped. That contradicts the " +
				"planar layout specs/045-texture-attachments.md section 12 chose, and " +
				"the choice should be revisited rather than left standing on a " +
				"measurement that no longer holds")
		}
	})
}

func binaryPutFloat(b []byte, f float32) {
	u := math.Float32bits(f)
	b[0], b[1], b[2], b[3] = byte(u), byte(u>>8), byte(u>>16), byte(u>>24)
}

func readFloat(b []byte) float32 {
	return math.Float32frombits(uint32(b[0]) | uint32(b[1])<<8 |
		uint32(b[2])<<16 | uint32(b[3])<<24)
}
