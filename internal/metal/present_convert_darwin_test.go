// Copyright 2026 The golang.design Initiative Authors.
// All rights reserved. Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

//go:build darwin

package metal

import (
	"math"
	"os"
	"testing"
	"time"

	"golang.design/x/accel/internal/mtl"
)

// The present conversion, checked against the bytes it produces.
//
// This is the part of the on-screen path with a real chance of being wrong in a
// way nothing else notices. A drawable is BGRA8Unorm and the render path works
// in RGBA32Float, so presenting converts — and a reversed swizzle produces a
// perfectly plausible image, as does an unclamped value that wraps. Both would
// pass every test that only checks the calls succeeded.
//
// Done against a texture this test owns rather than against a drawable, because
// a drawable is released the moment it is presented and reading one back would
// mean not presenting it — which is the half of the path worth exercising.
func TestThePresentConversionWritesTheRightBytes(t *testing.T) {
	devs, err := mtl.Devices()
	if err != nil || len(devs) == 0 {
		if os.Getenv("ACCEL_REQUIRE_METAL") != "" {
			t.Fatalf("this job promises Metal and found no device (err=%v)", err)
		}
		t.Skipf("no Metal device (err=%v)", err)
	}
	d := devs[0]
	defer func() {
		for _, x := range devs {
			x.Close()
		}
	}()

	const w, h = 4, 2
	var pipe presentPipeline
	defer pipe.close()
	state, err := pipe.get(d)
	if err != nil {
		t.Fatalf("compiling the conversion: %v", err)
	}

	// One distinguishable colour per pixel, plus a value above one and a value
	// below zero, so the clamp is asserted rather than assumed.
	src := make([]float32, w*h*4)
	want := make([][4]byte, w*h)
	set := func(i int, r, g, b, a float32, wb [4]byte) {
		src[i*4+0], src[i*4+1], src[i*4+2], src[i*4+3] = r, g, b, a
		want[i] = wb
	}
	// Metal rounds a normalized store to nearest, so 0.25 -> 64 and 0.5 -> 128.
	set(0, 0.25, 0.5, 0.75, 1, [4]byte{191, 128, 64, 255}) // stored B,G,R,A
	set(1, 1, 0, 0, 1, [4]byte{0, 0, 255, 255})
	set(2, 0, 1, 0, 1, [4]byte{0, 255, 0, 255})
	set(3, 0, 0, 1, 1, [4]byte{255, 0, 0, 255})
	set(4, 2, -1, 0.5, 1, [4]byte{128, 0, 255, 255}) // clamped both ways
	set(5, 0, 0, 0, 0, [4]byte{0, 0, 0, 0})
	set(6, 1, 1, 1, 1, [4]byte{255, 255, 255, 255})
	set(7, 0.5, 0.5, 0.5, 0.5, [4]byte{128, 128, 128, 128})

	buf, err := d.NewBuffer(len(src)*4, mtl.StorageShared)
	if err != nil {
		t.Fatalf("buffer: %v", err)
	}
	defer buf.Close()
	copy(buf.Bytes(), asBytes(src))

	tex, err := d.NewRenderTarget(mtl.PixelFormatBGRA8Unorm, w, h)
	if err != nil {
		t.Fatalf("target: %v", err)
	}
	defer tex.Close()
	out, err := d.NewBuffer(w*h*4, mtl.StorageShared)
	if err != nil {
		t.Fatalf("buffer: %v", err)
	}
	defer out.Close()

	q := d.NewQueue()
	defer q.Close()
	cb := q.Begin()
	enc, err := cb.Render([]mtl.RenderAttachment{{
		Texture: tex, Load: mtl.LoadActionDontCare, Store: mtl.StoreActionStore,
	}}, nil)
	if err != nil {
		t.Fatalf("encoder: %v", err)
	}
	enc.SetPipeline(state)
	enc.SetFragmentBuffer(buf, 0, 0)
	enc.SetFragmentBytes(presentDims(w, h), 1)
	enc.Draw(3, 0, 3, 1, 0)
	enc.End()
	blit := cb.Blit()
	blit.CopyTextureToBuffer(tex, out, 0)
	blit.End()
	cb.Commit()
	cb.Wait()

	got := out.Bytes()
	for i := range want {
		var px [4]byte
		copy(px[:], got[i*4:])
		if px != want[i] {
			t.Errorf("pixel %d is %v and should be %v; the source was %v — a reversed "+
				"swizzle or a missing clamp both produce a plausible image",
				i, px, want[i], src[i*4:i*4+4])
		}
	}
}

// asBytes reinterprets a float slice for the staging copy.
func asBytes(f []float32) []byte {
	out := make([]byte, len(f)*4)
	for i, v := range f {
		b := math.Float32bits(v)
		out[i*4+0] = byte(b)
		out[i*4+1] = byte(b >> 8)
		out[i*4+2] = byte(b >> 16)
		out[i*4+3] = byte(b >> 24)
	}
	return out
}

// The present target's own state machine, at the level that owns it.
//
// Through the internal type rather than through accel.Surface, because these
// are the transitions the layer above cannot reach once it has done its own
// checks — and each of them is a drawable that would otherwise be lost or a
// call that would reach Core Animation after the target is gone.
func TestThePresentTargetStateMachine(t *testing.T) {
	devs, err := mtl.Devices()
	if err != nil || len(devs) == 0 {
		if os.Getenv("ACCEL_REQUIRE_METAL") != "" {
			t.Fatalf("this job promises Metal and found no device (err=%v)", err)
		}
		t.Skipf("no Metal device (err=%v)", err)
	}
	md := devs[0]
	defer func() {
		for _, x := range devs {
			x.Close()
		}
	}()

	layer, err := mtl.NewOffscreenLayer()
	if err != nil {
		t.Skipf("no CAMetalLayer: %v", err)
	}
	defer layer.Close()

	d := &device{dev: md, queue: md.NewQueue()}
	defer d.queue.Close()
	target := &presentTarget{dev: d, layer: layer}
	if err := target.Configure(8, 8); err != nil {
		t.Fatalf("configure: %v", err)
	}

	t.Run("a zero extent is refused", func(t *testing.T) {
		if err := target.Configure(0, 8); err == nil {
			t.Error("a zero-width drawable size was accepted")
		}
	})

	t.Run("an image is spent after Discard", func(t *testing.T) {
		img, err := target.Acquire(time.Second)
		if err != nil {
			t.Fatalf("acquire: %v", err)
		}
		img.Discard()
		// Discarding twice must not release the drawable twice: the second
		// release is of an object the layer has already taken back, which is
		// the crash the ownership rule in objc_darwin.go describes.
		img.Discard()
		if err := img.Present(nil, 0); err == nil {
			t.Error("a discarded image was presented")
		}
	})

	t.Run("a frame that is not Metal memory is refused", func(t *testing.T) {
		img, err := target.Acquire(time.Second)
		if err != nil {
			t.Fatalf("acquire: %v", err)
		}
		defer img.Discard()
		if err := img.Present(nil, 0); err == nil {
			t.Error("a nil block was presented")
		}
	})

	t.Run("a closed target hands out nothing", func(t *testing.T) {
		other := &presentTarget{dev: d, layer: layer}
		if err := other.Configure(8, 8); err != nil {
			t.Fatalf("configure: %v", err)
		}
		if err := other.Close(); err != nil {
			t.Fatalf("close: %v", err)
		}
		// Twice is not an error, because a surface closing after a device has
		// already torn down its targets is ordinary.
		if err := other.Close(); err != nil {
			t.Errorf("closing twice gave %v", err)
		}
		if _, err := other.Acquire(time.Second); err == nil {
			t.Error("a closed target handed out a drawable")
		}
		if err := other.Configure(4, 4); err == nil {
			t.Error("a closed target was reconfigured")
		}
	})
}
