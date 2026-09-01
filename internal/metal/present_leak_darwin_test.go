// Copyright 2026 The golang.design Initiative Authors.
// All rights reserved. Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

//go:build darwin

package metal

import (
	"os"
	"testing"
	"time"

	"golang.design/x/accel/internal/mtl"
)

// A presented frame releases its command buffer, and so does a frame whose
// pass could not be encoded.
//
// Present began a command buffer per frame and closed it on neither path, so a
// frame loop leaked one retained MTLCommandBuffer per frame. The count is the
// only witness: Metal does not report a retained object and Go's heap does not
// hold it.
func TestPresentReleasesItsCommandBuffer(t *testing.T) {
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

	const w, h = 8, 8
	d := &device{dev: md, queue: md.NewQueue()}
	defer d.queue.Close()
	target := &presentTarget{dev: d, layer: layer}
	if err := target.Configure(w, h); err != nil {
		t.Fatalf("configure: %v", err)
	}
	defer target.pipe.close()
	buf, err := md.NewBuffer(w*h*16, mtl.StorageShared)
	if err != nil {
		t.Fatalf("buffer: %v", err)
	}
	defer buf.Close()
	src := &block{dev: d, buf: buf}

	before := mtl.LiveCommandBuffers()
	const frames = 20
	for range frames {
		img, err := target.Acquire(time.Second)
		if err != nil {
			t.Fatalf("acquire: %v", err)
		}
		if err := img.Present(src, 0); err != nil {
			t.Fatalf("present: %v", err)
		}
	}
	if held := mtl.LiveCommandBuffers() - before; held != 0 {
		t.Fatalf("%d presented frames left %d command buffers retained", frames, held)
	}

	// The failure path: a drawable with no texture makes the render pass
	// refuse, and the buffer begun for it has to go the same way.
	img := &presentImage{target: target, drawable: &mtl.Drawable{}, w: w, h: h}
	if err := img.Present(src, 0); err == nil {
		t.Fatal("a frame with no drawable texture was presented")
	}
	if held := mtl.LiveCommandBuffers() - before; held != 0 {
		t.Fatalf("a frame that failed to encode left %d command buffers retained", held)
	}
}
