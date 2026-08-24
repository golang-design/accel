// Copyright 2026 The golang.design Initiative Authors.
// All rights reserved. Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

//go:build darwin

package testkernels_test

import (
	"errors"
	"strings"
	"testing"
	"time"

	"golang.design/x/accel"
	"golang.design/x/accel/internal/mtl"
	"golang.design/x/accel/internal/testkernels"
)

// The frame loop against a real CAMetalLayer.
//
// # What this proves and what it does not
//
// specs/034-surface-present.md section 7 says two claims are never reported as
// one. Measured, there are three, and this covers the middle one:
//
//  1. headless render — a frame renders and the pixels come back;
//  2. the drawable path — a CAMetalLayer hands out a drawable, a command buffer
//     renders into it and presents it on that same command buffer, and the
//     drawable goes back;
//  3. the compositor handoff — a drawable reaching a screen, with the bounded
//     pool, the blocking acquire and the vsync that come with it.
//
// The layer here is attached to no window. That was measured rather than
// assumed: an unattached layer hands out eight drawables without presenting
// any, and setting maximumDrawableCount does not bound it — so it exercises
// every part of the drawable *lifetime* and none of the pool's back pressure.
// Claim 3 needs a display session and is not made here.
func TestTheFrameLoopAgainstAMetalLayer(t *testing.T) {
	const w, h = 16, 16
	d := openMetalDevice(t)

	layer, err := mtl.NewOffscreenLayer()
	if err != nil {
		t.Skipf("no CAMetalLayer on this machine: %v", err)
	}
	defer layer.Close()

	s, err := d.NewWindowSurface(accel.NativeHandle{
		Kind: accel.NativeMetalLayer, Ptr: layer.Pointer(),
	}, accel.SurfaceDescriptor{Width: w, Height: h, Images: 2, Label: "layer"})
	if err != nil {
		t.Fatalf("NewWindowSurface: %v", err)
	}
	defer s.Close()

	g, swap := presentGraph(t, d, s, w, h)

	// Several frames, because a drawable held across frames exhausts the pool
	// and the symptom is a loop that stops rather than an error. Four is more
	// than any pool size, so a leak appears here as a hang the test timeout
	// catches rather than as a wrong pixel.
	for frame := range 4 {
		f, err := s.Acquire(2 * time.Second)
		if err != nil {
			t.Fatalf("frame %d: acquire: %v", frame, err)
		}
		if err := g.BindPresent(swap, f); err != nil {
			t.Fatalf("frame %d: BindPresent: %v", frame, err)
		}
		fence := d.Queue().SubmitAfter(g, f.Acquired)
		if err := fence.Wait(); err != nil {
			t.Fatalf("frame %d: submit: %v", frame, err)
		}

		// What the graph rendered is still readable: presenting converts a copy
		// into the drawable and does not consume the frame's buffer.
		out := make([]float32, w*h*4)
		if err := d.Queue().ReadBuffer(f.View().Buffer, f.View().Offset, out); err != nil {
			t.Fatalf("frame %d: readback: %v", frame, err)
		}
		if out[0] != 0.25 {
			t.Fatalf("frame %d: pixel (0,0) is %v, want what the pass drew",
				frame, out[:4])
		}

		if err := s.Present(f, fence); err != nil {
			t.Fatalf("frame %d: present: %v", frame, err)
		}
	}
}

// presentGraph records the frame graph the loop replays: one pass filling the
// present slot.
func presentGraph(t *testing.T, d *accel.Device, s *accel.Surface, w, h int) (*accel.Graph, accel.Slot) {
	t.Helper()
	q := d.Queue()
	verts, err := d.NewBuffer(accel.BufferDescriptor{
		DType: accel.F32, Count: 9,
		Usage: accel.BufferStorage | accel.BufferCopyDst, Label: "verts",
	})
	if err != nil {
		t.Fatalf("buffer: %v", err)
	}
	t.Cleanup(func() { _ = verts.Close() })
	if err := q.WriteBuffer(verts, 0, []float32{-1, -1, 0, 3, -1, 0, -1, 3, 0}); err != nil {
		t.Fatalf("write: %v", err)
	}
	vv, err := verts.View(0, verts.Count())
	if err != nil {
		t.Fatalf("view: %v", err)
	}

	pipe, err := d.NewRenderPipeline(accel.RenderPipelineDescriptor{
		Vertex:        &testkernels.ScaledVSStage,
		Fragment:      &testkernels.TintedFSStage,
		VertexBuffers: posOnly(),
		Targets:       []accel.ColorTargetState{{Format: accel.RGBA32Float}},
		Label:         "present",
	})
	if err != nil {
		t.Fatalf("pipeline: %v", err)
	}
	t.Cleanup(func() { _ = pipe.Close() })

	r := d.NewRecorder()
	swap := r.PresentSlot(s, "swapchain")
	p := r.RenderPass(accel.RenderPassDescriptor{
		Color: []accel.ColorAttachment{{Slot: swap, Load: accel.LoadClear}},
		Width: w, Height: h, Label: "frame",
	})
	p.SetPipeline(pipe)
	p.SetVertexBuffer(0, vv)
	p.SetVertexUniform(0, testkernels.StageTransform{Scale: 1})
	p.SetFragmentUniform(0, testkernels.StageTint{Colour: accel.Vec4{0.25, 0.5, 0.75, 1}})
	p.Draw(accel.Draw{VertexCount: 3})

	g, err := r.Build()
	if err != nil {
		t.Fatalf("build: %v", err)
	}
	t.Cleanup(func() { _ = g.Close() })
	return g, swap
}

// A device with no on-screen path reports it rather than failing later.
//
// Decision 6: absence is reported, not discovered. The CPU backend has no
// drawable and never will, so a caller asking for a window surface is told
// immediately instead of getting one that never shows anything.
func TestTheCPUBackendReportsNoPresentPath(t *testing.T) {
	cpu, err := accel.OpenCPU(accel.CPUOptions{})
	if err != nil {
		t.Fatalf("OpenCPU: %v", err)
	}
	defer cpu.Close()

	_, err = cpu.NewWindowSurface(accel.NativeHandle{Kind: accel.NativeMetalLayer, Ptr: 1},
		accel.SurfaceDescriptor{Width: 8, Height: 8})
	if !errors.Is(err, accel.ErrNoPresent) {
		t.Errorf("the CPU backend gave %v, want ErrNoPresent", err)
	}
}

// A handle naming nothing is refused before any drawable is asked for.
func TestAWindowSurfaceRefusesTheWrongHandle(t *testing.T) {
	d := openMetalDevice(t)
	for _, c := range []struct {
		name string
		h    accel.NativeHandle
	}{
		{"no handle", accel.NativeHandle{}},
		{"a nil layer", accel.NativeHandle{Kind: accel.NativeMetalLayer}},
	} {
		t.Run(c.name, func(t *testing.T) {
			if _, err := d.NewWindowSurface(c.h, accel.SurfaceDescriptor{
				Width: 8, Height: 8,
			}); err == nil {
				t.Error("a handle that names nothing was accepted")
			}
		})
	}
}

// The rest of the windowed surface's state machine.
//
// Each of these is a place the drawable can be lost, and losing one is a frame
// loop that stops rather than an error: the pool empties and the next acquire
// waits for an image nobody returned.
func TestAWindowSurfaceReturnsEveryDrawable(t *testing.T) {
	const w, h = 8, 8
	d := openMetalDevice(t)
	layer, err := mtl.NewOffscreenLayer()
	if err != nil {
		t.Skipf("no CAMetalLayer: %v", err)
	}
	defer layer.Close()

	s, err := d.NewWindowSurface(accel.NativeHandle{
		Kind: accel.NativeMetalLayer, Ptr: layer.Pointer(),
	}, accel.SurfaceDescriptor{Width: w, Height: h, Images: 2, Label: "returns"})
	if err != nil {
		t.Fatalf("NewWindowSurface: %v", err)
	}
	defer s.Close()

	t.Run("an abandoned frame is discarded", func(t *testing.T) {
		f, err := s.Acquire(time.Second)
		if err != nil {
			t.Fatalf("acquire: %v", err)
		}
		if err := s.Discard(f); err != nil {
			t.Fatalf("discard: %v", err)
		}
		// And it is spent: neither presenting nor discarding it again is
		// allowed, because the drawable behind it is already gone.
		if err := s.Discard(f); err == nil {
			t.Error("a frame was discarded twice")
		}
		if err := s.Present(f, nil); err == nil {
			t.Error("a discarded frame was presented")
		}
	})

	t.Run("a resize reconfigures the layer", func(t *testing.T) {
		if err := s.Resize(w*2, h); err != nil {
			t.Fatalf("resize: %v", err)
		}
		if gw, gh := s.Extent(); gw != w*2 || gh != h {
			t.Errorf("the extent is %dx%d, want %dx%d", gw, gh, w*2, h)
		}
		f, err := s.Acquire(time.Second)
		if err != nil {
			t.Fatalf("acquire after resize: %v", err)
		}
		if got := f.View().Count; got != w*2*h*4 {
			t.Errorf("the image holds %d elements, want %d", got, w*2*h*4)
		}
		if err := s.Discard(f); err != nil {
			t.Fatalf("discard: %v", err)
		}
	})

	t.Run("closing twice is not an error", func(t *testing.T) {
		other, err := d.NewWindowSurface(accel.NativeHandle{
			Kind: accel.NativeMetalLayer, Ptr: layer.Pointer(),
		}, accel.SurfaceDescriptor{Width: 4, Height: 4, Label: "twice"})
		if err != nil {
			t.Fatalf("NewWindowSurface: %v", err)
		}
		if err := other.Close(); err != nil {
			t.Fatalf("close: %v", err)
		}
		if err := other.Close(); err != nil {
			t.Errorf("closing twice gave %v", err)
		}
		if _, err := other.Acquire(time.Second); err == nil {
			t.Error("a closed surface handed out a frame")
		}
	})

	t.Run("an NSView handle says what to pass instead", func(t *testing.T) {
		_, err := d.NewWindowSurface(accel.NativeHandle{
			Kind: accel.NativeNSView, Ptr: 1,
		}, accel.SurfaceDescriptor{Width: 4, Height: 4})
		if err == nil {
			t.Fatal("an NSView handle was accepted")
		}
		if !strings.Contains(err.Error(), "CAMetalLayer") {
			t.Errorf("the error should name what to pass instead, got %v", err)
		}
	})
}
