// Copyright 2026 The golang.design Initiative Authors.
// All rights reserved. Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

package accel_test

import (
	"errors"
	"strings"
	"testing"
	"time"

	"golang.design/x/accel"
	"golang.design/x/accel/internal/testkernels"
)

func newSurface(t *testing.T, d *accel.Device, w, h, images int) *accel.Surface {
	t.Helper()
	s, err := d.NewHeadlessSurface(accel.SurfaceDescriptor{
		Width: w, Height: h, Images: images, Label: "test surface",
	})
	if err != nil {
		t.Fatalf("NewHeadlessSurface: %v", err)
	}
	t.Cleanup(func() { _ = s.Close() })
	return s
}

// recordFrameGraph builds the graph the loop replays: one pass writing the
// present slot, tinted by a per-frame uniform.
func recordFrameGraph(t *testing.T, d *accel.Device, s *accel.Surface, w, h int) (*accel.Graph, accel.Slot, *accel.Buffer) {
	t.Helper()
	q := d.Queue()
	pos := []float32{-1, -1, 0, 3, -1, 0, -1, 3, 0}
	vb := newBuffer(t, d, "pos", len(pos), accel.BufferStorage|accel.BufferCopyDst)
	if err := q.WriteBuffer(vb, 0, pos); err != nil {
		t.Fatalf("write: %v", err)
	}

	r := d.NewRecorder()
	swap := r.PresentSlot(s, "swapchain")
	pass := r.RenderPass(accel.RenderPassDescriptor{
		Color: []accel.ColorAttachment{{
			Slot: swap, Load: accel.LoadClear,
		}},
		Width: w, Height: h, Label: "frame",
	})
	pass.SetPipeline(blendPipeline(t, d, accel.BlendState{}))
	pass.SetVertexBuffer(0, whole(t, vb))
	pass.SetVertexUniform(0, testkernels.StageTransform{Scale: 1})
	pass.SetFragmentUniform(0, testkernels.StageTint{Colour: [4]float32{1, 0, 0, 1}})
	pass.Draw(accel.Draw{VertexCount: 3})

	g, err := r.Build()
	if err != nil {
		t.Fatalf("build: %v", err)
	}
	t.Cleanup(func() { _ = g.Close() })
	return g, swap, vb
}

// specs/034-surface-present.md section 8: the headless frame loop runs several
// frames with double buffering and a resize in the middle, verified by readback,
// with no display.
//
// The loop here is character-for-character the one in section 1, which is the
// claim the headless surface exists to support: acquire, bind, submit after the
// acquire fence, present. A loop that skipped SubmitAfter for headless would be
// testing a different loop from the one a windowed surface runs.
func TestTheHeadlessFrameLoop(t *testing.T) {
	const w, h = 8, 8
	d := openDevice(t)
	q := d.Queue()
	s := newSurface(t, d, w, h, 2)
	g, swap, _ := recordFrameGraph(t, d, s, w, h)

	seen := map[int]bool{}
	for frame := range 4 {
		f, err := s.Acquire(time.Second)
		if err != nil {
			t.Fatalf("frame %d: acquire: %v", frame, err)
		}
		if err := g.BindPresent(swap, f); err != nil {
			t.Fatalf("frame %d: BindPresent: %v", frame, err)
		}
		fence := q.SubmitAfter(g, f.Acquired)
		if err := fence.Wait(); err != nil {
			t.Fatalf("frame %d: submit: %v", frame, err)
		}

		// Readback before present, which is what "presenting" means with no
		// compositor: the pixels are available.
		out := make([]float32, w*h*4)
		if err := q.ReadBuffer(f.View().Buffer, f.View().Offset, out); err != nil {
			t.Fatalf("frame %d: readback: %v", frame, err)
		}
		if out[0] != 1 || out[1] != 0 || out[2] != 0 {
			t.Fatalf("frame %d: pixel (0,0) is %v, want the drawn red", frame, out[:4])
		}
		seen[f.Index()] = true

		if err := s.Present(f, fence); err != nil {
			t.Fatalf("frame %d: present: %v", frame, err)
		}
	}

	// Double buffering means the images rotate. One image reused every frame
	// would pass every assertion above and would not be double buffering.
	if len(seen) != 2 {
		t.Errorf("the loop used %d distinct images over four frames, want 2", len(seen))
	}
}

// A graph built against a surface renders what the same graph renders into an
// ordinary offscreen target.
//
// The present slot is a slot, so this is really the claim that nothing about
// presenting changes what is drawn — and it is worth stating because the slot
// carries four extra checks that a plain slot does not.
func TestASurfaceFrameMatchesAnOffscreenTarget(t *testing.T) {
	const w, h = 8, 8
	d := openDevice(t)
	q := d.Queue()

	s := newSurface(t, d, w, h, 1)
	g, swap, _ := recordFrameGraph(t, d, s, w, h)
	f, err := s.Acquire(time.Second)
	if err != nil {
		t.Fatalf("acquire: %v", err)
	}
	if err := g.BindPresent(swap, f); err != nil {
		t.Fatalf("BindPresent: %v", err)
	}
	if err := q.SubmitAfter(g, f.Acquired).Wait(); err != nil {
		t.Fatalf("submit: %v", err)
	}
	viaSurface := make([]float32, w*h*4)
	if err := q.ReadBuffer(f.View().Buffer, f.View().Offset, viaSurface); err != nil {
		t.Fatalf("readback: %v", err)
	}

	// The same graph, bound to an ordinary buffer instead.
	offscreen := newBuffer(t, d, "offscreen", w*h*4,
		accel.BufferStorage|accel.BufferCopySrc|accel.BufferCopyDst)
	if err := g.Bind(accel.SlotBinding{Slot: swap, Buffer: whole(t, offscreen)}); err != nil {
		t.Fatalf("bind: %v", err)
	}
	if err := q.Submit(g).Wait(); err != nil {
		t.Fatalf("submit offscreen: %v", err)
	}
	viaBuffer := readback(t, d, offscreen)

	for i := range viaSurface {
		if viaSurface[i] != viaBuffer[i] {
			t.Fatalf("float %d is %v through the surface and %v offscreen", i,
				viaSurface[i], viaBuffer[i])
		}
	}
}

// Every rejection BindPresent owns, and the third is the one a format check
// alone accepts.
func TestBindPresentRejections(t *testing.T) {
	const w, h = 8, 8
	d := openDevice(t)

	t.Run("a frame from another surface", func(t *testing.T) {
		s := newSurface(t, d, w, h, 1)
		other := newSurface(t, d, w, h, 1)
		g, swap, _ := recordFrameGraph(t, d, s, w, h)
		f, err := other.Acquire(time.Second)
		if err != nil {
			t.Fatalf("acquire: %v", err)
		}
		err = g.BindPresent(swap, f)
		if err == nil {
			t.Fatal("a frame from another surface was accepted")
		}
		if !strings.Contains(err.Error(), "still the wrong image") {
			t.Errorf("got %v", err)
		}
	})

	t.Run("a frame from an earlier generation", func(t *testing.T) {
		s := newSurface(t, d, w, h, 1)
		g, swap, _ := recordFrameGraph(t, d, s, w, h)
		f, err := s.Acquire(time.Second)
		if err != nil {
			t.Fatalf("acquire: %v", err)
		}
		// Resize to the same extent, which is the case an extent check misses.
		if err := s.Resize(w, h); err != nil {
			t.Fatalf("resize: %v", err)
		}
		newFrame, err := s.Acquire(time.Second)
		if err != nil {
			t.Fatalf("acquire after resize: %v", err)
		}
		_ = f
		err = g.BindPresent(swap, newFrame)
		if err == nil {
			t.Fatal("a frame from a later generation bound to a stale graph")
		}
		if !strings.Contains(err.Error(), "generation 1") ||
			!strings.Contains(err.Error(), "generation 0") {
			t.Errorf("the error should name both generations, got %v", err)
		}
	})

	t.Run("an ordinary slot is not a present slot", func(t *testing.T) {
		s := newSurface(t, d, w, h, 1)
		r := d.NewRecorder()
		plain := r.Slot(accel.SlotDescriptor{
			Name: "plain", Kind: accel.BindingStorageBuffer,
			DType: accel.F32, Access: accel.AccessWrite, MinCount: 4,
		})
		out := newBuffer(t, d, "out", 4, accel.BufferStorage|accel.BufferCopyDst)
		r.UploadToSlot(plain, 0, 4, make([]float32, 4))
		_ = out
		g, err := r.Build()
		if err != nil {
			t.Fatalf("build: %v", err)
		}
		defer g.Close()
		f, err := s.Acquire(time.Second)
		if err != nil {
			t.Fatalf("acquire: %v", err)
		}
		err = g.BindPresent(plain, f)
		if err == nil {
			t.Fatal("an ordinary slot accepted a frame")
		}
		if !strings.Contains(err.Error(), "is not a present slot") {
			t.Errorf("got %v", err)
		}
	})
}

// A resize increments the generation and reconfigures the images.
//
// specs/034-surface-present.md section 4.1: attachment extents stay validated at
// build, so a resize forces a rebuild. The generation is what makes the stale
// graph fail loudly at bind rather than quietly at the wrong size.
func TestAResizeIncrementsTheGeneration(t *testing.T) {
	d := openDevice(t)
	s := newSurface(t, d, 8, 8, 2)
	if got := s.Generation(); got != 0 {
		t.Errorf("a new surface is at generation %d, want 0", got)
	}
	if err := s.Resize(16, 4); err != nil {
		t.Fatalf("resize: %v", err)
	}
	if got := s.Generation(); got != 1 {
		t.Errorf("after one resize the generation is %d, want 1", got)
	}
	if w, h := s.Extent(); w != 16 || h != 4 {
		t.Errorf("the extent is %dx%d, want 16x4", w, h)
	}
	f, err := s.Acquire(time.Second)
	if err != nil {
		t.Fatalf("acquire: %v", err)
	}
	if got := f.View().Count; got != 16*4*4 {
		t.Errorf("the image holds %d elements, want %d; the resize did not reallocate",
			got, 16*4*4)
	}
}

// An acquire that races a reconfiguration reports ErrSurfaceOutOfDate rather
// than reallocating underneath a graph built against the old extent.
func TestAnOutOfDateSurfaceReportsRatherThanReallocating(t *testing.T) {
	d := openDevice(t)
	s := newSurface(t, d, 8, 8, 2)
	s.Invalidate()

	_, err := s.Acquire(time.Second)
	if !errors.Is(err, accel.ErrSurfaceOutOfDate) {
		t.Fatalf("acquire on an out-of-date surface gave %v, want ErrSurfaceOutOfDate", err)
	}
	// And the recovery is the one section 1 writes: resize, rebuild, continue.
	if err := s.Resize(8, 8); err != nil {
		t.Fatalf("resize: %v", err)
	}
	if _, err := s.Acquire(time.Second); err != nil {
		t.Fatalf("acquire after the resize: %v", err)
	}
}

// Acquiring more images than the surface has reports expiry rather than
// blocking forever.
//
// A call the API described as non-blocking that waits on a compositor is worse
// than one that says so — and with every image held by the caller there is
// nothing to wait for at all.
func TestAcquireReportsExpiryRatherThanBlocking(t *testing.T) {
	d := openDevice(t)
	s := newSurface(t, d, 4, 4, 1)

	first, err := s.Acquire(time.Second)
	if err != nil {
		t.Fatalf("acquire: %v", err)
	}
	start := time.Now()
	_, err = s.Acquire(0)
	if !errors.Is(err, accel.ErrAcquireTimeout) {
		t.Fatalf("the second acquire gave %v, want ErrAcquireTimeout", err)
	}
	if elapsed := time.Since(start); elapsed > 100*time.Millisecond {
		t.Errorf("a zero timeout took %v, so it blocked", elapsed)
	}
	if err := s.Present(first, nil); err != nil {
		t.Fatalf("present: %v", err)
	}
	if _, err := s.Acquire(time.Second); err != nil {
		t.Fatalf("after presenting, acquire gave %v", err)
	}
}

// Presenting a frame twice is an error, because the second present hands the
// compositor an image the loop has already given up.
func TestAFramePresentsOnce(t *testing.T) {
	d := openDevice(t)
	s := newSurface(t, d, 4, 4, 2)
	f, err := s.Acquire(time.Second)
	if err != nil {
		t.Fatalf("acquire: %v", err)
	}
	if err := s.Present(f, nil); err != nil {
		t.Fatalf("present: %v", err)
	}
	if err := s.Present(f, nil); err == nil {
		t.Fatal("a frame was presented twice")
	}
}

// A slot too small for the render area is refused, and refused at the
// declaration rather than at execution.
//
// The build check was written as "check the size unless it is a slot" when
// slots became attachable, which is the exemption shape
// specs/009-sequencing.md records twice. It turns out the slot access
// declaration catches this first, with a better message because it is closer to
// the call -- so the build check is defence rather than the only guard. This
// asserts the message a caller actually sees.
func TestAnUndersizedSlotAttachmentIsRefused(t *testing.T) {
	d := openDevice(t)
	r := d.NewRecorder()
	small := r.Slot(accel.SlotDescriptor{
		Name: "too small", Kind: accel.BindingStorageBuffer,
		DType: accel.F32, Access: accel.AccessWrite, MinCount: 4,
	})
	p := r.RenderPass(accel.RenderPassDescriptor{
		Color: []accel.ColorAttachment{{Slot: small, Load: accel.LoadClear}},
		Width: 64, Height: 64, Label: "undersized",
	})
	p.SetPipeline(solidPipeline(t, d))
	p.Draw(accel.Draw{VertexCount: 3})

	g, err := r.Build()
	if err == nil {
		_ = g.Close()
		t.Fatal("a slot far too small for the render area was accepted")
	}
	if !strings.Contains(err.Error(), `slot "too small" is declared to hold at least`) {
		t.Errorf("the error should compare what the slot promises with what the pass "+
			"uses, got %v", err)
	}
}

// An attachment names one resource or one slot, and neither or both is refused.
func TestAnAttachmentNamesExactlyOneResource(t *testing.T) {
	d := openDevice(t)
	target := newBuffer(t, d, "target", 4*4*4,
		accel.BufferStorage|accel.BufferCopySrc|accel.BufferCopyDst)

	for _, c := range []struct {
		name string
		att  func(r *accel.Recorder) accel.ColorAttachment
		says string
	}{{
		name: "neither",
		att:  func(*accel.Recorder) accel.ColorAttachment { return accel.ColorAttachment{} },
		says: "names no resource",
	}, {
		name: "both",
		att: func(r *accel.Recorder) accel.ColorAttachment {
			s := r.Slot(accel.SlotDescriptor{
				Name: "s", Kind: accel.BindingStorageBuffer, DType: accel.F32,
				Access: accel.AccessWrite, MinCount: 64,
			})
			return accel.ColorAttachment{View: whole(t, target), Slot: s}
		},
		says: "names both a resource and a slot",
	}} {
		t.Run(c.name, func(t *testing.T) {
			r := d.NewRecorder()
			p := r.RenderPass(accel.RenderPassDescriptor{
				Color: []accel.ColorAttachment{c.att(r)},
				Width: 4, Height: 4, Label: "ambiguous",
			})
			p.SetPipeline(solidPipeline(t, d))
			p.Draw(accel.Draw{VertexCount: 3})
			g, err := r.Build()
			if err == nil {
				_ = g.Close()
				t.Fatal("an ambiguous attachment was accepted")
			}
			if !strings.Contains(err.Error(), c.says) {
				t.Errorf("the error should say %q, got %v", c.says, err)
			}
		})
	}
}
