// Copyright 2026 The golang.design Initiative Authors.
// All rights reserved. Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

package accel_test

import (
	"errors"
	"strings"
	"sync"
	"testing"
	"time"

	"golang.design/x/accel"
	"golang.design/x/accel/internal/kernels"
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
	pass.SetVertexUniform(0, kernels.StageTransform{Scale: 1})
	pass.SetFragmentUniform(0, kernels.StageTint{Colour: [4]float32{1, 0, 0, 1}})
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

	// One frame of the loop, at whatever extent the surface has now. The last
	// pixel is checked as well as the first, so a frame drawn at the old
	// extent into a resized surface is visible as an unwritten corner.
	frame := func(n int, g *accel.Graph, swap accel.Slot, w, h int) *accel.Frame {
		t.Helper()
		f, err := s.Acquire(time.Second)
		if err != nil {
			t.Fatalf("frame %d: acquire: %v", n, err)
		}
		if err := g.BindPresent(swap, f); err != nil {
			t.Fatalf("frame %d: BindPresent: %v", n, err)
		}
		fence := q.SubmitAfter(g, f.Acquired)
		if err := fence.Wait(); err != nil {
			t.Fatalf("frame %d: submit: %v", n, err)
		}

		// Readback before present, which is what "presenting" means with no
		// compositor: the pixels are available.
		out := make([]float32, w*h*4)
		if err := q.ReadBuffer(f.View().Buffer, f.View().Offset, out); err != nil {
			t.Fatalf("frame %d: readback: %v", n, err)
		}
		for _, px := range []int{0, w*h - 1} {
			if c := out[px*4 : px*4+4]; c[0] != 1 || c[1] != 0 || c[2] != 0 {
				t.Fatalf("frame %d: pixel %d of %dx%d is %v, want the drawn red",
					n, px, w, h, c)
			}
		}

		if err := s.Present(f, fence); err != nil {
			t.Fatalf("frame %d: present: %v", n, err)
		}
		return f
	}

	seen := map[int]bool{}
	for n := range 2 {
		seen[frame(n, g, swap, w, h).Index()] = true
	}

	// The resize in the middle. It invalidates the graph built against the
	// old generation, so the loop rebuilds and carries on at the new extent.
	const w2, h2 = 16, 16
	if err := s.Resize(w2, h2); err != nil {
		t.Fatalf("resize: %v", err)
	}
	stale, err := s.Acquire(time.Second)
	if err != nil {
		t.Fatalf("acquire after resize: %v", err)
	}
	if err := g.BindPresent(swap, stale); err == nil {
		t.Fatal("a graph built against the old generation bound a frame of the new one")
	}
	if err := s.Discard(stale); err != nil {
		t.Fatalf("discard: %v", err)
	}
	g2, swap2, _ := recordFrameGraph(t, d, s, w2, h2)
	for n := 2; n < 4; n++ {
		seen[frame(n, g2, swap2, w2, h2).Index()] = true
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

	// The same graph, bound to an ordinary buffer instead. A present slot is a
	// buffer slot -- specs/034-surface-present.md makes the presented image one
	// -- so what stands in for the swapchain image here is a buffer and not a
	// texture, which is also why a slot attachment did not change shape when an
	// attachment became a texture view.
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
		// Discarded before the resize, which is the rule: a resize invalidates
		// every frame acquired before it, and a frame has to go back before it
		// can be invalidated.
		if err := s.Discard(f); err != nil {
			t.Fatalf("discard: %v", err)
		}
		// Resize to the same extent, which is the case an extent check misses.
		if err := s.Resize(w, h); err != nil {
			t.Fatalf("resize: %v", err)
		}
		newFrame, err := s.Acquire(time.Second)
		if err != nil {
			t.Fatalf("acquire after resize: %v", err)
		}
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
	target := colourTarget(t, d, "target", 4, 4)

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
			return accel.ColorAttachment{View: view(t, target), Slot: s}
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

// A device with no on-screen path reports it rather than failing later.
//
// Decision 6: absence is reported, not discovered. The CPU backend has no
// drawable and never will, so a caller asking for a window surface is told at
// the call instead of getting one that never shows anything.
func TestADeviceWithNoPresentPathSaysSo(t *testing.T) {
	d := openDevice(t)
	_, err := d.NewWindowSurface(accel.NativeHandle{
		Kind: accel.NativeMetalLayer, Ptr: 1,
	}, accel.SurfaceDescriptor{Width: 8, Height: 8})
	if !errors.Is(err, accel.ErrNoPresent) {
		t.Errorf("the CPU backend gave %v, want ErrNoPresent", err)
	}
	if err != nil && !strings.Contains(err.Error(), d.Info().Name) {
		t.Errorf("the error should name the device, got %v", err)
	}
}

// A window surface with no extent is refused before a backend is asked.
//
// Before, because the refusal belongs to the descriptor rather than to the
// platform: a zero extent is wrong on every backend, and reporting it as
// "this device has no present path" would send the caller to the wrong place.
func TestAWindowSurfaceNeedsAnExtent(t *testing.T) {
	d := openDevice(t)
	for _, c := range []struct{ w, h int }{{0, 8}, {8, 0}, {-1, -1}} {
		_, err := d.NewWindowSurface(accel.NativeHandle{Kind: accel.NativeMetalLayer, Ptr: 1},
			accel.SurfaceDescriptor{Width: c.w, Height: c.h})
		if err == nil {
			t.Errorf("a %dx%d window surface was accepted", c.w, c.h)
			continue
		}
		if errors.Is(err, accel.ErrNoPresent) {
			t.Errorf("a %dx%d extent was reported as a missing present path, which sends "+
				"the caller to the wrong place: %v", c.w, c.h, err)
		}
	}
}

// A window surface with a negative image count is refused as a headless one
// is, and before a backend is asked, for the reason the extent is.
//
// The headless constructor refused fewer than one image; the windowed one
// went on to make a slice of that length, which is a panic for a negative
// count and a slice of nothing for the value the headless one refuses.
func TestAWindowSurfaceNeedsAnImageCount(t *testing.T) {
	d := openDevice(t)
	for _, images := range []int{-1, -8} {
		_, err := d.NewWindowSurface(accel.NativeHandle{Kind: accel.NativeMetalLayer, Ptr: 1},
			accel.SurfaceDescriptor{Width: 8, Height: 8, Images: images})
		if err == nil {
			t.Errorf("a window surface with %d images was accepted", images)
			continue
		}
		if errors.Is(err, accel.ErrNoPresent) {
			t.Errorf("%d images was reported as a missing present path, which sends "+
				"the caller to the wrong place: %v", images, err)
		}
		if !strings.Contains(err.Error(), "images") {
			t.Errorf("the refusal should say what is wrong: %v", err)
		}
	}
}

// A present slot records one generation's extent with that generation's number.
//
// PresentSlot read the extent and the generation under two lock acquisitions,
// so a resize between them recorded the old extent against the new number. A
// frame of that generation then passed the generation check and failed the
// size check, which reads as a graph too small for a surface it was built
// against. A resizer runs beside the recorder and the invariant is checked
// for every graph: a frame from the recorded generation binds.
func TestPresentSlotRecordsOneGenerationsExtent(t *testing.T) {
	d := openDevice(t)
	s := newSurface(t, d, 16, 16, 2)

	// Several resizers, so that one is nearly always waiting on the surface's
	// lock: a waiter is what turns the gap between two acquisitions into a
	// resize, since a contended mutex hands off to whoever is queued.
	stop := make(chan struct{})
	var resizers sync.WaitGroup
	for n := 0; n < 8; n++ {
		resizers.Add(1)
		go func() {
			defer resizers.Done()
			// Alternating so that a stale extent is the larger one half of
			// the time, which is the half a size check can see.
			big := false
			for {
				select {
				case <-stop:
					return
				default:
				}
				w := 4
				if big {
					w = 16
				}
				big = !big
				// Refused while a frame is outstanding, which is fine: the
				// recorder below acquires and discards between graphs.
				_ = s.Resize(w, w)
			}
		}()
	}
	defer func() { close(stop); resizers.Wait() }()

	for i := 0; i < 500; i++ {
		r := d.NewRecorder()
		slot := r.PresentSlot(s, "swapchain")
		if slot == 0 {
			t.Fatal("PresentSlot failed")
		}
		// One element, which every extent covers, so the graph is built for
		// whatever the slot recorded.
		r.UploadToSlot(slot, 0, 1, []float32{0})
		g, err := r.Build()
		if err != nil {
			t.Fatalf("build %d: %v", i, err)
		}
		f, err := s.Acquire(time.Second)
		if err != nil {
			_ = g.Close()
			if errors.Is(err, accel.ErrSurfaceOutOfDate) {
				continue
			}
			t.Fatalf("acquire %d: %v", i, err)
		}
		err = g.BindPresent(slot, f)
		if errors.Is(err, accel.ErrTooSmall) {
			t.Fatalf("graph %d: the frame is from the recorded generation and the "+
				"recorded extent is another generation's: %v", i, err)
		}
		if err != nil && !strings.Contains(err.Error(), "generation") {
			t.Fatalf("graph %d: BindPresent: %v", i, err)
		}
		_ = s.Discard(f)
		_ = g.Close()
	}
}

// A closed device hands out no surfaces of either kind.
func TestAClosedDeviceMakesNoSurface(t *testing.T) {
	d, err := accel.OpenCPU(accel.CPUOptions{})
	if err != nil {
		t.Fatalf("OpenCPU: %v", err)
	}
	if err := d.Close(); err != nil {
		t.Fatalf("close: %v", err)
	}
	desc := accel.SurfaceDescriptor{Width: 8, Height: 8}
	if _, err := d.NewHeadlessSurface(desc); err == nil {
		t.Error("a closed device made a headless surface")
	}
	if _, err := d.NewWindowSurface(accel.NativeHandle{
		Kind: accel.NativeMetalLayer, Ptr: 1,
	}, desc); err == nil {
		t.Error("a closed device made a window surface")
	}
}

// A headless surface's images rotate, and a discarded frame goes back.
//
// Discard exists for the windowed case, where a frame holds a drawable the
// compositor lent it — but the accounting is the surface's and is the same
// either way, so it is asserted here where no display is needed: a frame
// discarded frees its slot, and one already spent cannot be spent again.
func TestADiscardedFrameGoesBack(t *testing.T) {
	d := openDevice(t)
	s := newSurface(t, d, 4, 4, 1)

	first, err := s.Acquire(time.Second)
	if err != nil {
		t.Fatalf("acquire: %v", err)
	}
	// One image, so nothing else is available until this one goes back.
	if _, err := s.Acquire(0); !errors.Is(err, accel.ErrAcquireTimeout) {
		t.Fatalf("the second acquire gave %v, want ErrAcquireTimeout", err)
	}
	if err := s.Discard(first); err != nil {
		t.Fatalf("discard: %v", err)
	}
	if _, err := s.Acquire(time.Second); err != nil {
		t.Fatalf("after discarding, acquire gave %v", err)
	}

	// Spent means spent, whichever way it was spent.
	if err := s.Discard(first); err == nil {
		t.Error("a frame was discarded twice")
	}
	if err := s.Present(first, nil); err == nil {
		t.Error("a discarded frame was presented")
	}
	if err := s.Discard(nil); err == nil {
		t.Error("a nil frame was discarded")
	}

	other := newSurface(t, d, 4, 4, 1)
	f, err := other.Acquire(time.Second)
	if err != nil {
		t.Fatalf("acquire: %v", err)
	}
	if err := s.Discard(f); err == nil {
		t.Error("a frame from another surface was discarded")
	}
}

// Every refusal the surface constructors and the present slot own.
//
// Portable, and deliberately so: none of these needs a display or a drawable,
// and putting them behind a platform tag would mean the coverage job that runs
// on Linux never reaches the code a Linux caller can still call.
func TestSurfaceAndPresentSlotRefusals(t *testing.T) {
	d := openDevice(t)

	t.Run("a headless surface with no extent", func(t *testing.T) {
		for _, c := range []struct{ w, h int }{{0, 8}, {8, 0}} {
			if _, err := d.NewHeadlessSurface(accel.SurfaceDescriptor{
				Width: c.w, Height: c.h,
			}); err == nil {
				t.Errorf("a %dx%d surface was accepted", c.w, c.h)
			}
		}
	})

	t.Run("a surface with a negative image count", func(t *testing.T) {
		if _, err := d.NewHeadlessSurface(accel.SurfaceDescriptor{
			Width: 8, Height: 8, Images: -1,
		}); err == nil {
			t.Error("a surface with -1 images was accepted")
		}
	})

	t.Run("an unlabelled surface still has a label", func(t *testing.T) {
		s, err := d.NewHeadlessSurface(accel.SurfaceDescriptor{Width: 4, Height: 4})
		if err != nil {
			t.Fatalf("NewHeadlessSurface: %v", err)
		}
		defer s.Close()
		// It appears in every error about the surface, so an empty one would
		// make those errors name nothing.
		if s.Label() == "" {
			t.Error("a surface with no label given has none, and its errors name nothing")
		}
	})

	t.Run("a resize to nothing", func(t *testing.T) {
		s := newSurface(t, d, 8, 8, 1)
		if err := s.Resize(0, 8); err == nil {
			t.Error("a resize to a zero width was accepted")
		}
		if got := s.Generation(); got != 0 {
			t.Errorf("a refused resize left the generation at %d, and a resize that did "+
				"not happen must not invalidate a graph", got)
		}
	})

	t.Run("a closed surface", func(t *testing.T) {
		s, err := d.NewHeadlessSurface(accel.SurfaceDescriptor{Width: 4, Height: 4})
		if err != nil {
			t.Fatalf("NewHeadlessSurface: %v", err)
		}
		if err := s.Close(); err != nil {
			t.Fatalf("close: %v", err)
		}
		if err := s.Close(); err != nil {
			t.Errorf("closing twice gave %v", err)
		}
		if err := s.Resize(8, 8); err == nil {
			t.Error("a closed surface was resized")
		}
		if _, err := s.Acquire(time.Second); err == nil {
			t.Error("a closed surface handed out a frame")
		}
	})

	t.Run("a present slot with no surface", func(t *testing.T) {
		r := d.NewRecorder()
		if got := r.PresentSlot(nil, "none"); got != 0 {
			t.Errorf("PresentSlot with no surface returned slot %d", got)
		}
		if _, err := r.Build(); err == nil {
			t.Error("a recorder that failed a PresentSlot still built")
		}
	})

	t.Run("a present slot for another device's surface", func(t *testing.T) {
		other, err := accel.OpenCPU(accel.CPUOptions{})
		if err != nil {
			t.Fatalf("OpenCPU: %v", err)
		}
		defer other.Close()
		s, err := other.NewHeadlessSurface(accel.SurfaceDescriptor{Width: 4, Height: 4})
		if err != nil {
			t.Fatalf("NewHeadlessSurface: %v", err)
		}
		defer s.Close()

		r := d.NewRecorder()
		r.PresentSlot(s, "foreign")
		if _, err := r.Build(); err == nil {
			t.Error("a surface from another device was recorded as a present slot")
		} else if !strings.Contains(err.Error(), "different device") {
			t.Errorf("the error should say the surface is another device's, got %v", err)
		}
	})

	t.Run("BindPresent with no frame", func(t *testing.T) {
		s := newSurface(t, d, 4, 4, 1)
		g, swap := trivialPresentGraph(t, d, s)
		if err := g.BindPresent(swap, nil); err == nil {
			t.Error("BindPresent accepted no frame")
		}
		f, err := s.Acquire(time.Second)
		if err != nil {
			t.Fatalf("acquire: %v", err)
		}
		if err := s.Present(f, nil); err != nil {
			t.Fatalf("present: %v", err)
		}
		if err := g.BindPresent(swap, f); err == nil {
			t.Error("BindPresent accepted a frame that was already presented")
		}
	})
}

// trivialPresentGraph is the smallest graph with a present slot: an upload into
// it, which needs no pipeline and no stage.
func trivialPresentGraph(t *testing.T, d *accel.Device, s *accel.Surface) (*accel.Graph, accel.Slot) {
	t.Helper()
	r := d.NewRecorder()
	swap := r.PresentSlot(s, "swapchain")
	r.UploadToSlot(swap, 0, 4, make([]float32, 4))
	g, err := r.Build()
	if err != nil {
		t.Fatalf("build: %v", err)
	}
	t.Cleanup(func() { _ = g.Close() })
	return g, swap
}

// The rest of Present's and PresentSlot's refusals, all portable.
func TestPresentRefusals(t *testing.T) {
	d := openDevice(t)

	t.Run("no frame", func(t *testing.T) {
		s := newSurface(t, d, 4, 4, 1)
		if err := s.Present(nil, nil); err == nil {
			t.Error("Present accepted no frame")
		}
	})

	t.Run("a frame from another surface", func(t *testing.T) {
		s := newSurface(t, d, 4, 4, 1)
		other := newSurface(t, d, 4, 4, 1)
		f, err := other.Acquire(time.Second)
		if err != nil {
			t.Fatalf("acquire: %v", err)
		}
		err = s.Present(f, nil)
		if err == nil {
			t.Fatal("a frame from another surface was presented")
		}
		if !strings.Contains(err.Error(), other.Label()) {
			t.Errorf("the error should name the surface the frame came from, got %v", err)
		}
	})

	t.Run("a submission that failed", func(t *testing.T) {
		s := newSurface(t, d, 4, 4, 1)
		f, err := s.Acquire(time.Second)
		if err != nil {
			t.Fatalf("acquire: %v", err)
		}
		// A graph that cannot run: the fence carries the failure, and Present
		// must not show a frame whose contents were never produced.
		bad := failedFence(t, d)
		err = s.Present(f, bad)
		if err == nil {
			t.Fatal("a frame was presented after the submission that rendered it failed")
		}
		if !strings.Contains(err.Error(), "failed") {
			t.Errorf("the error should say the submission failed, got %v", err)
		}
	})

	t.Run("a positive timeout with every image held", func(t *testing.T) {
		s := newSurface(t, d, 4, 4, 1)
		if _, err := s.Acquire(time.Second); err != nil {
			t.Fatalf("acquire: %v", err)
		}
		// It reports rather than waits out the timeout: with every image held
		// by the caller there is nothing to wait for, and sleeping would be a
		// stall with no possible outcome.
		//
		// # Why the timeout is a second and the bound is not a wall clock
		//
		// The property is "it did not wait for the timeout", and the first
		// version asserted it as "under 5ms" against a 10ms timeout -- a 2x
		// margin, on a machine that also runs the rest of this suite. It failed
		// on a loaded one while the property held, which is the measurement
		// being wrong rather than the code.
		//
		// A one-second timeout against a 100ms bound is a 10x margin for the
		// same property, and a scheduler delay that eats 100ms would break far
		// more than this test. The wall clock cannot be removed entirely --
		// "returned early" is a claim about elapsed time -- so the fix is a
		// margin wide enough that only a real regression closes it.
		const timeout = time.Second
		start := time.Now()
		_, err := s.Acquire(timeout)
		if !errors.Is(err, accel.ErrAcquireTimeout) {
			t.Fatalf("the second acquire gave %v, want ErrAcquireTimeout", err)
		}
		if elapsed := time.Since(start); elapsed > timeout/10 {
			t.Errorf("it waited %v of a %v timeout for an image only the caller "+
				"could return", elapsed, timeout)
		}
	})

	t.Run("an unnamed present slot still has a name", func(t *testing.T) {
		s := newSurface(t, d, 4, 4, 1)
		r := d.NewRecorder()
		slot := r.PresentSlot(s, "")
		if slot == 0 {
			t.Fatal("an unnamed present slot was refused")
		}
		r.UploadToSlot(slot, 0, 4, make([]float32, 4))
		g, err := r.Build()
		if err != nil {
			t.Fatalf("build: %v", err)
		}
		defer g.Close()
		for _, gs := range g.Slots() {
			if gs.Descriptor.Name == "" {
				t.Error("a slot with no name given has none, and its errors name nothing")
			}
		}
	})

	t.Run("BindPresent on a closed graph", func(t *testing.T) {
		s := newSurface(t, d, 4, 4, 1)
		g, swap := trivialPresentGraph(t, d, s)
		if err := g.Close(); err != nil {
			t.Fatalf("close: %v", err)
		}
		f, err := s.Acquire(time.Second)
		if err != nil {
			t.Fatalf("acquire: %v", err)
		}
		if err := g.BindPresent(swap, f); err == nil {
			t.Error("a closed graph bound a present frame")
		}
	})
}

// failedFence returns a fence whose submission failed, for the paths that must
// not proceed after one.
func failedFence(t *testing.T, d *accel.Device) *accel.Fence {
	t.Helper()
	// A graph submitted twice without waiting: the second is refused, and the
	// refusal arrives through the fence rather than through the call.
	r := d.NewRecorder()
	src := newBuffer(t, d, "src", 16, accel.BufferStorage|accel.BufferCopySrc|accel.BufferCopyDst)
	dst := newBuffer(t, d, "dst", 16, accel.BufferStorage|accel.BufferCopySrc|accel.BufferCopyDst)
	r.CopyBuffer(whole(t, dst), whole(t, src))
	g, err := r.Build()
	if err != nil {
		t.Fatalf("build: %v", err)
	}
	t.Cleanup(func() { _ = g.Close() })
	if err := d.Queue().Submit(g).Wait(); err != nil {
		t.Fatalf("submit: %v", err)
	}
	if err := g.Close(); err != nil {
		t.Fatalf("close: %v", err)
	}
	// Submitting a closed graph is refused, and the refusal rides the fence.
	f := d.Queue().Submit(g)
	if err := f.Wait(); err == nil {
		t.Fatal("submitting a closed graph succeeded, so this fence carries no failure")
	}
	return f
}

// A resize with a frame outstanding is refused, and the refusal names the
// count.
//
// specs/034-surface-present.md section 8 says a resize invalidates every frame
// acquired before it. That was free while a frame was a buffer view; a windowed
// frame holds a drawable the compositor lent it, and a caller who dropped one
// and acquired again would leak it — which is the pool exhaustion Discard
// exists to prevent, and whose symptom is a loop that stops with no error and
// no stack.
//
// The surface keeps no list of outstanding frames, so it cannot return them for
// the caller. Refusing turns the silent leak into a call they fix in one line,
// and the frame loop of section 1 already satisfies the rule: the resize there
// follows an acquire that failed, so nothing is outstanding.
func TestAResizeWithAFrameOutstandingIsRefused(t *testing.T) {
	d := openDevice(t)
	s := newSurface(t, d, 8, 8, 2)

	f, err := s.Acquire(time.Second)
	if err != nil {
		t.Fatalf("acquire: %v", err)
	}
	err = s.Resize(16, 16)
	if err == nil {
		t.Fatal("a resize with a frame outstanding was accepted, which leaks its drawable")
	}
	if !strings.Contains(err.Error(), "1 frame(s) outstanding") {
		t.Errorf("the error should say how many are outstanding, got %v", err)
	}
	if got := s.Generation(); got != 0 {
		t.Errorf("a refused resize left the generation at %d; a resize that did not "+
			"happen must not invalidate a graph", got)
	}
	if gw, gh := s.Extent(); gw != 8 || gh != 8 {
		t.Errorf("a refused resize changed the extent to %dx%d", gw, gh)
	}

	// And the documented fix works.
	if err := s.Discard(f); err != nil {
		t.Fatalf("discard: %v", err)
	}
	if err := s.Resize(16, 16); err != nil {
		t.Fatalf("after discarding, resize gave %v", err)
	}
}

// A surface presents normally after a discard and a resize.
//
// This is not the stale-frame drop in [accel.Surface.Present]: that branch
// handles a frame acquired before a resize, and Resize refuses while a frame
// is outstanding, so no sequence of public calls reaches it. What a caller can
// do is discard the frame, resize, and acquire again -- and the frame they get
// must present, or the discard has left the surface in a state the resize did
// not recover from.
func TestASurfacePresentsAfterADiscardAndAResize(t *testing.T) {
	d := openDevice(t)
	s := newSurface(t, d, 8, 8, 2)

	f, err := s.Acquire(time.Second)
	if err != nil {
		t.Fatalf("acquire: %v", err)
	}
	if err := s.Discard(f); err != nil {
		t.Fatalf("discard: %v", err)
	}
	if err := s.Resize(4, 4); err != nil {
		t.Fatalf("resize: %v", err)
	}

	// A second frame from the new generation presents normally, which is what
	// makes the drop above about staleness rather than about the surface being
	// broken by the resize.
	after, err := s.Acquire(time.Second)
	if err != nil {
		t.Fatalf("acquire after resize: %v", err)
	}
	if err := s.Present(after, nil); err != nil {
		t.Fatalf("present after resize: %v", err)
	}
}
