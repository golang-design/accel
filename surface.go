// Copyright 2026 The golang.design Initiative Authors.
// All rights reserved. Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

package accel

import (
	"errors"
	"fmt"
	"sync"
	"time"

	"golang.design/x/accel/internal/driver"
)

// The surface and present path of specs/034-surface-present.md.
//
// # Why present is not a graph node
//
// Present is a handoff to a compositor whose completion is not the device's to
// signal. A graph describes device work; acquisition and presentation are paired
// to one external Frame. Keeping present outside the graph preserves that
// boundary, and the ordering that does matter — rendering finishes before the
// image is shown — is expressed by the submission fence, which the API already
// has.

// ErrSurfaceOutOfDate reports that a surface's images no longer match its
// configuration, so the frame loop must resize and rebuild.
//
// An error value rather than a silent reallocation: a graph is built against a
// specific extent, so reallocating underneath it would leave the graph
// describing an image that no longer exists, and the failure would appear later
// as a size mismatch with no cause attached.
var ErrSurfaceOutOfDate = errors.New("accel: the surface is out of date")

// ErrAcquireTimeout reports that no image became available in time.
var ErrAcquireTimeout = errors.New("accel: acquiring a frame timed out")

// SurfaceDescriptor configures a surface.
type SurfaceDescriptor struct {
	Width, Height int

	// Images is how many rotate. Two is double buffering, which is the least
	// that lets the device render one frame while the compositor holds another.
	Images int

	Label string
}

// Surface hands out images to render into and takes them back to present.
//
// One type rather than an interface per backend, because the state a frame loop
// depends on — the generation counter, the rotation, what counts as out of date
// — is the same everywhere and is the part that must not vary. A backend
// supplies only where the pixels live.
type Surface struct {
	dev   *Device
	label string
	impl  surfaceImpl

	// present is the on-screen target, or nil for a headless surface. What it
	// changes is Present: acquire takes an image from it and Present converts
	// the frame into that image.
	present driver.PresentTarget

	mu sync.Mutex
	// gen increments on every resize. A graph records the generation it was
	// built against, so a frame from before a resize is refused by identity
	// rather than by extent -- two generations can share an extent.
	gen           uint64
	width, height int
	inFlight      int
	next          int
	outOfDate     bool
	closed        bool
}

// surfaceImpl is where the pixels live. The headless surface rotates ordinary
// buffers; a windowed one would hand out drawables.
type surfaceImpl interface {
	// image returns the view for one rotation index.
	image(i int) BufferView

	// reconfigure resizes the underlying images.
	reconfigure(w, h int) error

	// count is how many images rotate.
	count() int

	closeImpl() error
}

// Frame is one acquired image.
//
// It is not a naked view: BindPresent takes a Frame so it can check the surface
// identity and the generation, which a view cannot carry. A frame from another
// surface with the same format and extent is the case a format check alone
// accepts.
type Frame struct {
	surface *Surface
	gen     uint64
	index   int
	view    BufferView

	// Acquired is signalled when the image is safe to render into. On a
	// headless surface it is already signalled, because nothing else holds the
	// image; a windowed surface signals it when the compositor releases one.
	Acquired *Fence

	// image is the backend's acquired drawable, for a windowed surface.
	image driver.PresentImage

	presented bool
}

// View is where the frame's pixels are, for a caller that wants to read them
// back rather than present them.
func (f *Frame) View() BufferView { return f.view }

// Index is which of the rotating images this is, for a diagnostic that wants to
// say the loop is rotating rather than reusing one.
func (f *Frame) Index() int { return f.index }

// NewHeadlessSurface makes a surface with no window.
//
// Not a mock. It has the same generation counter, the same acquire timeout, the
// same rotation and the same out-of-date behaviour as a windowed one, which is
// what lets the whole frame path run in CI with no display. A mock would agree
// with the interface and disagree with the state machine — and the state machine
// is what a frame loop is.
func (d *Device) NewHeadlessSurface(desc SurfaceDescriptor) (*Surface, error) {
	if err := d.state.checkOpen("NewHeadlessSurface"); err != nil {
		return nil, err
	}
	label := desc.Label
	if label == "" {
		label = "headless surface"
	}
	if desc.Width <= 0 || desc.Height <= 0 {
		return nil, fmt.Errorf("accel: NewHeadlessSurface %q: the extent is %dx%d",
			label, desc.Width, desc.Height)
	}
	images := desc.Images
	if images == 0 {
		images = 2
	}
	if images < 1 {
		return nil, fmt.Errorf("accel: NewHeadlessSurface %q: %d images", label, images)
	}

	h := &headless{dev: d, label: label, images: make([]*Buffer, images)}
	s := &Surface{dev: d, label: label, impl: h, width: desc.Width, height: desc.Height}
	if err := h.reconfigure(desc.Width, desc.Height); err != nil {
		return nil, err
	}
	return s, nil
}

// Label is the surface's label, which appears in every error about it.
func (s *Surface) Label() string { return s.label }

// Generation is how many times the surface has been reconfigured.
//
// Exposed because a stale-frame error names two of them, and a caller
// reconciling a rebuild wants to see the number the graph was built against.
func (s *Surface) Generation() uint64 {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.gen
}

// Extent is the surface's current size.
func (s *Surface) Extent() (int, int) {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.width, s.height
}

// Acquire hands out the next image.
//
// It takes a timeout and can report expiry, because it can genuinely block: the
// swapchain may be full, or the compositor may not have released an image. A
// call described as non-blocking that waits on a compositor is worse than one
// that says so.
func (s *Surface) Acquire(timeout time.Duration) (*Frame, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.closed {
		return nil, fmt.Errorf("accel: Acquire: surface %q is closed", s.label)
	}
	if s.outOfDate {
		return nil, fmt.Errorf("%w: %q, generation %d", ErrSurfaceOutOfDate, s.label, s.gen)
	}
	if s.inFlight >= s.impl.count() {
		// Every image is held. A windowed surface would wait here for the
		// compositor; there is nothing to wait for when the caller holds them
		// all, so the timeout expires rather than deadlocking.
		if timeout <= 0 {
			return nil, fmt.Errorf("%w: %q has %d images and the caller holds all of "+
				"them; present one before acquiring another", ErrAcquireTimeout,
				s.label, s.impl.count())
		}
		return nil, fmt.Errorf("%w: %q after %v", ErrAcquireTimeout, s.label, timeout)
	}

	i := s.next
	f := &Frame{
		surface: s, gen: s.gen, index: i, view: s.impl.image(i),
		Acquired: signalledFence(),
	}
	if s.present != nil {
		// The drawable is taken here rather than at Present, because this is
		// where a compositor makes a caller wait: a windowed Acquire is the
		// call that blocks, and taking the image later would move the wait to
		// where the loop cannot act on it.
		img, err := s.present.Acquire(timeout)
		if err != nil {
			return nil, fmt.Errorf("accel: Acquire on %q: %w", s.label, err)
		}
		f.image = img
	}
	s.next = (s.next + 1) % s.impl.count()
	s.inFlight++
	return f, nil
}

// Present hands the frame back, to be shown once the fence signals.
//
// The fence is the ordering: it is the submission that rendered into the image,
// and present must follow it. Passing nil presents immediately, which is only
// correct when nothing was submitted.
func (s *Surface) Present(f *Frame, after *Fence) error {
	if f == nil {
		return fmt.Errorf("accel: Present: no frame")
	}
	if f.surface != s {
		return fmt.Errorf("accel: Present: the frame came from %q and this is %q",
			f.surface.label, s.label)
	}
	if f.presented {
		return fmt.Errorf("accel: Present: frame %d of %q was already presented or "+
			"discarded", f.index, s.label)
	}
	if after != nil {
		if err := after.Wait(); err != nil {
			return fmt.Errorf("accel: Present: the submission that rendered frame %d "+
				"failed: %w", f.index, err)
		}
	}

	s.mu.Lock()
	stale := f.gen != s.gen
	s.mu.Unlock()

	if stale {
		// The surface was reconfigured while the caller held this frame, so its
		// pixels describe an extent that no longer exists. Dropped rather than
		// shown, and its drawable goes back rather than leaking -- the buffer
		// behind it may already be closed, so this must happen before anything
		// reads f.view.
		//
		// Reaching here means Resize was called with the frame outstanding,
		// which Resize refuses. It is kept because a refusal that is also
		// handled is cheaper than a rule that is only stated.
		f.presented = true
		if f.image != nil {
			f.image.Discard()
			f.image = nil
		}
		return nil
	}

	if f.image != nil {
		blk, base := blockFor(f.view.Buffer)
		if err := f.image.Present(blk, base+f.view.Offset*f.view.DType.Size()); err != nil {
			f.presented = true
			s.mu.Lock()
			s.inFlight--
			s.mu.Unlock()
			return fmt.Errorf("accel: Present on %q: %w", s.label, err)
		}
		f.image = nil
	}

	s.mu.Lock()
	defer s.mu.Unlock()
	f.presented = true
	s.inFlight--
	return nil
}

// Discard hands a frame back without presenting it.
//
// Every acquired frame is either presented or discarded. On a windowed surface
// the frame holds a drawable the compositor lent it, and one abandoned rather
// than returned exhausts the pool -- whose symptom is a frame loop that stops,
// with no error and no stack pointing at the cause.
//
// A caller reaches this when a frame cannot be rendered after all: a graph that
// failed to build, a resize noticed between acquire and submit, a frame skipped
// for timing.
func (s *Surface) Discard(f *Frame) error {
	if f == nil {
		return fmt.Errorf("accel: Discard: no frame")
	}
	if f.surface != s {
		return fmt.Errorf("accel: Discard: the frame came from %q and this is %q",
			f.surface.label, s.label)
	}
	if f.presented {
		return fmt.Errorf("accel: Discard: frame %d of %q was already presented or "+
			"discarded", f.index, s.label)
	}
	f.presented = true
	if f.image != nil {
		f.image.Discard()
		f.image = nil
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if f.gen == s.gen {
		s.inFlight--
	}
	return nil
}

// Resize reconfigures the surface, which increments the generation and
// invalidates every graph built against the old one.
//
// specs/034-surface-present.md section 4.1: attachment extents stay validated at
// build, so a resize forces a rebuild. That is the cost of a build-time check
// that catches a mismatched attachment before any device work happens.
func (s *Surface) Resize(w, h int) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.closed {
		return fmt.Errorf("accel: Resize: surface %q is closed", s.label)
	}
	if w <= 0 || h <= 0 {
		return fmt.Errorf("accel: Resize: surface %q to %dx%d", s.label, w, h)
	}
	// A resize invalidates every frame acquired before it, and a windowed frame
	// holds a drawable the compositor lent it. This surface keeps no list of
	// outstanding frames, so it cannot return them -- and a caller who dropped
	// one and acquired again would leak it, which is the pool exhaustion
	// [Surface.Discard] exists to prevent, with the symptom that has no error
	// and no stack.
	//
	// Refused rather than tracked, because refusing turns a silent leak into a
	// call the caller fixes in one line. The frame loop of
	// specs/034-surface-present.md section 1 already satisfies it: the resize
	// there follows an acquire that *failed*, so nothing is outstanding.
	if s.inFlight > 0 {
		return fmt.Errorf("accel: Resize: surface %q has %d frame(s) outstanding; a "+
			"resize invalidates every frame acquired before it, and a frame holding a "+
			"drawable must go back before it can be invalidated -- present or discard "+
			"them first", s.label, s.inFlight)
	}
	if err := s.impl.reconfigure(w, h); err != nil {
		return err
	}
	if s.present != nil {
		if err := s.present.Configure(w, h); err != nil {
			return err
		}
	}
	s.width, s.height = w, h
	s.gen++
	s.inFlight = 0
	s.next = 0
	s.outOfDate = false
	return nil
}

// Invalidate marks the surface out of date, so the next Acquire reports it.
//
// This is what a windowed surface's compositor does when the window changes
// under it. Exposed because the headless surface is not a mock: a test of the
// out-of-date path needs the same entry point the real event takes.
func (s *Surface) Invalidate() {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.outOfDate = true
}

// Close releases the surface's images.
func (s *Surface) Close() error {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.closed {
		return nil
	}
	s.closed = true
	err := s.impl.closeImpl()
	if s.present != nil {
		if perr := s.present.Close(); err == nil {
			err = perr
		}
		s.present = nil
	}
	return err
}

// signalledFence is a fence that has already completed.
//
// A headless surface has nothing to wait for: no compositor holds the image, so
// the acquire is immediate. The fence still exists so the frame loop of
// specs/034-surface-present.md section 1 is character-for-character the same as
// a windowed one -- a loop that omitted SubmitAfter for headless would be
// testing a different loop.
func signalledFence() *Fence {
	f := newFence()
	f.signal()
	return f
}

// headless rotates ordinary buffers and presents by leaving the pixels readable.
type headless struct {
	dev    *Device
	label  string
	images []*Buffer
}

func (h *headless) count() int { return len(h.images) }
func (h *headless) image(i int) BufferView {
	v, _ := h.images[i].View(0, h.images[i].Count())
	return v
}

func (h *headless) reconfigure(w, h2 int) error {
	if err := h.closeImpl(); err != nil {
		return err
	}
	for i := range h.images {
		b, err := h.dev.NewBuffer(BufferDescriptor{
			DType: F32, Count: w * h2 * 4,
			Usage: BufferStorage | BufferCopySrc | BufferCopyDst,
			Label: fmt.Sprintf("%s image %d", h.label, i),
		})
		if err != nil {
			return fmt.Errorf("accel: surface %q: allocating image %d: %w", h.label, i, err)
		}
		h.images[i] = b
	}
	return nil
}

func (h *headless) closeImpl() error {
	var first error
	for i, b := range h.images {
		if b == nil {
			continue
		}
		if err := b.Close(); err != nil && first == nil {
			first = err
		}
		h.images[i] = nil
	}
	return first
}

// PresentSlot records a rebindable binding point for a surface's image.
//
// A dedicated slot type rather than an ordinary slot plus a format, because a
// format cannot prove that the eventual image is presentable, belongs to the
// right surface, or comes from the generation the graph was built for. The slot
// records the device, the surface identity, the generation and the extent, and
// [Graph.BindPresent] checks all of them.
//
// specs/034-surface-present.md section 2.
func (r *Recorder) PresentSlot(s *Surface, name string) Slot {
	if name == "" {
		name = "present"
	}
	if s == nil {
		r.fail("PresentSlot %q: no surface", name)
		return 0
	}
	if s.dev != r.state.dev {
		r.fail("PresentSlot %q: surface %q belongs to a different device", name, s.label)
		return 0
	}
	// The extent and the generation are one reading, because they are one
	// fact: a resize changes both. Read under two acquisitions they could be
	// the old extent against the new number, and a frame of that generation
	// then passed the generation check and failed the size check.
	w, h, gen := s.configuration()
	slot := r.slotImpl(SlotDescriptor{
		Name: name, Kind: BindingStorageBuffer, DType: F32,
		Access: AccessWrite, MinCount: w * h * 4,
	})
	if slot == 0 {
		return 0
	}
	if r.state.present == nil {
		r.state.present = map[Slot]presentSlot{}
	}
	r.state.present[slot] = presentSlot{surface: s, gen: gen, width: w, height: h}
	return slot
}

// configuration is the surface's extent and generation as one reading.
func (s *Surface) configuration() (w, h int, gen uint64) {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.width, s.height, s.gen
}

// presentSlot is what a present slot records beyond an ordinary one.
type presentSlot struct {
	surface       *Surface
	gen           uint64
	width, height int
}

// BindPresent binds an acquired frame to a present slot.
//
// It takes a Frame and never a naked view, so it can check the surface identity
// and the generation. An ordinary render target with the same format and extent
// is the case a format check alone accepts, and a frame from before a resize is
// the case an extent check alone accepts when the two generations happen to
// share a size.
func (g *Graph) BindPresent(slot Slot, f *Frame) error {
	if err := g.state.checkOpen("BindPresent"); err != nil {
		return err
	}
	ps, ok := g.present[slot]
	if !ok {
		return fmt.Errorf("accel: BindPresent: slot %d is not a present slot; "+
			"PresentSlot is what records the surface a frame must come from", slot)
	}
	if f == nil {
		return fmt.Errorf("accel: BindPresent: no frame for %q", ps.surface.label)
	}
	if f.surface != ps.surface {
		return fmt.Errorf("accel: BindPresent: the frame came from surface %q and %q "+
			"was recorded; a frame with a matching format and extent is still the "+
			"wrong image", f.surface.label, ps.surface.label)
	}
	if f.gen != ps.gen {
		return fmt.Errorf("accel: BindPresent: the frame is from generation %d of %q "+
			"and the graph was built against generation %d; a resize invalidates the "+
			"graph, and the two generations can share an extent",
			f.gen, ps.surface.label, ps.gen)
	}
	if f.presented {
		return fmt.Errorf("accel: BindPresent: frame %d of %q has already been presented",
			f.index, ps.surface.label)
	}
	return g.Bind(SlotBinding{Slot: slot, Buffer: f.view})
}

// NativeHandleKind says what a [NativeHandle] points at.
type NativeHandleKind = driver.NativeHandleKind

const (
	// NativeMetalLayer is a `CAMetalLayer*` the caller created and attached to
	// a window. It must be created and resized on the main thread.
	NativeMetalLayer = driver.NativeMetalLayer

	// NativeNSView is an `NSView*`. accel makes the layer, and that call must
	// itself be on the main thread.
	NativeNSView = driver.NativeNSView
)

// NativeHandle is a platform-tagged pointer to a window resource the caller
// owns.
//
// Tagged rather than bare, because a backend given the wrong kind of pointer
// sends a message to an object that does not answer it, and the crash names
// neither the caller nor the mistake.
type NativeHandle = driver.NativeHandle

// ErrNoPresent reports that a device has no on-screen path.
//
// Reported rather than discovered, which is decision 6: a caller asks and is
// told, instead of finding out when a frame does not appear.
var ErrNoPresent = driver.ErrNoPresent

// NewWindowSurface makes a surface that presents to a window the caller owns.
//
// # What is yours and what is accel's
//
// accel does not create windows. specs/034-surface-present.md section 6 puts
// the window, its event loop, input, focus and DPI on your side of the line,
// and everything from the swapchain inward on accel's — because window creation
// is an operating system concern with no relation to GPU work, and absorbing it
// would drag a windowing library and an opinion about event loops into a
// library whose subject is device work.
//
// So you create the window and hand over a native handle. accel owns the
// drawables, acquire, present and resize.
//
// # Main-thread obligation on macOS
//
// A `CAMetalLayer` must be created and resized on the main thread. accel cannot
// check this and does not try: call this from the main thread, and call
// [Surface.Resize] from it too. Given a [NativeNSView] this call creates the
// layer, so the obligation applies to this call itself.
//
// # What the frame loop looks like
//
// The same as a headless one, which is the point of the headless surface
// existing: [Surface.Acquire], [Graph.BindPresent], submit after the frame's
// fence, then [Surface.Present]. Nothing in the loop changes when the pixels
// start going to a screen.
func (d *Device) NewWindowSurface(h NativeHandle, desc SurfaceDescriptor) (*Surface, error) {
	if err := d.state.checkOpen("NewWindowSurface"); err != nil {
		return nil, err
	}
	label := desc.Label
	if label == "" {
		label = "window surface"
	}
	if desc.Width <= 0 || desc.Height <= 0 {
		return nil, fmt.Errorf("accel: NewWindowSurface %q: the extent is %dx%d",
			label, desc.Width, desc.Height)
	}
	// The image count belongs to the descriptor, as the extent does, so it is
	// refused before the backend is asked and with the headless surface's
	// rule. It was checked after, and only for zero: a negative count reached
	// make and took the process down.
	images := desc.Images
	if images == 0 {
		images = 2
	}
	if images < 1 {
		return nil, fmt.Errorf("accel: NewWindowSurface %q: %d images", label, images)
	}
	p, ok := d.dev.(driver.Presenter)
	if !ok {
		return nil, fmt.Errorf("%w: %q", ErrNoPresent, d.info.Name)
	}
	target, err := p.NewPresentTarget(h, desc.Width, desc.Height)
	if err != nil {
		return nil, fmt.Errorf("accel: NewWindowSurface %q: %w", label, err)
	}

	// The rotating images are buffers, exactly as the headless surface's are,
	// and presenting converts one into a drawable. specs/033-render-api.md
	// makes an attachment a buffer view, so the graph renders into a buffer
	// either way -- and that is what keeps the frame loop identical.
	hl := &headless{dev: d, label: label, images: make([]*Buffer, images)}
	s := &Surface{
		dev: d, label: label, impl: hl, present: target,
		width: desc.Width, height: desc.Height,
	}
	if err := hl.reconfigure(desc.Width, desc.Height); err != nil {
		_ = target.Close()
		return nil, err
	}
	return s, nil
}
