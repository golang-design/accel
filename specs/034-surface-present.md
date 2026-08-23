---
title: "Surfaces, acquisition, resize, and present"
status: drafted
layer: device
depends_on:
  - 001-device-resources.md
  - 003-command-graph.md
  - 005-graphics.md
  - 033-render-api.md
---

# Surfaces, acquisition, resize, and present

[005](005-graphics.md)'s third child spec: how a rendered frame reaches a screen,
and how the same code path runs with no screen at all.

This is the smallest of the four specs and the one with the most external
surface, because it is where accel meets an operating system it does not own. The
line is drawn in section 5, and it is drawn where the predecessor project drew
it, for reasons that project measured.

## 1. The frame loop, in full

```go
// Once, when the frame graph is built.
swap := rec.PresentSlot(surface, "swapchain")
// ... the graph writes into swap like any attachment ...
g, err := rec.Build()

// Per frame, for the life of the window.
for !done {
	frame, err := surface.Acquire(timeout)
	if errors.Is(err, accel.ErrSurfaceOutOfDate) {
		surface.Resize(newExtent)
		g = rebuild()          // see section 4
		continue
	}
	g.BindPresent(swap, frame)
	fence := queue.SubmitAfter(g, frame.Acquired)
	surface.Present(frame, fence)
}
```

```
        acquire                      submit                    present
 ┌──────────────┐          ┌───────────────────────┐      ┌──────────────┐
 │ surface hands│  Frame   │ graph writes the bound│ Fence│ queue hands  │
 │ out a Frame  │ ───────▶ │ present slot          │─────▶│ it to the    │
 │ + a fence    │          │ (device work)         │      │ compositor   │
 └──────────────┘          └───────────────────────┘      └──────────────┘
        ▲                                                         │
        └───────────────── the image rotates back ────────────────┘
```

## 2. The acquired image is a typed present slot

A graph cannot name a swapchain texture at record time, because which texture the
frame gets is decided at acquire time. That much is forced. What is a decision is
the **type** of the slot, and this spec adds a dedicated one rather than reusing
an attachment slot with a format.

An ordinary attachment slot plus a format cannot prove that the eventual texture
is presentable, belongs to the right surface, or comes from the surface
generation the graph was built for. So `PresentSlot` records, internally:

| Recorded in the slot | What it rejects |
| --- | --- |
| device | a frame from another device |
| surface identity | a frame from another surface with the same format |
| surface generation | a frame from before a resize |
| format and extent | a graph built for a different swapchain configuration |
| render-target usage | an ordinary texture that happens to match |
| final state `Present` | nothing — this is what it *adds*, see below |

`BindPresent` takes a `Frame`, never a naked texture, and verifies all of it. The
last row is the one that is not validation: recording that the slot's final state
is `Present` makes the present transition **representable to the graph planner**,
so the transition to present-ready is emitted as the pass's store, which is the
part of presenting that genuinely is device work.

This is [003](003-command-graph.md)'s second kind of variation with a stronger
slot type. The graph is built once and replayed until resize or surface
reconfiguration changes that type-level contract.

## 3. Present is not a graph node

Present is a queue operation taking the submission fence it must follow.

The reason is a boundary, not a convenience: present is not work on the device,
it is a handoff to a compositor whose completion is not the device's to signal.
Acquisition and presentation are paired to one external `Frame`, while a graph
describes only device work. Keeping present outside preserves that; the ordering
that does matter — rendering finishes before the image is shown — is expressed by
the submission fence, which is a thing the API already has.

`Acquire` takes a timeout and can report expiry. It can genuinely block: the
swapchain may be full, or the compositor may not have released an image. A call
the API described as non-blocking that waits on a compositor is worse than one
that says so and returns.

## 4. Resize, and the rebuild it forces

`Resize` reallocates the swapchain, increments the surface generation, and
invalidates any transient sized to the old extent.

**A resize means rebuilding the graphs that render into the surface.** Stated
plainly because it is a cost. It is cheap per resize event and would be
unacceptable per frame, which is exactly why it is worth being sure resize is
the only thing that triggers it.

### 4.1 Decision: attachment extents stay validated at build

005 leaves this open — whether making render area and attachment extents dynamic
would let a resize rebind rather than rebuild — and this spec closes it in
005's own direction, with the reasoning made explicit because the trade is real.

| | Build-validated extents (chosen) | Dynamic extents |
| --- | --- | --- |
| Resize cost | rebuild the graph | rebind |
| A pass whose area exceeds an attachment | build error naming both | undefined behaviour or a runtime error per backend |
| Transient sizing | the planner knows every extent, so it can size and alias | the planner sizes for a maximum it must be told |
| Frequency of the cost | once per resize event | — |

The second row decides it. Extents are what attachment-size validation is made
of, and moving them out of build moves that whole class of error out of build
with them — into a place where 003 promises errors do not arrive. Paying a
rebuild at an event that happens when a human drags a window edge, in exchange
for keeping every extent error at build, is the trade this project has made
everywhere else. **This closes 005's fourth open question**, and 033 §8 points
here.

### 4.2 Out of date is an error value, not a silent reallocation

A backend may report the swapchain as out of date on acquire, because the window
changed underneath it. `ErrSurfaceOutOfDate` is explicit.

Hiding it and reallocating would leave the caller's stale graphs pointing at
freed textures, and the caller has to rebuild either way. An error they must
handle is strictly better than a success that invalidated their graphs without
telling them.

## 5. Headless is the same code path

A surface with no window rotates ordinary offscreen textures and "presents" by
making the frame's pixels available for readback. The frame loop in section 1 is
character-for-character identical.

This is the predecessor's design, and it earned its keep: it is what lets the
entire frame path — acquire, render, present, rotate, resize — run in CI on every
platform with no display and no compositor. It is also what makes the CPU
backend's graphics path complete rather than nearly complete, since a CPU backend
with no present would leave the frame loop untested exactly where its state
machine lives.

The headless surface is not a mock. It is a `Surface` implementation with the
same generation counter, the same acquire timeout, the same rotation, and the
same out-of-date error on a resize that raced an acquire. A mock would agree with
the interface and disagree with the state machine.

## 6. Windowing is out of scope, and the line is here

**accel does not create windows.** The caller supplies a native handle and accel
owns everything from the swapchain inward.

| Caller owns | accel owns |
| --- | --- |
| The OS window and its lifetime | The swapchain and its textures |
| The event loop, input, DPI changes | `EGLSurface`, `CAMetalLayer` drawables, DXGI swapchain |
| Choosing a window visual — accel reports the constraint | Present, acquire, resize |

Window creation is an operating system and event loop concern with no relation to
GPU work: input, focus, DPI, menu bars, main-thread affinity. Absorbing it would
drag a windowing library, its own cross-platform test matrix, and an opinion
about event loops into a library whose subject is device work — in a codebase
that cannot use cgo, where every windowing backend is hand-written syscall or
`purego` binding.

The boundary needs traffic in **both** directions, and the predecessor proved
this concretely: its `WindowVisualID` reports the native visual an X11 window
must be created with for the EGL config to be compatible. So accel reports its
constraints — required visual or pixel format, supported present modes, supported
extents — and accepts a platform-tagged native handle:

| Platform | Handle |
| --- | --- |
| X11 | `Display*` plus window XID |
| Wayland | `wl_display` plus `wl_surface` |
| Windows | `HWND` |
| macOS | `CAMetalLayer*`, or `NSView*` |

**Caller obligation on macOS.** A `CAMetalLayer` must be created and resized on
the main thread. accel accepts a layer the caller created and attached and
documents the requirement; given an `NSView*` it will create the layer, and then
that call itself must be made on the main thread, and says so.

**The honest cost.** accel cannot ship a runnable windowed example without either
a second package or a test-only window shim. That is not hypothetical: it is
exactly why the predecessor needed an `app/` package. The answer here is a
test-only shim, small and unexported, sufficient for the present tests below and
explicitly not a windowing library.

## 7. The Metal drawable path is the risk, and it is proven first

The predecessor implemented on-screen present for X11/EGL and Win32/ANGLE and
**never implemented Metal's `CAMetalLayer` drawable path**, so present worked on
GL and not on Metal.

That is a warning rather than a detail. Metal is the backend where present
differs most from the others:

- the drawable is owned by the layer, not by accel;
- `nextDrawable` can block, or return nothing, and both are ordinary;
- the completion-handler lifetime rule in
  [`conventions.md`](../docs/conventions.md) applies directly — a handler that
  outlives what it captures is a use-after-free with a plausible-looking frame as
  its symptom;
- and the drawable must be presented on the command buffer that rendered it, not
  afterwards, which is a different call shape from every other backend.

So the ordering in this spec's implementation is: headless surface first, on
every backend, because it is what the frame-loop state machine is tested against;
then **Metal's drawable path**, before any other on-screen backend. Proven early
rather than left as the last brick again.

**Two claims, never reported as one.** "Headless render verified" and "windowed
present verified on a display" are separate states in the status table. Metal's
drawable path needs a real display session, which CI does not have, so it is
honestly tracked as verified-on-a-machine rather than verified-in-CI.

## 8. Done

- a graph built against a surface renders a frame and the readback matches what
  the same graph produces into an ordinary offscreen target;
- `BindPresent` rejects: a frame from another surface, a frame from an earlier
  resize generation, and an ordinary render-target texture with the same format
  and extent — the third being the one a format check alone would accept;
- a frame acquired from the matching surface binds and reaches the present state,
  and the graph carries the transition as the pass's store rather than as a node;
- the headless frame loop runs several frames with double buffering and a resize
  in the middle, verified by readback, on every backend and with no display;
- a resize increments the generation, and a graph built before it fails
  `BindPresent` with a stale-surface error naming both generations;
- an acquire that races a resize reports `ErrSurfaceOutOfDate` rather than
  reallocating;
- `Acquire` with an expired timeout reports expiry rather than blocking; and
- Metal's `CAMetalLayer` drawable path presents on a machine with a display,
  tracked as its own claim.

## 9. Open questions

- **Present modes.** FIFO, mailbox, and immediate have different names and
  different availability on every backend, and the choice interacts with
  `Acquire`'s blocking behaviour. Reported as a capability and defaulted to the
  one every backend has (FIFO); which others to expose is undecided until a
  caller needs one.
- **Frame pacing.** Nothing here measures or limits frame rate. Whether accel
  should report the compositor's refresh interval, or stay out of it entirely, is
  open — it is information the swapchain has and the caller cannot easily get.
- **Multiple surfaces on one device.** The design admits it and nothing tests it.
  The state to watch is per-surface generation counters interacting with graphs
  that write two swapchains, which no corpus case builds.
