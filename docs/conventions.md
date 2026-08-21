# Backend conventions

Normative table of places where GPU backends genuinely disagree, and what
`accel` guarantees on top of them.

Every entry here was found empirically, most of them by a debugging session that
cost hours because the symptom looked like a mathematics bug rather than a
convention mismatch. They are recorded so nobody has to rediscover them. A
backend that violates an `accel` guarantee is broken, not merely different.

## How to read this

Each entry states the divergence, the guarantee `accel` presents to callers, and
where the correction belongs. "Correction belongs in the backend" means the
public API never exposes the divergence.

---

## Coordinate systems and clip space

### Clip-space depth range

| Backend | Native NDC z |
| --- | --- |
| OpenGL / OpenGL ES | `[-1, 1]` |
| Metal | `[0, 1]` |
| Vulkan | `[0, 1]` |
| D3D12 | `[0, 1]` |

**Divergence.** A projection matrix producing GL-convention `[-1, 1]` silently
loses everything in the near half on Metal, Vulkan, and D3D12: those clip `z < 0`
away entirely. The symptom is not a distorted image, it is *no coverage at all*
for near geometry, which reads like a broken transform rather than a range
mismatch.

**Guarantee.** `accel` presents a single clip convention to kernels. The backend
remaps: a Metal vertex stage emits `z' = (z + w) / 2` and a fragment stage
recovers the caller's range as `position.z * 2 - 1`.

**Correction belongs in the backend**, in the emitted shader, never in the
caller's matrix.

### Face winding

**Divergence.** Metal's default front-facing winding is the opposite of GL's for
the same vertex order.

**Why it is nasty.** Getting this backwards keeps *back* faces instead of front
faces. The silhouette is still correct, so the image looks broadly right while
every per-pixel attribute is from the wrong surface. It presents as a shading
bug, not a culling bug.

**Guarantee.** Winding is specified explicitly in the pipeline descriptor and
means the same thing on every backend.

### Texture readback origin

**Divergence.** Reading a render target back yields **bottom-origin** rows on
both GL and Metal: source row `r` is screen row `h - 1 - r`. On Metal this holds
*despite* its top-left texture origin, so reasoning from the documented origin
gives the wrong answer.

**Compute buffers are not flipped.** A storage buffer written by a compute
kernel reads back linearly. So a pipeline that reads its G-buffer from a compute
SSBO never sees the flip, and one that reads the same data from a *texture* does.
A test covering only the compute path will not catch it.

**Diagnostic that identifies it fast.** Compare coverage counts and overlap
between a flipped and unflipped interpretation. Equal pixel counts with roughly
half overlap is the flip fingerprint, and it distinguishes a flip from an
interpolation error in one measurement.

**Guarantee.** `accel` returns readback in caller order. The backend flips.

---

## Compute model

### Fragment-shader buffer writes are unordered

**Divergence.** In core GLES 3.1 and GL 4.x, buffer writes from a fragment shader
have no ordering guarantee with respect to other fragments covering the same
pixel. `layout(early_fragment_tests) in` controls *whether* a fragment runs, not
what order the survivors write in.

**Consequence.** Writing a G-buffer from a fragment shader into a storage buffer
produces a nondeterministic winner wherever geometry overlaps: the depth buffer
ends up correct while the buffer holds the farther fragment's attributes. Writing
through render targets with the depth test on does not have this problem, because
the ROP applies the depth test and the colour write as one indivisible operation.

**Guarantee.** `accel` does not offer unordered fragment writes as a way to
produce a deterministic buffer. Where rasterizer-ordered access is genuinely
needed it is a queryable capability (`ARB_fragment_shader_interlock`, Metal raster
order groups) and absent by default.

### Passing buffers to functions

**Divergence.** GLSL ES 3.1 cannot pass a storage buffer block to a function.
MSL can, as a pointer parameter.

**Guarantee.** Kernel helper functions take values, not buffers. Buffer indexing
stays in the entry point. The kernel compiler rejects a buffer parameter on a
helper with an error naming the restriction, rather than emitting code that fails
inside the driver.

### Integer literals in GLSL

**Divergence.** GLSL forbids mixing `uint` with `int` literals, and idiomatic
kernel code writes `gid*4` constantly.

**Guarantee.** The thread id binds to an `int` local in emitted GLSL. Explicit
conversions in the source still work.

### Reserved words

`out`, `in`, `buffer`, `uniform` and friends are GLSL keywords and are entirely
ordinary Go identifiers, `out` especially so for an output buffer. Emitted GLSL
suffixes colliding names.

---

## Resources

### Depth texture storage mode

**Divergence.** On macOS a depth texture must be created private. Shared storage
is invalid for depth and fails at creation.

### Readback stride by format

**Divergence.** Bytes per pixel must come from the format. Assuming 4 gives a
buffer a quarter of the size a 32-bit-per-channel format needs, and the failure
is an out-of-range panic during readback rather than a wrong image.

| Format | Bytes per pixel |
| --- | --- |
| RGBA8 | 4 |
| RGBA16F | 8 |
| RGBA32F | 16 |
| Depth32F | 4 |

---

## Execution and lifetime

### GL has no command buffers

**Divergence.** GL is a synchronous state machine bound to a thread-current
context, with no native command buffer.

**Guarantee.** The GL backend owns a goroutine locked to an OS thread that holds
the context, and replays a recorded command list on it. This is invisible to the
caller, and is why [`design.md`](design.md) decision 1's recording model costs GL
nothing: it was going to record anyway.

### Objective-C object lifetime across completion handlers

**Divergence.** A Metal command buffer completion handler runs *after* the
enclosing autorelease pool has drained. Releasing an autoreleased object from
inside the handler is a use-after-free, and it crashes inside `objc_msgSend` with
a stack that points nowhere useful.

**Guarantee.** Backend-internal. Every thread that makes Objective-C calls holds
its own pool, drained on that same OS thread, and completion handlers never
release objects they did not retain.

---

## Testing implications

Three rules follow from the table and are worth stating separately, because each
was learned by getting it wrong.

1. **The CPU backend is the oracle.** Every convention above is a way for a GPU
   backend to disagree with it. See [`design.md`](design.md) decision 3.

2. **A parity test proves only the path it exercises.** The readback-origin entry
   is the case in point: a compute-path test passes while the texture path is
   mirrored. Cover each path that has its own convention.

3. **Measure and attribute before theorising.** Every entry here looked like a
   mathematics error at first. Coverage counts, overlap between competing
   interpretations, and substituting one input at a time localise a convention
   bug in a couple of iterations. Re-deriving the mathematics does not, because
   the mathematics is usually right.
