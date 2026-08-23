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

**Built and not yet built.** Every entry here is normative, and not every entry
has code behind it. The CPU backend and Metal are built; the GL, GLSL, Vulkan
and D3D12 entries state what a backend **will be required to do** and bind
whoever writes it. The graphics entries — clip depth range, face winding,
readback origin — are implemented today only by the CPU reference rasterizer,
since there is no GPU render path yet. Where an entry is a requirement rather
than a description, it says so.

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

**Guarantee.** A `//accel:helper` **may** take a resource slice, and the access
it makes is attributed to the caller's binding rather than to the helper — which
is why access inference records a mode per binding and merges it into the
caller's (`internal/kernelc/front/front.go:547`). MSL lowers such a parameter to
a `device T*`, const-qualified where it is never written through
(`internal/kernelc/emit/msl.go:300`).

A target that cannot pass a buffer to a function must **inline** the helper. That
always terminates: the call graph is checked acyclic and is finite.

> Earlier revisions of this file stated the opposite — that the compiler rejects
> a buffer parameter on a helper. It never did, and
> [`specs/013`](../specs/013-kernel-subset.md) §2 builds the access-propagation
> machinery precisely because helpers take slices.

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
| `RGBA8Unorm`, `RGBA8UnormSRGB`, `BGRA8Unorm` | 4 |
| `R16Float` | 2 |
| `RG16Float`, `R32Float` | 4 |
| `RG32Float`, `RGBA16Float` | 8 |
| `RGBA32Float` | 16 |
| `Depth32Float` | 4 |
| `Depth24PlusStencil8` | reports **0**, and is not host-copyable — see below |

`Device.FormatInfo(f).BytesPerPixel` is the single source of truth. This table is
a reader's summary of it and will go stale first.

`Depth24PlusStencil8` reports **0** bytes per pixel rather than a number: it has
no single defined value, because backends are free to store it as 24-plus-8
packed or as 32-plus-8 padded, and inventing a stride would be wrong somewhere.

**No depth format is host-copyable**, not only that one. `format.go:171` sets
`HostCopyable = !depth`, so `Depth32Float` cannot be read back directly either.
Reading any of them reports that the format *is device-private, which several
backends require of depth formats and this one enforces so the rule is not
discovered in production* (`texturealloc.go:155`). The way to get depth to the
host is a texture-to-buffer transfer node, which is device-to-device and works
on every backend — see [`005`](../specs/005-graphics.md)'s handoff section.

> An earlier revision said `Depth24PlusStencil8` alone was non-host-copyable and
> that the error names `Depth32Float` as the format to use instead. Neither was
> true.

### A driver.Block may be a handle, not your allocation

**Rule for every backend, not a divergence.** A `driver.Block` reaching a
backend is not always one that backend allocated. accel's shared transient pool
hands graphs a handle so it can grow — swap the allocation underneath — without
invalidating the operands and executables that captured it at build time.

So a backend that type-asserts a `Block` to its own concrete type must call
`driver.Unwrap` first, and must do it **at use, never at compile**. Resolving
while lowering a plan caches the pre-growth allocation, which then survives a
growth as a stale pointer, and the symptom is one graph reading another's
memory rather than a failure.

Missing the unwrap is loud, which is deliberate: the assertion fails and the
error names the wrapper — *"names a `*accel.poolBlock`, which is not Metal
memory"*. Missing the *timing* is silent, which is why it is written here.
See [`specs/031-shared-transients.md`](../specs/031-shared-transients.md).

---

## Execution and lifetime

### GL has no command buffers

**Divergence.** GL is a synchronous state machine bound to a thread-current
context, with no native command buffer.

**Guarantee.** The GL backend owns a goroutine locked to an OS thread that holds
the context, and replays a recorded command list on it. This is invisible to the
caller, and is why the recording model costs GL
nothing: it was going to record anyway.

### Apple GPUs flush a subnormal result to zero

**Divergence.** A subnormal *stored* in a buffer survives a round trip, and a
subnormal the compiler folds survives, but arithmetic that **produces** one
returns zero. Measured on an M2, with the runtime operand in a buffer so nothing
is folded:

| Expression | Result |
| --- | --- |
| store `2⁻¹⁴⁹`, read it back | `0x00000001`, preserved |
| `ldexp(1.0f, -149)` as a constant | preserved, folded on the host |
| `x + 0.0f` where x is `2⁻¹⁴⁹` | **0** |
| `2⁻¹⁴⁸ * 0.5f` | **0** |
| `(2⁻⁷⁰)²` | **0** |

The CPU backend preserves them, so this is a real difference between the oracle
and the device, not a property of f32.

**Guarantee.** None, and that is the point:
[`008`](../specs/008-numerics.md) makes exactness a property of *(class, domain,
profile)*, and the normal-result condition belongs to a comparison rather than
to the machine. So `probe.Profile.SubnormalsPreserved` is **false** for Metal,
`ExactAvailable` stays true because it asks only about rounding and contraction,
and a comparison whose values reach the subnormal range is one Metal cannot be
held to. The probe pins the measurement in both directions: a device that began
preserving them would widen the domain and the test says so, which is the one
direction [`009`](../specs/009-sequencing.md)'s risk row permits.

### Metal fuses a multiply-add unless a pragma says otherwise

**Divergence.** Metal compiles with `-ffp-contract=fast` by default, so `a*b+c`
becomes `fma(a, b, c)` and differs from a separately rounded product in the last
bit. `MTLCompileOptions` looks like the control for this and is not:
`MTLMathMode.safe` disables reassociation and denormal flushing and leaves the
multiply-add free to fuse. Measured on an M2 with x = 1 + 2⁻¹², where the two
answers are 2⁻¹¹ and 2⁻¹¹ + 2⁻²⁴.

**Guarantee.** Every emitted kernel carries `#pragma METAL fp contract(off)`,
and a device test asserts both that the pragma disables contraction and that the
default does not. [`008`](../specs/008-numerics.md) §6 requires contraction to
be controlled rather than observed, and a compile option that silently does not
control it is the worst version of observed.

### Private buffers are addressable on Apple silicon

**Divergence.** Metal documents `-[MTLBuffer contents]` as nil for
`MTLStorageModePrivate`. On an Apple M2 it returns a valid pointer, for every
storage mode, because unified memory means the allocation genuinely is
addressable and only the API contract says otherwise. Measured, not remembered:
a buffer created at mode 2 reports `storageMode = 2` and a non-nil `contents`.

**Guarantee.** The requested storage mode decides host visibility, and
`-contents` is consulted only for modes already known to be mappable. Asking
the object instead would make every buffer mappable on Apple silicon and not on
an Intel Mac, which turns a portability rule into a machine-specific one, and
`Block.Bytes()` is the one place [`006`](../specs/006-backends.md) §1 puts that
rule.

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
   backend to disagree with it.

2. **A parity test proves only the path it exercises.** The readback-origin entry
   is the case in point: a compute-path test passes while the texture path is
   mirrored. Cover each path that has its own convention.

3. **Measure and attribute before theorising.** Every entry here looked like a
   mathematics error at first. Coverage counts, overlap between competing
   interpretations, and substituting one input at a time localise a convention
   bug in a couple of iterations. Re-deriving the mathematics does not, because
   the mathematics is usually right.
