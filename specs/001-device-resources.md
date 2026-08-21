---
title: "Devices, memory, and resources"
status: drafted
layer: device
depends_on: []
---

# Devices, memory, and resources

The bottom of layer 1: how a device is obtained, how memory is allocated, and
what a resource is.

Everything else in layer 1 sits on this. [002](002-compute-model.md) binds the
buffers allocated here, [003](003-command-graph.md) plans transients into the
pools defined here, [005](005-graphics.md) renders into the textures formatted
here, [006](006-backends.md) reports per backend what each memory kind is
actually worth, and [007](007-tensor-layer.md) is the consumer that decides
whether the numbers are right, because it allocates thousands of buffers and
views most of them.

This spec is long in one place on purpose. Alignment and layout are the part of
a device API that callers cannot discover from the type system, that every
backend disagrees about, and that produces wrong numbers rather than errors when
guessed. Sections 3 and 4 are that part.

---

## 1. Devices

A device is opened by requesting a backend, or by asking for the best available
one. Enumeration reports what is present before anything is opened, so a caller
can choose on the basis of reported capabilities rather than by trying and
catching failures.

Opening never falls back silently. A caller asking for a specific backend that is
unavailable gets an error saying so. Automatic selection is a distinct, explicit
call. Silent fallback turns "my GPU code is slow" into a mystery, and the
predecessor hit exactly this when a renderer quietly ran on the CPU path.

A device carries one or more queues. Whether compute and graphics queues are
distinct is a backend property and is reported, not assumed.
[006](006-backends.md) R8 makes queue topology a backend responsibility, because
the target backends disagree completely: Vulkan exposes queue families with
capability bits, D3D12 has typed command queues, Metal has one general queue, and
GL and the CPU backend have exactly one.

### 1.1 Limits are a separate record from capabilities

`Capabilities` (in `compute.go`) answers "can this device do X". Allocation needs
a different question answered: "what numbers must I round to". Those numbers are
not capabilities. They are always present and always have a value, so they live
in their own record, queried at open.

```go
// Limits are the device's numeric allocation and binding constraints. Unlike
// Capabilities, every field always has a value: the question is never whether
// the limit exists, only what it is.
//
// Every field is queried at device open, never assumed from the platform. A
// backend that cannot query one reports the portable floor from spec 001
// section 3.1, which is conservative and therefore safe.
type Limits struct {
	// MinStorageBufferOffsetAlignment and MinUniformBufferOffsetAlignment are the
	// byte alignments a bound view's offset must satisfy. Suballocation is built
	// on these: a buffer that may be bound must start at a multiple of the
	// strictest alignment its declared usage implies.
	MinStorageBufferOffsetAlignment int
	MinUniformBufferOffsetAlignment int

	// MinBufferCopyOffsetAlignment and MinBufferCopyRowPitchAlignment constrain
	// transfers rather than bindings, and they are why a texture readback can
	// cost a repack. See section 4.
	MinBufferCopyOffsetAlignment   int
	MinBufferCopyRowPitchAlignment int

	// MinTexturePlacementAlignment is the alignment a texture's backing memory
	// must start at inside a pool. It is far coarser than any buffer alignment on
	// some backends, which is why textures and buffers do not share a pool.
	MinTexturePlacementAlignment int

	// MaxBufferBytes is the largest single buffer, MaxPoolBytes the largest single
	// device allocation, and MaxPools the driver's cap on live allocations.
	// MaxPools is the number that makes pooling mandatory rather than merely
	// efficient.
	MaxBufferBytes int
	MaxPoolBytes   int
	MaxPools       int

	MaxTextureExtent2D    int
	MaxTextureExtent3D    int
	MaxTextureArrayLayers int
}

// Limits reports this device's numeric constraints.
func (d *Device) Limits() Limits { panic(ErrNotImplemented) }
```

Splitting `Limits` from `Capabilities` is not tidiness. A caller writes
`if caps.Subgroups` and writes `round(n, lim.MinStorageBufferOffsetAlignment)`,
and those are different kinds of code. Mixed into one record, half the fields are
booleans that gate a path and half are integers that appear in arithmetic, and
the second half get left at zero by a backend that forgot to fill them in, which
is a division by zero or a silent alignment of 1. **A backend that cannot query a
limit reports the portable floor, never zero**, and a zero-valued limit on an
opened device is a test failure (section 11.2).

---

## 2. Memory is allocated from pools, not per resource

The predecessor allocated one device allocation per buffer. That is fine for a
renderer with a handful of buffers and wrong for a model with thousands of
tensors: allocation is expensive, drivers cap the number of allocations, and per
resource allocation forecloses the aliasing that
[003](003-command-graph.md) depends on.

So: a caller allocates a **pool** and suballocates from it. A convenience path
allocates a single buffer from an implicit pool, for callers who genuinely have a
handful.

Pools have a **memory kind**, which is the property that actually matters for
performance:

| Kind | Meaning |
| --- | --- |
| `Device` | Fast for the GPU, not host-visible. Weights, activations. |
| `Upload` | Host-writable, GPU-readable. Staging. |
| `Readback` | GPU-writable, host-readable. Results. |
| `Shared` | Host-visible and device-local where the hardware is unified. |

`Shared` is a real capability on unified-memory hardware, not an alias for
something else, and using it there removes a copy entirely. It is reported per
device rather than assumed from the platform.

[006](006-backends.md) resolves what each kind costs per backend, and two entries
there are load-bearing here. `Device` is `emul` on GL, because GL buffers are
opaque and its usage hints are advisory: the backend cannot make memory
device-local and cannot prevent a host read, so it reports the kind satisfiable
and does not pretend the hint did anything. `Shared` is `cap` on Metal, Vulkan
and D3D12 and `no` on GL and WebGPU.

**Requesting a kind the device reports absent is an error naming the kind and the
device.** It is never an `Upload` pool handed back under a different name. A
caller who sized a KV cache against "no staging copy" and silently got staging
has a performance mystery of exactly the kind section 1 rejects for device
selection.

### 2.1 Pool policy

A pool is created with a kind, a size, and a policy that selects the allocator in
section 5.

```go
// PoolPolicy selects how a pool carves itself up. See spec 001 section 5.
type PoolPolicy int

const (
	// PoolGeneral is a general-purpose pool: arbitrary allocation and free order,
	// O(1) allocate and free, bounded fragmentation. The default, and what a
	// caller holding weights and caches wants.
	PoolGeneral PoolPolicy = iota

	// PoolLinear allocates by bumping a cursor and frees only by resetting the
	// whole pool. Individual Close is a no-op against the memory. This is what a
	// Graph's transient pool uses, because spec 003 computes every offset ahead of
	// time and the pool's contents die together.
	PoolLinear
)

// PoolDescriptor describes a pool to create.
type PoolDescriptor struct {
	Kind   MemoryKind
	Bytes  int
	Policy PoolPolicy

	// Textures reserves the pool for textures. Texture placement alignment is far
	// coarser than buffer alignment on some backends, and some backends forbid the
	// mixture outright, so it is a pool property rather than a per-allocation one.
	// See section 4.4.
	Textures bool

	// Label appears in allocation errors and in backend debug tooling.
	Label string
}
```

`Device.NewPool(kind, bytes)` stays as the two-argument convenience and means
`PoolDescriptor{Kind: kind, Bytes: bytes, Policy: PoolGeneral}`.

---

## 3. Alignment and layout

This is the section a caller cannot infer and cannot test their way into. Getting
it wrong does not produce an error, it produces numbers that are off by one
field, which reads as a mathematics bug. It is the buffer-side analogue of every
entry in [`conventions.md`](../docs/conventions.md).

### 3.1 Offset alignment: the guarantee, and where the real number comes from

A bound resource's byte offset is not free. Every backend requires bound storage
and uniform ranges to start at a multiple of a device-reported alignment, and the
values differ by two orders of magnitude across devices of the same backend.

Two things are stated separately below, and conflating them is how a spec becomes
confidently wrong. The first column is the **accel guarantee**, which this
document defines and every backend must satisfy. The second is **where the real
number comes from**, which is a query, not a remembered constant. No per-backend
constant appears in this table, because [006](006-backends.md) is right that a
confidently wrong pin in a normative spec is worse than an unknown: it gets built
on.

| Constraint | accel guarantee | Resolved at device open by |
| --- | --- | --- |
| Bound storage buffer offset | a multiple of **256 bytes** is always sufficient | Vulkan `VkPhysicalDeviceLimits.minStorageBufferOffsetAlignment`; D3D12 fixed by the API per view kind; Metal the platform family's buffer offset alignment; GL `GL_SHADER_STORAGE_BUFFER_OFFSET_ALIGNMENT`; WebGPU `limits.minStorageBufferOffsetAlignment`; CPU reports the portable floor |
| Bound uniform buffer offset | a multiple of **256 bytes** is always sufficient | Vulkan `minUniformBufferOffsetAlignment`; D3D12 constant buffer placement, fixed by the API; Metal the platform family's buffer offset alignment; GL `GL_UNIFORM_BUFFER_OFFSET_ALIGNMENT`; WebGPU `limits.minUniformBufferOffsetAlignment`; CPU reports the portable floor |
| Copy source or destination offset | a multiple of **16 bytes** is always sufficient | Vulkan's buffer copy offset rules; D3D12 texture data placement alignment; Metal blit offset rules; GL has no equivalent; CPU reports the portable floor |
| Buffer to texture row pitch | accel presents tightly packed rows, the backend pads. See 4.2 | D3D12 `D3D12_TEXTURE_DATA_PITCH_ALIGNMENT`; Vulkan `bufferRowLength` in texels with a texel-block and 4-byte offset rule; Metal `bytesPerRow` rules including `minimumLinearTextureAlignmentForPixelFormat:`; GL `GL_PACK_ALIGNMENT` and `GL_UNPACK_ALIGNMENT` |
| Texture placement inside a pool | textures get their own pools. See 4.4 | D3D12 resource placement alignment, far coarser than any buffer alignment; Vulkan `VkMemoryRequirements.alignment` per texture; Metal heap placement alignment; GL and CPU: not applicable |

**Why 256 is the guaranteed number.** It is at least as large as any bound-offset
alignment any target backend requires, so a caller who rounds to it is correct on
every device without querying anything. Every backend either requires 256 or
requires less, so "round to 256" is never insufficient and is sometimes wasteful.
A caller who wants the waste back queries `Limits` and rounds to the real number
for the device in hand. That is the standard shape for this class of constraint:
a portable constant that is always safe, plus a query for callers who care.

**What it costs.** 256 bytes of granularity against
[007](007-tensor-layer.md)'s thousands of tensors is a real number, so here it
is. A model with 5,000 separately allocated weight planes wastes at worst 255
bytes each, about 1.25 MiB, which against a multi-gigabyte weight set is noise.
It stops being noise when allocations are small and numerous: 100,000 per-head
slices padded to 256 bytes waste up to 25 MiB, and a 128-element f16 head slice
is 256 bytes of payload carrying up to 100 percent overhead.

**The escape hatch is already in the design.** The alignment floor applies to a
**bound** view's offset, not to a view's existence. A view used only as a
transfer endpoint needs only the copy alignment, and a view that is never bound
at all (because the address it represents travels as a `u32` a kernel reads)
needs no alignment beyond its dtype. Both 003 and 007 already prefer that route:
003 records a KV cache write offset as buffer contents rather than as a rebound
view, and 007 measures the alternative at 128 binding updates per token for a
64-layer model. So the case that would pay the alignment tax hardest is the case
the design already routes around, and the tax lands on per-layer slabs, which are
megabytes each.

**Suballocation aligns by intended use.** A buffer's requirement is derived from
its declared `BufferUsage`:

```
required = 4                                             // dtype floor, always
if usage & UsageStorage:  required = max(required, lim.MinStorageBufferOffsetAlignment)
if usage & UsageUniform:  required = max(required, lim.MinUniformBufferOffsetAlignment)
if usage & (UsageCopySrc|UsageCopyDst):
                          required = max(required, lim.MinBufferCopyOffsetAlignment)
if usage & UsageIndirect: required = max(required, 4)    // indirect args are u32 triples
```

This is a second reason usage is declared at creation and not inferred. The
allocator needs the number before it places the allocation, and a buffer that
later turns out to be bound as a uniform cannot be moved: its address is already
in descriptors and, on two backends, in a recorded command buffer.

### 3.2 Storage buffers hold scalars, not structs

**Decision: a storage buffer is a tightly packed array of one scalar dtype. It
has no struct layout, because it has no structs.**

This falls out of the existing surface rather than being imposed on it.
`BufferDescriptor` is a `DType` plus a `Count`, and [004](004-kernel-authoring.md)
maps a storage buffer parameter to a Go `[]T` where `T` ranges over the scalar
dtypes. There is nowhere for a struct to enter. Making that explicit closes the
std140-versus-std430 question for storage buffers entirely:

| Property | Rule |
| --- | --- |
| Element stride | exactly `DType.Size()` bytes, no padding ever |
| Element `i` at byte | `i * DType.Size()`, relative to the view's base offset |
| Base offset units | `BufferView.Offset` counts elements of `BufferView.DType` |
| Byte order | device native, which accel requires to equal host native (3.5) |
| Struct members | not expressible |
| Matrices | not expressible: a matrix is a flat scalar array plus indexing in the kernel |

That is std430's rule for scalar arrays, which every backend supports for shader
storage: GLSL ES 3.1 SSBOs take `std430`, HLSL structured and byte-address
buffers are tightly packed, MSL is a pointer to `T`, SPIR-V takes an explicit
`ArrayStride` equal to the scalar size, and WGSL storage arrays of scalars have
the scalar's stride. There is no divergence to correct, which is exactly why
restricting to scalars is worth doing.

**The honest cost: a caller with array-of-structs data must split it into
planes.** A particle system with a position, a velocity and a mass per particle
is three buffers here, not one. That is a structure-of-arrays discipline forced
by the API. It is the right default for a compute API (coalesced loads want SoA,
and a workgroup reading only positions should not pull velocities through cache),
and [007](007-tensor-layer.md) reached the same conclusion independently for
quantized weights, where quants, scales and zero points are three plane buffers
for the same stated reason: layer 1 types a buffer by dtype and an interleaved
block struct has no dtype.

The cost is sharper for graphics, where a vertex buffer genuinely is
array-of-structs. [005](005-graphics.md) handles that with an explicit vertex
input layout of strides, offsets and attribute formats, which is a struct layout
owned by the pipeline rather than by the buffer. Such a buffer is declared here
as `DType: U8` with `Count` equal to its size in bytes, and 005's layout
interprets it.

### 3.3 Uniform buffers use std140, and the Go side conforms

Uniform buffers are the only place structs exist, and they need a convention
because a kernel's by-value struct parameter has fields at addresses both sides
must agree on.

**Decision: uniform buffer contents use std140, exactly.**

- **std140 is the portable intersection.** GLSL ES 3.1 permits `std140` on
  uniform blocks and does not permit `std430` on them. Every other backend can
  produce std140 (HLSL constant buffers already follow equivalent 16-byte packing
  rules, MSL and SPIR-V take explicit offsets, and WGSL's uniform address space
  imposes 16-byte array and struct strides that std140 satisfies). Picking std430
  for uniforms would leave the GL backend unable to express a uniform block, and
  the only remedy would be promoting every uniform to a storage buffer, losing
  the constant-cache path that makes uniforms worth having.
- **A bespoke accel layout buys nothing.** Any invented layout still needs a host
  encoder and a shader-side decode, and would additionally differ from what every
  shader author already knows.
- **It costs padding, and the padding is visible.** std140 rounds a
  three-component vector to sixteen bytes, aligns a nested struct to sixteen, and
  gives an array of scalars a stride of sixteen. An array of 64 floats in a
  uniform block occupies 1024 bytes, not 256. That is why arrays belong in
  storage buffers and why this decision is confined to uniforms, which are small
  by nature.

This **narrows** [004](004-kernel-authoring.md), which says a by-value struct
parameter is "`constant T&` on the GPU under std140 or std430 layout". 001 pins
it to std140. 004's mechanism is unchanged and is what makes the choice
affordable: the generator emits a per-kernel encoder and decoder, the GPU layout
owns the padding, and the caller's Go struct never declares a padding field. An
unsafe cast from a Go struct to uniform bytes stays rejected, because it is
silently correct for a struct of four floats and silently wrong for the first one
containing a three-float member.

#### std140 rules, stated so they can be checked

| Type | Alignment | Size consumed |
| --- | --- | --- |
| `f32`, `i32`, `u32` | 4 | 4 |
| 2-component vector | 8 | 8 |
| 3-component vector | 16 | 12, and the next member then aligns by its own rule |
| 4-component vector | 16 | 16 |
| array of any type | 16, element stride rounded up to a multiple of 16 | stride times length |
| column-major matrix with `N` columns | 16, as an array of `N` column vectors | 16 times `N` |
| nested struct | 16 | its own size rounded up to a multiple of 16 |
| the block itself | 16 | rounded up to a multiple of 16 |

**How a Go struct maps.** The generator walks the Go struct with `go/types`,
assigns each field the std140 offset from the table, and emits the pair of
functions. The Go struct's own field offsets are irrelevant to the device layout
and are never assumed to match it.

```go
// Authored by the caller. Ordinary Go: no padding fields, no tags.
type Params struct {
	Scale   float32
	Origin  [3]float32
	Steps   uint32
	Inverse [4][4]float32
}

// Generated. The offsets are std140's, not Go's.
//
//	Scale   at   0,  4 bytes
//	Origin  at  16, 12 bytes consumed, aligned to 16 as a 3-vector
//	Steps   at  28,  4 bytes, occupying the tail of Origin's 16-byte slot
//	Inverse at  32,  4 columns of 16 bytes, 64 bytes
//	size        96, already a multiple of 16
const ParamsBlockSize = 96

func encodeParams(dst []byte, p Params) { /* generated */ }
func decodeParams(src []byte) Params    { /* generated */ }
```

**Forbidden in a uniform struct, each for a reason that is not taste:**

| Forbidden | Why |
| --- | --- |
| pointers, slices, maps, interfaces, channels | no device representation exists |
| `bool` | one byte in Go, four on every device |
| `int`, `uint`, `uintptr` | platform-width: a struct whose layout depends on `GOARCH` has no single device layout |
| `float64` | 002 has no f64 dtype, and 007 states f64 exists for training numerics |
| arrays of structs | legal in std140 and a padding trap; excluded until something needs it |
| runtime-sized anything | the block size is baked into the pipeline |
| unexported fields | the generator cannot address them, and skipping one silently produces the wrong block size |

Violations are reported by the kernel compiler at generate time, naming the
struct, the field and the reason, never at dispatch.

#### The uniform buffer's dtype

A std140-encoded block has no scalar dtype, and `BufferDescriptor` requires one.
The resolution, stated explicitly because otherwise this section contradicts
`memory.go`:

**A uniform buffer is declared `DType: U8` with `Count` equal to the encoded
block size in bytes.** On a `BindingUniformBuffer` slot, dtype means bytes rather
than elements, and the graph builder validates the bound range's byte length
against the pipeline's declared block size rather than against a dtype. That is
the same exception a vertex buffer takes in 3.2, and there are exactly two.

```go
params, err := pool.Alloc(accel.BufferDescriptor{
	DType: accel.U8,
	Count: kernels.ParamsBlockSize, // generated constant, 96 above
	Usage: accel.UsageUniform | accel.UsageCopyDst,
	Label: "params",
})
```

### 3.4 Size versus stride versus allocation size

Four numbers are easy to conflate and each has its own rule.

| Number | Definition | Rounded? |
| --- | --- | --- |
| element size | `DType.Size()` | never |
| buffer size | `DType.Size() * Count` | never: this is the caller's number |
| allocation size | buffer size rounded up to the pool's granularity | yes, to 256 by default |
| bound range size | the byte length a binding exposes to a kernel | rounded up to 4 for storage, to 16 for uniform per std140's block rule |

`PoolStats.Used` reports allocation sizes, not buffer sizes, so a caller
comparing `Used` against the sum of their buffer sizes sees the alignment tax
rather than suspecting a leak. That is deliberate: the tax should be visible in
the number a caller already looks at.

### 3.5 Byte order

**accel requires the host and the device to share byte order, and every
supported platform is little-endian.**

All target backends run on little-endian hosts in every configuration accel
targets, and the CPU backend is the host. So `Buffer.Write` is a memory copy and
never a byte swap, `ViewAs` is a reinterpretation and never a conversion, and a
kernel reading `u32` sees exactly the bytes the host wrote.

A big-endian `GOARCH` is not supported: `Open` fails naming the architecture,
rather than producing silently byte-swapped results. Stating this is cheap, and
the alternative is a bug class that only appears on hardware nobody here owns.

---

## 4. Textures, formats, row pitch, and copy alignment

Textures exist for graphics and for sampled reads. They carry a format, extent,
mip levels, array layers, and a usage set.

Formats are distinct from buffer dtypes even where they name the same width,
because texture formats carry sampling and colour-space semantics a buffer dtype
does not. Bytes per pixel always comes from the format; see
[`conventions.md`](../docs/conventions.md) for why assuming it is a real bug.

Depth formats are constrained by backends in ways colour formats are not,
including a macOS requirement that depth textures be device-private. The backend
enforces this rather than the caller learning it.

### 4.1 The format table

**BPP** is bytes per pixel, which [`conventions.md`](../docs/conventions.md)
already fixes for four of these and this table completes. **Render** is usable as
an attachment. **Sample** is readable by a sampler, `filter` meaning linear
filtering is also available and `point` meaning nearest only. **Storage** is
usable as a storage image a kernel writes. Backend marks follow
[006](006-backends.md): `yes` is architecturally guaranteed, `cap` is queryable
and device-dependent, `no` is architecturally absent, `?` is unknown and must be
measured at first contact rather than remembered.

| Format | BPP | Channels | Render | Sample | Storage | CPU | Metal | Vulkan | D3D12 | GLES 3.1 | WebGPU |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `RGBA8Unorm` | 4 | R,G,B,A 8-bit normalized | yes | filter | yes | yes | yes | yes | yes | yes | yes |
| `RGBA8UnormSRGB` | 4 | as above, sRGB transfer on RGB | yes | filter | **no** | yes | yes | yes | yes | yes | yes |
| `BGRA8Unorm` | 4 | B,G,R,A 8-bit normalized | yes | filter | cap | yes | yes | cap | yes | no | yes |
| `R16Float` | 2 | R half | yes | filter | cap | yes | yes | cap | yes | cap | cap |
| `RG16Float` | 4 | R,G half | yes | filter | cap | yes | yes | cap | yes | cap | cap |
| `RGBA16Float` | 8 | R,G,B,A half | yes | filter | yes | yes | yes | yes | yes | cap | yes |
| `R32Float` | 4 | R float | yes | point | yes | yes | yes | yes | yes | cap | yes |
| `RG32Float` | 8 | R,G float | yes | point | cap | yes | yes | cap | yes | cap | cap |
| `RGBA32Float` | 16 | R,G,B,A float | yes | point | yes | yes | yes | cap | yes | **cap** | cap |
| `Depth32Float` | 4 | depth | depth only | point, compare | no | yes | yes | yes | yes | yes | yes |
| `Depth24PlusStencil8` | **opaque** | depth + stencil | depth only | point, compare | no | yes | cap | yes | yes | yes | yes |

Four rows need their reasoning stated, or they read as arbitrary.

**`RGBA8UnormSRGB` is never storage-capable.** An sRGB format's meaning is a
transfer function applied by fixed-function hardware on read and write, and a
storage image write bypasses fixed function. No target backend offers an sRGB
storage image, so accel reports it absent everywhere rather than emulating it and
producing a slightly different curve per backend. A kernel that wants sRGB output
writes linear values into a linear format and lets a later pass convert, or
applies the curve itself.

**`BGRA8Unorm` exists because swapchains ask for it**, not because a caller
should pick it. It is the native presentable format on several platforms, so
[005](005-graphics.md)'s surface reports it and a frame graph writes it. GLES has
no such storage image format, hence `no` in that column.

**Filterability is not sampleability.** The 32-bit float formats are sampleable
everywhere and linearly filterable only where the device says so
(`float32-filterable` in WebGPU, a format feature bit in Vulkan, a family query in
Metal). The table says `point` because point sampling is the guarantee. Assuming
linear is how a G-buffer resolve ends up with nearest-neighbour artefacts on one
vendor only.

**`Depth24PlusStencil8` has no defined bytes per pixel.** The name follows
WebGPU's convention and means "at least 24 bits of depth, exact layout
unspecified". Metal's 24-bit depth-stencil format is a device capability rather
than universal, while D3D12 and Vulkan pack it into a 32-bit unit. Because the
layout is device-defined, **this format is not host-copyable**: `Texture.Read` on
it is an error, and buffer copies of it are an error, each naming the format and
pointing at `Depth32Float`. That is the honest answer for a format whose point is
to let the driver choose, and it means 005's depth readback path uses
`Depth32Float`, which has a defined layout.

```go
// FormatInfo describes what a format is and what this device can do with it.
// Every field is a device answer, not a constant, because several rows above are
// per device.
type FormatInfo struct {
	BytesPerPixel int // 0 when the layout is opaque, as for Depth24PlusStencil8
	Channels      int
	IsDepth       bool
	IsStencil     bool
	IsSRGB        bool

	Renderable   bool
	Sampleable   bool
	Filterable   bool
	StorageRead  bool
	StorageWrite bool
	HostCopyable bool
}

// FormatInfo reports what this device can do with a format. A format the device
// does not support at all reports every capability false rather than erroring, so
// a caller can survey formats in a loop.
func (d *Device) FormatInfo(f Format) FormatInfo { panic(ErrNotImplemented) }
```

**Cross-check against [005](005-graphics.md), and one finding.** 005's deferred
renderer writes albedo as `RGBA8Unorm`, normals as `RGBA16Float`, world position
as `RGBA32Float` and depth as `Depth32Float`, then reads all four from compute
and writes an HDR storage texture. Against the table, `RGBA32Float` as a colour
attachment is `cap` on GLES 3.1, not `yes`, because colour-renderable 32-bit
float targets are an extension there rather than core. So **005's worked example
is not guaranteed to run on the GL backend**, and a GL device reporting that
capability absent needs a different G-buffer layout (world position reconstructed
from depth, which a bandwidth-conscious renderer does anyway). This is a gap in
005 against 006's GL column, surfaced here because this spec owns the format
table.

### 4.2 Row pitch: the guarantee, and who pays

**Guarantee: at the accel API boundary, texture data is tightly packed. Row `r`
of mip level `m` begins at byte `r * width(m) * BytesPerPixel`, with no padding
between rows and none between layers or mip levels beyond what that arithmetic
implies.**

This holds for `Texture.Read`, `Recorder.CopyTextureToBuffer`, and
`Recorder.CopyBufferToTexture`. A caller sizes a readback buffer as
`width * height * bpp` and is always right.

The backends do not all agree. D3D12 requires the row pitch of a buffer in a
texture copy to be a multiple of `D3D12_TEXTURE_DATA_PITCH_ALIGNMENT` (256) and
the copy's placement within the buffer to be a multiple of
`D3D12_TEXTURE_DATA_PLACEMENT_ALIGNMENT` (512). Vulkan expresses the same idea
differently, as a `bufferRowLength` counted in texels with the buffer offset
constrained to multiples of the texel block size and of 4. Metal takes a
`bytesPerRow` with per-format rules. GL has `GL_PACK_ALIGNMENT` and
`GL_UNPACK_ALIGNMENT`, defaulting to 4, which silently pads any row whose byte
length is not a multiple of 4. The two D3D12 constants are the only per-backend
numbers stated as literals anywhere in this spec, because they are fixed by the
API rather than queried per device; everything else is a `Limits` query.

**Where the padding happens: inside the backend, in an intermediate buffer the
caller never sees.** When the caller's tight pitch does not satisfy the backend's
requirement, the backend allocates scratch with the padded pitch, does the device
copy into it, then repacks row by row into the caller's tight buffer. That is one
extra full-size copy of the image.

**Who pays: the caller whose row length is not already aligned, on the backends
that require alignment, and nobody else.** A 1024-wide `RGBA8Unorm` target has a
4096-byte row and repacks nowhere. A 100-wide one has a 400-byte row, not a
multiple of 256, so it repacks on D3D12. The cost is proportional to the image and
otherwise silent, which is the wrong kind of silent on a performance path, so it
is reported:

```go
// CopyStats reports what a transfer does. It is a plan-time fact, not a
// measurement: the backend knows at build whether a copy's pitch needs padding,
// so a recorded copy carries this from Graph.NodeStats as soon as Build returns
// (spec 003, Statistics). The immediate path has no node, so its repacks are
// counted in Queue.Stats instead of returned per call.
type CopyStats struct {
	Bytes    int
	Repacked bool // an intermediate padded-pitch buffer was used
	RowPitch int  // the pitch the backend used on the device side
}

// AlignedRowPitch reports the row pitch this device uses for a texture copy of
// the given format and width. When it equals width*BytesPerPixel there is no
// repack. A caller who sizes their own buffer to this value and uses the aligned
// copy entry point avoids the repack entirely.
func (d *Device) AlignedRowPitch(f Format, width int) int { panic(ErrNotImplemented) }
```

The escape hatch is deliberate and narrow. A caller willing to handle padded rows
(a video encoder, a screenshot writer about to compress anyway) asks for the
aligned pitch, sizes their buffer to `AlignedRowPitch(f, w) * h`, and uses the
copy variant that writes padded rows. Everyone else gets tight packing and
possibly a repack. The default is correct and simple, the fast path is available
by asking.

### 4.3 Row pitch composes with the readback origin flip, and the order is normative

[`conventions.md`](../docs/conventions.md) guarantees readback arrives in caller
row order, and that the backend flips where its native order is bottom-origin.
This spec guarantees rows are tightly packed. Both corrections land in the same
call, and a backend author who applies them in the wrong order produces an image
correct in one axis and scrambled in the other. The flip fingerprint in
conventions.md (equal pixel counts, roughly half overlap) will **not** identify
that, because a pitch error changes the pixel count.

So the order is normative:

> **Depad first, then flip.** The backend repacks padded device rows into tight
> rows, then reverses row order if its native origin requires it. A backend may
> fuse the two passes provided the observable result is identical to doing them
> in that order.

As an invariant a test can check: after `Texture.Read` returns, byte
`(r*width + x) * bpp` holds the pixel the fragment stage wrote at window
coordinate `(x, r)` with `r = 0` the top row, on every backend, for every width,
whether or not `width * bpp` is a multiple of anything.

### 4.4 Texture placement, and why textures get their own pools

A texture's backing memory has a coarser alignment than any buffer's. D3D12 fixes
a resource placement alignment far above any buffer alignment, with a smaller
tier for small textures; Vulkan reports the requirement per texture through
`VkMemoryRequirements`. Some backends additionally restrict which memory types a
texture may occupy, so a heap holding both buffers and textures is not always
expressible.

**Decision: a pool is either a buffer pool or a texture pool, chosen at
creation.** Mixing them would mean either padding every buffer to the texture
alignment (tens of kilobytes of granularity applied to a model's thousands of
tensors is not a tax, it is a fatal multiplier) or running two allocators with
different granularities inside one pool, which is the complexity of two pools
without the clarity.

The cost: a caller who wants one budget for both keeps two pools and adds the
numbers. That is a real inconvenience for a renderer and the right trade against
a tensor workload that would otherwise be destroyed by it.

---

## 5. Suballocation

A pool is a single device allocation. This section is how it is carved up.

Note the division of labour with [003](003-command-graph.md), because both
documents place memory. **003 owns transient placement within one graph**: it
computes an interference relation over DAG reachability, packs transients
size-descending first-fit, and hands the pool a set of precomputed offsets. **001
owns how a pool is carved up in general**, which is what serves caller
allocations whose lifetimes accel cannot know. The two meet at `PoolLinear`: a
graph's transient pool is linear precisely because 003 already solved placement
offline, so that pool needs no runtime allocator at all.

### 5.1 The workload, before choosing

An allocator is only justifiable against a workload, so here is the one that
matters, from [007](007-tensor-layer.md) and [005](005-graphics.md):

| Client | Count | Size range | Lifetime |
| --- | --- | --- | --- |
| model weights | thousands, one to three planes per tensor | kilobytes to hundreds of megabytes | allocated once at load, freed at shutdown |
| KV cache | a handful | hundreds of megabytes | the session's |
| graph transients | hundreds per graph | kilobytes to megabytes | placed by 003, freed together |
| render targets | tens | megabytes | rebuilt on resize |
| staging blocks | a ring of a few | fixed, megabytes | recycled continuously |
| convenience buffers | tens | anything | arbitrary |

Two properties dominate. First, **the great majority of allocations are made once,
in a burst, and freed together at the end**. Second, **the churn that does exist
is at a small number of fixed sizes** (the staging ring) rather than at arbitrary
ones.

### 5.2 The choice: TLSF for general pools, bump for linear pools

**Decision: `PoolGeneral` uses TLSF (two-level segregated fit). `PoolLinear`
bumps a cursor.**

TLSF keeps a two-dimensional array of free lists indexed by the exponent and a
fixed number of mantissa bits of a block's size, plus a bitmap over each level.
Allocation finds the smallest size class guaranteed to satisfy the request, takes
its head, and splits the remainder. Free coalesces with physically adjacent free
neighbours through boundary tags and pushes the result onto its class. Both
operations are a handful of bit operations and pointer writes.

| Property | TLSF | Bump | Buddy | First-fit list | Fixed size classes |
| --- | --- | --- | --- | --- | --- |
| allocate | O(1) | O(1) | O(log n) | O(n) | O(1) |
| free | O(1) | unsupported | O(log n) | O(1), O(n) with coalescing | O(1) |
| internal fragmentation | bounded by mantissa resolution, a few percent | zero | up to 2x | zero | up to the class gap |
| external fragmentation | good-fit, low in practice | not applicable | low | high, order-dependent | none within a class, high between |
| implementation size | about 250 lines | about 20 | about 150 | about 80 | about 100 |

Why the others lose against this workload:

- **Buddy** rounds every allocation to a power of two. A 640 MiB KV cache becomes
  1 GiB. That is fatal on an 8 GiB device, and it happens to the one allocation
  the caller sized deliberately.
- **First-fit free list** is O(n) in live blocks, so loading thousands of weight
  tensors becomes quadratic in the tensor count, on the path a user experiences as
  "loading the model".
- **Fixed size classes** are excellent for the staging ring and bad for weights,
  whose sizes come from a model's dimensions and do not cluster. The gap between
  classes is wasted per allocation and unbounded across classes.
- **Bump** cannot free, which is right for a graph's transient pool and wrong for
  anything a caller closes individually.

Why TLSF wins: O(1) both ways, good-fit rather than first-fit so a large free
block is not chopped up to serve a small request, internal fragmentation bounded
by a resolution the implementation chooses (four mantissa bits gives at most about
6 percent overshoot), and small enough to read in one sitting and test
exhaustively. The workload's burst of thousands of same-phase allocations is the
case it handles best, because nothing is freed during the burst so the free lists
stay short.

**Alignment folds in cleanly.** The allocator's granularity is the pool's
alignment floor (256 by default, section 3.1), every block size and offset is a
multiple of it, and a per-allocation alignment stricter than the floor is served
by over-allocating and trimming the head back onto the free list.

### 5.3 Fragmentation behaviour, stated honestly

**There is no compaction, and there cannot be within a pool's life.** A device
address is baked into descriptor sets on Vulkan, descriptor heap entries on
D3D12, an argument buffer or encoded binding on Metal, and, on the two backends
with a native replayable command object, into recorded commands. Moving an
allocation would mean rewriting every binding naming it and invalidating every
graph that recorded it. So external fragmentation inside a pool is permanent for
that pool's lifetime.

The consequence, which callers must be told rather than left to discover: **a
long-running process that allocates and frees buffers of varying sizes in one
general pool will eventually fail an allocation while `PoolStats.Free` reports
plenty of space.** That is not a bug, it is what a non-compacting allocator is.

The mitigation is not a cleverer allocator, it is **separating pools by lifetime
class**, and the table in 5.1 is that separation:

- weights in their own pool, allocated once, never freed individually,
- the KV cache in its own pool or as its own allocation,
- transients in a graph-owned `PoolLinear`,
- staging in a ring of fixed-size blocks,
- everything else in a general pool that is expected to fragment.

Grouping by lifetime rather than by kind is the single technique that makes
non-compacting allocation work, and it is why `PoolDescriptor` takes a policy and
a label instead of accel trying to guess.

`PoolStats` therefore reports enough to see fragmentation coming:

```go
// PoolStats reports a pool's occupancy.
type PoolStats struct {
	Size int
	Used int // sum of allocation sizes, which includes alignment padding
	Free int // Size - Used

	// LargestFree is the biggest single allocation this pool can still serve. The
	// gap between Free and LargestFree is fragmentation, and it is the number that
	// predicts the failure in section 5.3 rather than reporting it afterwards.
	LargestFree int

	// Allocations is the live count and Blocks the number of free blocks. Rising
	// Blocks against flat Allocations is fragmentation accumulating.
	Allocations int
	Blocks      int
}
```

### 5.4 When a pool cannot satisfy an allocation

It fails. It does not grow, it does not spill into another pool, and it never
returns a smaller buffer.

```
accel: alloc "kv_cache_layer_7" (48.0 MiB, align 256) failed in pool "weights"
       (Device, 2.0 GiB): 61.4 MiB free in 214 blocks, largest 12.3 MiB.
       The pool has space but not contiguous space; see spec 001 section 5.3.
```

The message carries the failing allocation's label, the pool's label and kind,
the request with its alignment, and both `Free` and `LargestFree`, because those
two numbers together distinguish exhaustion from fragmentation. Those are
different problems with different fixes and are indistinguishable from a bare
failure.

### 5.5 Whether pools grow: resolved

The previous draft left this open, leaning fixed. It resolves as follows, and the
resolution is forced by what a device allocation is rather than chosen for taste.

**A pool never grows.** A pool is exactly one device allocation
(`VkDeviceMemory`, a `MTLHeap`, an `ID3D12Heap`, a GL buffer object), and no
target backend can resize one in place. Growing would mean allocating a second,
larger allocation and copying, which invalidates every address already handed
out, which 5.3 has just established is impossible. The choice is not between
fixed and growable, it is between fixed and lying.

**The implicit pool behind `Device.NewBuffer` grows by adding blocks.** That is
what a caller actually means by growth and it is expressible: the implicit pool
is a *set* of fixed-size device allocations (default 64 MiB per block, or the
request rounded up when the request is larger), and when no block can serve a
request the set adds one. Nothing moves, no address is invalidated, and the
driver's cap on live allocations is respected by keeping blocks large.

**Explicitly created pools do not grow, because their purpose is to make the
requirement a number the caller committed to.** 003 exposes a graph's memory
requirement before submission and 007 documents sizing a machine by compiling the
worst-case plan and asking. A pool that quietly grew would make both numbers
advisory, and the failure a fixed pool reports at allocation time would instead
surface as an out-of-device-memory error from the driver at an unrelated moment.

The cost, stated: a caller who mis-sizes a pool gets an error and must recreate
it, which for a weights pool means reloading. That is why the requirement is
reported in advance in three places (`Graph.Memory`, `Plan.Memory`, and 007's KV
cache sizing formula) rather than left to trial.

---

## 6. Buffers

A buffer is a typed, sized range within a pool: a dtype (see
[002](002-compute-model.md)), an element count, and a usage set declared at
creation.

Usage is declared up front because backends need it: it decides the underlying
allocation flags, and a mismatch is a validation error at build rather than
undefined behaviour at execution. Section 3.1 adds the second reason: usage also
decides alignment, and alignment is decided before placement.

| Usage | Implies | Alignment contribution |
| --- | --- | --- |
| `UsageStorage` | bound to a `BindingStorageBuffer` slot | `MinStorageBufferOffsetAlignment` |
| `UsageUniform` | bound to a `BindingUniformBuffer` slot, contents std140 | `MinUniformBufferOffsetAlignment` |
| `UsageIndex` | index buffer for `DrawIndexed` (005) | index width, 2 or 4 |
| `UsageVertex` | vertex buffer, interpreted by 005's vertex layout | 4 |
| `UsageIndirect` | source of dispatch or draw arguments | 4 |
| `UsageCopySrc` | source of a transfer | `MinBufferCopyOffsetAlignment` |
| `UsageCopyDst` | destination of a transfer, and of `Buffer.Write` | `MinBufferCopyOffsetAlignment` |

Declaring a usage a buffer does not need costs alignment and possibly a stricter
memory type. Declaring one it does need late is a build error naming the buffer,
the node and the missing usage. The asymmetry is intentional: over-declaring
wastes bytes, under-declaring is a bug, so the error lands on the side that is
wrong.

### 6.1 Views

A view is a sub-range of a buffer, optionally reinterpreted at a different dtype
where the sizes work out. Views are what let the tensor layer slice a KV cache or
address one attention head without copying. A view does not own memory and cannot
outlive its buffer.

```go
// BufferView is a sub-range of a [Buffer], possibly at a different dtype. It is a
// value: copying it is fine and does not copy any memory.
//
// Offset and Count are in elements of this view's DType, never in bytes. The byte
// range described is [Offset*DType.Size(), (Offset+Count)*DType.Size()) relative
// to the start of Buffer.
type BufferView struct {
	Buffer *Buffer
	DType  DType
	Offset int
	Count  int
}
```

**Elements, not bytes, and the rule is uniform.** `Buffer.View(offset, count)`
takes elements of the buffer's dtype. `Buffer.ViewAs(d, offset, count)` takes
elements of `d`, the new dtype, not of the buffer's. Anything else would make
`ViewAs` require the caller to do the conversion arithmetic `ViewAs` exists to
do. [007](007-tensor-layer.md) states the same rule for tensor strides for the
same reason: byte arithmetic at this boundary reintroduces the dtype-width
multiplication that typing the buffer removed, and it is actively wrong for
sub-byte quantized data.

**Alignment of a view's offset depends on what the view is for:**

| The view is | Byte offset must be a multiple of |
| --- | --- |
| bound to a storage slot | `MinStorageBufferOffsetAlignment` |
| bound to a uniform slot | `MinUniformBufferOffsetAlignment` |
| a transfer source or destination | `MinBufferCopyOffsetAlignment` |
| an indirect argument source | 4 |
| never bound and never copied | `DType.Size()` |

The last row matters more than it looks. A view is a plain value and creating one
is not an operation the device sees, so a caller may make a view at any element
offset and pass it around freely. The alignment applies only when the view
reaches a binding or a copy. The check therefore lives at graph build (or at the
call, for the immediate transfer path), and the error names the view's buffer,
the required alignment, its `Limits` source, and the offending byte offset.

**`ViewAs` legality, exactly.** Let `d` be the requested dtype. `ViewAs` succeeds
when all of:

1. `offset >= 0` and `count >= 0`;
2. `(offset + count) * d.Size() <= buffer.Count() * buffer.DType().Size()`, so
   the byte range lies inside the buffer;
3. the resulting byte offset `offset * d.Size()` is a multiple of `d.Size()`,
   which is automatic given rule 1 and is stated so the byte formula is
   unambiguous;
4. when the view covers the whole buffer, `d.Size()` divides the buffer's byte
   length, which is the "sizes work out" case the current API documents.

It fails otherwise, with an error naming both dtypes, both sizes, and the
resulting byte range.

**The bit-pattern guarantee.** `ViewAs` reinterprets, it never converts. The
bytes are unchanged, and a `u32` view of an `f32` buffer sees the IEEE 754
binary32 encoding of each value in the device's native byte order, which 3.5
requires to equal the host's. This is a guarantee, not an accident: it is what
lets a kernel read a quantized plane as `u8` and a scale plane as bit-packed
`u16`, which [007](007-tensor-layer.md) depends on, and what lets a debug path
dump any buffer as `u8`.

One caveat that must not be lost. Reinterpreting f16 as u16 and back is exact.
Reinterpreting **across widths** (an `f32` buffer viewed as `u8`) is exact
byte-wise and is only meaningful if the caller knows the byte order, which 3.5
fixes as little-endian. accel does not reorder.

**Overlapping views in one dispatch.** The rule is the one the hardware can
actually keep:

| Case | Legality |
| --- | --- |
| two overlapping views bound, both `AccessRead` | legal, always |
| two overlapping views bound statically, at least one writing | **rejected at graph build**, naming both slots and the overlapping byte range |
| two overlapping views reaching one node through **rebindable slots**, at least one writing | **rejected at `Rebind` and at `Submit`**: at build there is no view to compare. This is [003](003-command-graph.md)'s check V21 |
| views into different buffers of the same pool | cannot overlap: suballocation is disjoint by construction |
| views into transients 003 aliased | cannot overlap in one dispatch: 003 aliases only transients that do not interfere, so two aliased transients are never both live at one node |

**The first two rows are the same rule checked at two different times, and the
split is not a choice.** Where a view is bound at record time the builder knows
its buffer, offset and length, so overlap is a decidable comparison and the error
lands at build with the recording call site attached. Where the slot is
rebindable there is nothing to compare until something is bound: 003 tracks
hazards against the *slot*, not against whatever will occupy it, so binding one
buffer to two slots the builder treated as independent invalidates the inferred
edge set. That case is caught at `Rebind` and again at `Submit`, in release
builds too, and 003 §V21 owns it. Neither spec's check subsumes the other and a
backend or builder implementing only one leaves the other case as a race.

The rejection is a rejection rather than undefined behaviour because
[002](002-compute-model.md) gives no way to order two writes to one address from
different bindings, so the result would be a nondeterministic winner, which is
precisely the failure mode [`conventions.md`](../docs/conventions.md) documents
for unordered fragment writes and refuses to offer.

**The one undefined behaviour that remains** is inside a single binding: two
invocations of one dispatch writing the same element through the same view. That
is a kernel-level race, not decidable by the builder, and 002 owns it. The CPU
backend catches it with the race detector, which [006](006-backends.md) calls the
strongest property that backend has.

---

## 7. Resource lifetime

Resources are Go objects with an explicit `Close`. They are not finalizer-managed:
GPU memory is a scarce resource with a driver-imposed cap, and leaving its release
to the garbage collector means release happens at an unpredictable time under
memory pressure that the collector cannot see.

A resource freed while a submission using it is in flight is a use-after-free.
The implementation keeps a submission's resources alive until its fence signals,
so the safe behaviour is the default, and closing early is reported rather than
crashing.

This section says how.

### 7.1 The mechanism: an internal reference count plus a per-submission retain set

Every device-backed resource carries an unexported atomic reference count and a
closed flag. The count is **not** exposed and callers never manipulate it: the
API is `Close`, and the count is how `Close` is made safe.

```go
// Internal, embedded in Buffer, Texture, Sampler, ComputePipeline and Graph.
type resourceState struct {
	refs   atomic.Int32 // 1 for the caller handle, plus 1 per in-flight submission
	closed atomic.Bool  // the caller has called Close
	label  string
}
```

The count starts at 1, which is the caller's handle. Then:

- **Submit retains.** `Queue.Submit` walks the graph's resource set (every bound
  buffer, texture and sampler, every pipeline, and the graph's transient pool)
  and increments each count once, storing the retained set on the submission.
- **Fence signal releases.** When the submission's fence signals, the backend
  decrements every count in that set. On Metal this happens in the command buffer
  completion handler, which is exactly the hazard
  [`conventions.md`](../docs/conventions.md) records: the handler runs after the
  enclosing autorelease pool has drained, so the backend releases only objects it
  retained itself, which the retain set makes true by construction.
- **Reaching zero frees.** The memory returns to its pool and the backend object
  is destroyed at the moment the count reaches zero, which is either inside
  `Close` (nothing in flight) or inside the fence completion path (something
  was).

The retain set is per submission, not per graph, because a graph may be submitted
many times over its life and each submission has its own fence.

### 7.2 What `Close` does

`Close` sets the closed flag, decrements the caller's reference, and returns.

- **Nothing in flight.** The count reaches zero, the memory returns to the pool
  immediately, and `Close` returns nil. This is the ordinary path and it is
  synchronous, which is the entire reason `Close` exists rather than a finalizer.
- **Something in flight.** The count is above zero after the decrement. The
  memory is **not** returned, the resource stays valid for the running
  submission, and `Close` returns a `*LifetimeError` naming the resource and the
  number of submissions holding it. The release happens when the last fence
  signals. Nothing crashes and nothing leaks.

That is what "reported rather than crashing" means concretely: the operation
succeeds in the sense that the caller's handle is gone and the memory will come
back, and it reports in the sense that the caller learns their teardown ordering
was wrong. A caller who does not want the error waits on the fence first, which
is one line.

**After `Close` the handle is dead.** Any subsequent use (`Write`, `Read`,
`View`, binding it into a graph) is a `*LifetimeError`, never a use-after-free,
because the closed flag is checked before anything touches device state. That is
one atomic load on paths that already do more work than that.

**`Pool.Close` and `Device.Close` are ordered, not recursive.** Closing a pool
with live buffers is a `*LifetimeError` naming the pool and counting the live
buffers, and the pool is not freed. Closing a device with live pools behaves the
same. The API could close children recursively and deliberately does not: a
caller who closed a pool out from under a buffer they still hold has a bug, and
turning that bug into a silent success makes the next use of the buffer undefined
instead of reported.

```go
// LifetimeError reports a resource used or released at the wrong time.
type LifetimeError struct {
	Op       string // "Close", "Write", "Bind", ...
	Resource string // the resource's Label
	Reason   string // "in flight", "closed", "has live children"
	InFlight int    // submissions still holding it, when Reason is "in flight"
	Children int    // live children, when Reason is "has live children"
}

func (e *LifetimeError) Error() string { panic(ErrNotImplemented) }
```

```
accel: Close "hdr_target": 2 submissions still in flight. The texture stays
       valid until they complete and its memory is released then. Wait on the
       fence before Close to avoid this.

accel: Write "logits": buffer is closed.

accel: Close pool "weights": 1284 live buffers. Close them first; pools do not
       close recursively (spec 001 section 7.2).
```

### 7.3 The exact rule for a `BufferView` outliving its `Buffer`

`BufferView` is a value with an exported `*Buffer` field, so it can be copied,
stored, and constructed by hand. That makes "a view must not outlive its buffer"
a rule needing a mechanism, not just a sentence.

**The rule, in three parts:**

1. **A view holds no reference.** Creating a view does not retain the buffer and
   holding a view does not keep device memory alive. A view is an address
   expression, not an owner. If views retained, a cached view would silently pin a
   weight tensor forever, and 007 caches views by the thousand.
2. **The Go pointer keeps the Go object alive, so a view is never a dangling
   pointer in the memory-safety sense.** What can be gone is the *device* memory.
   The closed flag lives on the `Buffer` object the view still points at, so the
   check is always possible.
3. **Every use of a view checks the buffer.** Binding a view whose buffer is
   closed is a `*LifetimeError` at graph build. Copying to or from one is the same
   error at the call. There is no path by which a view reaches the device without
   the buffer being consulted, because the buffer is where the device handle
   lives.

So the rule is enforced rather than trusted, and the failure is a named error
rather than a crash. A hand-constructed `BufferView` with nonsense fields is
equally safe: `Offset` and `Count` are validated against the live buffer at every
use, so the worst outcome is a rejection.

What is deliberately not offered: a view does not become valid again if its
buffer's memory is reallocated to something else. The closed flag is monotonic
and a `Buffer` object is never reused for a different allocation, so a stale view
stays an error forever rather than aliasing whatever landed at that address.

---

## 8. Transfers

Host to device, device to host, and device to device, all recorded as graph nodes
so they participate in dependency tracking and barrier computation like any other
work.

Texture-to-buffer and buffer-to-texture transfers are first-class. This is not an
incidental convenience: a rasterized G-buffer feeding a compute pass needs
exactly this, and the predecessor could not do it on-device at all, so the data
went out to the host and back every frame.

Readback follows caller row order regardless of the backend's native origin, per
[`conventions.md`](../docs/conventions.md), and tightly packed rows per 4.2, in
the composition order fixed by 4.3.

### 8.1 Two paths, and which is which

| Path | Entry point | In a graph | Blocks |
| --- | --- | --- | --- |
| recorded | `Recorder.CopyToBuffer`, `CopyBuffer`, `CopyTextureToBuffer`, `CopyBufferToTexture` | yes, a node with declared access | no, it runs when the graph runs |
| immediate | `Buffer.Write`, `Buffer.Read`, `Texture.Read` | no | `Write` no, `Read` yes |

The recorded path is what a hot loop uses. It is a node, so 003 infers its edges,
computes its barriers, and orders it against everything else.

The immediate path is a convenience for setup, teardown and debugging. It exists
because writing initial weights and reading final logits should not require
building a graph.

**A correction to the current doc comment.** `memory.go` says of `Buffer.Write`:
"It is recorded, not immediate: for a transfer that participates in graph
dependency tracking, record it with `Recorder.CopyToBuffer` instead." The two
clauses contradict each other and the second is the intended meaning. The comment
should read: "It is immediate, not recorded: it does not participate in graph
dependency tracking. For a transfer that does, record it with
`Recorder.CopyToBuffer`." The decision is unchanged, only its statement.

### 8.2 `Write` is asynchronous, `Read` is synchronous, and both are precise

**`Buffer.Write` returns once the caller's data has been copied out of the Go
slice, not once the device has the bytes.** The caller may reuse or modify their
slice the moment `Write` returns. This matches
[003](003-command-graph.md)'s rule that nothing in the API blocks implicitly.

For that to be safe it needs a mechanism, not a hope:

> **Pending writes form a batch that the next submission flushes as its
> prologue.** `Write` appends a staged copy to the device's pending transfer
> batch. `Queue.Submit` and `Queue.SubmitAfter` emit the pending batch ahead of
> the graph's own work on the same queue, so every `Write` issued before a
> `Submit` is visible to that submission, and `Fence.Wait` on that submission
> also proves the writes landed.

Two consequences, stated rather than left implicit:

- **If the caller never submits again, the batch never flushes.** That is
  harmless for correctness, since nothing reads the buffer, and wrong for a
  caller who wrote and then expects to read back some other way. So
  `Buffer.Read`, `Texture.Read` and `Device.Close` all flush the pending batch
  first, and an explicit `Queue.Flush()` exists for the caller who wants it
  without a read.
- **The staging ring can fill.** `Write` stages into a ring of `Upload` blocks. If
  the ring is full, `Write` blocks until a block is recycled by a completed
  submission. That is the one case where `Write` blocks, it is reported through
  queue statistics, and the fix is to submit more often or to use the recorded
  path.

**`Buffer.Read` blocks**, and its doc comment already says so. It flushes pending
writes, submits a one-shot copy into a `Readback` block, waits on the fence, and
copies out. That is a full round trip to the device and back, documented as
inappropriate in a hot loop for the same reason `Queue.Run` is.

**Partial updates are first-class in both directions.** `Write(offset, data)` and
`Read(offset, into)` address elements of the buffer's dtype and touch only the
range they name. The range must lie inside the buffer and must satisfy the copy
alignment from 3.1. That last part is a real constraint on `Device` pools: a
caller updating one f32 at element 7 needs the backend to round the copy out to
the alignment, which it does by reading, merging and writing back. A caller doing
many small partial updates should instead fill one staging buffer and record one
copy, which is 8.4.

### 8.3 `MemoryShared` makes the copy vanish, and that is the whole point

On unified-memory hardware a `Shared` pool is host-visible and device-local at
once. There is no staging block and no copy:

| Operation | `Device` pool | `Shared` pool |
| --- | --- | --- |
| `Write` | memcpy into staging, plus a device copy in the next submission's prologue | memcpy into the mapping, and that is all |
| `Read` | flush, device copy into readback, fence wait, memcpy out | memcpy out of the mapping once a fence proves the device is done |
| bytes moved per write | 2x the payload | 1x |
| device round trips per read | 1 | 0 |

That difference is why `Shared` is a memory kind and not a hint. On an Apple
silicon machine a model's weights live in `Shared` and are written once with no
staging at all; on a discrete GPU the same code costs a staging copy, which is
correct rather than surprising because the kind was requested explicitly and its
availability was reported.

**The ordering obligation does not vanish with the copy.** A `Shared` buffer
written by the host while the device reads it is a data race, and there is no copy
in between to hide it. The fence rules are identical: write before submit, or wait
on the fence before writing again. accel does not detect this, and it is the one
place where `Shared` is genuinely more dangerous than `Upload` plus a copy.

### 8.4 Worked example: uploading a model's weights efficiently

The naive version, and why it is slow:

```go
// Slow. Do not do this.
for _, t := range tensors {
	buf, _ := weights.Alloc(descFor(t))
	buf.Write(0, t.Data) // stages, appends to the pending batch
}
queue.Flush()
```

Every `Write` stages the tensor into an `Upload` block, so the whole model passes
through the staging ring. The ring is a few megabytes, so `Write` blocks whenever
it fills and the upload becomes a sequence of small submissions paced by fence
waits. The data is copied twice (host slice to staging, staging to device) and
the device idles between blocks.

The efficient version stages in bulk, copies in one recorded graph, and on
unified memory skips staging entirely:

```go
// Fast. One staging pool, one graph, one submission, one fence per chunk.
kind := accel.MemoryUpload
if dev.Info().Capabilities.SharedMemoryKind {
	kind = accel.MemoryShared // no staging copy at all on unified memory
}

staging, err := dev.NewPool(kind, chunkBytes) // sized to a chunk, not the model
weights, err := dev.NewPool(accel.MemoryDevice, totalWeightBytes)

rec := dev.NewRecorder()
for _, t := range chunk {
	dst, _ := weights.Alloc(accel.BufferDescriptor{
		DType: t.DType, Count: t.Count,
		Usage: accel.UsageStorage | accel.UsageCopyDst,
		Label: t.Name,
	})
	src, _ := staging.Alloc(accel.BufferDescriptor{
		DType: t.DType, Count: t.Count,
		Usage: accel.UsageCopySrc,
		Label: t.Name + ".staging",
	})
	src.Write(0, t.Data) // memcpy into the mapping, no device work
	dv, _ := dst.View(0, t.Count)
	sv, _ := src.View(0, t.Count)
	rec.CopyBuffer(dv, sv) // one node per tensor
}
g, err := rec.Build()          // validated, ordered, barriers computed, once
fence := dev.Queue().Submit(g) // one submission for the whole chunk
fence.Wait()                   // one fence wait for the whole chunk
```

The accounting, honestly:

| | naive | chunked graph | `Shared` |
| --- | --- | --- | --- |
| host copies per byte | 2 | 2 | 1 |
| submissions | one per ring fill | one per chunk | one per chunk |
| fence waits | one per ring fill | one per chunk | one per chunk |
| peak extra memory | ring size | chunk size | none beyond the weights |

Chunk size is the tuning knob and trades peak staging memory against submission
count. On a device reporting `SharedMemoryKind`, the staging pool and the copy
nodes both disappear and the weights are written straight into the memory the
device will read, which is the copy vanishing that this spec and 002's memory
model both promise.

---

## 9. Error taxonomy

Every failure in this spec is a typed error carrying the numbers a caller needs
to fix it. A bare `errors.New` here is a defect, for the same reason
[003](003-command-graph.md) says a build error naming only "type mismatch" is a
defect: the caller cannot act on it.

```go
// Sentinels for errors.Is. Each typed error below unwraps to one of these, so a
// caller can branch on the class without depending on the struct.
var (
	ErrOutOfDeviceMemory = errors.New("accel: out of device memory")
	ErrFragmented        = errors.New("accel: pool has space but not contiguous space")
	ErrAlignment         = errors.New("accel: alignment violation")
	ErrUsage             = errors.New("accel: usage violation")
	ErrLifetime          = errors.New("accel: lifetime violation")
	ErrFormat            = errors.New("accel: format not usable this way")
)

// AllocError reports a failed suballocation. Free and LargestFree together
// distinguish exhaustion from fragmentation, which have different fixes.
type AllocError struct {
	Label       string
	Pool        string
	Kind        MemoryKind
	Requested   int
	Alignment   int
	Free        int
	LargestFree int
	PoolSize    int
}

// AlignmentError reports an offset or pitch that does not meet a device
// requirement. Required always comes from Limits, never from a constant.
type AlignmentError struct {
	What     string // "view offset", "copy offset", "row pitch"
	Resource string
	Offset   int
	Required int
	Source   string // the Limits field that imposed it
}

// UsageError reports a resource used in a way it did not declare.
type UsageError struct {
	Resource string
	Node     NodeID
	Slot     int
	Declared BufferUsage
	Needed   BufferUsage
	Site     string // the recording call site, per spec 003
}

// FormatError reports a format the device cannot use as asked.
type FormatError struct {
	Format Format
	Want   string // "renderable", "storage", "host copyable", "filterable"
	Device string
}
```

Realistic messages, which are the actual specification of these types:

```
accel: alloc "blk.31.ffn_up.weight" (176.0 MiB, align 256) failed in pool
       "weights" (Device, 6.0 GiB): 210.4 MiB free in 3 blocks, largest
       88.0 MiB. Fragmented, not exhausted (spec 001 section 5.3).

accel: alloc "kv" (2.5 GiB) failed in pool "session" (Device, 2.0 GiB): request
       exceeds pool size. Pools do not grow (spec 001 section 5.5); size the
       pool from Plan.Memory() before allocating.

accel: view offset of "kv_cache.k" (byte 8, element 4 of f16) is not a multiple
       of 256 required for a storage binding on this device
       (Limits.MinStorageBufferOffsetAlignment). Align the slice, or pass the
       offset as buffer contents (spec 003).

accel: node 41 slot 2 ("scores") binds buffer "logits" declared
       UsageStorage|UsageCopySrc but needs UsageCopyDst.
       Recorded at model/attention.go:118.

accel: node 12 binds overlapping views of "kv_cache.k" to slots 1 (read) and 3
       (write), bytes [4096, 8192) in common. Overlapping writable bindings are
       rejected (spec 001 section 6.1).

accel: Texture.Read on Depth24PlusStencil8 is not supported: the format's layout
       is device-defined and has no bytes per pixel. Use Depth32Float for
       readback (spec 001 section 4.1).

accel: NewPool(MemoryShared, 512 MiB): this device reports no unified memory
       (Metal, "AMD Radeon Pro 5500M"). Use MemoryUpload plus a copy, or check
       Capabilities.SharedMemoryKind first.

accel: ViewAs(F32) on "quants" (u8, 1048577 elements): byte length 1048577 is
       not a multiple of 4.
```

Each message names the resource by its `Label`, which is why `Label` is on every
descriptor and why its doc comment already says it is worth setting.

---

## 10. Open questions

- **~~Whether pools grow.~~ Resolved in 5.5.** A pool is one device allocation
  and no backend can resize one, so pools do not grow. The implicit pool behind
  `Device.NewBuffer` grows by adding blocks, which is what callers mean and is
  expressible without invalidating an address.
- **Whether 256 should stay the default suballocation granularity.** 3.1 makes
  256 always sufficient and queries the real number for callers who care.
  Unresolved is whether the *default* granularity should be the queried number,
  which would make identical code produce different pool layouts and different
  `PoolStats` on different devices, and would make a memory requirement computed
  on one machine wrong on another. Leaning toward keeping 256 as the default with
  the queried value as an explicit pool option. 007's small-view case is the
  evidence that would decide it.
- **Whether a general pool should offer defragmentation at a quiescent point.**
  Compaction is impossible while addresses are baked into descriptors, but a
  caller who has closed every graph and waited every fence has no baked
  addresses, so a "repack while idle" operation is expressible and would fix the
  long-running fragmentation in 5.3. It requires every live buffer to be
  updatable in place, which means a level of indirection on every device address,
  which costs something on every access. Not designed, and it should not be
  designed out.
- **Whether uniform buffers should exist at all, given std140's padding.**
  Storage buffers are tightly packed, universally available, and can hold
  everything a uniform can. The argument for uniforms is the constant cache,
  which is a real win for values every invocation reads. The argument against is
  a second layout convention (3.3) and a second alignment (3.1) for a resource
  whose payload is usually under 256 bytes. Keeping them, because the cache win
  is measurable on every backend, but a measurement showing otherwise would
  simplify this spec considerably.
- **Whether `Depth24PlusStencil8` should be replaced by a depth-stencil format
  with a defined layout.** 4.1 makes it non-host-copyable because its layout is
  device-defined, and it is `cap` rather than `yes` on Metal. A 32-bit-float
  depth plus 8-bit stencil format has a defined layout and wider guaranteed
  support. Not changed here, because the enum is part of the current surface and
  005 renders against it, but it is the format most likely to be wrong.
- **Whether `AlignedRowPitch` plus a padded-row copy variant earns its API
  surface.** 4.2 offers it as the escape from the repack. It is two entry points
  and a concept, for a cost that only appears on some backends at unaligned
  widths. The alternative is always repacking and reporting it in `CopyStats`,
  which is simpler and slower for the one caller who cares.
- **Sparse and virtual memory** for models larger than device memory. Out of
  scope for v0, but it should not be designed out. Note that 5.5's block-set
  implicit pool is the shape a sparse pool would take, so the two do not
  conflict.
- **Multi-device.** One device per instance for v0. Tensor-parallel inference
  wants more, and the queue model should not foreclose it. The resource model
  here is single-device throughout: a `Buffer` names a pool, a pool names a
  device, and nothing crosses. Peer-to-peer transfer would be a new transfer kind
  rather than a change to any of this.

---

## 11. Testing

### 11.1 Round trips and dtypes

- Every dtype round-trips host to device to host unchanged, at every memory kind
  the device reports, for buffers whose sizes are not multiples of anything
  interesting (1 element, 3 elements, 65,537 elements).
- A partial `Write` followed by a full `Read` changes only the range written, on
  every memory kind, including a range whose byte offset is not copy-aligned, so
  the backend's read-merge-write path is exercised.
- `Write` then `Submit` then `Fence.Wait` makes the written bytes visible to that
  submission, which is the ordering 8.2 promises. The negative form is also a
  test: a `Write` issued after `Submit` is not visible to it.
- `Read` flushes pending writes, so `Write` then `Read` with no submission in
  between returns the written data.

### 11.2 Alignment and layout

- **Every alignment in `Limits` is a positive power of two on every enumerated
  device.** A backend that forgot to fill one in reports zero and fails this,
  which is the cheapest possible catch for the failure mode 1.1 names.
- A view bound at an offset that is not `MinStorageBufferOffsetAlignment`-aligned
  is rejected at graph build, and the error names the required alignment and its
  `Limits` field.
- The same view used only as a copy source at the same offset is accepted, which
  proves the alignment rule is per use and not per view.
- **std140 encoding is checked against the device, not against the encoder.** A
  kernel reads each field of a uniform struct containing a scalar, a 3-vector, a
  scalar occupying the 3-vector's tail, and a 4x4 matrix, writes each to a
  distinct storage buffer element, and the host asserts the values. This is the
  test that catches an encoder agreeing with itself and disagreeing with the
  shader, which no host-side round trip can catch. It runs on every backend,
  because std140 agreement is exactly the kind of thing that holds on three
  backends and not the fourth.
- A uniform struct containing a forbidden type (`bool`, `int`, `float64`) fails
  at generate time with an error naming the field.
- A storage buffer's element stride equals `DType.Size()`, checked by having a
  kernel write its global id into element `i` and asserting the host sees a dense
  ramp with no gaps, for every dtype including the narrow ones.

### 11.3 Views and aliasing

- A view aliases its parent: writing through the view is visible through the
  parent at the right offset, verified with a dtype whose size is not 4 so an
  elements-versus-bytes confusion cannot pass.
- `ViewAs` at an incompatible dtype is rejected, and the error names both dtypes
  and the byte length.
- `ViewAs` preserves bits: an f32 buffer viewed as u32 reads back the IEEE
  encodings, checked against `math.Float32bits` for values including a negative
  zero, an infinity and a NaN, on every backend. This is 6.1's bit-pattern
  guarantee and it is the one a backend could break by converting.
- Two overlapping views bound to one dispatch with at least one writing is
  rejected at build, naming both slots and the overlapping byte range. Two
  overlapping read-only views are accepted and produce the expected results.
- A view of a closed buffer is rejected at every use site and never crashes.
- A hand-constructed `BufferView` with an out-of-range `Count` is rejected rather
  than reading past the allocation.

### 11.4 Allocation

- Pool exhaustion reports a clear error rather than a driver crash, and the error
  distinguishes exhausted from fragmented by carrying both `Free` and
  `LargestFree`.
- **A fragmentation scenario is constructed deliberately**: allocate many blocks,
  free every other one, then request a block larger than any hole. It fails with
  `ErrFragmented` while `Free` exceeds the request, which is the behaviour 5.3
  promises rather than a bug to be fixed.
- `PoolStats.Used` accounts for alignment padding, verified by allocating N
  one-byte buffers and asserting `Used` is `N * granularity`, not `N`.
- The allocator is fuzzed: random allocate and free sequences never overlap two
  live allocations, never exceed the pool, and always coalesce back to one free
  block when everything is freed. Overlap is checked by writing a per-allocation
  pattern and reading it back.
- Allocation is O(1): allocating 10,000 buffers takes time linear in the count,
  not quadratic. This is the property that rules out the first-fit free list, so
  it is a measured guard rather than an assumption.
- A `PoolLinear` pool refuses individual frees and resets as a whole, and 003's
  transient placement runs against it unchanged.

### 11.5 Textures and formats

- Readback row order matches the caller's convention on every backend, tested
  through both the texture path and the compute-buffer path, since they diverge.
- **Row pitch and origin compose correctly**, tested at a width whose byte length
  is deliberately not a multiple of 256: render a pattern encoding both row and
  column, read it back, assert every pixel. A backend that depads after flipping
  fails this and passes a square power-of-two test, which is why the width is
  chosen badly on purpose.
- `AlignedRowPitch` equals `width * bpp` exactly when `CopyStats.Repacked` is
  false, over a sweep of widths, on every backend.
- Every format in 4.1 is created and its reported `FormatInfo` matches the
  table's guarantees. A `yes` cell that reports false on a real device is a bug
  in the table and is corrected by evidence, per 006's discipline.
- `Texture.Read` on `Depth24PlusStencil8` errors and names `Depth32Float`.
- Depth textures are private where the backend requires it, and the depth
  readback path through a transfer node works on macOS, which is 005's test seen
  from this side.
- A texture pool and a buffer pool cannot be mixed, and the error says so.

### 11.6 Lifetime

- Closing a resource with work in flight does not crash and is reported: the
  submission completes with correct results, the error names the in-flight count,
  and the memory returns to the pool after the fence, verified through
  `PoolStats`.
- Using a closed resource is a `*LifetimeError` at every entry point, enumerated
  by a table-driven test so a new entry point that forgets the check is caught.
- Closing a pool with live buffers is reported and the pool is not freed, so the
  buffers still work afterwards.
- **The retain set is exact**: a graph submitted twice holds its resources across
  both submissions, and a resource bound only in the second submission is not
  retained by the first. Checked through `PoolStats` at each step rather than by
  inspecting the count.
- Metal specifically: closing a resource inside a completion handler's window
  does not crash, which is the
  [`conventions.md`](../docs/conventions.md) autorelease hazard exercised
  deliberately rather than waited for.

### 11.7 Devices

- Usage validation rejects a buffer used in a way it did not declare, and the
  error names the node, the slot and the recording call site.
- Requesting `MemoryShared` on a device reporting it absent errors naming the
  kind and the device, and never silently returns an `Upload` pool.
- Opening an unavailable backend errors naming it and never returns a different
  backend, which is 006's rule tested from this side.
- `Limits` is populated on every enumerated device, with no zero-valued field.
