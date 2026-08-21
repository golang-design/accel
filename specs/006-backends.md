---
title: "Backends: the contract, the set, and the capability matrix"
status: drafted
layer: device
depends_on:
  - 001-device-resources.md
  - 002-compute-model.md
  - 003-command-graph.md
---

# Backends

What a backend has to do, which backends exist, what each one can and cannot do,
and how [003](003-command-graph.md)'s graph reaches each one.

This spec answers 003's open question about graph lowering, and 002's open
question about whether f16 storage and f16 arithmetic are one capability or two.
It answers both concretely enough that the answers can be wrong and be caught.

## The oracle rule, stated once

The CPU backend enforces the **intersection** of what every backend allows, and
computes the **exact** semantics every backend is required to produce.

This generalises [002](002-compute-model.md)'s shared-memory poison pattern into
a principle. A rule that only one backend enforces is a rule callers discover by
shipping. So: `Device` memory is not mappable on the CPU backend even though it
trivially could be, non-uniform barrier arrival fails there even though it would
merely be undefined elsewhere, and a workgroup larger than the portable floor is
rejected under the oracle's strict mode. The oracle is the strictest device, not
the most permissive one.

The cost is that a kernel can pass on a GPU and fail on the CPU. That is the
intended direction: a portability bug should surface on the device that runs
everywhere, on a laptop, with no provisioning.

---

## 1. What a backend is responsible for

A backend implements an unexported interface. Adding one touches no public API
([`000-decisions.md`](000-decisions.md) layering rule 2), and no backend-specific type
appears in a public signature (rule 3). Restated as responsibilities:

**R1. Existence and identity.** Report whether the backend can run on this
machine without opening a device, and enumerate the devices it sees with a name,
a vendor, and whether the device is hardware or a software rasterizer. Probing
must not panic when the driver library is absent, stale, or broken, because
enumeration runs on machines that have none of it.

**R2. Capabilities.** Fill in the capability record from
[002](002-compute-model.md) by **querying the device**, never by assuming from
the platform. Every `cap` and every `?` in section 3's matrix is resolved here. A
backend that cannot determine a capability reports it absent, not present.

**R3. Memory.** Allocate pools of the four memory kinds in
[001](001-device-resources.md), suballocate buffers from them, and map the kinds
that are host-visible. A backend that cannot honour a kind reports it absent, and
the caller gets an error naming the kind, rather than an allocation that quietly
behaves differently.

**R4. Resources.** Buffers, views, textures, samplers, and their lifetime,
including the platform constraints that belong in the backend rather than in the
caller (macOS depth textures must be private, bytes per pixel comes from the
format).

**R5. Shader ingestion.** Declare which shader IR the backend consumes and turn a
compiled kernel into a pipeline object. The backend does **not** compile Go.
[004](004-kernel-authoring.md) owns the emitter and its IR; the backend is a
consumer of whatever 004 produces for its target:

| Backend | Consumes |
| --- | --- |
| CPU | the Go function itself |
| Metal | MSL source text |
| Vulkan | SPIR-V binary |
| D3D12 | DXBC or DXIL binary (see 2.4) |
| OpenGL ES | GLSL ES source text |
| WebGPU | WGSL source text |

**R6. Pipelines.** Bake the static parts of the compute model into the pipeline:
workgroup size, shared memory size, binding layout. All three are needed at
compile time by at least one backend, which is why 002 puts them in the pipeline
descriptor rather than at dispatch.

**R7. Graph lowering.** Turn a built, validated `Graph` into whatever this
backend can resubmit cheaply, and expose the rebinding points that
[003](003-command-graph.md) promises stay cheap. Section 4 is this responsibility
in detail.

**R8. Queues, submission, and completion.** Report the device's queue topology
rather than letting callers assume it, per [001](001-device-resources.md):
whether compute and graphics are separate queues, and whether transfers have
their own. The four backends disagree completely (Vulkan exposes queue families
with capability bits, D3D12 has typed command queues, Metal has one general
queue, GL and the CPU backend have exactly one), so this is reported, never
inferred from the platform. Then: submit, return a fence, and keep a
submission's resources alive until the fence signals.

**R9. Convention correction.** Every divergence in
[`conventions.md`](../docs/conventions.md) is corrected inside the backend:
clip-space depth range, face winding, readback row order, GLSL reserved words and
integer literal rules, Objective-C pool discipline. A backend that leaks one of
these to the caller is broken, not merely different.

**R10. Teardown.** Release everything, on the thread that is allowed to release
it, and survive being closed while work is in flight.

### Required core, optional interfaces

The predecessor put everything on one interface and expressed absence as a stub
that fails when called: `newWindowSurface` returned `ErrUnsupported` on Metal,
`setComputeTexture` was an empty method on GL, and the Vulkan backend's whole
texture, sampler, and render half returned `ErrUnsupported`. That is absence
discovered at the call, which is exactly what decision 6 forbids.

So the contract splits:

- A **core** interface every backend implements in full: R1 to R8 for compute,
  plus R9 and R10. There is no stub in the core. A backend that cannot do
  something in the core is not a backend.
- **Optional** interfaces discovered by type assertion, each mirrored by a
  capability bit so the answer is available before anything is called:
  graphics and render passes, presentation to a window, texture sampling in
  compute, native graph replay, subgroup operations, indirect dispatch.

One more shape lesson from the predecessor, worth naming because it looks
harmless: `backendBuffer.bytes() []byte` returned the buffer's contents as a Go
slice. That worked only because the Metal backend created every buffer with
`StorageModeShared`. There is no `bytes()` for device-local memory on a discrete
GPU, so this one method quietly foreclosed [001](001-device-resources.md)'s
memory kinds. Buffer access here is a mapping operation on a pool whose kind
permits mapping, and mapping a `Device` pool is an error on every backend
including the CPU.

---

## 2. The backend set

### 2.1 CPU (pure Go)

**Reached cgo-free**: it is Go. No driver, no library, no build tags
([`000-decisions.md`](000-decisions.md) layering rule 4).

**Can**: everything in the compute model, exactly, including the parts real
hardware treats as undefined. Every dtype, every atomic including float add,
subgroups at a fixed configurable size, shared memory with poison, barriers with
non-uniform-arrival detection.

**Cannot**: be fast in the way a GPU is fast (section 5), and cannot validate
anything about a driver. A backend bug that is a driver quirk is invisible here
by construction, which is why parity tests still need real devices.

**Difficulty**: moderate, and none of it is in the API. The work is the execution
model and the exactness contract, both in section 5.

### 2.2 Metal (darwin)

**Reached cgo-free**: `purego` plus an Objective-C runtime shim, calling
`objc_msgSend` against `Metal.framework`. Proven in the predecessor
(`gpu/backend_darwin.go`, `gpu/mtl`), including compute, render to texture, MRT,
and depth.

**Can**: the whole model. Metal is the most complete of the GPU backends because
MSL is the least restrictive shading language of the set (it can pass buffers to
functions, which GLSL ES cannot), unified memory makes `Shared` genuinely free on
Apple silicon, and the API is already a command-buffer API.

**Cannot**: resubmit a command buffer. `MTLCommandBuffer` is single-submit, so
Metal has no reusable command object without indirect command buffers, which have
their own cost (section 4.3).

**Difficulty**: moderate, with a sharp edge. The edge is object lifetime:
completion handlers run after the enclosing autorelease pool has drained, and
getting that wrong crashes inside `objc_msgSend` with a useless stack. See
[`conventions.md`](../docs/conventions.md).

### 2.3 Vulkan

**Reached cgo-free**: `purego` against `libvulkan.so.1` or `vulkan-1.dll`, with
create-info structs marshalled as C-layout Go structs. Proven in the predecessor
for the entire compute path (`gpu/backend_vk.go`): instance, device, memory,
descriptor sets, pipeline, command buffer, dispatch, readback, verified in CI on
lavapipe.

**Can**: more of the compute model than anything else, and it is the only backend
where the advanced capabilities (subgroups, f16, atomic float add, cooperative
matrix) are reachable through a documented, queryable extension mechanism rather
than a vendor deal.

**Cannot**: consume text. Vulkan takes SPIR-V, so this backend is blocked on
Go-to-SPIR-V, which is why [004](004-kernel-authoring.md) makes SPIR-V the
argument for having a real IR rather than an AST walk. The predecessor
sidestepped it by compiling GLSL with `glslang` in CI, which is fine for a proof
and not shippable, because it makes a build-time external tool part of the
runtime story, and there is no cgo-free GLSL-to-SPIR-V path.

**Difficulty**: large. Roughly forty structs marshal correctly through purego
before anything runs, and descriptor sets, memory type selection, and
synchronisation are each a subsystem. The predecessor showed the marshalling
works, which removes the risk but not the labour.

### 2.4 D3D12 (windows)

**Reached cgo-free**: `syscall` directly. Windows exposes its API through
`syscall.SyscallN` with no purego needed, and D3D12 objects are COM: a pointer to
a vtable, called as `vtbl[index](obj, args...)`. Device creation on WARP is
proven in the predecessor on a stock runner (`gpu/dxprobe_windows_test.go`).

**Can**: everything the model needs, in principle, and it is the only backend
that runs on Windows with zero provisioning because WARP ships in the OS.

**Cannot**, and this is the column's real constraint: **reach shader model 6
through the compiler that is present.** `D3DCompile` in `d3dcompiler_47.dll` tops
out at SM 5.1. SM 5.1 has no wave intrinsics and no native 16-bit types, so on
that path the D3D12 column loses subgroups, f16 arithmetic, and the packed 8-bit
dot product. SM 6 means DXIL, which means `dxcompiler.dll` (DXC), which is not on
a stock Windows install or a stock runner, and unsigned DXIL needs developer mode
enabled. Loading DXC by syscall is cgo-free but it is a shipped binary
dependency, which cuts against the same lean bet that ruled out cgo. Emitting
DXIL directly from the kernel compiler avoids the dependency and is a large piece
of work of the same weight as Go-to-SPIR-V. This is unresolved (open question 4).

**Difficulty**: large. COM vtable indices are transcribed by hand from header
inheritance order, one interface at a time, and a wrong index is an immediate
crash with no diagnostic. The shader question above sits on top of that.

### 2.5 OpenGL ES 3.1

**Reached cgo-free**: `purego` against `libGLESv2` plus EGL for the context, or
`syscall` on Windows. Fully proven in the predecessor (`gpu/backend_gl.go`, 829
lines): compute with SSBOs and UBOs, render to texture, MRT, depth, float
targets, and windowed present, all green in CI on Mesa llvmpipe.

**Can**: the baseline compute model. Compute shaders, shared memory, barriers,
integer atomics, and indirect dispatch are all GLES 3.1 core.

**Cannot**: any of the modern capabilities. No subgroups in core GLES 3.1, no
f16 or bf16 arithmetic, no float atomics, and no memory kinds, because GL buffers
are opaque and its usage hints are advisory. It also has no command buffer at
all, so the backend owns a `runtime.LockOSThread` goroutine holding the context
and replays a recorded list onto it. GLSL ES adds its own restrictions that the
kernel compiler must respect: no buffer parameters on helper functions, no mixing
`uint` and `int` literals, reserved words that are ordinary Go identifiers. All
three are in [`conventions.md`](../docs/conventions.md).

**Difficulty**: moderate, and the lowest-risk of the GPU backends because it is
the one the predecessor finished. Its value is reach, not capability: it is the
fallback for hardware and drivers too old for Vulkan.

### 2.6 WebGPU and WASM

**Reached cgo-free**: under `GOOS=js`, `syscall/js` against the browser's
`navigator.gpu`. This satisfies decision 2 trivially, since there is no C
anywhere in the path.

The alternative, a native WebGPU implementation (`wgpu-native` or Dawn) loaded by
purego on desktop, is technically cgo-free and is **rejected for v0**: it makes a
large third-party shared library a runtime dependency on platforms that already
have Vulkan, Metal, and D3D12, which is the dependency posture cgo-free exists to
avoid.

**Can**: the WGSL compute baseline, which is deliberately the portable
intersection: workgroups, workgroup storage, barriers, i32 and u32 atomics,
indirect dispatch.

**Cannot**: float atomics (WGSL atomics are i32 and u32 only, architecturally),
bf16 arithmetic, or `Shared` memory in [001](001-device-resources.md)'s sense.
Two more problems are structural rather than featural. First, WebGPU is
asynchronous where `accel` is synchronous: adapter request, device request, and
buffer mapping are all promises, and reconciling that with a blocking Go API on a
single-threaded JS event loop is the actual design problem, not the API surface.
Second, every `syscall/js` call crosses a boundary that costs orders of magnitude
more than a Go call, so a software-replayed graph of a thousand nodes pays a
thousand crossings per submission. The mitigation, encoding the node list into a
typed array and replaying it with a loop on the JS side in one crossing, is
plausible and unproven (open question 5).

**Difficulty**: unknown, and the honest estimate is large for reasons unrelated
to graphics. Deferred past v0, but the graph model should not be shaped in a way
that forecloses the batched-crossing trick.

### 2.7 Not in the set

CUDA, per [`000-decisions.md`](000-decisions.md): reachable without cgo in principle,
not in the first milestone, and the largest single gap for training workloads.
Metal on iOS, OpenCL, and WebGL are not planned.

---

## 3. Capability matrix

**How to read a cell.**

| Mark | Meaning |
| --- | --- |
| `yes` | Architecturally guaranteed. Every device on this backend has it. |
| `cap` | Queryable. Present on some devices, absent on others, reported per device. |
| `emul` | Provided by the backend on top of something else, at a stated cost. |
| `no` | Architecturally absent. The API reports it absent and errors if required. |
| `gated` | Present in the hardware and the API, unreachable through the shader compiler that ships with the OS. See 2.4. |
| `n/a` | The question does not apply, because a prerequisite capability is `no`. |
| `?` | **Unknown.** Resolved by measurement at first contact, then recorded. |

A `?` here is a completed cell, not a gap: it names something that must be
measured rather than remembered. Every `cap` and `?` is resolved by the query
listed under the table, at device open, and once measured on real hardware the
result belongs in [`conventions.md`](../docs/conventions.md), which is what that
document is for. Version-pinned extension claims are deliberately absent from
this table, because a confidently wrong version pin in a normative spec is worse
than an unknown: it gets built on.

### Compute model (from [002](002-compute-model.md))

| Capability | CPU | Metal | Vulkan | D3D12 | GLES 3.1 | WebGPU |
| --- | --- | --- | --- | --- | --- | --- |
| Max workgroup size, max invocations | yes | yes | yes | yes | yes | yes |
| Shared (workgroup) memory | yes | yes | yes | yes | yes | yes |
| Execution + memory barriers | yes | yes | yes | yes | yes | yes |
| Atomics i32/u32, storage | yes | yes | yes | yes | yes | yes |
| Atomics i32/u32, shared | yes | yes | yes | yes | yes | yes |
| Atomic float add, storage | yes | cap | cap | no | no | no |
| Atomic float add, shared | yes | ? | cap | no | no | no |
| Subgroup ops (shuffle, ballot, reduce) | yes | cap | cap | gated | no | cap |
| Subgroup size reported | yes | yes | yes | gated | n/a | cap |
| f16 storage | yes | yes | yes | emul | emul | emul |
| f16 arithmetic | yes | yes | cap | gated | no | cap |
| bf16 storage | yes | emul | emul | emul | emul | emul |
| Denormals preserved f32 | yes | ? | ? | ? | ? | ? |
| Denormals preserved f16 | yes | ? | ? | ? | ? | ? |
| Inf/NaN produced | yes | ? | ? | ? | ? | ? |
| bf16 arithmetic | yes | ? | cap | ? | no | no |
| i8/u8 storage | yes | yes | yes | emul | emul | emul |
| i8 packed dot product | yes | ? | cap | gated | no | no |
| Indirect dispatch | yes | yes | yes | yes | yes | yes |
| Max storage binding size | yes | yes | yes | yes | yes | yes |
| Max bindings per kind | yes | yes | yes | yes | yes | yes |
| FP contraction control | yes | ? | yes | yes | ? | ? |
| Reports hardware vs software device | yes | yes | yes | yes | yes | yes |
| Reports queue topology | yes | yes | yes | yes | yes | yes |

`gated` means the capability exists in the hardware and the API but is
unreachable through the shader compiler that ships with the OS; see 2.4. It is
not `cap`, because no query resolves it: a distribution decision does.

`emul` on the narrow dtypes is the important entry, and it is what answers
002's open question. **f16 storage and f16 arithmetic are two capabilities, not
one.** Storage is universally available because a 16-bit float is reachable by
bit packing on every backend in the table (`packHalf2x16` in GLSL ES,
`f32tof16` in HLSL, `pack2x16float` in WGSL), at the cost of a shift and a
convert per access. bf16 is easier still, being the top half of an f32.
Arithmetic is not emulable at any acceptable cost and is a real capability.

The design consequence is the one 002 already wanted: narrow dtypes are storage
formats that convert to f32 on load and back on store, that default works
everywhere, and native narrow arithmetic is an opt-in that requires a capability.
A quantized model runs on every backend in this table. It runs *fast* only where
the arithmetic capability is present.

### Graph lowering (see section 4)

| Property | CPU | Metal | Vulkan | D3D12 | GLES 3.1 | WebGPU |
| --- | --- | --- | --- | --- | --- | --- |
| Native replayable object | no | cap | yes | yes | no | no |
| Lowering strategy | replay | re-encode, ICB optional | reusable primary CB | closed command list | replay on context thread | replay, batched crossing |
| Rebinding without re-record | yes | yes | yes | yes | yes | yes |

### Memory kinds (from [001](001-device-resources.md))

| Kind | CPU | Metal | Vulkan | D3D12 | GLES 3.1 | WebGPU |
| --- | --- | --- | --- | --- | --- | --- |
| `Device` | yes (enforced) | yes | yes | yes | emul | yes |
| `Upload` | yes | yes | yes | yes | yes | yes |
| `Readback` | yes | yes | yes | yes | yes | yes |
| `Shared` | yes | cap | cap | cap | no | no |

`Device` is `emul` on GL because GL buffers are opaque and its usage hints are
advisory: the backend cannot make the memory device-local, and cannot prevent a
read. It reports the kind as satisfiable, does not pretend the hint did anything,
and is the reason the CPU backend enforces non-mappability instead: without one
device enforcing it, nothing does until a discrete GPU does.

`Shared` is never assumed from the platform. An Apple silicon Mac has it and an
Intel Mac does not, and both are darwin.

### Graphics

| Property | CPU | Metal | Vulkan | D3D12 | GLES 3.1 | WebGPU |
| --- | --- | --- | --- | --- | --- | --- |
| Render passes, rasterization | yes (reference) | yes | yes (unbuilt) | yes (unbuilt) | yes | yes |
| Presentation to a window | no | cap | cap | cap | cap | n/a |
| Multisampling | no | cap | cap | cap | cap | cap |
| Rasterizer-ordered access | yes | cap | cap | cap | cap | no |

[005](005-graphics.md) decides that the CPU backend rasterizes, so the graphics
half has an oracle too, and the reasoning there is the reasoning here: half the
entries in [`conventions.md`](../docs/conventions.md) are graphics conventions,
and those are exactly the ones that cost the predecessor hours. This spec adopts
that decision rather than restating it, and the graphics rows above are the
backend-side view of 005's surface. Multisampling is excluded by 005 because the
sample pattern is not standardized across vendors, so the oracle would be weaker
precisely where MSAA is used.

**Rasterizer-ordered access** is in the table because
[`conventions.md`](../docs/conventions.md) names it as a queryable capability and
this spec owns the matrix. It is `ARB_fragment_shader_interlock` or
`GL_EXT_fragment_shader_interlock` on GL, raster order groups on Metal,
`VK_EXT_fragment_shader_interlock` on Vulkan, and rasterizer ordered views on
D3D12. WGSL has no equivalent, so WebGPU is `no`. The CPU backend has it for free
by rasterizing in primitive order, which is another instance of the oracle
holding the guarantee that hardware makes optional. It is absent by default
everywhere, per conventions.md: unordered fragment writes are not offered as a
way to produce a deterministic buffer.

The predecessor's Vulkan and D3D12 backends never grew a render path at all, so
`unbuilt` is a statement about evidence: the API supports it, nobody in this
lineage has done it cgo-free, and the estimate is unproven.

### Where the answers come from

Each backend resolves its `cap` and `?` cells at device open:

- **Metal**: `supportsFamily:` for feature sets,
  `maxTotalThreadsPerThreadgroup` on the pipeline state,
  `maxThreadgroupMemoryLength`, `threadExecutionWidth` for the SIMD group size,
  `hasUnifiedMemory` for `Shared`, `supportsIndirectCommandBuffers` in the family
  query.
- **Vulkan**: `VkPhysicalDeviceLimits` (`maxComputeWorkGroupSize`,
  `maxComputeWorkGroupInvocations`, `maxComputeSharedMemorySize`,
  `maxStorageBufferRange`, `maxPerStageDescriptorStorageBuffers`),
  `VkPhysicalDeviceSubgroupProperties` for size and supported operation classes,
  the feature structs for 16-bit storage, float16, atomic float, and integer dot
  product, `VkPhysicalDeviceMemoryProperties` heaps for `Shared`,
  `VkPhysicalDeviceType` for software versus hardware.
- **D3D12**: `D3D12_FEATURE_DATA_D3D12_OPTIONS1` (`WaveOps`, `WaveLaneCountMin`,
  `WaveLaneCountMax`), `OPTIONS4.Native16BitShaderOpsSupported`,
  `D3D12_FEATURE_DATA_ARCHITECTURE` (`UMA`, `CacheCoherentUMA`) for `Shared`,
  `D3D12_FEATURE_DATA_SHADER_MODEL` for the reachable model, and the adapter's
  software flag. Workgroup limits are fixed by the API rather than queried.
- **GLES**: `GL_MAX_COMPUTE_WORK_GROUP_SIZE`,
  `GL_MAX_COMPUTE_WORK_GROUP_INVOCATIONS`, `GL_MAX_COMPUTE_SHARED_MEMORY_SIZE`,
  `GL_MAX_SHADER_STORAGE_BLOCK_SIZE`, `GL_MAX_COMPUTE_SHADER_STORAGE_BLOCKS`,
  the extension string, and `GL_RENDERER` for software detection.
- **WebGPU**: `GPUAdapter.limits`, `GPUAdapter.features` for `shader-f16` and
  subgroups, and the adapter's fallback flag.
- **CPU**: its own configuration. See section 5.

The API-guaranteed minimums differ enough that portability is not obvious
(GLES 3.1 and Vulkan guarantee far less per workgroup than D3D12 fixes by
contract, and WebGPU sits between them), so the strict oracle mode in section 5
exists to make a kernel's portability testable rather than assumed.

---

## 4. How a graph lowers

[003](003-command-graph.md) leaves this open and says the answer constrains how
much replay is worth. Here is the answer, decomposed, because the parts have very
different distributions.

Replay saves three separable things:

1. **Plan-once**: validation, DAG construction, barrier computation, transient
   live-range analysis, pool assignment, and allocation. This is `accel`'s own
   work, it is the largest of the three for a graph of any size, and **every
   backend gets it, including software replay.**
2. **Encode-once**: turning planned nodes into driver commands. Only a backend
   with a resubmittable command object gets this.
3. **Driver-side**: state pre-resolution inside the driver when it is handed a
   pre-built object. Real on some drivers, unmeasured, and never the reason to
   choose the model.

**The conclusion**: (1) is backend-independent and is most of the value, so the
recording model is justified even where lowering is software replay. That is the
sentence 003 was waiting for. (2) is a bonus on two backends out of six. A design
that depended on (2) would be a design that only pays off on Vulkan and D3D12,
and this one does not.

### 4.1 Vulkan: a reusable primary command buffer

003 names secondary command buffers. That framing is worth correcting: a
**primary** command buffer recorded without `ONE_TIME_SUBMIT` is already
resubmittable, and it is the natural lowering of a `Graph`. Secondary command
buffers are for *composition*, so they are the right tool if `Graph` later gains
sub-graphs, and not the right tool for the base case. Some drivers also charge
real overhead for `vkCmdExecuteCommands`, so using secondaries for the base case
would be paying for composition nobody asked for.

**Rebinding without re-recording** is the constraint that makes this work.
[003](003-command-graph.md) promises bound resources can change between
submissions. If the command buffer baked in a descriptor, that promise would cost
a re-record. So a binding slot lowers to an entry in a `VkDescriptorSet` that the
command buffer references by handle: the set's *contents* are updated host-side
between submissions, the command buffer stays valid, and rebinding costs a
descriptor write. Push descriptors and push constants are baked into the recorded
commands and are therefore usable only for values that do not vary.

**Honest caveat**: re-recording a primary command buffer in Vulkan is itself
cheap, and for a small graph it may beat the bookkeeping that keeps a recorded one
valid. Which wins is a measurement, not a deduction (open question 1).

### 4.2 D3D12: a closed command list, and bundles mostly do not apply

Same shape as Vulkan: a command list that has been closed can be executed many
times, so that is the replayable object.

**Bundles are not the answer**, and the reason is specific: a bundle cannot
contain a resource barrier, and cannot change render targets. `accel`'s graph is
dispatches separated by computed barriers, so a graph with any dependency between
its nodes cannot be one bundle. Bundles would apply to a barrier-free run of
nodes inside a graph, which is a narrow case, and their documented benefit is
amortising CPU recording of small repeated command sequences, which is the thing
the closed command list already gives us. Bundles are therefore not planned;
revisit only if measurement shows recording cost dominating.

Rebinding lowers to descriptor heap writes with the heap bound by the command
list, mirroring 4.1.

### 4.3 Metal: re-encode by default, indirect command buffers as an option

Metal is the interesting case because it has the least to gain. A
`MTLCommandBuffer` is single-submit, so every submission needs a fresh command
buffer and fresh encoders no matter what. Replay's category (2) saving is
therefore **zero by default on Metal**: the graph is re-encoded per submission
from the planned node list.

That is fine, and stating why matters. Metal encoding is a cheap CPU-side call
per command, and for a graph of a hundred nodes the re-encode is a rounding error
next to the plan-once saving. It stops being fine somewhere in the thousands of
nodes, which a large model reaches.

`MTLIndirectCommandBuffer` is the escape: compute commands encoded once into an
ICB, then executed from a fresh command buffer per submission with a single
`executeCommandsInBuffer`. The costs are real: it requires argument buffers for
resource binding, it requires explicit residency calls for every resource the ICB
touches, and it is a capability, not a guarantee. So ICB lowering is behind the
same optional interface as everything else in this spec, is off by default, and
ships only with a measurement against re-encode. This is the one place where the
graph model's value on a backend depends on an optimisation that is not yet
written.

### 4.4 OpenGL ES: software replay on the context thread

GL has no command buffer at all, so the recorded node list is replayed by the
backend onto its context-owning thread.

What replay saves here is entirely category (1), plus one thing that is not on
the list and is worth naming: the recorded list crosses to the context thread
**once per submission** instead of once per command. The predecessor's backend
marshalled each `backend` method onto the context thread and waited, which is a
channel round trip per call. Submitting a planned graph turns N round trips into
one. This is why [`conventions.md`](../docs/conventions.md) says the recording
model costs GL nothing: GL was going to record anyway, and now it records for a
reason.

### 4.5 CPU and WebGPU

CPU replays the planned node list directly. Category (1) only, and it is enough:
the CPU backend's per-submission overhead is dominated by kernel execution.

WebGPU replays into a fresh `GPUCommandEncoder` per submission. Compute has no
equivalent of render bundles, so there is no native object. The interesting
variable is the crossing cost from 2.6: the difference between a per-node
`syscall/js` call and a single crossing carrying an encoded op list could be an
order of magnitude, and it makes the graph model *more* valuable on WebGPU than
on any backend with a native object. Unproven (open question 5).

---

## 5. The CPU backend in depth

### Two execution strategies, chosen from what the pipeline declares

**Cooperative kernels** (any use of shared memory, barriers, or subgroup
operations) run as **one goroutine per invocation**, with the workgroup's
invocations sharing a shared-memory slice. Go has no coroutines, and a barrier
requires every invocation to be suspendable mid-kernel, so goroutines are the
only construct with the right shape. Workgroups are distributed across a bounded
worker pool sized from `GOMAXPROCS`, since a large dispatch would otherwise
create an unbounded number of goroutines.

**Flat kernels** (no shared memory, no barriers, no subgroups) run as a **plain
loop over invocations** on one goroutine per workgroup. There is nothing to
suspend, so there is nothing to pay a scheduler for. This is ordinary Go code
over ordinary slices, and it runs at the speed of ordinary Go code.

The strategies must agree. A conformance test runs a set of flat kernels under
both strategies and requires identical results, which keeps the fast path from
drifting into a second implementation.

### Barriers, and what the oracle catches

A barrier is a reusable counting barrier over the workgroup's invocations. Three
behaviours make it an oracle rather than an implementation:

- **Non-uniform arrival is detected.** An invocation that returns from the kernel
  while others are waiting at a barrier is a failure with the workgroup id and
  the invocation ids that did not arrive, not a hang and not a silently different
  result. [002](002-compute-model.md) calls this out as a large part of why the
  CPU backend is worth having, and it is: on real hardware this is undefined
  behaviour that usually appears to work.
- **Shared memory is poisoned**, not zeroed, at workgroup start, so a kernel that
  reads before writing fails loudly here instead of working on whichever backend
  happens to zero.
- **`go test -race` finds missing barriers.** This is unique to this backend and
  it is the strongest argument for the goroutine model over anything cleverer.
  Shared memory is a Go slice and the invocations are real goroutines, so a
  missing barrier around a shared-memory read-after-write is a genuine Go data
  race that the race detector reports with both stacks. No GPU backend can offer
  that at any price. It is worth a slower oracle.

Deliberate nondeterminism follows from the same design: for a *correct* kernel
the CPU backend is deterministic, and for a racy one it is not, which is the
right direction. A shuffle mode that permutes invocation scheduling order under a
fixed seed makes ordering assumptions fail reproducibly rather than occasionally.

### Subgroups

Emulated at a **configurable fixed size, defaulting to 4**. Not 1: a subgroup
size of 1 makes shuffle and ballot degenerate and hides exactly the bugs the
emulation exists to find. Small enough that a normal workgroup spans several
subgroups, so cross-subgroup errors and tail handling are exercised.

The size is configurable because a kernel that assumes a subgroup size is wrong
on the next device, and the way to prove a kernel does not assume one is to run
it at 1, 4, 32, and 64 and require the same answer. That is a test the oracle can
run and no single piece of hardware can.

### Strict mode

The oracle enforces the intersection, and how tight the intersection is depends on
what the caller is targeting. So the CPU device is configurable between:

- **Permissive** (default): limits generous enough to run anything, for
  development.
- **Strict portable**: workgroup and shared-memory limits set to the floor
  guaranteed across the backends in section 3, `Shared` memory absent, subgroups
  absent, float atomics absent, narrow arithmetic absent. A kernel that runs
  under strict portable runs on every backend in the matrix. A kernel that does
  not is telling you which capability it needs, at build time, on a laptop.
- **Mimic**: limits and capabilities loaded from a capability record captured
  from a real device, so a failure reported from a machine nobody has can be
  reproduced on one that is available.

### Exactness, and the one thing that threatens it

Integer results are bit-exact everywhere and that is a hard requirement.

Floating point is not, and promising it would be a lie. FMA contraction, sqrt and
transcendental implementations, and reduction order all legitimately differ
between an emitted shader and a Go function. So the contract is:

- **Exact**: integer kernels, and f32 kernels restricted to `+`, `-`, `*`, `/`
  with contraction forbidden and a fixed reduction order.
- **ULP-bounded**: everything else, with the bound stated per operation class
  (transcendentals, `rsqrt`, `fma`-permitted paths), and the bound is part of the
  conformance suite rather than a per-test constant someone tuned until it
  passed.

Forbidding contraction is a **requirement on the emitter**, and its feasibility
is not uniform: SPIR-V has the `NoContraction` decoration, HLSL has `precise`,
MSL has `-ffp-contract`, and GLSL ES 3.1 and WGSL are unresolved.
[004](004-kernel-authoring.md) carries this as an open question from the emitter
side and puts `a*b+c` in its class B; this is the same question seen from the
device side, and it is why FP contraction control is a matrix row with `?` cells
rather than an assumption. It is a real risk to the exact tier, not a formality.

**The oracle can also disagree with itself.** Go permits fusing a multiply and an
add into an FMA unless a conversion specifies the rounding, so the same Go kernel
can produce different f32 results on arm64 (which has FMA) and amd64 (which may
not fuse). An oracle that differs between the developer's machine and CI is worse
than no oracle. The CPU backend therefore requires an explicit `float32(...)`
conversion at each rounding point in its evaluation of a kernel's f32 arithmetic,
and a conformance test runs the same kernels on arm64 and amd64 and requires
bit-identical results. This is a stated obligation of the kernel compiler's CPU
target, not an accident of how the Go code is written.

### Performance expectations

It is a correctness tool. Stating the shape anyway, because "how slow" decides
whether it is usable for real work:

- **Flat kernels**: within a small factor of hand-written Go over slices, and
  parallel across workgroups. This is genuinely usable for small real work: a
  modest tensor operation, a test fixture, a unit-test-sized model. It will not
  compete with a BLAS.
- **Cooperative kernels**: dominated by goroutine scheduling and barrier
  synchronisation, roughly a scheduler operation per invocation per barrier
  interval. Usable for correctness at thousands to low millions of invocations,
  not for production work. A tiled GEMM will run and will be slow, and that is
  the correct tradeoff for the thing whose job is to be right.

The disclosed goal: the CPU backend must be fast enough that the full conformance
suite runs in the time a Go test suite is allowed to take, and fast enough that
layer 2 can be developed against it before any GPU backend is finished
([`000-decisions.md`](000-decisions.md) decision 3). Beyond that, it is not optimised
at the cost of exactness or of the checks in this section.

---

## 6. Selection and enumeration

Three operations, deliberately distinct, per [001](001-device-resources.md)'s
no-silent-fallback rule:

**Enumerate** reports every device every compiled-in backend can see, before
anything is opened, with its backend, name, hardware-or-software flag, and full
capability record. Enumeration never opens a device and never panics: a backend
whose library is missing, stale, or broken contributes an entry explaining why it
found nothing, because "Vulkan is not installed" and "Vulkan is installed and
reports no compute queue" are different problems for the caller.

**Open a named backend** succeeds or returns an error naming the backend and the
reason. It never returns a different backend. This is the rule the predecessor
violated by convenience, and it turned "my GPU code is slow" into a mystery.

**Open the best available** is a separate call that takes an explicit policy: a
preference order, whether software devices are acceptable, whether the CPU
backend is a candidate, and required capabilities. It fails rather than
descending into something the caller did not sanction.

Two constraints on the defaults:

- **The CPU backend is never selected automatically unless the policy names it.**
  It is a first-class backend and it is not a fast path, and a caller who wanted
  a GPU and got the CPU should hear about it as an error.
- **Software GPU devices are treated as their own class.** lavapipe and WARP are
  real devices that report as software, and one may well be slower than the CPU
  backend. Automatic selection must be able to see that distinction, which is why
  it is a matrix row.

**Environment override**, narrowly. `ACCEL_BACKEND` may **restrict the candidate
set for automatic selection only**. It never redirects an explicit request, since
that would reintroduce the exact mystery decision 3 exists to prevent, and
whatever it did appears in the opened device's identity string, so a surprising
device is self-explaining in a log.

---

## 7. Testing and CI

### One conformance suite, parameterized over backends

There is one suite. It runs against the CPU backend first, where a failure is a
bug in the suite or in the model, and then against every device enumeration
found, where a failure is a backend bug or a convention divergence.

Its content is 001, 002, and 003's testing sections plus the capability
discipline of this spec:

- **Every capability-gated path is exercised present and absent.** The absent
  case is the one that regresses, because it is the code nobody runs on their
  own machine. The CPU backend's strict mode makes both cases runnable on any
  machine.
- **Every capability the matrix marks `cap` or `?` is measured and recorded** at
  device open, and the run reports the record, so the matrix in section 3 is
  corrected by evidence rather than by memory.
- **Each path with its own convention is covered separately.** Readback origin is
  the standing example: a compute-buffer test passes while the texture path is
  mirrored. See [`conventions.md`](../docs/conventions.md).
- **The oracle comparison uses the two-tier numeric contract** from section 5,
  exact where exactness is required and ULP-bounded elsewhere, never a tolerance
  chosen until the test went green.

### What CI should depend on, and the rule the ANGLE break taught

The predecessor provisioned four software drivers to get device coverage: Mesa
llvmpipe for GLES, Mesa lavapipe for Vulkan, WARP for D3D12, and ANGLE for GL on
Windows. Three of those were stable for years. The fourth broke, and how it broke
is the useful part: **ANGLE was not installed, it was scavenged** from whatever
browser happened to be on the runner image. When an image dropped the DLL the job
failed for reasons unrelated to the change that triggered it, and the fix (search
Edge, then Chrome, then the WebView runtime, then sweep Program Files) is a job
that depends on Microsoft's browser packaging decisions. As of the 2026-08
runner image, Edge and Chrome no longer ship `libEGL.dll` at all, and Firefox is
the only remaining source. Copying the DLLs out also fails, because every shipped
ANGLE build links siblings that live beside it.

The rule that generalises:

> **A CI job may be blocking only if its driver comes from a package the workflow
> installs explicitly at a pinned version, or from the operating system itself.
> Anything scavenged from software installed for another purpose is
> non-blocking.**

Applied:

| Tier | Jobs | Provisioning |
| --- | --- | --- |
| **1, blocking, every commit** | CPU backend on linux, macOS, windows; `CGO_ENABLED=0` build gate for every `GOOS`; a grep for `import "C"` across the module including tests | none |
| **2, blocking** | Vulkan on lavapipe (apt `mesa-vulkan-drivers`); GLES on llvmpipe (apt `libegl1 libgles2 libgl1-mesa-dri`, `EGL_PLATFORM=surfaceless`); D3D12 on WARP (in the OS); Metal on a macOS runner (in the OS) | one apt install, or nothing |
| **3, non-blocking, reported** | GLES on Windows via ANGLE; WebGPU in a headless browser | scavenged or heavyweight |
| **4, manual or nightly** | real hardware: discrete NVIDIA and AMD, Apple silicon, an Android or mobile GPU if reachable | not GitHub-hosted |

Tier 1 is what makes decision 3 pay. It needs no provisioning at all, it runs on
every platform, and it is the reason a broken tier 2 job never blocks a fix.

**Windows GLES is dropped to non-blocking, and the coverage hole is stated
plainly** rather than papered over with a better scavenging script. Windows is
already covered by D3D12 on WARP and by Vulkan, GLES on Windows exists for old
drivers rather than for capability, and a blocking job whose input is a browser
vendor's packaging decision will break again. The Windows GLES backend stays
build-gated and compile-verified; its runtime path is verified on Linux and on
hardware, not in CI on Windows.

### Green CI does not prove hardware correctness

Every tier 2 GPU job except Metal runs on a **software rasterizer**, and software
implementations have the most forgiving memory models in existence. A missing
barrier, a race on shared memory, or an assumption about subgroup size can pass
on llvmpipe, lavapipe, and WARP, and fail on the first real mobile GPU. Metal on
a macOS runner is the only tier 2 job touching a real driver, and it is one
vendor.

So: tier 2 proves the *plumbing* (the calls marshal, the pipeline compiles, the
result comes back). It does not prove the *memory model*. Tier 4 exists for that,
its findings go into [`conventions.md`](../docs/conventions.md), and no claim of
the form "verified on Vulkan" is made from a lavapipe run alone.

### Per-backend entry gate

Before a backend joins the conformance run it must pass, in this order: a
probe (the library loads, a device is created) run in CI; a capability report
that populates every row of section 3's matrix for that device; a buffer
round trip for every dtype; a barrier-and-shared-memory reduction matching the
oracle; and a tiled GEMM, which [002](002-compute-model.md) names as the proof
that the model is sufficient. A backend that cannot run the GEMM is not
finished, whatever else it can do.

---

## 8. Open questions

1. **Does a reusable Vulkan primary command buffer actually beat re-recording?**
   Vulkan re-recording is cheap, and keeping a recorded buffer valid across
   rebinding costs descriptor bookkeeping. Section 4.1 asserts the reusable
   buffer is the right lowering; that assertion is a hypothesis until a graph of
   realistic size is measured both ways. If re-recording wins, the graph model
   still pays through plan-once, but 003's framing of native replay as the
   payoff needs softening.

2. **Can the CPU backend's cooperative path be made faster without losing what
   makes it an oracle?** Goroutine-per-invocation is what buys suspendable
   barriers and `go test -race`. Anything faster (an interpreter with explicit
   continuations, a compile-time transform that splits kernels at barriers into
   sequentially executed phases) gives up the race detector, which is the single
   most valuable property the backend has. Unresolved whether both can be kept,
   perhaps as two modes.

3. ~~**What exactly is the numeric contract, per operation class?**~~
   **Answered by [008](008-numerics.md)**, which owns the two tiers this section
   names, derives reduction bounds from a stated error model rather than choosing
   them, and forbids a test from carrying a hardcoded tolerance at all. The part
   that is not answered is the part that was never a design question: whether
   contraction can actually be forbidden on Metal is a measurement, it is 008's
   first open question, and until it is taken this backend's exact tier is a
   hypothesis. If it fails, the exact tier here shrinks to integers and
   conversions, which is a materially weaker oracle and would be visible
   immediately, because 008 makes the collapse fail loudly rather than widen a
   tolerance.

4. **How does D3D12 get a shader model 6 shader?** `D3DCompile` reaches SM 5.1,
   which costs the column its subgroups, native f16, and packed 8-bit dot
   product. The options are shipping or locating `dxcompiler.dll` (cgo-free by
   syscall but a binary dependency, and unsigned DXIL needs developer mode), or
   emitting DXIL from the kernel compiler (no dependency, and a piece of work the
   size of Go-to-SPIR-V). [004](004-kernel-authoring.md) carries the same
   asterisk from the emitter side, so this needs one decision taken jointly, not
   two. Until it is taken, four cells in the D3D12 column read `gated` and the
   backend is a compute baseline only.

5. **Is WebGPU worth v0, and does the batched crossing work?** Section 2.6's
   mitigation for `syscall/js` overhead, encoding a graph's node list into a
   typed array and replaying it in one crossing, is plausible and unmeasured. If
   it works, WebGPU is a good fit for the recording model and browsers become a
   target. If it does not, WebGPU is a per-node crossing cost that no amount of
   planning recovers, and `GOOS=js` runs the CPU backend.

6. **How does enumeration survive a broken driver?** R1 says probing must not
   panic, and a Go `recover` does not cover the case that actually happens:
   `dlopen` of a mismatched or corrupt driver, or a driver that aborts the
   process during instance creation, kills the process outright with no Go frame
   to recover in. The predecessor never hit this because it opened one backend
   per platform. Enumerating four means loading four drivers on a machine that
   was never tested with them. Options are probing in a subprocess (robust,
   slow, and awkward on `GOOS=js`), caching a probe result on disk, or accepting
   the risk and documenting that enumeration can take the process down. None is
   obviously right, and this decides how safe `Enumerate` really is.

7. **Where does cooperative matrix support land?** [002](002-compute-model.md)
   defers it and notes it is where most GEMM throughput lives. It is a Vulkan
   extension, a Metal simdgroup matrix type, and a D3D wave matrix, with no
   portable shape and no emulation that is worth writing on the CPU backend
   beyond a correctness stand-in. It is not in this matrix because adding a row
   of `cap` and `?` would imply a design that does not exist yet.

## Testing

Consolidated, since sections 5 and 7 carry the detail:

- Every backend that opens passes the same conformance suite, compared against
  the CPU oracle under the two-tier numeric contract.
- The CPU backend's flat and cooperative execution strategies produce identical
  results on kernels eligible for both.
- The same f32 kernel is bit-identical on arm64 and amd64.
- A subgroup-using kernel produces the same result at emulated subgroup sizes
  1, 4, 32, and 64, and matches its no-subgroup fallback.
- Non-uniform barrier arrival, and a read of unwritten shared memory, both fail
  on the CPU backend with the workgroup and invocation identified.
- A missing barrier in a shared-memory read-after-write is reported by
  `go test -race` on the CPU backend.
- A kernel that passes under strict portable mode runs on every enumerated
  device; a kernel needing an absent capability fails at graph build with the
  capability and the device named.
- Opening a named unavailable backend returns an error naming it, and never
  returns a different backend.
- `ACCEL_BACKEND` cannot change the result of an explicit open.
- The module contains no `import "C"`, verified by a CI grep over all files
  including tests, and every `GOOS` builds with `CGO_ENABLED=0`.

## Amendment: numeric behaviour rows

[002](002-compute-model.md) added denormal preservation (f32 and f16) and
Inf/NaN production to the matrix. All three vary by backend, all three change
results, and none of them can be discovered except by asking, so they are
capabilities rather than assumptions.

Every GPU cell starts `?`. That is deliberate and follows this spec's own rule
about confidently wrong numbers: flush-to-zero behaviour is exactly the kind of
detail that is easy to state from memory and wrong on the device in front of you.
The CPU backend is `yes` for all three because Go's float32 arithmetic preserves
denormals and produces infinities and NaNs, which is also what makes it the
strictest oracle of the set: a kernel that relies on a denormal surviving will
pass there and may not elsewhere.

Filling these in is a measurement task, one small kernel per cell, and it should
happen before any kernel is written that depends on the answer.
