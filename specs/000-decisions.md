# accel: design

`accel` is a backend-selectable, cgo-free foundation for running compute and
graphics work on a GPU from Go.

This document holds the decisions everything else is built on. It is normative:
if a spec in `specs/` contradicts it, the spec is wrong. Bounded designs live in
`specs/`; empirical backend behaviour lives in [`conventions.md`](conventions.md).

## Two layers, not one

The library is two layers with a hard boundary between them.

**Layer 1, the device.** Memory, kernels, pipelines, command recording,
submission, and presentation. It knows nothing about tensors, neural networks,
or meshes. Its vocabulary is buffers, textures, workgroups, and barriers.

**Layer 2, the tensor.** Dtypes, shapes, views, an operator set, a computation
graph, and memory planning. It is written entirely in terms of layer 1 and adds
no backend-specific code of its own.

The boundary matters because the two layers have genuinely different users. A
renderer wants layer 1 directly. An inference engine wants layer 2 and should
never touch a bind group. Keeping the tensor layer free of backend code is what
makes adding a backend a layer 1 concern only.

## Locked decisions

### 1. The unit of submission is a recordable, replayable command graph

**This is the decision the rest of the design hangs on.**

The obvious model, and the one WebGPU uses, is a one-shot command encoder: begin
a pass, set a pipeline, dispatch, end, submit, discard. `accel` does not use that
as its primary model.

Instead, work is recorded into a **`Graph`**: a reusable object, immutable once
built, that can be submitted many times with its inputs rebound between
submissions. Single-shot submission still exists, as a convenience that records a
one-use graph and submits it.

Why, concretely. A transformer layer is on the order of a hundred operations, and
a model is dozens of layers. Under one-shot encoding, every token costs a full
re-encode of thousands of commands, every intermediate allocation is negotiated
per operation, and no operation can be fused with its neighbour because the API
has already forgotten what came before. Ollama's `ml.Context` has exactly this
shape for exactly this reason: `Forward` records, `Compute` executes, `Reserve`
pre-plans the allocation. A device API offering only immediate submission forces
its tensor layer to give all of that up.

Recording costs graphics nothing. A render pass is already recorded into a
command buffer under every backend, and a frame whose structure is stable across
frames benefits from replay as much as a model does. Every backend has a native
expression of this: Vulkan secondary command buffers, D3D12 bundles, Metal
indirect command buffers, and, where nothing native exists, a recorded list the
backend replays itself.

The cost is real and accepted: a recorded graph is harder to build and harder to
debug than a straight-line encoder, and validation moves from "when you call it"
to "when you build it". The alternative is a layer 1 that cannot carry layer 2,
discovered only after layer 1 has users.

### 2. cgo-free is a hard requirement

Every backend reaches its driver through `purego` or raw syscalls. No `import "C"`
anywhere in the module, ever, including in tests.

This is the project's central bet: cross-compilation, fast builds, no toolchain
dependency, and a `CGO_ENABLED=0` binary that still uses the GPU.

The cost, stated honestly: the ML ecosystem is C and CUDA all the way down, so
this rules out linking cuBLAS, cuDNN, or GGML. Everything layer 2 needs must be
written as kernels in this repo. That is a large amount of work, and it will not
beat vendor libraries on raw throughput for a long time, if ever.

### 3. A pure-Go CPU backend is a first-class backend

Not a fallback, not a mock. It implements the same interface as every GPU backend
and produces the same results.

Without it every test needs a device, and that cost is not theoretical: the
predecessor project provisioned Mesa llvmpipe, lavapipe, WARP, and
ANGLE-scavenged-from-an-installed-browser across four CI jobs to get coverage,
and the ANGLE one broke when a runner image dropped the DLL. A CPU backend gives
a device-free reference oracle that runs under `go test ./...` on every platform
with no provisioning, and it is what makes the tensor layer testable before any
GPU backend is finished.

It is also the cross-backend oracle. Every GPU backend is verified against it.

**Be precise about what that proves.** Decision 5 has the CPU backend run the
*same kernel source* as every GPU backend, so agreement between them proves the
lowering is correct: the compiler, the bindings, the dispatch, and the backend's
conventions. It does not prove the kernel computes the right thing. A wrong
formula is wrong identically everywhere and every parity test passes.

Algorithm correctness therefore needs a second, independent implementation:
a naive reference written separately, in the test, not derived from the kernel
source. Both checks are required and they catch different failures. Treating
cross-backend agreement as sufficient is the trap this decision is most likely
to lead someone into.

### 4. The compute model is designed in, not retrofitted

Workgroup size is part of the pipeline descriptor. Shared memory, barriers,
atomics, and subgroup operations are in the layer 1 API from the start. Dtypes
include f16, bf16, and 8-bit integers alongside f32.

This corrects a specific, known mistake. The predecessor emitted
`layout(local_size_x = 1) in;`, had no shared memory, no barriers, and no atomics,
and supported only `[]float32`. Each of those alone makes a tiled GEMM
impossible, and without a tiled GEMM there is no attention, no softmax, no
reduction, and so no transformer. Adding them afterwards is not an extension but
a rewrite, because they change what a kernel signature means.

### 5. Kernels are authored in Go

A restricted subset of Go compiles to each backend's shading language. One kernel
source runs as ordinary Go on the CPU backend and as native shader code on the
GPU, which is what makes decision 3's oracle exact rather than approximate.

The predecessor proved this works, including helper functions that compile
correctly on both Metal and GLSL and match a Go run bit for bit. It also showed
the compiler must be built on `go/types` rather than a bare AST walk; skipping
that is a debt that surfaces later as confusing failures.

Note the limit this places on decision 3: sharing the source is what makes the
oracle exact, and equally what stops it from being independent. See decision 3.

### 6. Capabilities are queryable, and absence is explicit

Backends differ in what they support, and the API says so rather than failing at
dispatch time. A caller asks what a device can do and gets a typed answer. An
operation a backend cannot perform returns a clear error naming the missing
capability, never a silent wrong result.

## What this is not

Stated plainly so the scope is not read as a promise:

- **Not a training framework.** There is no autodiff and none is designed. The
  tensor layer targets inference first. Training becomes a real conversation only
  after a competitive GEMM exists.
- **No CUDA backend at v0.** CUDA's driver API is reachable without cgo in
  principle, but it is not in the first milestone. This is the largest single gap
  for anyone whose workload is training.
- **Not a portable performance guarantee.** A cgo-free kernel written here will
  not match a vendor library at v0.
- **Not a WebGPU implementation.** The model is deliberately different (decision
  1) and the API does not aim to match `wgpu`.

## Layering rules

1. Layer 2 imports layer 1. Layer 1 never imports layer 2.
2. Backends implement an unexported interface. Adding one touches no public API.
3. No backend-specific type appears in a public signature.
4. The CPU backend is always buildable on every platform and is never build-tagged
   away.
