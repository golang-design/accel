---
title: "Compute model: workgroups, shared memory, barriers, atomics, subgroups"
status: drafted
layer: device
depends_on:
  - 001-device-resources.md
---

# Compute model

Implements [`design.md`](../docs/design.md) decision 4. This spec exists because
its subject was the predecessor project's central mistake, and the mistake is
worth stating precisely so the design is judged against it.

## The mistake being corrected

The predecessor emitted `layout(local_size_x = 1) in;`. One invocation per
workgroup, no shared memory, no barriers, no atomics, and `[]float32` as the only
buffer element type.

Each of those alone is fatal for the workloads layer 2 targets:

- One invocation per workgroup leaves the hardware almost entirely idle, since
  the scheduler's unit is a subgroup of 32 or 64 lanes.
- Without shared memory a tiled GEMM cannot exist, because tiling *is* staging a
  block into fast memory shared by a workgroup.
- Without barriers a workgroup cannot cooperate at all, so no reduction, no
  softmax, no scan, no attention.
- With f32 only, inference runs at 2 to 4 times the memory traffic it should and
  cannot touch quantized weights.

None of these can be added later without changing what a kernel signature means,
which is why they are specified before anything is built.

## Workgroups

Workgroup size is part of the **pipeline descriptor**, fixed at pipeline
creation, not at dispatch. Backends need it at compile time: it appears in the
GLSL layout qualifier and in Metal's threads-per-threadgroup.

A dispatch specifies a count of **workgroups**, not a count of threads. This is
the opposite of the predecessor's `Dispatch(x, y, z)` thread count, and the
change is deliberate: a thread count makes the workgroup size invisible to the
caller, which is exactly how it ended up being 1.

A kernel can read its global id, its local id within the workgroup, its
workgroup id, and the workgroup size.

Every device reports a maximum workgroup size and a maximum total invocation
count per workgroup. Exceeding either is a graph build error, not a dispatch
failure.

## Shared memory

A kernel declares workgroup-shared storage of a given dtype and element count,
fixed at pipeline creation because every backend needs the size statically.

Shared memory is uninitialised at workgroup start. A kernel that reads before
writing gets undefined values, and the CPU backend deliberately fills it with a
poison pattern rather than zeroes, so that a kernel accidentally relying on
zero-initialisation fails loudly on the oracle instead of silently working on one
backend and not another.

Devices report their shared-memory budget per workgroup. Requesting more is a
build error.

## Barriers

Two kinds, and conflating them is a common source of wrong results:

- **Execution barrier**: all invocations in the workgroup reach this point
  before any proceeds.
- **Memory barrier**: writes issued before are visible to reads issued after,
  scoped to shared memory, to storage buffers, or to both.

The common case is both together, and the API provides that as one call so the
easy path is the correct one.

**Uniform control flow is required.** A barrier reached by some invocations of a
workgroup and not others is undefined behaviour on real hardware, and it will
appear to work in testing. This is a documented caller obligation. The kernel
compiler rejects a barrier inside divergent control flow where it can prove
divergence, and the CPU backend detects non-uniform arrival at runtime and fails
the test rather than papering over it. The CPU backend catching this is a large
part of why it is worth having.

Barriers here are *intra*-kernel. Inter-node synchronisation is computed by the
graph builder and is not the caller's concern.

## Atomics

Atomic add, min, max, exchange, and compare-exchange on 32-bit integer and
unsigned types in storage and shared memory.

Atomic **float** add is a queryable capability, not a guarantee. Support is
genuinely uneven across backends, and it is the one people reach for in
reductions, so its absence must be visible rather than discovered.

Atomic operations return their previous value.

## Subgroups

Subgroup operations, a shuffle, a broadcast, a ballot, and a reduction across
the lanes the hardware schedules together, are the difference between a fast
reduction and a slow one.

They are a **queryable capability** with a device-reported subgroup size, not a
guarantee. Subgroup size varies across hardware, some backends do not expose
these at all, and a kernel that assumes a size is wrong on the next device.

Every kernel using subgroups must have a correct path that does not. The CPU
backend implements them at a fixed subgroup size to keep the oracle exact.

## Dtypes

Buffer elements are typed. The set:

| dtype | Notes |
| --- | --- |
| `f32` | Universal. |
| `f16` | Capability. The inference workhorse. |
| `bf16` | Capability. Wider range than f16 at equal width; support is thinner. |
| `i32`, `u32` | Universal. Atomics operate on these. |
| `i8`, `u8` | Storage and conversion. Quantized weights. |

Arithmetic in a kernel is f32 unless the kernel says otherwise. Narrow types are
storage formats that convert on load and store by default, which is what most
inference kernels want: read f16, accumulate f32, write f16. A kernel that wants
native narrow arithmetic asks for it and requires the corresponding capability.

That default matters for correctness, not just convenience. Accumulating a long
dot product in f16 loses accuracy badly, and making f32 accumulation the default
means the naive kernel is the numerically sound one.

## Capabilities

Queried before use, never discovered by failure:

- max workgroup size per dimension, and max invocations per workgroup
- shared memory bytes per workgroup
- subgroup support and size
- f16, bf16 arithmetic support
- atomic float add support
- max storage buffer binding size, and max bindings per kind

A graph requiring an absent capability fails at build with an error naming the
capability and the device.

## Open questions

- **Cooperative matrix / tensor core primitives** (Vulkan cooperative matrix,
  Metal simdgroup matrix, D3D wave matrix). This is where most of the achievable
  GEMM throughput is on modern hardware. Exposing them portably is hard and the
  abstraction is not obvious. Deferred, but the workgroup and subgroup design
  must not foreclose it.
- **Whether f16 arithmetic is worth a separate capability from f16 storage.**
  Several backends store but do not compute. Probably yes, unresolved.
- **Indirect dispatch**, a device-written workgroup count. Needed for anything
  data-dependent and interacts with the graph model's immutability. See
  [003](003-command-graph.md).

## Testing

- A reduction using shared memory and barriers, matching the CPU oracle exactly
  for f32 and within a stated tolerance for f16.
- A tiled GEMM: the motivating case, and the proof the model is sufficient. If
  it cannot be written against this API, this spec has failed.
- Non-uniform barrier arrival is detected by the CPU backend and reported.
- Shared memory poison: a kernel that reads shared memory before writing fails
  on the CPU backend.
- Atomic contention: many invocations atomically accumulating produce the exact
  expected total.
- A subgroup-using kernel and its no-subgroup fallback produce identical results.
- Every capability-gated path is exercised both present and absent.
