---
title: "Devices, memory, and resources"
status: drafted
layer: device
depends_on: []
---

# Devices, memory, and resources

The bottom of layer 1: how a device is obtained, how memory is allocated, and
what a resource is.

## Devices

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

## Memory is allocated from pools, not per resource

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

## Buffers

A buffer is a typed, sized range within a pool: a dtype (see
[002](002-compute-model.md)), an element count, and a usage set declared at
creation.

Usage is declared up front because backends need it: it decides the underlying
allocation flags, and a mismatch is a validation error at build rather than
undefined behaviour at execution.

**Views.** A view is a sub-range of a buffer, optionally reinterpreted at a
different dtype where the sizes work out. Views are what let the tensor layer
slice a KV cache or address one attention head without copying. A view does not
own memory and cannot outlive its buffer.

## Textures

Textures exist for graphics and for sampled reads. They carry a format, extent,
mip levels, array layers, and a usage set.

Formats are distinct from buffer dtypes even where they name the same width,
because texture formats carry sampling and colour-space semantics a buffer dtype
does not. Bytes per pixel always comes from the format; see
[`conventions.md`](../docs/conventions.md) for why assuming it is a real bug.

Depth formats are constrained by backends in ways colour formats are not,
including a macOS requirement that depth textures be device-private. The backend
enforces this rather than the caller learning it.

## Transfers

Host to device, device to host, and device to device, all recorded as graph nodes
so they participate in dependency tracking and barrier computation like any other
work.

Texture-to-buffer and buffer-to-texture transfers are first-class. This is not an
incidental convenience: a rasterized G-buffer feeding a compute pass needs
exactly this, and the predecessor could not do it on-device at all, so the data
went out to the host and back every frame.

Readback follows caller row order regardless of the backend's native origin, per
[`conventions.md`](../docs/conventions.md).

## Lifetime

Resources are Go objects with an explicit `Close`. They are not finalizer-managed:
GPU memory is a scarce resource with a driver-imposed cap, and leaving its release
to the garbage collector means release happens at an unpredictable time under
memory pressure that the collector cannot see.

A resource freed while a submission using it is in flight is a use-after-free.
The implementation keeps a submission's resources alive until its fence signals,
so the safe behaviour is the default, and closing early is reported rather than
crashing.

## Open questions

- **Whether pools grow.** A fixed-size pool makes planning simple and makes
  running out a caller problem. A growable pool is friendlier and complicates
  aliasing. Leaning fixed, with the graph builder reporting its requirement in
  advance.
- **Sparse and virtual memory** for models larger than device memory. Out of
  scope for v0, but it should not be designed out.
- **Multi-device.** One device per instance for v0. Tensor-parallel inference
  wants more, and the queue model should not foreclose it.

## Testing

- Every dtype round-trips host to device to host unchanged.
- A view aliases its parent: writing through the view is visible through the
  parent at the right offset.
- Reinterpreting a view at an incompatible dtype is rejected.
- Usage validation rejects a buffer used in a way it did not declare.
- Closing a resource with work in flight does not crash and is reported.
- Readback row order matches the caller's convention on every backend, tested
  through both the texture path and the compute-buffer path, since they diverge.
- Pool exhaustion reports a clear error rather than a driver crash.
