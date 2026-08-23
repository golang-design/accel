---
title: "The rest of the MSL target, and the Metal numeric profile"
status: drafted
layer: device
depends_on:
  - 002-compute-model.md
  - 008-numerics.md
  - 010-kernel-corpus.md
  - 021-metal-bringup.md
---

# The rest of the MSL target

The second of [009](009-sequencing.md)'s three M6 children. [021](021-metal-bringup.md)
made the device an oracle with nine straight-line kernels; this widens the MSL
subset until the rest of [010](010-kernel-corpus.md)'s corpus runs on it, and
records the numeric profile everything downstream is derived from.

## 1. The probes come first, and the order is normative

[009](009-sequencing.md)'s risk table fixes it: *MSL cannot meet
exact/contraction or primitive ceilings | M6 probes before other Metal numeric
tests | Change lowering/domain or reject primitive; never widen from
observation.*

So the first work here is [008](008-numerics.md)'s probes against Metal and the
recorded profile, before any test derives a bound from it. A probe that misses a
normative ceiling is answered by changing the lowering or narrowing the
supported domain. **A bound is never widened to match what the device happened
to report**, because a bound derived from an observation is a bound that says
nothing.

One expectation to carry in, and one already answered:

- Apple silicon's SIMD width is 32, so the CPU sweep at 1, 4, 32 and 64 does not
  map one-to-one. The width is read from a pipeline's `threadExecutionWidth`,
  which [021](021-metal-bringup.md) already does.
- Contraction is **already controlled**: 021 found that `MTLMathMode.safe` does
  not disable it and put `#pragma METAL fp contract(off)` in every emitted
  kernel, with a device test asserting both directions. The probe here confirms
  it rather than discovering it.

## 2. What the emitter gains

Everything [021](021-metal-bringup.md) §5 refuses by name:

| Construct | MSL |
| --- | --- |
| workgroup-shared memory | `threadgroup T *name [[threadgroup(k)]]`, extent fixed at pipeline creation |
| `Thread.Barrier` | `threadgroup_barrier(mem_flags::mem_threadgroup)` |
| atomics | `atomic_fetch_*_explicit` on `device atomic_uint` and friends |
| subgroups | `simd_sum`, `simd_min`, `simd_max`, `simd_broadcast_first`, `simd_is_first` |
| helper functions | `static` free functions, emitted before their callers |
| `kmath` intrinsics | `metal::sqrt`, `rsqrt`, `exp`, `log`, `sin`, `cos`, `tanh`, `abs`, `min`, `max` |
| narrow storage | `half` and its conversions |

**The cooperative lowering is not re-emitted.** MSL has real barriers, so a
Metal kernel is the *authored* structure rather than the resumable state machine
[018](018-cooperative-lowering.md) generates for the CPU. Both come from one IR,
which is what makes the differential meaningful: the CPU runs a program counter
and Metal runs a barrier, and they must still agree.

## 3. The host side of a uniform

[021](021-metal-bringup.md)'s deviation 1 closes here. `kernel.Kernel` gains a
generated encoder so a backend holding a `[]any` can produce std140 bytes
without reflection and without a second layout implementation beside the
generated codec.

## 4. Done

- the Metal numeric profile is recorded, from probes, before anything derives
  from it;
- every kernel in [010](010-kernel-corpus.md)'s v0 list carries MSL and compiles
  on the device;
- CPU and Metal agree on every corpus kernel within [008](008-numerics.md)'s
  budget for that kernel, and bit for bit where the budget is exactness;
- the portable tiled GEMM matches the higher-precision reference on Metal at
  dimensions that are not multiples of any tile dimension; and
- a uniform-carrying dispatch runs, which is deviation 1 retired.

## Testing

The differential against the CPU backend is the whole strategy, and
[021](021-metal-bringup.md) established that it works by showing it detect an
edited kernel. What this child adds is breadth: the same comparison over the
corpus rather than over one kernel.
