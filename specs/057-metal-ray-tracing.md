---
title: "Metal ray tracing: acceleration structures and intersection_query through purego"
status: drafted
layer: device
depends_on:
  - 021-metal-bringup.md
  - 022-msl-target.md
  - 053-ray-tracing.md
  - 056-cpu-intersector.md
---

# Metal ray tracing

[053](053-ray-tracing.md)'s fourth child. The first backend, after
[056](056-cpu-intersector.md) exists to check it against.

## 1. What Metal gives, and that it is reachable cgo-free

Everything needed is an Objective-C selector or MSL source, which is exactly what
[021](021-metal-bringup.md)'s shim already reaches:

| need | Metal | reached by |
| --- | --- | --- |
| structure object | `MTLAccelerationStructure` | `newAccelerationStructureWithSize:` |
| sizing | `accelerationStructureSizesWithDescriptor:` | a struct return, see §2 |
| build | `MTLAccelerationStructureCommandEncoder` | a new encoder kind |
| refit | `refitAccelerationStructure:…` | same encoder |
| traversal | `intersection_query` in MSL | the emitter, no host call |

**No new FFI mechanism.** The one thing that is not an ordinary
`objc_msgSend` is the sizes query, which returns a struct by value —
`objc_msgSend_stret` on some ABIs, and on arm64 a small struct returns in
registers so the ordinary send works. That is an arm64-specific fact and it is
recorded here rather than discovered: if this ever runs on x86-64 macOS, the
sizes query is the call that needs the `_stret` variant.

## 2. Where it does not match [054](054-acceleration-structures.md)

**Metal sizes a structure from a descriptor, then the caller allocates.** 054
puts counts on the object and contents on the build, which fits Vulkan and DXR;
Metal wants the descriptor at sizing time too. The reconciliation is that 054's
descriptor is fixed at creation, so it is available when Metal needs it — the
mismatch is in *when the allocation happens*, not in what is known.

So `NewAccelerationStructure` on Metal performs the sizes query and the
allocation together, and the build node encodes into the already-allocated
object. The caller sees 054's shape and the divergence stays in the backend,
which is [006](006-backends.md)'s contract working as intended.

**A new encoder kind ends the current one.** `MTLAccelerationStructureCommandEncoder`
is neither compute nor blit, so the pass logic in `internal/metal` grows a third
case beside `compute()` and `blit()` — and, like those, ending the previous
encoder is what orders the build against a dispatch that reads the structure.
This is the same machinery [023](023-metal-graph.md) already has, extended by
one, not a new lifetime rule.

## 3. The MSL side

`intersection_query` is MSL 2.4 and Metal 3, so it is **capability-gated at
pipeline creation** rather than assumed — a device predating it must be refused
by name, and the profile must report the bit honestly.

That last clause is not decoration. [005](005-graphics.md)'s
`RasterizerOrderedAccess` is reported on the CPU profile and is *unreachable* —
the ordering holds, and nothing can observe it because no fragment stage binds a
written slice ([STATUS.md](STATUS.md)). A bit nobody can act on is a weaker
failure than a false one, and still the wrong shape: a caller reads it and plans
around a thing they cannot use. So the Done list below asserts this capability
against a device that actually traverses, rather than against the profile that
claims it.

The emitter grows one case: [055](055-ray-queries.md)'s two intrinsics lower to
an `intersection_query` with the corresponding acceptance mode, and the hit
record is read back field by field into the IR's struct.

## 4. Done

- **The Metal closest hit matches the CPU intersector** on `Hit`, `Instance` and
  `Primitive` exactly, and on `T` and `Bary` within
  [008](008-numerics.md)'s bound, over
  [055](055-ray-queries.md) §7's non-degenerate ray set.
- **A build then a trace in one graph needs no explicit barrier** and produces
  the right answer, which is what says the encoder boundary ordered them.
- **Two structures built in one submission do not alias**, the case a single
  reused encoder would break.
- **A device without the capability is refused at pipeline creation**, naming it.
- **The capability bit is true only where a traversal actually runs**, asserted
  by tracing rather than by reading the bit — the check 005's ROA claim did not
  have.
- **A refit produces the same hits as a rebuild** for moved vertices with
  unchanged topology, which is the only thing that makes refit worth having.
