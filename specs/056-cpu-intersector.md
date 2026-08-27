---
title: "The CPU intersector: a reference BVH, and what makes it an oracle"
status: drafted
layer: device
depends_on:
  - 006-backends.md
  - 008-numerics.md
  - 053-ray-tracing.md
  - 054-acceleration-structures.md
  - 055-ray-queries.md
---

# The CPU intersector

[053](053-ray-tracing.md)'s third child, and the one that must exist before the
Metal path is worth writing — [009](009-sequencing.md)'s sequencing rule, since
a backend built first is a backend compared against nothing.

## 1. It is a reference, not an implementation

[035](035-cpu-rasterizer.md)'s stance, applied again. This is the definition of
what a ray query means in accel, and every other backend is checked against it.
That sets the priorities in an order that is not the obvious one:

1. **Right**, always.
2. **Legible**, because a reference nobody can read cannot settle an argument.
3. **Fast enough that the differential runs in CI**, and no faster.

The third is a real constraint rather than a disclaimer. An oracle too slow to
run is an oracle that stops being run, which is how [011](011-conformance-harness.md)'s
§2 profile rule ended up enforced nowhere. A few thousand rays over a few
thousand triangles must fit a test that runs on every push.

**So: a binned-SAH build and a stack-based traversal.** Unremarkable, textbook,
and chosen because a reader can check it against the literature rather than
against its author.

## 2. Build

$$
\text{cost}(\text{split}) = C_t + \frac{A_L}{A}\,n_L\,C_i + \frac{A_R}{A}\,n_R\,C_i
$$

Surface-area heuristic over 12 bins per axis, split at the minimum, leaf at 4
primitives or when no split beats the leaf cost.

**Determinism is a requirement, not a property.** The same geometry must produce
the same tree, so: bins are computed from the centroid bounds in a fixed axis
order, ties in the cost go to the lower axis then the lower bin, and the
primitive partition is stable. Nothing sorts by pointer, nothing iterates a map.

That matters because [054](054-acceleration-structures.md) §6 asserts `Size()`
is stable across identical builds, and because a non-deterministic reference
turns every differential failure into a question about the reference.

## 3. Traversal, and the deliberate shuffle

Closest-hit traversal is an ordered descent with an explicit stack: visit the
nearer child first, cull a subtree whose entry distance exceeds the current best
$t$. Standard, and the ordering is a performance property — the *answer* is the
minimum either way.

**Under a seed, the intersector visits candidates in a shuffled order.** Not to
be fast, and not to be random: to make [055](055-ray-queries.md) §4's rule
*checkable*. A kernel whose answer changes when the order changes is one that was
never portable, and the CPU backend is where that must be discovered.

This is [002](002-compute-model.md)'s subgroup emulation reasoning, one
subsystem over: the emulated width sweep exists so a kernel that assumes 32 lanes
fails on a laptop rather than on a phone. Same shape, same seed plumbing
(`kernel.Options.ShuffleSeed`), same reason.

**Closest-hit and any-hit must both be shuffle-invariant**, and asserting that
is what the shuffle buys. A leaf-ordering bug that happens to return the right
answer for one traversal order is exactly what it catches.

## 4. Watertightness, and the honest limit

A ray passing exactly through a shared triangle edge must hit one of the two
triangles and not zero — a miss there is a visible crack, and it is the classic
failure of a naive Möller–Trumbore.

**Watertight edge handling** (Woop's shear-and-scale formulation) makes the
edge tests agree in sign for shared edges. It is chosen because the alternative
— an epsilon — is a tuned tolerance, which [008](008-numerics.md) §9 rejects on
principle and which this project has spent the week removing elsewhere.

**What it does not buy.** Watertight against *itself* is not watertight against
Metal, whose intersector this spec does not control. So the differential in
[055](055-ray-queries.md) §7 excludes rays constructed to graze a shared edge,
and that exclusion is asserted rather than assumed — a generator that silently
stopped producing edge cases would make the test pass by testing nothing.

The crack-freeness claim is therefore about the CPU backend alone, and it is
stated as such rather than as a portable guarantee accel cannot make.

## 5. Where it lives

`internal/raytrace`, beside `internal/raster`, and for the same reason: it is a
reference implementation the device layer calls, not part of the public surface.

The traversal entry point is a plain function over the built structure and a ray.
The kernel-side intrinsics of [055](055-ray-queries.md) lower to a call to it on
the CPU target, which is what keeps traversal out of
[018](018-cooperative-lowering.md)'s state machine.

## 6. Done

- **A ray through a shared edge hits exactly one of the two triangles**, over a
  sweep of edge-grazing rays. Removing the watertight formulation fails this
  with a miss, which is the crack.
- **The same geometry builds the same tree**, byte for byte, across runs and
  across GOMAXPROCS.
- **Closest hit is invariant under the shuffle seed**, over a scene with
  overlapping primitives, and **any-hit existence is too**.
- **A deliberately order-dependent kernel is caught by the shuffle** — the
  positive control, since the three above pass for an intersector that ignores
  the seed entirely.
- **A ray missing every primitive descends no leaf**, asserted on a counter, so
  culling is measured rather than assumed from timing.
- **The differential over a few thousand rays completes inside the CI budget**,
  asserted as a bound rather than left to notice.
