---
title: "Ray tracing: acceleration structures, ray queries, and what the oracle can prove"
status: drafted
layer: device
depends_on:
  - 001-device-resources.md
  - 002-compute-model.md
  - 003-command-graph.md
  - 005-graphics.md
  - 008-numerics.md
---

# Ray tracing

The third thing layer 1 can do with a device, after compute and rasterization:
build a spatial index over geometry and ask it what a ray hits.

This is the **normative parent**. It fixes the boundaries and the
cross-component invariants and defers the surface to children, which is
[005](005-graphics.md)'s shape and for [005](005-graphics.md)'s reason — the
last time this project designed a whole subsystem in one spec, that spec could
not be finished and its dependents were marked done anyway
([STATUS.md](STATUS.md)).

## 1. What is missing today, precisely

The vertex and fragment stages exist ([032](032-stage-abi.md)), so a triangle
can be *drawn*. Nothing can ask where a triangle **is**. Concretely, none of the
following has any representation:

- an acceleration structure, at either level, as a resource or as a descriptor;
- a build or refit operation, as a graph node or otherwise;
- a ray, a hit, or a traversal call in the kernel subset;
- a capability bit a caller could query before relying on any of it.

`grep -ri "acceleration\|raytrac\|intersect" ` over the Go tree returns the
rasterizer's triangle-edge functions and nothing else.

## 2. Decision 1: ray queries, not ray-tracing pipelines

Two execution models exist in the industry and accel takes one.

| | ray-tracing pipeline | ray query (inline) |
| --- | --- | --- |
| shape | traversal *calls back* into raygen/miss/closest-hit/any-hit/intersection shaders | a kernel calls traversal and gets a hit back |
| needs | a shader binding table with strides, offsets and per-instance indices; shader recursion; a second dispatch model | a function call |
| reaches | Vulkan `VK_KHR_ray_tracing_pipeline`, Metal intersection function tables, DXR | Vulkan `VK_KHR_ray_query`, Metal `intersection_query`, DXR 1.1 inline |
| CPU oracle | a second execution model, with recursion, that the scheduler does not have | the traversal is an ordinary call from an existing kernel |

```
    ray-tracing pipeline                  ray query
    ─────────────────────                 ─────────
    dispatch rays                         dispatch a compute grid
        │                                     │
        ├─ raygen shader                      └─ kernel body
        │     └─ traceRay ──┐                        ├─ q.Trace(as, ray)
        │                   ├─ any-hit               ├─ ordinary loop
        │                   ├─ intersection          └─ writes its own output
        │                   └─ closest-hit / miss
        └─ SBT decides which shader runs
```

**Decision 3 of [000](000-decisions.md) settles it.** The CPU backend is the
oracle and is never build-tagged away, so anything accel exposes must run there.
A ray query is a call inside a kernel the CPU scheduler already runs. A
ray-tracing pipeline is a dispatch model with recursion, a binding table, and
shader selection at traversal time — a second interpreter, not a new intrinsic.

The cost is stated rather than hidden: **a ray query cannot express what a
pipeline is best at.** Divergent shading per material, where each hit runs
different code chosen by the geometry, is the case the SBT exists for; expressed
as a query it becomes a branch in one kernel and the divergence is the caller's
problem. Path tracers with many materials will feel this. That is the trade, and
it buys an oracle.

**A pipeline is not foreclosed.** It is additive: the acceleration structures in
[054](054-acceleration-structures.md) are the same objects, and a pipeline adds
stages and a table above them. If it is ever built, this decision is what it
revisits.

## 3. Decision 2: an acceleration structure is a resource, and its build is a node

It behaves like a texture in every way that matters to
[003](003-command-graph.md): the device owns it, the caller holds a handle, and
what it contains is opaque.

The build is **not** a constructor. It reads vertex and index buffers, writes
device memory, and takes real time, so it is a node in the graph and gets the
same treatment every other node does — inferred edges against the buffers it
reads, a barrier before anything traverses it, and a place in the plan a caller
can see.

```
  vertex buffer ─┐
  index buffer  ─┼─▶ [build BLAS] ─┐
                 │                  ├─▶ [build TLAS] ─▶ [dispatch: kernel traces]
  instance buffer ──────────────────┘
```

**Two levels, because every target has two** and the split is not accel's to
invent: a bottom-level structure indexes geometry in its own space, a top-level
structure indexes instances of bottom-level ones with transforms. Flattening
them would mean rebuilding geometry whenever an object moves, which is the cost
the two-level split exists to avoid.

## 4. Decision 3: closest-hit is exact, any-hit order is not

This is where ray tracing meets [008](008-numerics.md), and it is the part that
needs deciding before a line is written.

**A closest hit is defined by a minimum**, so it is a deterministic answer:

$$t^{*} = \min\{\,t \in [t_{\min}, t_{\max}] : \text{ray}(t) \text{ hits a primitive}\,\}$$

The *value* $t^{*}$ is a computed float and carries a derived bound like any
other. The *identity* of the primitive at $t^{*}$ is an integer and is either
right or wrong — except where two primitives are within the bound of each other,
which is a silhouette or a coincident surface, and there the answer is
legitimately either.

**Any-hit visitation is an order and no target defines it.** Traversal visits
candidates in whatever order its BVH gives, and the BVH differs per backend by
construction. So:

> A kernel whose result depends on the *order* candidate hits are visited is
> order-dependent, and this spec refuses to make it portable.

That is [002](002-compute-model.md) §4's atomics rule applied to traversal, and
the same test shape catches it: a result that changes when the CPU intersector
shuffles its visitation order is a result that was never portable. The CPU
backend therefore **shuffles deliberately** under a seed, exactly as the subgroup
emulation does, so the failure is found on a laptop rather than on a phone.

**What this means for the differential**, and it is a real weakening: two
backends are compared on the closest hit's primitive id and on $t^{*}$ within a
derived bound, over rays constructed to avoid coincident surfaces. Any-hit
accumulation (transparency counts, occlusion sums) is compared only where the
kernel is order-independent by construction, which for a sum over hits it is.

## 5. Decision 4: geometry is triangles and AABBs, and intersection is fixed-function

A bottom-level structure holds either triangles or axis-aligned bounding boxes.
Triangles get a built-in intersection test; AABBs get a hit at the box, and what
is inside is the caller's problem.

**Custom intersection shaders are out**, for decision 1's reason: they are the
same callback model the SBT exists to route. A caller who wants spheres puts
AABBs in the structure and tests the sphere in their own kernel after the query
returns the box. That is more work for them and one execution model for accel.

## 6. What the caller sees, in shape

Not the API — the children own that — but the boundary this parent fixes:

```go
// Build-time: descriptors, and a node in the recorder.
blas := dev.NewAccelerationStructure(accel.BottomLevelDescriptor{ … })
rec.BuildAccelerationStructure(blas, geometry…)   // a node, not a call

// Kernel-side: a query, in an ordinary compute kernel.
//accel:kernel workgroup=64
func Shade(t accel.Thread, scene accel.AccelStruct, rays []Ray, out []float32) {
    i := t.GlobalID().X
    hit := accel.TraceClosest(scene, origin, direction, tMin, tMax)
    if hit.Hit { … }
}
```

Three properties this parent fixes and no child may change:

1. **Traversal is a call in the existing kernel subset**, not a stage. Its
   result is a value; it writes nothing.
2. **An acceleration structure binds like a resource**, so the graph infers its
   hazards and a build before a trace produces a barrier without the caller
   asking.
3. **A device that cannot do it says so**, through a capability, and a plan that
   needs it is refused at build naming the capability — [002](002-compute-model.md)
   §7's rule, not a new one.

## 7. The children

| | Child | Covers |
| --- | --- | --- |
| 1 | [054](054-acceleration-structures.md) | the two levels as resources, build and refit as nodes, and their validation |
| 2 | [055](055-ray-queries.md) | the kernel-side query, the hit record, and the order-dependence rule |
| 3 | [056](056-cpu-intersector.md) | the reference BVH and traversal, and what makes it an oracle rather than an implementation |
| 4 | [057](057-metal-ray-tracing.md) | `MTLAccelerationStructure` and `intersection_query` through purego, and the differential |

**Order matters here and it is not the numbering.** 056 comes before 057: the
CPU intersector is what every later comparison is against, and building the
Metal path first would mean checking it against nothing. That is the same
sequencing [009](009-sequencing.md) used for the compute path, and the same
reason.

## 8. Out of scope, and stated so it is not mistaken for an omission

- **Ray-tracing pipelines and shader binding tables** — decision 1.
- **Custom intersection shaders** — decision 4.
- **Motion blur / deformable acceleration structures.** Vulkan and DXR expose
  motion structures; Metal exposes them differently; and nothing in accel's
  target set needs them before the static case works.
- **Ray tracing from a fragment stage.** Legal in the targets, and it needs
  [005](005-graphics.md)'s handoff question answered first — the one
  [STATUS.md](STATUS.md) records as having no owning child.
- **Hardware BVH serialization.** Compaction and serialized structures are a
  memory optimisation over a working traversal, not a prerequisite for one.

## 9. Open questions

- **Does the CPU intersector need to be fast, or only right?** It is the oracle,
  so right is mandatory. But a reference BVH that takes minutes over a real
  scene makes the differential untestable in CI, which is how the oracle stops
  being run. Leaning toward a plain binned-SAH build and a stack-based traversal
  — unremarkable, and fast enough that a few thousand rays over a few thousand
  triangles fits a test.
- **Is `tMin` exclusive or inclusive?** The targets differ, and a self-hit at
  $t=0$ is the single most common ray-tracing bug. This needs deciding in
  [055](055-ray-queries.md) and stating in one place, not left to whichever
  backend a caller tested on.
- **Should an instance transform be a full 3×4 or a TRS?** A full matrix is what
  every target takes and what a caller already has. A TRS is smaller and
  refittable without a rebuild. Leaning toward 3×4 because it is the target
  shape, and a caller who wants TRS composes it.
- **Does a query carry a cull mask in v0?** Every target has one and it costs a
  uniform. Leaning yes, because adding it later changes the query's signature,
  which is the mistake [005](005-graphics.md) records for the sample-count field
  it left out.
