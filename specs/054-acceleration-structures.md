---
title: "Acceleration structures: two levels, and a build that is a graph node"
status: drafted
layer: device
depends_on:
  - 001-device-resources.md
  - 003-command-graph.md
  - 053-ray-tracing.md
---

# Acceleration structures

[053](053-ray-tracing.md)'s first child. What the resource is, how it is built,
and what the graph knows about it.

## 1. Two levels, and what each holds

**Bottom level** indexes geometry in one object's own space. **Top level**
indexes *instances* of bottom-level structures, each with a transform.

$$
\text{TLAS} = \{(\,\text{BLAS}_i,\ M_i \in \mathbb{R}^{3\times4},\ \text{mask}_i,\ \text{id}_i\,)\}
$$

A ray is traced against a TLAS. Traversal transforms it into each instance's
space by $M_i^{-1}$, descends the BLAS there, and reports the hit in world space.

**Why the split is not accel's to collapse.** A scene with a thousand instances
of one mesh holds one BLAS and a thousand transforms. Flattening would hold a
thousand copies of the geometry, and moving an object would rebuild it. Every
target draws this line in the same place, which is the strongest evidence there
is that it is the right line.

## 2. A build is a node, not a constructor

```go
blas := dev.NewAccelerationStructure(accel.AccelDescriptor{
    Level: accel.BottomLevel,
    Geometry: []accel.GeometryDescriptor{{
        Kind:        accel.GeometryTriangles,
        Vertices:    accel.VertexRange{Buffer: vb, Format: accel.Float32x3, Stride: 12},
        Indices:     accel.IndexRange{Buffer: ib, Format: accel.Uint32},
        Opaque:      true,
    }},
})

rec.BuildAccelerationStructure(blas)      // a node in the graph
```

The descriptor is fixed at creation and the build reads it. That ordering is
deliberate: a structure's *size* depends on its geometry counts, and a device
allocates for it once, so counts belong to the object and contents belong to the
build.

**What the graph infers.** The build node declares a read of every vertex and
index range in the descriptor and a write of the structure. Traversal declares a
read of the structure. So build-then-trace produces a
read-after-write edge and a barrier with no caller involvement, and
trace-then-rebuild produces a write-after-read — which is
[003](003-command-graph.md) doing its ordinary job on a new resource kind rather
than a new rule.

**A `StageAccelerationBuild` stage bit** joins [003](003-command-graph.md)'s
mask, because a build is neither a dispatch nor a transfer and a barrier that
called it one would be wrong in the direction that costs correctness.

## 3. Refit, and the rule that makes it safe

A refit updates a structure whose **topology is unchanged** and whose vertex
positions moved. It is much cheaper than a build and it is a footgun: refitting
after the index buffer changed gives a structure that traverses and returns
wrong hits, with no error anywhere.

$$
\text{refit valid} \iff
\begin{cases}
\text{same geometry count} \\
\text{same primitive count per geometry} \\
\text{same index contents}
\end{cases}
$$

The first two are host-visible and **refused at record time**. The third is
device data and is not checkable, exactly as
[046](046-segmented-extents.md) §1's sum is not — so this spec does what 046
does: state that the caller owns it, and say what goes wrong, rather than
pretend a check exists.

**A refit before any build is refused**, because there is nothing to refit and
the result would be a traversal over uninitialised memory.

## 4. Validation

Each is host-visible and therefore refused at record time, naming the value.

| | Refused |
| --- | --- |
| V-AS1 | a geometry whose vertex range is not a multiple of its stride |
| V-AS2 | an index count that is not a multiple of three, for triangles |
| V-AS3 | a vertex or index buffer without the usage the build needs |
| V-AS4 | a TLAS instance naming a structure that is bottom-level-built but never built |
| V-AS5 | a TLAS naming a TLAS, or a BLAS holding instances |
| V-AS6 | a refit whose geometry or primitive counts differ from the build's |
| V-AS7 | a traversal of a structure the plan never builds |
| V-AS8 | a build on a device whose capability is absent, naming the capability |

V-AS7 is the one worth naming. A structure bound to a kernel and never built is
a traversal over whatever the allocation held, which returns *plausible hits* —
the failure mode this project refuses, and the same shape as
[026](026-tensor-decode.md)'s stale-version rule.

## 5. What is opaque, and what is not

The contents are opaque: no layout, no traversal order, no node count. A caller
gets back a handle and the memory it cost.

**The memory it cost is not opaque**, because a caller sizing a scene needs it
and every target reports it. `AccelerationStructure.Size()` returns the device
bytes, available after the build node's fence — not after `Build` returns, since
a device may compact.

## 6. Done

- **A build-then-trace graph has an inferred edge and a barrier between them**,
  asserted on the plan rather than on the image, so a barrier the planner would
  drop is visible before it is a wrong pixel.
- **A trace of an unbuilt structure is refused at build**, naming the structure.
- **A refit with changed counts is refused, and a refit with changed indices is
  not** — the second asserted as a documented hazard with a test showing the
  wrong answer it produces, so the boundary is a fact rather than a hope.
- **One BLAS instanced twice with different transforms is hit at both places**,
  which is the whole reason for two levels and fails for any implementation that
  quietly flattens.
- **`Size()` is non-zero after the fence and stable across identical builds.**
- **CPU and Metal agree** on the closest hit for a fixed scene, per
  [053](053-ray-tracing.md) §4's comparison rule.
