---
title: "Dispatch-shape accessors, and their place in the uniformity seeds"
status: drafted
layer: device
depends_on:
  - 002-compute-model.md
  - 012-kernel-pipeline.md
---

# Dispatch shape

**One thing:** a kernel can ask how big its dispatch is.

[002](002-compute-model.md) §1.3's last three rows and their §3.3 seed entries.
Split out of 002 as an independent chunk — see [STATUS.md](STATUS.md).

## 1. Three accessors that are specified and absent

`t.WorkgroupSize()`, `t.NumGroups()` and `t.GlobalSize()` appear in §1.3's table
of built-in ids and again in §3.3's workgroup-uniform seed set. None exists:
not on `internal/kernel.Thread`, not in the intrinsic table, not in the IR.

A kernel that needs its dispatch shape today takes it as a uniform field, which
every corpus kernel does — `GEMMDims.M`, `RaggedDims.Batch`, and so on. That
works and costs a uniform slot, and it is why this has not bitten: the shape is
available, just not from the thread.

**So this is convenience, not correctness**, and it is scoped here rather than
folded into a larger change for that reason. It is the smallest of 002's five
pieces and the only one with no hazard behind it.

## 2. What gets built

| accessor | is | already known to the backend as |
| --- | --- | --- |
| `t.WorkgroupSize()` | the declared workgroup extent | a compile-time constant from the `accel:kernel` directive |
| `t.NumGroups()` | how many workgroups this dispatch has | the grid the recorder set |
| `t.GlobalSize()` | `WorkgroupSize × NumGroups`, per axis | derived, not a third input |

`WorkgroupSize` is a constant the generator already has, so it lowers to a
literal rather than a read — which is worth stating because it means using it in
a loop bound keeps that bound compile-time uniform.

`GlobalSize` is derived rather than passed. Two numbers that must agree
eventually disagree, and the multiplication is free.

**All three are workgroup-uniform seeds** in §3.3's lattice, which is what makes
them usable in a barrier's control flow. That lattice is itself unbuilt — the
enforcement today is `checkBarrierPlacement`'s syntactic rule in
[018](018-cooperative-lowering.md) — so this spec adds the seeds to §3.3's table
and does not depend on the analysis existing.

## 3. Done

- **Each accessor returns what the dispatch was recorded with**, on both
  backends, for a grid whose three axes differ so a transposed lowering is
  visible.
- **`GlobalSize` equals `WorkgroupSize × NumGroups` per axis** rather than being
  bound separately.
- **A kernel using `WorkgroupSize()` as a loop bound still compiles with a
  barrier in that loop**, which is the property that makes the accessor worth
  having over a uniform field.
- **CPU and Metal agree**, exactly: these are integers.
