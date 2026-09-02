---
title: "Uniform loads: a kernel declares the bindings its dispatch never writes"
status: implemented
layer: compiler
depends_on:
  - 002-compute-model.md
  - 003-command-graph.md
  - 019-cooperative-diagnostics.md
---

# Uniform loads

**One thing:** let a kernel name the read-only bindings no invocation of its
dispatch writes, so that a load from one at a workgroup-uniform index is
workgroup-uniform and may bound a barrier's control flow, and enforce the
promise where each half of it is visible.

A child of [002](002-compute-model.md) §3.3. 002 owns the uniformity analysis
and its seeds; this owns the one seed 002 refused to add and the reason it can
be added soundly.

## 1. The problem

002 §3.3 seeds every load from a storage buffer non-uniform, "because another
invocation may have written the location", and closes the escape hatch a
read-only binding suggests: [003](003-command-graph.md) permits one buffer
bound to a read binding and a write binding of one dispatch, so a binding the
kernel never writes can still change under a workgroup mid-dispatch.

Wiring the analysis into the build (which [019](019-cooperative-diagnostics.md)
requires and which did not happen until 2026-09-02) refused four production
kernels: `GroupedMatVec`, `GroupedMatMul`, `AttentionRagged` and
`AttentionRaggedF16`. Each reads a routing table (`offsets`, and for the ragged
pair `lengths`) at a group-derived index and returns early, or bounds a
segment loop, on what it finds, with barriers after. 002's prescribed fix,
bounding the loop by the binding's *extent*, does not apply: the predicate is
the table's *contents*, and restructuring `GroupedMatMul` to keep every lane at
its barriers makes each expert's workgroup walk every token block, an
Experts-fold slowdown of the prefill kernel.

## 2. The rule

A kernel may carry

```go
//accel:uniform offsets, lengths
```

naming bindings from its own signature. For each named binding:

- **The compiler refuses** the directive when the name is not a binding, when
  the body writes the binding (directly or through a helper, after access
  propagation), when the list is empty, and on a helper, which has no dispatch
  to promise anything about.
- **The analysis** treats `IndexExpr` on a declared binding as the level of its
  index: a load at `t.GroupIndex()` is workgroup-uniform, a load at
  `t.LocalIndex()` is not. Only the kernel's own parameter object qualifies; a
  helper's parameter of the same name is analysed on its own summary.
- **The record** carries `kernelabi.UniformLoad` on the binding's `Access`, so
  the graph can see the declaration without the compiler.
- **The graph refuses** a dispatch that binds a concrete write binding over
  bytes a declared binding reads, at record time, as `ErrUniformLoadAliased`.
  A slot on either side is covered at Bind by 003's V24, which already refuses
  any slot overlapping any writer; two concrete resources are the one pair V24
  does not compare, which is why the record-time half exists.

### 2.1 Why this is sound where 002's escape hatch was not

002's objection is the alias: a read binding and a write binding of one
dispatch over the same bytes. The declaration turns that alias from permitted
into refused, for the declared binding only. With no writer of the same
dispatch over those bytes, the value a workgroup loads is the value every
workgroup loads: a dispatch's own writes are the only writes that can land
between two invocations' loads within it, and the graph orders every other
node's writes against the dispatch by its inferred hazards.

The residual is a kernel that writes the bytes through a *different* binding
whose range the graph cannot see through, which is exactly V24's domain and is
refused there. What is not covered, and is the caller's, is a write from
outside the graph while a submission is in flight, which [001](001-device-resources.md)
already forbids for every resource.

### 2.2 What it costs

- A kernel author has to know which of their tables is routing data, which is
  a fact they already hold: it is the table the host wrote before the dispatch.
- In-place dispatch over a declared binding is refused, which is the point.
- The check at record time is a pairwise walk over a dispatch's bindings,
  which are few.

## 3. Testing

- Frontend: the four refusals, each by message; the marking of exactly the
  named bindings.
- Analysis: a declared load at a group index is `Workgroup`, at a lane index
  `Non`, and an undeclared load `Non`; a barrier loop bounded by a declared
  load is accepted and refused without the declaration.
- Build: a lane-bounded barrier loop and a routing-table loop without its
  declaration are refused at generation with the barrier's and the predicate's
  positions; the same routing loop generates with it.
- Graph: a concrete writer over a declared binding's bytes is
  `ErrUniformLoadAliased` at Build, a disjoint range of the same buffer runs,
  and the slot case is V24's `ErrRebindOverlap`.
- Corpus: the four kernels declare their tables and the differential compares
  them on Metal unchanged.

## 4. Outcome — 2026-09-02

Built as specified in one pass. `uniform.AcceptBarriers` runs in the build,
the four kernels carry the declaration, and no kernel body changed. One
deviation from the first draft: a bind-time check was written and removed
once it was shown V24 already refuses every case it would have caught.
