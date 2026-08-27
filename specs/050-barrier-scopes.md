---
title: "Barrier storage classes, and what a barrier makes visible on each backend"
status: drafted
layer: device
depends_on:
  - 002-compute-model.md
  - 018-cooperative-lowering.md
  - 022-msl-target.md
---

# Barrier scopes

**One thing:** a barrier says *which memory* it orders, and every backend makes
that much visible.

[002](002-compute-model.md) §2.3, §2.4, §2.5 and §3.1's subgroup-scope row. Split
out of 002 because it is one of five independent chunks that spec owns, none
blocking the others — see [STATUS.md](STATUS.md)'s split plan.

## 1. The gap this closes, stated exactly

002 §2.5 is normative that `Barrier` orders **shared and storage** memory. The
MSL emitter emits threadgroup scope only:

```
internal/kernelc/emit/msl.go:1042
    threadgroup_barrier(mem_flags::mem_threadgroup)
```

`mem_device` appears nowhere in the repository. And the three calls §2.3 and
§2.5's table name — `BarrierShared`, `BarrierStorage`, `SubgroupBarrier` — do
not exist: no `Thread` method, no IR op, no intrinsic entry, no emitter case.
Only `Thread.Barrier` exists.

**Nothing is broken today, and that is a fact with a shelf life.** No kernel in
the corpus writes a storage buffer, barriers, and reads that buffer expecting
another lane's write — checked mechanically, not assumed. So the divergence is
latent. It becomes a wrong answer the first time somebody writes that kernel,
and 002 will have told them it was legal.

```
        what 002 §2.5 promises          what Metal emits
        ┌──────────────────┐            ┌──────────────────┐
        │ shared + storage │            │ shared only      │
        └──────────────────┘            └──────────────────┘
                    ▲                            ▲
                    └──── a kernel written to ───┘
                         the left is wrong on
                         the right, silently
```

## 2. What gets built

| | is | so that |
| --- | --- | --- |
| `Thread.BarrierShared()` | shared-memory scope only | the common case says so, and costs no device fence |
| `Thread.BarrierStorage()` | storage scope, and shared | a lane can publish through a buffer |
| `Thread.Barrier()` | unchanged: shared **and** storage | 002 §2.5 stays true rather than being narrowed to match the emitter |
| `Thread.SubgroupBarrier()` | subgroup scope | §3.1's row, which has no call today |

Each needs: the `Thread` method, an IR op, an intrinsic-table entry, the CPU
scheduler's handling, and an MSL case emitting the matching
`mem_flags` — `mem_threadgroup`, `mem_threadgroup | mem_device`, and
`simdgroup_barrier` respectively.

**`Barrier` keeps its meaning rather than being redefined.** Narrowing it to
threadgroup scope would make the emitter correct by making the guarantee weaker,
and every kernel already written against §2.5 would keep compiling while meaning
something else. The direction that cannot silently break a caller is to make the
implementation meet the spec.

## 3. Done

Each assertion names the mutation it catches.

- **A kernel publishes through a storage buffer across `BarrierStorage` and
  every lane reads the published value**, on both backends. This fails today on
  Metal and is the accepting half of the whole spec.
- **`BarrierShared` emits threadgroup scope and `BarrierStorage` emits device
  scope**, asserted on the emitted MSL rather than on a result — a barrier
  lowered at the wrong scope is a different program that passes any test whose
  data happens to fit in one threadgroup.
- **`Barrier` still orders both**, so the guarantee 002 §2.5 states is the one
  the corpus gets. Removing `mem_device` from its lowering fails this.
- **`SubgroupBarrier` is refused where the subgroup size is not known**, naming
  the capability, rather than lowering to a workgroup barrier that happens to
  work.
- **CPU and Metal agree** on the publishing kernel, which is what says the two
  schedulers implement one memory model rather than two.
