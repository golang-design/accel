---
title: "Barrier storage classes, and what a barrier makes visible on each backend"
status: implemented
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

## 1.1 Half of it is built — 2026-08-27

**`Barrier` now emits the scope §2.5 specifies.** The MSL lowering is
`threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device)`, which
is verbatim the target text §2.5's lowering table gives. 118 barriers across the
corpus changed and nothing else did, and CPU and Metal still agree — so the
guarantee is now delivered rather than promised.

**The assertion is on the emitted text, not on a result**, and that was forced
rather than chosen. A kernel whose data fits in one workgroup gets the right
answer either way, and Apple hardware may make a storage write visible across a
threadgroup barrier regardless — "may" being what undefined behaviour looks like
from the inside. §2.5's table gives the target text per backend, which is what
makes this checkable at all.

**What the old code got wrong is worth keeping.** The emitter emitted the narrow
scope *and carried a comment asserting the narrow rule as if it were the spec's*.
Code and its own justification were wrong together, so reading either against the
other would not have caught it — only reading both against §2.5 did.

What remains of this spec is §2's masked variants: `BarrierShared`,
`BarrierStorage` and `SubgroupBarrier`, which let a caller ask for the cheaper
scope deliberately. That matters on tiled mobile hardware for the reason §2.5
gives, and it is additive now that the default is correct.

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

## 4. Built — 2026-08-28

All five assertions. `BarrierShared`, `BarrierStorage` and `SubgroupBarrier`
exist as `Thread` methods, IR ops, intrinsic entries and MSL cases, with the
target text transcribed from [002](002-compute-model.md) §2.5's lowering table
rather than derived.

**The scopes are asserted on the emitted text, and that is forced.** §1.1
already recorded why for `Barrier` and it holds for all three: a workgroup's
data fits in one threadgroup, so it gets the right answer at any scope, and a
result cannot tell three apart. What the *results* check is that the rendezvous
happens at all — `PublishStorage` and `PublishShared` each have an accepting
half, because a lowering that emitted the right `mem_flags` and rendezvoused
nobody would satisfy a text assertion alone.

### 4.1 Two things this spec did not know it needed

**The `Subgroup` uniformity level had no seed.** The lattice has had three
levels since M4 and the analysis could reach two: `SubgroupID`,
`SubgroupInvocationID` and every subgroup result are named in 002 §3.3's seed
table and none was implemented, so each fell through to "the join of its
arguments", which for a nullary intrinsic is `Workgroup`. A predicate on
`SubgroupLane` read as workgroup-uniform and a barrier under it was **accepted**.

Nothing could have caught it before, because the only rule that distinguishes
the levels is the one this spec adds. 002 §12 names the discriminating pair —
"a `SubgroupBarrier` controlled by `SubgroupID` is accepted while the same
control around a workgroup barrier is rejected" — and writing it as a test made
three of its four rejections pass.

**A per-subgroup arrival check.** The CPU scheduler compared every active
invocation against one expected barrier. That is right for a workgroup barrier
and wrong for this one: lanes of different subgroups are under no obligation to
be at the same subgroup barrier, and the control the acceptor now permits puts
them at different ones deliberately. The `BarrierID` carries its scope and the
check narrows accordingly.

### 4.2 A mutation that proved nothing, and the kernel written because of it

Dropping the scope flag from the generated lowering left every test green. The
reason is that `SubgroupPublish`'s barrier is at the **top level**, so every
lane is at it and a workgroup-wide check passes anyway — the narrowing was
unreachable from the corpus, and a passing suite said nothing about it.

`SubgroupStagger` is the reachable shape: a subgroup barrier inside a loop whose
trip count is `SubgroupIndex`, so the subgroups are genuinely out of step at
every epoch. With it, the same mutation fails with "it returned while its peers
wait at barrier 2". **An assertion whose subject is unreachable is not an
assertion**, and the way that was found was reinstating the fault rather than
reading the test.

### 4.3 Deviation 1 — a divergent subgroup barrier is still refused

[018](018-cooperative-lowering.md)'s split cannot resume inside a branch, so a
`SubgroupBarrier` under an `if` is refused even though §5.3 makes it legal. The
loop form is what is built, and it is enough for the rule §12 states because a
subgroup-uniform loop bound produces the same staggering.

The refusal for a *subgroup operation* in a conditional was improved along the
way: it fell through to the Go emitter's "no lowering for intrinsic
SubgroupAddF32", which names no position, no kernel and no way forward, and
reads as an unimplemented operation rather than an unplaceable one. It now says
which it is, cites §5.3 for the legality, and gives the shape that works.
