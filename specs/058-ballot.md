---
title: "A ballot a kernel can spell, and the first capability Metal does not have"
status: implemented
layer: device
depends_on:
  - 002-compute-model.md
  - 004-kernel-authoring.md
  - 013-kernel-subset.md
  - 020-cooperative-atomics.md
  - 022-msl-target.md
---

# Ballot

**One thing:** `t.Ballot(pred)` returns a value a kernel can ask questions of.

[002](002-compute-model.md) §5.2's `Ballot` row and its `KernelMask` type. Split
out of 002 as its own chunk — see [STATUS.md](STATUS.md)'s split plan.

## 1. What exists, and what a kernel can reach

Almost all of it is built, and none of it is reachable:

| piece | where | reachable from a kernel |
| --- | --- | --- |
| the mask type, with 002 §5.2's whole method set | `internal/kernel.Mask` | no — unexported package |
| `Thread.SubgroupBallot(pred) Mask` | `internal/kernel/subgroup.go` | no — no intrinsic entry |
| `ir.OpBallot` | the IR | no — nothing produces it |
| `SubBallot`, and the scheduler case that combines it | `internal/kernel/schedule.go` | no — nothing suspends at it |
| `CapSubgroupBallot`, and its `//accel:requires` spelling | the capability table | yes, and it gates nothing |

So this is the shape [010](010-kernel-corpus.md)'s rule names and
[001](001-device-resources.md)'s reporter observed about `pagetable`: **the
mechanism is built and unreachable**. The gap is three lines of table and one
type decision, and the type decision is the reason it is a spec.

## 2. Why the type is the whole design

`Ballot` is the only subgroup operation whose result is not a `bool`, an
`f32`, or an id. 002 §5.2 fixes what it must be and why:

> `M` is `accel.KernelMask`, an opaque value type with methods rather than a
> `uint64`. The dtype set has no 64-bit integer, and Vulkan's ballot is 128 bits
> wide, so a `uint64` would foreclose a real device.

That is a **new kind in the kernel subset**, and [013](013-kernel-subset.md) is
what says which kinds exist. The subset admits scalars, `ID3`, fixed arrays,
slices, and one texture kind. A mask is none of those:

```
        a kernel may hold                what a mask needs
        ┌──────────────────┐             ┌──────────────────┐
        │ scalar, ID3,     │             │ an opaque value  │
        │ array, slice,    │  ──────▶    │ with methods,    │
        │ texture          │             │ 128 bits wide    │
        └──────────────────┘             └──────────────────┘
```

**The methods are the part that makes it a kind rather than a scalar.** A
kernel writes `t.Ballot(p).Count()`, and `Count` is a method call on a value the
compiler must lower — so each of 002 §5.2's five methods is an intrinsic in its
own right, keyed on the mask receiver:

| method | result | lowers to |
| --- | --- | --- |
| `Count()` | `i32` | a population count |
| `Bit(lane)` | `bool` | a shift and a test |
| `LowestSet()` | `u32` | a count of trailing zeros |
| `CountLower(lane)` | `i32` | a masked population count |
| `Any()` | `bool` | a comparison against zero |

The alternative — expose the bits and let a kernel do its own arithmetic — is
what 002 §5.2 rejects, and the rejection has a maintenance clause worth
repeating: *"if it is incomplete the fix is more methods, not exposing the
representation."*

## 3. Metal cannot spell it, and that is the point of building it

[022](022-msl-target.md) §5: `simd_ballot` returns a `simd_vote`, not an
integer. 002 §5.2's capability grouping already accounts for this — ballot is
`CapSubgroupBallot` and the vote and shuffle families are not, *"because
grouping them under ballot would refuse a device that has all five."*

So this is the **first kernel-visible capability the first backend does not
have**, and everything else in the corpus runs on both. Three properties that
have never been exercised together become live:

1. A kernel is *legal* and its Metal lowering is *refused*, by name and
   position, per [004](004-kernel-authoring.md)'s rule that a target-specific
   rejection names the target.
2. `Device.Supports` answers false on a real device for something a kernel can
   spell, so the capability query stops being decorative.
3. The corpus differential must **skip** rather than fail, and skipping has to
   be a stated reason rather than a silent absence — [011](011-conformance-harness.md)
   §2's ground: a test that quietly does not run is worse than one that fails.

**That is why this is worth building before the reductions.** The remaining
subgroup reductions are more of what already works; ballot is the case that
proves the capability system does anything at all.

## 4. What gets built

- `Thread.SubgroupBallot(pred bool) KernelMask` reachable from a kernel — see §4.1 on the name — with `KernelMask` on the root package aliasing
  `internal/kernel.Mask`, exactly as `Thread` and `ID3` are aliased.
- `MaskKind` in the IR, and its admission to [013](013-kernel-subset.md)'s
  subset with the five methods as intrinsics.
- The Go lowering, which is the existing `Mask` methods called directly.
- The MSL refusal, naming `simd_vote` and 022 §5.
- The capability wired to the intrinsic, so `//accel:requires subgroup_ballot`
  is what a kernel declares and a device without it refuses the pipeline.

**`NotEmpty` is renamed `Any`.** 002 §5.2 names five methods and the type has
five, one under a different name. The spec's name wins: a caller reading the
table and writing `m.Any()` is the case the API exists for.

### 4.1 The `Thread` spelling, which is not this spec's to change

002 §5.2's table writes `t.Ballot(pred)`, `t.Elect()`, `t.Any(pred)`. The code
writes `t.SubgroupBallot`, `t.SubgroupElect`, `t.SubgroupAny` — and so does
**every one of the sixteen** subgroup methods, including the ten that shipped
milestones ago and are called from the corpus.

So the divergence is not this operation's, and renaming one to match the table
would leave fifteen that do not. The prefixed form is also the better name on a
`Thread` that carries workgroup and dispatch accessors beside these: `t.Any` is
ambiguous where `t.SubgroupAny` is not.

`Ballot` therefore ships as `t.SubgroupBallot`, consistent with its family, and
**002 §5.2's table is what should change**. That is an edit to a normative
document about ten shipped methods, so it is recorded here and left for a
decision rather than taken in passing. The mask's own method names are a
different case and are matched: nothing shipped depends on them, and `Any` was
one rename.

## 5. Done

Each assertion names the mutation it catches.

- **`Ballot(lane < k).Count()` is `min(k, subgroupSize)`**, swept across
  emulated sizes 1, 4, 32 and 64. The accepting half, and the sweep is what
  makes it more than one arithmetic identity.
- **An inactive lane's bit is zero**, so `Ballot(true).Count()` under divergent
  control is the *active* count and not the subgroup size. 002 §5.1's rule 2,
  and the one it says everyone gets wrong.
- **Each of the five methods against a reference computed from the same
  predicate**, so a method that returned a plausible number — `LowestSet`
  returning zero for an empty mask rather than the width — disagrees.
- **`CountLower(lane)` equals an exclusive scan over the ballot**, which is the
  use the method exists for and the one that catches an off-by-one in it.
- **The Metal lowering is refused, naming `simd_vote`**, rather than emitted
  approximately.
- **The corpus differential skips this kernel with a reason**, and a test
  asserts the skip is *declared* rather than inferred from the kernel carrying
  no MSL — otherwise a kernel that lost its Metal half for an unrelated reason
  would silently join the skipped set.
- **A device without the capability refuses the pipeline**, naming the
  capability, rather than compiling a kernel it cannot run.

## 6. Built — 2026-08-28

Every assertion in §5, and the two guards §5 asked for turned out to exist
already and to fire on their own.

**`TestEverySubgroupRendezvousIsRegistered` pinned `Ballot` as the one
unreachable rendezvous** and now reads zero — the count was written as an exact
number rather than a floor precisely so a second one could not be exempted
alongside it. **`TestEveryKernelLowersToMSLOrSaysWhyNot`** demanded a declared
reason for the missing Metal artifact, which is §5's last-but-one assertion,
enforced before this spec was written. Neither needed adding.

### 6.1 What the capability system did on its own

Two things happened without being asked for, and both are the system working:

- The kernel declared `//accel:requires subgroup_ballot` and generation
  **refused it**, because the body also reads `SubgroupLane` and that needs
  `subgroup_basic`. Capability inference caught a directive that was true and
  incomplete.
- The corpus differential's completeness check did not have to be told to skip
  this kernel. It compares kernels that carry MSL, and the reason this one does
  not is now written down where a reader finds it.

### 6.2 Deviation 1 — the authored differential runs at one lane

`Thread.SubgroupBallot` returns a mask holding only the calling lane's bit,
because combining is the scheduler's job and not the method's — the same shape
every authored subgroup collective has. So the authored/lowering comparison runs
at subgroup size 1, where that **is** the right answer, exactly as
[020](020-cooperative-atomics.md)'s subgroup reduction does.

What it checks is therefore the five methods and the plumbing, not the
rendezvous. The rendezvous is checked by the sweep, where the generated path
runs at real widths and every lane must see the whole subgroup's vote.

### 6.3 Deviation 2 — `t.SubgroupBallot`, not `t.Ballot`

§4.1 has the reasoning. The name ships prefixed, consistent with its fifteen
siblings, and 002 §5.2's table is what should change — an edit to a normative
document about ten shipped methods, left for a decision rather than taken here.

### 6.4 Two gaps found after it was called built

Both were found by asking what a *caller* can write that this does not handle,
rather than by reading the code again.

**A workgroup barrier under a mask method was accepted.** `Count`, `Any` and
`LowestSet` take no arguments, and the analysis computed an intrinsic's level as
the join over its *arguments* — a join over zero operands is workgroup-uniform.
So `if m.Count() > 1 { t.Barrier() }` compiled, and different subgroups can take
different branches of it. The methods lower to ordinary calls rather than
rendezvous, so `IsSubgroupRendezvous` did not reach them either.

The receiver is what carries the level, and it is now folded in. Two of the five
methods were rejected before the fix **for the wrong reason** — `Bit` and
`CountLower` take a lane operand, and the obvious test passes `SubgroupLane` —
so a test written with only those two would have reported the analysis correct.
The three nullary ones are the ones that matter, which is why all five are rows.

**§5's inactive-lane assertion is not reachable from the corpus.** The only way
to make a lane inactive is to put the ballot inside a conditional, and
[018](018-cooperative-lowering.md)'s split cannot resume inside a branch — the
same limitation §4.3 records for the subgroup barrier. So the kernel that would
witness 002 §5.1's rule 2 is refused by the compiler.

It is checked at the **scheduler seam** instead, where a hand-written cooperative
kernel suspends at a ballot from inside a branch, which is what the mask's other
rules already use. That is a smaller claim than §5 made and it is the true one:
the rule holds of the runtime, and no kernel a caller can write today exercises
it. The honest fix is 018's, and this is recorded rather than quietly satisfied
by a test that does not reach the case.
