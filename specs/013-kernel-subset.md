---
title: "Kernel subset: control flow, helpers, and the full rejection corpus"
status: drafted
layer: device
depends_on:
  - 012-kernel-pipeline.md
---

# Kernel subset

The second of [009](009-sequencing.md)'s three M2 children. [012](012-kernel-pipeline.md)
made one kernel compile and run; this one makes the authored language the whole
v0 subset, and makes everything outside it fail with a position.

## 1. What it adds

The rest of [004](004-kernel-authoring.md)'s subset, which is:

- `for` in all three forms, three-clause, condition-only, and infinite;
- `break` and `continue`;
- helper functions marked `//accel:helper`, emitted ahead of their callers and
  lowered from the same helper IR as their callers;
- the full scalar type set, including the `F16` and `BF16` storage wrappers with
  their conversions;
- struct field selection on non-uniform structs, compound assignment, and
  explicit conversions across the scalar set; and
- the bounded scalar math intrinsics in `accel/kmath`, each carrying its numeric
  class from [008](008-numerics.md).

The IR node set does not grow. [004](004-kernel-authoring.md) closed it and
[012](012-kernel-pipeline.md) declared it whole; this child makes the remaining
nodes reachable. **If a construct here needs a node that is not already
declared, that is a finding about 004 and is resolved there, not by adding a
node in passing.** Recording that as an explicit rule is the point: the estimate
[009](009-sequencing.md) warns about is blown by exactly this kind of quiet
extension.

## 2. Helpers are where the storage restrictions become real

A helper that takes a resource slice can be called from two kernels whose
bindings differ in access mode, so the inferred read/write/atomic mode is a
property of the call site rather than of the helper. The transitive dependency
digest exists for the same reason: a helper edited without its callers being
regenerated is a silent divergence between what the source says and what runs.

Two analyses become load-bearing here and are cheap only while the call graph is
small:

- **Recursion is rejected**, direct and mutual, naming the cycle. No target can
  express it, so this is permanent rather than sequencing.
- **The call graph is acyclic and finite**, which is what lets a helper be
  emitted ahead of its callers in every target that requires declaration before
  use.

## 3. The rejection corpus

[009](009-sequencing.md)'s M2 done criteria ask for one negative test per
rejected v0 construct, asserting message and position. This child owns that
corpus, and the corpus is the deliverable rather than a side effect: it is the
executable form of the subset.

Each entry states which kind of exclusion it is, because
[004](004-kernel-authoring.md) draws the distinction and a reader who cannot see
it will argue with a wall that was only ever a schedule:

| Rejected | Kind | Message says |
| --- | --- | --- |
| recursion, closures, function values | permanent | no target can express it |
| slices of slices, interfaces, maps, channels, strings | permanent | no memory model on a GPU |
| `defer`, `panic`, `goto`, labeled branches | permanent | no structured control-flow lowering |
| allocation | permanent | no allocator |
| `range`, `switch`, `select` | permanent, node set | outside the closed IR node set |
| generic kernels, generic methods | sequencing | out of scope for v0 |
| multiple helper results | sequencing | out of scope for v0 |
| shared parameters, barriers, atomics, subgroups | sequencing | cooperative kernels arrive at M4 |

Every message carries a source position, and a test asserts the position and not
only the text. A diagnostic that names the right problem at the wrong line sends
a reader to the wrong place, which for a compiler is most of the cost.

## 4. Testing

- every row above has a negative test asserting message and position;
- each `for` form lowers and matches the authored function under
  [004](004-kernel-authoring.md)'s level 5, including a loop whose bound is a
  parameter and one whose trip count is zero;
- `break` and `continue` inside nested loops match the authored function;
- a helper called from two kernels with different access modes produces the
  correct mode at each call site;
- editing a helper without regenerating its callers fails the dependency digest,
  naming both; and
- direct and mutual recursion are rejected naming the cycle.

## 5. Open questions

- **Whether `for range` over an integer should be admitted.** Go 1.22's
  `for range n` is exactly the bounded loop kernels write most, and it lowers to
  a three-clause loop mechanically. It is rejected today because
  [004](004-kernel-authoring.md)'s node set names `range` as outside the set,
  and admitting it is an amendment to 004 rather than a decision here. Worth
  making, since the alternative is every kernel in the corpus spelling out an
  induction variable the IR immediately normalizes.
