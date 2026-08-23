---
title: "Cooperative diagnostics: shared-memory definition, arrival, and conflicting access"
status: implemented
layer: device
depends_on:
  - 018-cooperative-lowering.md
---

# Cooperative diagnostics

The second of [009](009-sequencing.md)'s three M4 children, and the one that
makes the CPU backend an oracle rather than an executor.
[018](018-cooperative-lowering.md) made cooperative kernels run; this makes the
ways they go wrong *reported* instead of *observed*.

## 1. Why these three, and why deterministically

A cooperative kernel has three failure modes that a GPU turns into silence:

| Failure | On a GPU | Here |
| --- | --- | --- |
| Reading shared memory nothing wrote | whatever the last workgroup left there | reported, per element |
| Invocations arriving at different barriers, or not arriving | a hang, or undefined results | reported with both positions |
| Two invocations touching one location unordered | a race whose result depends on the hardware | reported with both accesses |

Each is a wrong answer that reproduces on some machines and not others, which is
the class of bug this project exists to remove. And each is detectable exactly
on this backend, because it controls the schedule.

**Deterministically, on the first offending run.** Not "usually caught under
stress". A diagnostic that fires on an unlucky interleaving is one a developer
learns to re-run past, and [009](009-sequencing.md)'s done criteria say so
explicitly: *on the first offending run rather than on an unlucky
interleaving*. The mechanism for each below is a check on state the scheduler
already has, not a sampling of a race.

## 2. Shared-memory definition tracking

A shadow bit per shared element, cleared at workgroup start and set on write. A
read of an element whose bit is clear is reported with the element index, the
source position, and the invocation.

**Why a shadow bit and not a sentinel value.** A sentinel is a value the kernel
could legitimately compute. Then the check either misses the read, because the
kernel wrote the sentinel itself, or fires on a correct kernel. Neither is
acceptable in an oracle, and [009](009-sequencing.md) asks for the strong form:

> a kernel reading shared memory it never wrote fails for **every** stored bit
> pattern, so the test cannot pass because a sentinel happened to compare unequal

The test therefore sweeps the stored pattern across every bit pattern the dtype
admits — for narrow types exhaustively, for 32-bit types over the boundary
classes plus a random sample — and the diagnostic must fire for all of them.
A sentinel-based implementation fails that sweep by construction, which is the
point of writing it that way.

## 3. Barrier arrival

Every generated suspension point carries a stable barrier ID and a source
position ([004](004-kernel-authoring.md)). At each rendezvous epoch every active
invocation must suspend at the same ID.

An invocation that returns, reaches a different ID, or runs into the next epoch
while peers wait is reported with the expected and observed positions and the
offending invocation ids.

**What this deliberately does not do** is count live invocations against the
number blocked. That count falling short is *one* way an arrival becomes
impossible and not the only one, so inferring from it alone misses the case
where an invocation is still running and will never arrive. Keying the epoch on
barrier identity also means arriving at A while a peer waits at B is a reported
mismatch rather than a silent pairing — one rule covering two failures, which is
why [002](002-compute-model.md) §3.4 says the CPU backend catching this is a
large part of why it is worth having.

## 4. Conflicting access

Two invocations touching one location with no happens-before between them, at
least one writing, is a race. The scheduler knows the epoch boundaries, so
within an epoch it records the accesses each invocation made and reports an
unordered conflicting pair at the epoch's end.

Deterministic because the report does not depend on the two accesses actually
interleaving: they are compared after the fact, from records, so the diagnostic
fires whether or not the schedule happened to expose the race.

This is not `go test -race`, and the distinction is worth stating because the
two are easy to conflate. The race detector checks the CPU **runtime** — the
scheduler, the executable, the queue — and it runs over this milestone for that
reason. Kernel races are found by the instrumentation here, which is why they
are found deterministically rather than probabilistically.

## 5. Testing

- Every row of §1's table has a negative test asserting the message, the source
  position, the workgroup, and the invocation.
- §2's bit-pattern sweep, which is the criterion a sentinel implementation
  cannot meet.
- Non-uniform arrival, two invocations reaching different barrier IDs, and an
  invocation returning before a barrier are three separate cases, because they
  are three different mistakes with the same symptom on a GPU.
- Each diagnostic is asserted to fire on the **first** run, by running the
  offending kernel repeatedly with different scheduler orders — including the
  shuffled order — and requiring a report every time. A diagnostic that passes
  this at one seed and not another is exactly what §1 forbids.
- A correct cooperative kernel produces no diagnostic under any of those orders,
  because a checker that reports everything reports nothing.
- A benchmark reports the instrumentation's cost, since it is on by default in
  developer mode and off in strict mode, and the gap is what justifies having
  two modes.

## 6. Outcome — complete 2026-08-23

All three diagnostics of §1's table are built and §5's cases pass. Each was
confirmed by reinstating the weaker implementation and watching the right test
fail, which for a checker is the only evidence worth having: a diagnostic that
never fires and a diagnostic that cannot fire look identical from a passing
suite.

| Diagnostic | The weaker version, and what it fails |
| --- | --- |
| Shared-memory definition | A sentinel comparison fails the bit-pattern sweep on its first entry |
| Barrier arrival | Inferring from a count of who is blocked stops reporting both mismatch cases |
| Conflicting access | Reporting every shared access fails the concurrent-readers case |

### 6.1 The instrumentation is in the generated lowering

Nothing outside the kernel can see a shared-memory access: it is an ordinary
slice index in generated Go. So the lowering carries the calls, which is what
[004](004-kernel-authoring.md) means by calling the CPU lowering *instrumented*
rather than merely generated. A nil tracker makes each one a no-op the compiler
removes, so strict mode pays nothing for what developer mode wants, and there is
one lowering rather than two to keep in step.

**One bug this found in itself.** A store's own index was instrumented as a
read, so every write also reported a read of the element it was about to
define — a diagnostic that fires on every correct kernel, which is how people
learn to ignore diagnostics.

### 6.2 The barrier position had to be machine-independent

The generated file is committed, so an absolute path in it would differ between
machines and fail the freshness check on every checkout but the one that ran the
generator. It is a base name and a line, which is stable and still actionable
since a kernel and its generated file share a package.

### 6.3 What it added beyond §§2–4

- **`kernel.Diagnostic` and `Diagnostics`**, carrying kernel, workgroup,
  invocation, the conflicting invocation, and element. A report saying only that
  something raced is one nobody can act on.
- **`kernel.SharedTracker`**, with `Read`, `ReadAt`, `Write`, `Epoch`, and
  `Reset`. `ReadAt` returns its index so the instrumentation sits inside an
  expression: a load appears wherever a value does, and hoisting it into a
  statement would change the order of evaluation.
- **`kernel.BarrierID`**, an index and a position, because a mismatch report is
  only useful if a reader can see which two lines disagreed.
- **`kernel.Options`** and `DispatchCooperativeWith`, which is the
  developer/strict distinction [006](006-backends.md) §5 asks for. Diagnostics
  are on by default: the checks are what make this backend an oracle rather than
  an executor.
- **`emit.Package.Fset`**, so a generated diagnostic can name a line in the
  author's file.

## 7. What it does not build

- **No atomics or subgroups**, so no diagnostics for them.
  [020](020-cooperative-atomics.md) adds both together, because a diagnostic for
  an operation that does not exist has nothing to check.

## Correction: the shuffled-order requirement was unenforceable — 2026-08-24

§5 required each diagnostic asserted "including the shuffled order" and §6
declared §5's cases pass. Neither could be true: the shuffled order was accepted
as an option and never reached the scheduler, so every run was the deterministic
one and the "including" clause selected nothing.

[018](018-cooperative-lowering.md)'s correction records the wiring, which landed
2026-08-24. §5's requirement now has a mechanism behind it; running the
diagnostic corpus under a swept seed is the work that discharges it, and it has
not been done.
