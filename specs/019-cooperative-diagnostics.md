---
title: "Cooperative diagnostics: shared-memory definition, arrival, and conflicting access"
status: drafted
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

## 6. What it does not build

- **No atomics or subgroups**, so no diagnostics for them.
  [020](020-cooperative-atomics.md) adds both together, because a diagnostic for
  an operation that does not exist has nothing to check.
