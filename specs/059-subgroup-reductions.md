---
title: "The remaining subgroup reductions, and the cost of one opcode per type"
status: drafted
layer: device
depends_on:
  - 002-compute-model.md
  - 008-numerics.md
  - 020-cooperative-atomics.md
  - 022-msl-target.md
---

# Subgroup reductions

**One thing:** §5.2's reduction row, minus the three that are built.

[002](002-compute-model.md) §5.2. The last of that spec's five successors — see
[STATUS.md](STATUS.md)'s split plan.

## 1. What the row promises and what exists

> `t.SubgroupAddF32(v float32) float32`, plus `Mul`, `Min`, `Max` over
> f32/i32/u32 and `And`, `Or`, `Xor` over i32/u32

Enumerated, that is **seventeen** operations. Three exist:

| | f32 | i32 | u32 |
| --- | --- | --- | --- |
| `Add` | **built** | — | — |
| `Mul` | — | — | — |
| `Min` | **built** | — | — |
| `Max` | **built** | — | — |
| `And` | n/a | — | — |
| `Or` | n/a | — | — |
| `Xor` | n/a | — | — |

Plus the scans: §5.2 gives inclusive and exclusive add-scans, built for f32 and
absent for the integer types the row admits.

## 2. Why this is a spec and not an afternoon

**Each operation costs ten edits across six files.** Measured on
`SubgroupMinF32`, which is the most recently added and therefore the fairest
sample:

| file | edits |
| --- | --- |
| `ir/ir.go` | opcode, name-table row |
| `intrin/intrin.go` | table entry |
| `emit/coop.go` | carrier, runtime-name mapping, and the family switch |
| `emit/msl.go` | Metal spelling |
| `kernel/subgroup.go` | `SubgroupOp` constant, `String` case, `Thread` method |
| `kernel/schedule.go` | `combineOne` case |

Seventeen operations is around **170 mechanical edits**, and mechanical edits in
seventeen near-identical shapes are where a transposition survives review: the
`Min` case that says `>` is a wrong answer that compiles, in a family where
every neighbour looks the same.

**And the frame has nowhere to put the value.** `Frame` carries `SubF32`,
`SubBool`, `SubMask` and `SubLane`. An i32 or u32 reduction has no carrier, so
this is not "more of the same" at the runtime seam even though it is at every
other one.

```
     what a reduction needs to move        what Frame has
     ┌───────────────────────────┐         ┌──────────────┐
     │ f32   built               │────────▶│ SubF32       │
     │ i32   ── nothing ──       │    ✗    │ SubBool      │
     │ u32   ── nothing ──       │    ✗    │ SubMask      │
     └───────────────────────────┘         └──────────────┘
```

Three ways to close that, and the choice is the design:

1. **A field per type.** `SubI32`, `SubU32`. Two more fields, no ambiguity, and
   `Frame` is per invocation per workgroup so the size is not free.
2. **One `uint32` word, reinterpreted.** The i32 case bit-casts. Smaller, and it
   reintroduces exactly the kind of type confusion the typed bindings exist to
   prevent — a reduction that read the wrong interpretation would produce a
   plausible number.
3. **Generic over the carrier.** `Frame[T]` is not available: the scheduler
   holds `[]Frame` for a whole workgroup and the kernel's entry point is a
   non-generic function pointer in a record the generator writes.

**Option 1**, on the grounds that `Frame` already spends a `Mask` (16 bytes) on
one operation, and that a bit-cast carrier is a wrong answer that compiles.

## 3. What f32 `Mul` costs that the others do not

Reductions over f32 are the one place [008](008-numerics.md) has something to
say. `Add` is already bounded by §7's reduction rule; `Mul` is not, because a
product of *n* values has a different error growth and a different overflow
story — a subgroup of 64 lanes each holding 4 multiplies to 2^128, which is
finite in f32 only just.

**So `Mul` over f32 needs a derived bound before it needs an opcode**, and this
spec does not assume §7's applies. `Min` and `Max` are exact for every type, and
the integer operations are exact by construction, so seventeen operations split
into sixteen that need no numeric work and one that does.

## 4. Integer semantics, stated so the CPU and Metal can be compared

002 §4.3 fixes the atomics' overflow behaviour and says nothing about
reductions. The same rule applies and is stated here rather than inferred:

| | rule |
| --- | --- |
| `AddU32` | wraps modulo 2^32 |
| `AddI32` | wraps, two's complement |
| `MulU32`, `MulI32` | wrap likewise |
| `Min`, `Max`, `And`, `Or`, `Xor` | exact, no overflow to have |

**Wrapping rather than saturating**, matching the atomics, because a reduction
that saturated would disagree with the same arithmetic written as a loop — and a
kernel author checking a subgroup reduction against a scalar fallback is exactly
what [020](020-cooperative-atomics.md)'s fallback pattern asks them to do.

## 5. Metal, and what it does not spell

Metal has `simd_product`, `simd_and`, `simd_or`, `simd_xor` and the integer
overloads of `simd_min`/`simd_max`. **This is not verified against a device**,
and it is the first thing to check: the emitter's table is transcribed from a
target's documentation, and [058](058-ballot.md) is the standing reminder that a
plausible spelling can be absent — `simd_ballot` exists and returns the wrong
type.

If an operation turns out to be unspellable, it takes [058](058-ballot.md) §3's
route: a declared reason in `unlowered`, the refusal naming the target, and the
differential skipping with that reason rather than silently.

## 6. What gets built

The seventeen operations, the two carriers, and the integer scans. In three
slices, so that a slice can land and be verified rather than seventeen arriving
at once:

1. **`Min` and `Max` over i32 and u32.** Four operations, exact, and the two new
   carriers. The slice that proves the shape.
2. **`And`, `Or`, `Xor` over i32 and u32.** Six, exact, no new machinery.
3. **`Mul` over the three types, and the integer scans.** The one that needs
   §3's derived bound first.

## 7. Done

Each assertion names the mutation it catches.

- **Each reduction equals a scalar loop over the same lanes**, computed without
  a subgroup operation, at emulated sizes 1, 4, 32 and 64 — the sweep
  [020](020-cooperative-atomics.md) §4 uses, for the reason it gives.
- **`Min` and `Max` disagree on an input where they must**, so a transposed
  comparison in one of seventeen near-identical cases fails rather than passing
  as its neighbour.
- **An inactive lane contributes nothing, not an identity.** 002 §5.2's rule:
  a reduction over one active lane is that lane's value, which for `Mul` over
  `0` and for `And` over `0` are the two cases where "identity" and "absent"
  give different answers.
- **The integer reductions wrap**, matching §4's table and 002 §4.3's atomics,
  asserted at the boundary rather than by a value that happens not to overflow.
- **CPU and Metal agree exactly** for every integer operation, and within
  [008](008-numerics.md)'s bound for f32 `Mul` once §3's derivation exists.
- **Each new carrier moves the value it names.** A reduction whose result read
  the wrong frame field would return a plausible number, which is what the
  bit-cast carrier §2 rejects would make routine.
