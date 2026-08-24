---
title: "The rest of the MSL target, and the Metal numeric profile"
status: implemented
layer: device
depends_on:
  - 002-compute-model.md
  - 008-numerics.md
  - 010-kernel-corpus.md
  - 021-metal-bringup.md
---

# The rest of the MSL target

The second of [009](009-sequencing.md)'s three M6 children. [021](021-metal-bringup.md)
made the device an oracle with nine straight-line kernels; this widens the MSL
subset until the rest of [010](010-kernel-corpus.md)'s corpus runs on it, and
records the numeric profile everything downstream is derived from.

## 1. The probes come first, and the order is normative

[009](009-sequencing.md)'s risk table fixes it: *MSL cannot meet
exact/contraction or primitive ceilings | M6 probes before other Metal numeric
tests | Change lowering/domain or reject primitive; never widen from
observation.*

So the first work here is [008](008-numerics.md)'s probes against Metal and the
recorded profile, before any test derives a bound from it. A probe that misses a
normative ceiling is answered by changing the lowering or narrowing the
supported domain. **A bound is never widened to match what the device happened
to report**, because a bound derived from an observation is a bound that says
nothing.

### The recorded profile — 2026-08-23

Measured on an Apple M2, through kernels carrying the same
`#pragma METAL fp contract(off)` the emitter emits, because the question is what
*this project's generated code* does and not what Metal does in the abstract:

| Condition (008 §3) | Metal | CPU oracle |
| --- | --- | --- |
| round to nearest even | yes | yes |
| contraction off | yes | yes |
| subnormals preserved | **no** | yes |
| inf/NaN produced | yes | yes |

`ExactAvailable` is true on both, and it asks only about rounding and
contraction — the normal-result condition belongs to a comparison rather than to
the machine. So **a bit-for-bit differential against the CPU oracle is
justified, over the domain where no result is subnormal.**

The one divergence is recorded rather than worked around. Apple GPUs flush a
subnormal *result* to zero while preserving a stored one, so `x + 0.0f` at
2⁻¹⁴⁹ returns zero and `(2⁻⁷⁰)²` returns zero. That narrows Metal's exact
domain; it does not widen any bound, which is what
[009](009-sequencing.md)'s risk row forbids. See
[`conventions.md`](../docs/conventions.md).

**The probe harness had to change to measure this at all.** `probe.Ops` gained
`MulAdd`, because contraction is a decision a compiler makes *within* one
expression: composing `Mul` then `Add` through two separate kernels cannot fuse
however the device is configured, so the old shape would have reported
contraction off on a backend that contracts everything it compiles. A confident
wrong answer is worse than no probe, which is the argument
[008](008-numerics.md) already makes about probing with Go constants.

One expectation to carry in, and one already answered:

- Apple silicon's SIMD width is 32, so the CPU sweep at 1, 4, 32 and 64 does not
  map one-to-one. The width is read from a pipeline's `threadExecutionWidth`,
  which [021](021-metal-bringup.md) already does.
- Contraction is **already controlled**: 021 found that `MTLMathMode.safe` does
  not disable it and put `#pragma METAL fp contract(off)` in every emitted
  kernel, with a device test asserting both directions. The probe here confirms
  it rather than discovering it.

## 2. What the emitter gains

Everything [021](021-metal-bringup.md) §5 refuses by name:

| Construct | MSL |
| --- | --- |
| workgroup-shared memory | `threadgroup T *name [[threadgroup(k)]]`, extent fixed at pipeline creation |
| `Thread.Barrier` | `threadgroup_barrier(mem_flags::mem_threadgroup)` |
| atomics | `atomic_fetch_*_explicit` on `device atomic_uint` and friends |
| subgroups | `simd_sum`, `simd_min`, `simd_max`, `simd_broadcast_first`, `simd_is_first` |
| helper functions | `static` free functions, emitted before their callers |
| `kmath` intrinsics | `precise::sqrt`, `rsqrt`, `precise::exp`, `precise::log`, `precise::sin`, `precise::cos`, `precise::tanh`, `fabs`, `min`, `max` — the `precise::` namespace is not decoration, see §4 |
| narrow storage | `half` and its conversions |

**The cooperative lowering is not re-emitted.** MSL has real barriers, so a
Metal kernel is the *authored* structure rather than the resumable state machine
[018](018-cooperative-lowering.md) generates for the CPU. Both come from one IR,
which is what makes the differential meaningful: the CPU runs a program counter
and Metal runs a barrier, and they must still agree.

## 3. The host side of a uniform — built 2026-08-23

[021](021-metal-bringup.md)'s deviation 1 is retired. `kernel.Uniform` gained
one field:

```go
// Encode writes v into dst in std140 layout, and reports an error rather
// than panicking when v is the wrong type.
Encode func(dst []byte, v any) error
```

The generator fills it with a closure over the codec it already emits, so
**nothing here re-implements std140**:

```go
{Name: "p", Type: "ScaleParams", Size: 16, Encode: func(dst []byte, v any) error {
    return accel.EncodeKernelUniform(dst, v, ScaleParamsCodec{}.Encode)
}},
```

The two alternatives were both worse. Reflecting over the Go struct would put a
second layout implementation beside the generated one, and two implementations
of a padding rule disagree eventually. Asking callers to encode would move a
compiler-owned fact into code written by hand.

**A nil `Encode` is refused by name**, because it means a record generated
before this field existed. Binding zeros would be the worst answer available: a
uniform block of zeros is a plausible set of parameters, so the kernel would run
and quietly compute something else.

The buffer index is `emit.MSLUniformIndex(len(bindings), i)`, exported by the
emitter so the scheme has one definition rather than two.

## 4. Outcome — 2026-08-23

**All twenty-nine corpus kernels lower to MSL, compile on the device, and agree
with the CPU oracle.** Twenty-two agree **bit for bit**; the other seven reach a
bounded primitive and agree within a ceiling derived from §6's table, named per
kernel.

| | |
| --- | --- |
| carry MSL | 29 / 29 |
| compile on an M2 | 29 / 29 |
| agree exactly | 22 |
| agree within a §6 ceiling | 7 |

**What the differential proves, and what it does not.** It compares two
lowerings of one IR against each other, not against a higher-precision
reference: the CPU runs a resumable state machine with a program counter and
Metal runs the authored structure with a real barrier. So a disagreement is the
transform's. It is **not** §8's composed budget, which needs a reference the CPU
corpus tests already supply; what a ceiling here bounds is the divergence
attributable to two implementations of a bounded primitive each sitting up to
its own ceiling from correctly rounded, on opposite sides. Every ceiling is
recorded with its derivation, and a kernel reaching no bounded primitive keeps
zero — which is what stops a tolerance spreading from the kernels that need one
to the kernels that do not.

Confirmed by reinstating a fault: changing `exp` to `exp2` in one kernel's
emitted MSL reports 3,965,045 ULP against a ceiling of 16.

**The `precise::` namespace, not the default one.** `metal::exp` measured 18 ULP
against 008's ceiling, and `sin`/`cos` 1.9e-3, so the emitter uses
`precise::exp`, `precise::log`, `precise::sin`, `precise::cos`, `precise::tanh`
and `precise::sqrt`. That changed the **lowering** and never the bound, which is
the rule 008 exists to enforce: a ceiling raised until a run passes is a
tolerance, not a contract. `rsqrt`, `min`, `max` and `fabs` need no qualifier.

**Two things the oracle had to be told.** The CPU emulates subgroups at a width
a caller chooses and defaults to 4, while this device executes 32, so a
reduction over 64 elements was two different computations; the oracle is now
opened at the device's reported width, which is what
[006](006-backends.md) §5 makes the option for. And the f16 bindings move as
`[]uint16`, because the API boundary carries bit patterns.

**`numeq` gained `ULPDistance` and `WithinULP`.** Ordered-bit, so a distance is
meaningful across a binade boundary, which is how §6 defines the `sqrt` ceiling
and why. The sign is folded on the unsigned pattern rather than by testing a
converted integer for negative — widening `uint32` to `int64` never produces a
negative, which put the two zeros 2³¹ steps apart in the first version. NaN is
checked before the distance, because every comparison against one is false and
it would otherwise pass every ceiling there is.

## 5. Still outstanding

- `simd_ballot` returns a `simd_vote` rather than an integer, so `Ballot` is
  refused by name. The shuffles and the prefix sums are **built** — 2026-08-24,
  [020](020-cooperative-atomics.md) §6.4 and §6.5 — and this device now reports
  `SubgroupShuffle`, which it did not while nothing emitted them.
- `atomic<float>` is a Metal *version* capability rather than a spelling, so an
  f32 atomic is refused until the capability table can make the family query.
- An **array** member of a uniform block: std140 gives it a 16-byte stride
  whatever its element type, so it cannot be one C array a caller indexes with
  one index, and reconciling that rewrites the index expression rather than the
  declaration. No corpus kernel needs it.

## 6. Done — all met 2026-08-23

- the Metal numeric profile is recorded, from probes, before anything derives
  from it;
- every kernel in [010](010-kernel-corpus.md)'s v0 list carries MSL and compiles
  on the device;
- CPU and Metal agree on every corpus kernel within [008](008-numerics.md)'s
  budget for that kernel, and bit for bit where the budget is exactness;
- the portable tiled GEMM matches the higher-precision reference on Metal at
  dimensions that are not multiples of any tile dimension — asserted directly
  rather than inferred from the differential, because the differential says
  Metal agrees with the CPU and this says Metal is *right*, against a straight
  triple loop that shares none of the kernel's structure; and
- a uniform-carrying dispatch runs, which is deviation 1 retired.

## Testing

The differential against the CPU backend is the whole strategy, and
[021](021-metal-bringup.md) established that it works by showing it detect an
edited kernel. What this child adds is breadth: the same comparison over the
corpus rather than over one kernel.
