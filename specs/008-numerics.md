---
title: "Numerics: exactness classes, derived tolerances, and the contraction problem"
status: drafted
layer: device
depends_on:
  - 002-compute-model.md
  - 004-kernel-authoring.md
  - 006-backends.md
---

# Numerics

What "the same answer" means when two backends run one kernel, and what a test is
allowed to assert.

This spec exists because the same question was open in three places at once:
[004](004-kernel-authoring.md) asked how floating-point contraction is controlled
per target, [006](006-backends.md) asked what the numeric contract is per
operation class, and [007](007-tensor-layer.md) asked what tolerance an operator
comparison should use. They are one question, and while it had no owner every
testing section in this directory was spending numbers that did not exist.

It also exists because of a specific failure mode that is easy to walk into. A
comparison needs a tolerance; the obvious way to get one is to run the test, see
the difference, and pick something slightly larger. A suite built that way has
tolerances that encode whatever bug was present on the day each test was written,
and it cannot tell a regression from a rounding difference ever again.

**The rule this spec is built to enforce: a tolerance is derived from a stated
error model and a problem size, never chosen to make a test pass.** Everything
below is either that model, or the exactness that makes a model unnecessary.

---

## 1. Two tiers

[006](006-backends.md) §5 names two tiers and defers the content. The content:

| Tier | Meaning | Asserted as |
| --- | --- | --- |
| **Exact** | Bit-identical. Every backend, every run, every architecture. | Equality of bit patterns, never of values, so a negative zero and a NaN payload are compared too. |
| **Bounded** | Differs, by no more than a bound this spec derives from the operation and the problem size. | The derived bound, computed by the harness. |

There is no third tier, and in particular there is no "close enough" tier. An
operation whose bound cannot be derived is an operation that has no test here
until someone derives it.

**Exact is the goal wherever it is reachable, because a bounded comparison cannot
catch a small bug.** A tiled GEMM with a wrong tail predicate is wrong by an
amount that looks exactly like rounding.

---

## 2. The classes

[004](004-kernel-authoring.md) introduced classes A to E for kernel exactness.
They are promoted here to normative, with their tiers and their bounds.

| Class | Contents | Tier | Bound |
| --- | --- | --- | --- |
| **A** | Integer arithmetic; loads, stores, indexing; bit operations; comparisons | Exact | none needed |
| **B** | f32 `+`, `-`, `*` with contraction forbidden and evaluation order fixed | Exact | none needed, **conditional on §4** |
| **C** | f32 `a*b+c` where a target may contract to an FMA | Bounded | §4.3, and it is **not** a small ULP |
| **D** | Conversions between f32 and the narrow types | Exact, given §5's pinned rounding | none needed |
| **E** | Division, `sqrt`, and the transcendentals: `exp`, `log`, `sin`, `cos`, `pow`, `rsqrt`, `tanh`, … | Bounded | §6, per operation, per backend, measured |
| **F** | Atomic float add | Bounded, **and not reproducible against itself** | §7 |
| **G** | Reductions and dot products over many terms, where the summation order differs | Bounded | §3, derived from the length |

**Division is in class E and not class B, which is a correction to
[004](004-kernel-authoring.md)'s original table.** It is tempting to group `/`
with the other three arithmetic operators, and it is wrong: the SPIR-V
environment specification requires `OpFAdd`, `OpFSub` and `OpFMul` to be
correctly rounded and requires `OpFDiv` only to be within **2.5 ULP**, and Metal
under its default floating-point mode is permitted to compute `x/y` as
`x * (1/y)`, which is a different function. A GPU divide is an approximation
instruction with a Newton step, not an IEEE operation.

This is not academic for v0. The GEMM does not divide, but `RMSNorm`, `Softmax`
and `Recip` are all in [007](007-tensor-layer.md)'s v0 operator set and all of
them do, so the first three normalization tests written would have asserted a bit
equality no GPU owes them. The cost of finding this after those tests exist is
that somebody widens a tolerance; the cost of finding it now is one table row.

Class B is the load-bearing one and it is the one at risk. If §4 fails on a
backend, that backend's class B collapses into class C and the exact tier there
shrinks to classes A and D. [006](006-backends.md) says this outright and it is
worth repeating: that would be a materially weaker oracle, not a documentation
change.

---

## 3. Reduction order, and the bound that comes out of it

The most common bounded comparison in the project is "this kernel summed `K`
things in a different order than the reference did". It has a textbook answer and
using it removes the temptation to guess.

For summation of `x_1 … x_K` in f32 with unit roundoff `u = 2^-24`, any
evaluation order satisfies

```
| computed - exact |  <=  gamma_K * sum |x_i|,   where gamma_K = K*u / (1 - K*u)
```

Two consequences the harness relies on:

1. **The bound is on the sum of magnitudes, not on the result.** A dot product
   whose terms cancel has a large *relative* error and a small absolute one, and a
   tolerance expressed as a relative error against the result is wrong for exactly
   the inputs a test should include. The harness compares against
   `gamma_K * sum|a_i * b_i|`, which it computes in f64 alongside the reference.
2. **Both sides are bounded by it, so the comparison is against the bound, not
   between the implementations.** A tiled GEMM and a naive triple loop are two
   evaluation orders; neither is the truth. The f64 reference is the truth, and
   both must lie within `gamma_K` of it.

For a tree reduction of depth `d = ceil(log2 K)` the bound tightens to
`gamma_d`, which is why a workgroup tree reduction is *more* accurate than a
sequential loop, not less. The harness uses the bound for the order the kernel
actually implements, not a single number for all reductions, because using the
loose bound everywhere would hide a genuine error in the tree kernels.

**Accumulation is f32 even when the data is f16**, per
[002](002-compute-model.md) §6.4, and the bound above is then in f32 with the
inputs' conversion error from §5 added once per element. A bound derived as if
accumulation were f16 would be roughly `2^13` times looser and would accept
almost anything.

**Worked, so the numbers are checkable.** A dot product of length 4096 whose terms
are all near 1.0: `gamma_4096 = 4096 * 2^-24 / (1 - 4096 * 2^-24)` is about
`2.44e-4`, and `sum|x_i|` is about 4096, so the absolute bound is about `1.0`
against a result near 4096, a relative bound near `2.4e-4`. A tree reduction of
the same data has `d = 12` and a bound about 341 times tighter. Those are the
numbers a GEMM test asserts, and neither of them was chosen by running anything.

---

## 4. Contraction: the one that decides how much is exact

### 4.1 The problem

`a*b + c` may be evaluated as two rounded operations or as one fused
multiply-add with a single rounding. The two give different results. Go permits
the fusion, and so does every shading language, and neither side promises to make
the same choice.

This threatens two guarantees at once:

- **Backend against oracle.** The emitted MSL may contract where the Go path did
  not.
- **The oracle against itself.** Go's spec permits fusing unless an explicit
  conversion intervenes, so the same Go kernel can produce different f32 results
  on arm64, which has FMA, and amd64, which may not fuse.
  [006](006-backends.md) requires bit-identical results on both, so this one is
  not optional.

### 4.2 The decision

**Contraction is forbidden in class B, on both sides, and forbidding it is an
obligation on the emitter rather than a hope about the compiler.**

| Target | Mechanism | Status |
| --- | --- | --- |
| Go (CPU backend) | an explicit `float32(...)` conversion at each rounding point, emitted by the compiler's Go target, never left to how a human wrote the source | available, and required |
| MSL | the compiler's floating-point mode set away from its default: `mathMode` safe on current Metal, `fastMathEnabled = false` on older | available, and **the default is the wrong one** |
| SPIR-V | the `NoContraction` decoration on the arithmetic instruction | available by specification |
| HLSL | the `precise` qualifier | available by specification |
| GLSL ES 3.1 | **nothing.** The `precise` qualifier postdates it | absent |
| WGSL | nothing equivalent | absent |

**The Metal row is the one that changes what the backend must do, and it is wider
than contraction.** Metal's default floating-point optimization is *fast*, which
does not merely permit fusing a multiply and an add: it permits reassociation, it
permits computing a division as a multiplication by a reciprocal, and it assumes
no NaNs, no infinities, and no signed zeros. Two of accel's stated guarantees die
under it. Class B's fixed evaluation order dies to reassociation, and
[002](002-compute-model.md) §6.3's requirement that overflow produces an infinity
and that NaN propagates dies to the no-NaN assumption, which would make the
`InfNaNProduced` row in [006](006-backends.md)'s matrix `no` on Metal for a reason
that is a compiler flag rather than a device.

So: **the Metal backend compiles kernels with the safe floating-point mode, and
that is a correctness requirement rather than a tuning decision.** It costs
performance and the cost is accepted, because a backend that is faster by
assuming values accel promises to produce is not implementing this design. A
future opt-in relaxed mode is a capability, and it would have to say which of
these guarantees it drops.

The Go row deserves emphasis because it is counter-intuitive: the CPU backend's
exactness is not a property of running Go, it is a property of the *generated* Go
being written to round where the GPU rounds. A hand-written kernel that reads
naturally is not automatically in class B; the compiler's Go target puts it
there.

### 4.3 Where contraction cannot be forbidden

On GLES and WebGPU it cannot, so `a*b+c` there is class C. The honest bound is
unpleasant and is stated rather than softened:

**The difference between a contracted and an uncontracted `a*b+c` is bounded by
one rounding of the product, `0.5 ulp(a*b)`, which is a bound on the *absolute*
error and not on the relative error of the result.** Where `c` nearly cancels
`a*b`, the relative difference is unbounded. A test comparing a class C kernel
must therefore compare against `0.5 * ulp(|a*b|)` accumulated over the
expression's multiply-adds, and a test whose inputs are chosen to cancel is
testing cancellation, not the backend.

The practical consequence for v0 is small, because neither v0 backend is in this
row, and it is exactly why the row must be written down now: the first person to
bring up the GLES backend will otherwise discover that half the conformance suite
asserts bit equality it can no longer deliver, and the cheapest way out of that
discovery is to loosen the tolerances everywhere.

### 4.4 What is measured first

**Before any kernel depends on class B, one probe per target**: a kernel
computing `a*b+c` over inputs chosen so that a contracted and an uncontracted
evaluation differ in the last bit, run on the CPU backend on arm64 and amd64 and
on Metal, comparing bit patterns. It is a day of work and it decides how much of
this spec survives contact.

---

## 5. Conversions are exact, and that is a pinned decision

[002](002-compute-model.md) §6.2 pins every conversion's rounding mode. This spec
adds only the consequence: **class D is in the exact tier**, so a conversion is
compared by bit pattern and never by tolerance.

That is what makes the f16 and bf16 tables in 002 testable at all: they assert
exact output bits for exactly-halfway inputs, including the bf16 case where
round-to-nearest-even and truncation differ, so a `>>16` implementation fails on
the first case rather than drifting.

One asymmetry follows and is worth naming, because it looks like an
inconsistency. A *conversion* is exact; a *value that has been through a
conversion* carries its rounding error into whatever bound applies next. §3's
reduction bound therefore adds one `2^-11` relative term per f16 input, once, and
not per operation.

**Subnormals are where exactness and portability disagree**, and 002 already
decides it: strict portable mode flushes f32 and f16 subnormals to zero, so the
oracle is the strictest device. A comparison run in strict mode against a device
that preserves subnormals is comparing two different specified behaviours, so the
harness refuses it rather than absorbing the difference into a tolerance. Modes
must match, and the run reports which mode it used.

---

## 6. Transcendentals

Class E is bounded and the bounds are **per function and per backend**, because
every shading language specifies its own and they are not the same. Following
[006](006-backends.md)'s rule that a confidently wrong number in a normative spec
is worse than an unknown, this table is a shape to be filled by measurement, not
a set of remembered constants:

Two columns can be filled from specifications rather than measurement, because
the target specifies them normatively. The SPIR-V environment specification's
precision table is the source for the Vulkan column, and it is included here even
though Vulkan is not a v0 backend, because it is evidence about what GPU hardware
does generally and it is what put division in this class:

| Operation | accel requirement | CPU | Metal | Vulkan (specified) | D3D12 | GLES | WebGPU |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `/` | measured | ? | ? | **2.5 ULP** | ? | ? | ? |
| `sqrt` | measured | ? | ? | inherited from `1.0 / inversesqrt` | ? | ? | ? |
| `rsqrt` | measured; it is an approximation instruction on most hardware | ? | ? | **2 ULP** | ? | ? | ? |
| `exp`, `exp2` | measured | ? | ? | **3 ULP** | ? | ? | ? |
| `log`, `log2` | measured | ? | ? | **3 ULP** outside `[0.5, 2.0]` | ? | ? | ? |
| `sin`, `cos` | measured, and stated as an **absolute** bound over a stated argument range rather than a ULP bound, because argument reduction dominates for large arguments | ? | ? | absolute bound, not ULP | ? | ? | ? |
| `pow` | measured; it is usually `exp2(y*log2(x))` and inherits both errors | ? | ? | inherited from `exp2(y*log2(x))` | ? | ? | ? |
| `tanh` | measured | ? | ? | inherited | ? | ? | ? |

Note what the Vulkan column shows about the shape of the problem: `sqrt` is not
specified directly at all, it is specified as a composition of two other
approximations, so its error is inherited and larger than either. An accel
requirement of "correctly rounded `sqrt`" would exclude conformant devices, which
is why the requirement column says measured rather than naming a number now.

**How a cell is filled.** One probe kernel per function, evaluated over a fixed
input set (dense over the exponent range, plus the arguments where the function is
ill-conditioned), compared against a correctly rounded f64 evaluation on the host,
reported as maximum observed ULP. The measured number goes into
[`conventions.md`](../docs/conventions.md), which is what that document is for,
and the requirement column is then set from what every supported backend actually
delivers rather than from what one of them promises.

**Until a cell is measured, the function is not usable in a kernel that has a
conformance test.** That is a real restriction and it is the correct direction: an
unmeasured transcendental in a tested kernel means a tolerance somebody invented.

The CPU backend has its own problem here, and 004 already names it: the intrinsic
must be *some* Go function with a real body, so `sin` on the CPU path and `sin` on
the GPU are two implementations. Class E concedes this. What it does not concede
is that the concession may quietly widen: the CPU implementations are the Go
standard library's, they are correctly rounded or nearly so, and they are the
reference the GPU bounds are measured *against*, so the oracle is the tight side
of the comparison rather than another unknown.

---

## 7. Atomic float add, and reproducibility

Class F is the one case where a bound is not enough, because the operation is not
reproducible against **itself**: the hardware picks the accumulation order and it
varies between runs on one device.

Rules, all three of which exist to stop a flaky test entering the suite:

1. **An f32 atomic test asserts a bound, never a total.** The integer version of
   the same test asserts an exact total, and the two tests must not be written by
   copying one from the other.
2. **The bound is §3's**, with `K` the number of contributing invocations, since
   an atomic accumulation is a summation in an unknown order.
3. **A kernel using atomic float add is excluded from the determinism tests** in
   [003](003-command-graph.md), which assert bit-identical results across two
   submissions. 003 already lists this exclusion; this spec is where the reason
   lives.

---

## 8. What the harness does, and what a test may not do

The enforcement mechanism, because a rule with no mechanism is a preference.

```go
// Package numeq. The name avoids shadowing the standard library's cmp, which a
// test file comparing numbers is otherwise very likely to want.
//
// A comparison names its class and its problem size and the harness derives the
// bound. There is no entry point that takes a tolerance.
numeq.Exact(t, got, want)                         // classes A, B, D
numeq.Reduction(t, got, want, numeq.Sequential(K)) // class G, gamma_K
numeq.Reduction(t, got, want, numeq.Tree(K))       // class G, gamma_log2(K)
numeq.Approx(t, got, want, numeq.Div)              // class E, the measured bound for this device
numeq.Contracted(t, got, want, terms)              // class C, section 4.3
```

Three rules follow from there being no tolerance parameter:

- **A test may not contain a float literal as a tolerance.** Enforced by a
  vet-style check over the test corpus, because this is the rule most likely to be
  broken by someone in a hurry, and it is invisible in review once it is in.
- **`numeq.Exact` on a class that is not exact on this device fails loudly**, rather
  than falling back to a bound. If a backend's class B collapsed under §4, every
  test that assumed otherwise must fail, so the collapse is visible once rather
  than absorbed everywhere.
- **The reference is computed in f64 on the host**, separately from the kernel
  source, per [`000-decisions.md`](000-decisions.md) decision 3: cross-backend
  agreement proves the lowering, and only an independent reference proves the
  mathematics.

---

## 9. What this changes in the sibling specs

| Spec | Was | Now |
| --- | --- | --- |
| [004](004-kernel-authoring.md) | classes A to E, with contraction open | classes are here and normative; 004 keeps the language-side rule that the emitter must forbid contraction |
| [006](006-backends.md) | two tiers, bounds deferred | tiers are here, and 006 keeps the oracle rule and the arm64/amd64 requirement that motivates §4 |
| [007](007-tensor-layer.md) | "tolerances have to be stated per operator class and the numbers have not been chosen" | operator comparisons use §3's derived bounds; nothing per-operator is chosen |
| [002](002-compute-model.md) | pinned conversion rounding | unchanged, and §5 states that this is what puts class D in the exact tier |

---

## 10. Open questions

- **Does class B survive on Metal?** §4.4's probe decides it. If MSL cannot be
  made to stop contracting, the largest exact class on the only v0 GPU backend
  collapses to class C, and the oracle's strongest claim goes with it. This is the
  single most consequential unmeasured thing in the design and it should be
  measured before the GEMM is written, not after.
- **Is the f64 host reference enough for the GEMM?** §3's bound assumes the
  reference is exact. An f64 accumulation of an f32 dot product is not exact, only
  much better, and for `K` in the millions the reference's own error enters the
  comparison. Kahan or exact accumulation on the host would remove the question at
  some cost. Not needed at transformer shapes, and it should be revisited before
  anyone tests a reduction over a whole model.
- **How is a per-device measured ULP distributed?** §6 says the number goes into
  `conventions.md` and the harness uses it. A number in a document is not a number
  in a test. The options are a generated table keyed by device identity, a
  measurement run at suite startup (slow, and it makes the suite's own results
  device-dependent in a way that is hard to review), or a required floor that every
  backend must meet with the measurement used only to catch regressions. Leaning
  toward the last, undecided.
- **Do the narrow dtypes need their own reduction bound?** §3 assumes f32
  accumulation, which 002 makes the default and 004 makes the only thing that
  compiles. When native f16 arithmetic arrives as a capability-gated intrinsic, a
  kernel accumulating in f16 needs a bound with `u = 2^-11`, and at
  `K = 4096` that bound is loose enough to accept a badly wrong answer. The
  probable answer is that f16 accumulation is never in the conformance suite and
  is a caller's explicit choice, but it is not decided here.
- **Reproducibility across driver versions.** Everything above compares a run to a
  reference, not to a stored golden. Storing golden f32 outputs would catch
  drift that a bound absorbs, and would break on every driver update for reasons
  that are not bugs. Not attempted.

---

## 11. Testing

The tests of this spec are the tests that make the other specs' tests meaningful.

- **The contraction probe** of §4.4, on every backend and on both host
  architectures, is the first numeric test in the suite and gates class B.
- **The same f32 kernel is bit-identical on arm64 and amd64**, which is
  [006](006-backends.md)'s requirement and is a direct test of the Go target
  emitting explicit roundings.
- **The derived bound is tested against itself**: a summation whose exact result
  is known analytically (all terms equal, a power of two count) is checked to lie
  within `gamma_K` and to be *outside* `gamma_K / 100`, so a bound that is
  accidentally enormous fails.
- **A deliberately wrong kernel must fail a bounded comparison.** One test injects
  a one-element error into a reduction of length 4096 and asserts the comparison
  rejects it. Without this, a bound that is too loose passes everything and nobody
  notices, which is the failure this spec exists to prevent.
- **Conversion tables from 002 are asserted bit-exact**, including the bf16
  halfway case where truncation and round-to-nearest-even differ.
- **`numeq.Exact` on a device whose class B collapsed fails**, tested by forcing the
  collapse on the CPU backend through its configuration rather than by waiting for
  a backend that has the problem.
- **No test in the corpus contains a hardcoded tolerance**, checked mechanically.
  This is the one that keeps the rest honest and it is the one most likely to be
  worked around, so the check names the file and line rather than reporting a
  count.
