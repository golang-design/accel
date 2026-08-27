---
title: "Float to integer: saturating, NaN to zero, and the same answer everywhere"
status: implemented
layer: device
depends_on:
  - 002-compute-model.md
  - 008-numerics.md
  - 013-kernel-subset.md
---

# Float to integer

**One thing:** converting a float to an integer has one answer, and every
backend gives it.

[002](002-compute-model.md) §6.2's last rows. Split out of 002 as an independent
chunk — see [STATUS.md](STATUS.md)'s split plan.

## 1. What was unspecified, and why it compiled

*Past tense as of 2026-08-27: §2.1 records what shipped. This section is kept as
the statement of the problem, because the argument for the shape of the answer
is in it.*

002 §6.2 states a saturating contract: a value past the destination's range
clamps to its limit, and a NaN becomes zero. Neither the API nor the check
existed.

- `accel.ToI32`, `ToU32`, `ToI8`, `ToU8` — no matches anywhere in the tree.
- The front end's `conversion` (`internal/kernelc/front/build.go:1024`) accepts
  any pair `go/types` calls convertible and emits `ir.NewConvert` with **no
  float-source check**.

So `int32(f)` compiles today and means whatever the target does with it. The
three targets disagree exactly where it matters:

| | out of range | NaN |
| --- | --- | --- |
| Go (CPU lowering) | **undefined** by the language spec | undefined |
| MSL | implementation-defined | implementation-defined |
| SPIR-V `OpConvertFToS` | undefined | undefined |

## 1.1 It is live, not latent — corrected 2026-08-27

The paragraph that stood here said **"no kernel in the corpus converts a float to
an integer today — checked, not assumed"**, and called the gap latent. Both were
wrong. Three kernels do it, all in graphics stages, all converting an
interpolated coordinate for a texel fetch:

```
internal/testkernels/stages.go:173   x := int32(in.Texel[0])
internal/testkernels/stages.go:174   y := int32(in.Texel[1])
internal/testkernels/stages.go:210   accel.Fetch(src, int32(c[0]), int32(c[1]))
```

**How the error was made, because it generalises.** The check was a grep for
`int32\([a-z]+\)` — a bare identifier inside the conversion. `int32(in.Texel[0])`
is a selector and an index, so it could not match, and a second grep aimed at
arithmetic missed it the same way. The pattern's shape decided the answer and the
answer was reported as "checked".

**What actually found them was the type checker.** Adding the refusal below to
the front end named all three in one run, with positions. A conversion is a typed
relation, and only the thing that knows the types can enumerate it — a lesson
worth more than the finding, since this project greps for evidence constantly.

**So the order of work is forced, and it is the opposite of the cheap one.** A
refusal cannot land first: it would break three working stages that have no
replacement to move to. The saturating intrinsics of §2 must exist *before* the
bare conversion can be refused, and the refusal is the last step rather than the
first.

The three live sites are also the argument that the semantics matter. A texel
coordinate arriving slightly out of range — which
[032](032-stage-abi.md) §5 says returns zero from the fetch — currently produces
an undefined integer *before* the fetch ever sees it, so the out-of-range rule
those stages exist to demonstrate rests on a conversion that has no rule.

## 2. What gets built

**Two intrinsics — amended 2026-08-27, and it said four.** `ToI8` and `ToU8` are
removed from this spec's scope rather than left outstanding, because building
them would be wrong rather than merely unfinished: [002](002-compute-model.md)
makes the narrow integer kinds *storage*, converted to f32 on load, and
arithmetic on them is outside the subset. A kernel that converted into an `i8`
could not then use the result, so the function would exist to produce a value
nothing can consume. If narrow arithmetic ever arrives they arrive with it.

Intrinsics rather than a conversion rule, and that difference is the point.

$$
\mathrm{ToI32}(x) = \begin{cases}
0 & x \text{ is NaN} \\
-2^{31} & x < -2^{31} \\
2^{31}-1 & x > 2^{31}-1 \\
\lfloor x \rfloor_{0} & \text{otherwise}
\end{cases}
$$

where $\lfloor\cdot\rfloor_0$ is truncation toward zero. `ToU32` is the same with
its own upper limit and a lower bound of zero.

**A named call rather than a conversion expression**, because `int32(f)` reads
like Go and Go does not promise this. A reader who writes the conversion gets a
refusal naming `ToI32`; a reader who writes `ToI32` gets the semantics they
read. The alternative — making `int32(f)` mean the saturating thing — is a
kernel subset that looks like Go and is not, which
[013](013-kernel-subset.md) exists to avoid.

Each needs: the exported function, an IR op, an intrinsic entry, the CPU
lowering, the MSL lowering, and a front-end refusal for the bare conversion
naming the replacement.

## 2.1 Built — 2026-08-27, and what is not

**Built:** `kmath.ToI32` and `kmath.ToU32`, their IR ops, intrinsic entries, and
both lowerings; the three `stages.go` sites migrated onto them; and the front-end
refusal, which names `kmath.ToI32` or `kmath.ToU32` by the destination's
signedness and carries two positioned rejection cases.

**Deviation 1 — they are `kmath`, not `accel`.** §2 above says `accel.ToI32`.
They went where every other scalar-math intrinsic already is, because the
intrinsic table is keyed by package and function identity and `kmath` is the
package the front end resolves scalar math from. Putting these two in the root
package would have made them the only kernel-callable maths outside it.

**Deviation 2 — `ToI8` and `ToU8` are out of scope, not outstanding.** §2 said
four and is amended to two, with the reason there: converting *into* a narrow
integer produces a value the subset cannot then use. The refusal routes an `i8`
destination to `ToI32`, which is the widest sensible answer rather than a
function that is not there. This is an amendment rather than a gap, which is why
the spec can be `implemented` — a section removed for a stated reason is not a
section left unbuilt.

**The boundary differential is built** — `SaturatingConvert` in the corpus, and
a case in the cross-backend differential seeded with NaN, both infinities, and
the exact limits. No ULP ceiling: the outputs are integers of an exact class, so
any difference at all is a divergence.

**One mutation was neutral, and that is worth recording.** Changing the MSL's
`x <= -2147483648.0f` to `<` does *not* fail, because at exactly $-2^{31}$ the
fall-through `int(x)` returns the same value — the boundary is representable and
the clamp is redundant there. So the code is robust at that edge and the
mutation proves nothing. Diverging the NaN result does fail, naming the kernel.
Both facts are the test's discriminating power measured rather than assumed.

## 3. Done

- **The three live sites in `stages.go` keep working**, converting to the same
  texel indices they do today for in-range coordinates. This is the accepting
  half: three shipped stages already depend on this conversion, so the
  intrinsics replace something the corpus runs rather than enabling something
  new.
- **Every boundary converts to its limit**: $\pm\infty$, values just inside and
  just outside each destination's range, and the exact limits themselves.
- **NaN converts to zero**, for every destination, including a NaN whose payload
  differs — a lowering that tests `x != x` and one that tests a bit pattern
  behave differently on a signalling NaN.
- **CPU and Metal agree bit for bit** across that whole set. This is an exact
  class by [008](008-numerics.md) §2: the result is an integer and there is
  nothing to round.
- **The bare conversion is refused, naming the intrinsic and the reason.**
  Removing the refusal makes `int32(f)` compile again, which is what shipped
  before this spec.
- **A kernel that saturates on one backend and wraps on the other is what this
  prevents**, so the differential is over inputs chosen to lie outside the
  range rather than a uniform sweep that never leaves it.
