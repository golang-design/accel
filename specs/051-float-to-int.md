---
title: "Float to integer: saturating, NaN to zero, and the same answer everywhere"
status: drafted
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

## 1. What is unspecified today, and why it compiles

002 §6.2 states a saturating contract: a value past the destination's range
clamps to its limit, and a NaN becomes zero. Neither the API nor the check
exists.

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

**No kernel in the corpus converts a float to an integer today** — checked, not
assumed — so nothing is wrong now. This is the same shape as
[050](050-barrier-scopes.md): a gap that is latent because nobody has written
the kernel the spec says is legal.

## 2. What gets built

Four intrinsics, not a conversion rule, and the difference is the point.

$$
\mathrm{ToI32}(x) = \begin{cases}
0 & x \text{ is NaN} \\
-2^{31} & x < -2^{31} \\
2^{31}-1 & x > 2^{31}-1 \\
\lfloor x \rfloor_{0} & \text{otherwise}
\end{cases}
$$

where $\lfloor\cdot\rfloor_0$ is truncation toward zero. `ToU32`, `ToI8` and
`ToU8` are the same with their own limits and a lower bound of zero.

**A named call rather than a conversion expression**, because `int32(f)` reads
like Go and Go does not promise this. A reader who writes the conversion gets a
refusal naming `ToI32`; a reader who writes `ToI32` gets the semantics they
read. The alternative — making `int32(f)` mean the saturating thing — is a
kernel subset that looks like Go and is not, which
[013](013-kernel-subset.md) exists to avoid.

Each needs: the exported function, an IR op, an intrinsic entry, the CPU
lowering, the MSL lowering, and a front-end refusal for the bare conversion
naming the replacement.

## 3. Done

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
