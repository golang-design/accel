---
title: "Quantized weights: the int8 representation, its error bound, and the kernels"
status: drafted
layer: tensor
depends_on:
  - 001-device-resources.md
  - 008-numerics.md
  - 010-kernel-corpus.md
  - 007-tensor-layer.md
---

# Quantized weights

The first of [009](009-sequencing.md)'s M8 items. [000](000-decisions.md) makes
v0 "deliberately unquantized" so that the first token proves the layering rather
than throughput; this is the first thing built on top of that proof, and it is
what makes the tensor layer useful for a model somebody actually ships.

**Scope: symmetric int8 with a per-block scale.** Not because it compresses
best — it does not — but because it is the vertical slice this repository cuts
through every hard thing, the way [012](012-kernel-pipeline.md),
[021](021-metal-bringup.md) and [024](024-tensor-bringup.md) each did. It proves
the whole chain: a representation, a *derived* error bound, quantized kernels,
both backends agreeing, and a tensor-layer operator. Sub-byte formats then
become a second variant on machinery that works rather than two unproven things
at once.

## 1. The representation

A weight matrix is stored as two planes rather than one buffer of structs.
[001](001-device-resources.md) §3.2 forces that and gives the reason: layer 1
types a buffer by dtype, and an interleaved block struct has no dtype.

```
weights  w[0..K)  ──quantize──▶  quants  q[0..K)      i8, one per weight
                                 scales  s[0..K/32)   f16, one per block of 32
```

$$w_i \approx q_i \cdot s_{\lfloor i/32 \rfloor}, \qquad q_i \in [-127, 127]$$

**Why 32.** [010](010-kernel-corpus.md)'s tiled GEMM steps K in 16, and 32 is a
multiple of it, so **no K-step ever straddles a scale boundary**: a step reads
one scale, not two. `RowWidth` is 128, also a multiple, so the row kernels
inherit the same property. A block of 24 would have been smaller and would have
put a branch in the innermost loop.

**Why symmetric.** A zero point is a third plane, an extra load per block, and a
subtraction in the inner loop. It earns that where a distribution is not centred
on zero; transformer weight matrices are close to centred, and this spec's job
is the slice rather than the best ratio.

**Why 127 rather than 128.** The negative range of int8 reaches −128, and using
it would make the scale asymmetric about zero: `-128 * s` has no positive
counterpart. Clamping to ±127 keeps the representation symmetric, which is what
lets the error bound below be stated without a special case at one end.

**Storage.** Quants are an `I8` buffer. Scales are `F16`. Both are ordinary
dtypes, so nothing here needs [001](001-device-resources.md)'s `ViewAs`
reinterpretation — that is reserved for the sub-byte formats, where a nibble
plane genuinely has no dtype of its own.

## 2. Quantizing

Per block $b$ over its 32 weights:

$$s_b = \frac{\max_i |w_i|}{127}, \qquad q_i = \operatorname{round}\left(\frac{w_i}{s_b}\right)$$

Round to nearest, ties to even, which is the only rounding
[002](002-compute-model.md) admits.

**A block of all zeros gives $s_b = 0$**, and dividing by it would produce NaN
for every weight in the block. The scale is then set to zero and every quant to
zero, which reconstructs exactly: $0 \cdot 0 = 0$. This is the one special case,
and it is a real one — a pruned or padded matrix has such blocks.

**The clamp is not decoration either.** `round(w/s)` can reach 127 exactly at
the maximum, and floating-point rounding can push it to 128 for a weight one ulp
below. Without the clamp that wraps to −128 in an int8, turning the largest
weight in a block into the most negative one.

## 3. The error bound

This is the part [008](008-numerics.md) requires and the part a measurement
cannot supply. Quantization error is **derived**, not observed.

**Per weight**, rounding to nearest gives at most half a step:

$$|w_i - q_i s_b| \le \frac{s_b}{2} = \frac{\max_j |w_j|}{254}$$

**Per dot product** of length $K$ against activations $x$, the quantization term
is the sum of those, weighted by what multiplies them:

$$\left| \sum_i q_i s_{b(i)} x_i - \sum_i w_i x_i \right| \le \frac{1}{254} \sum_i |x_i| \cdot \max_{j \in b(i)} |w_j|$$

That is an *absolute* bound over the actual inputs, in the same shape as
[008](008-numerics.md) §8's forward propagation, and it is computable by a test
from the inputs it used.

**The accumulation term is separate and already specified.** The products are
summed in f32, so §7's reduction bound applies to the sum unchanged. The total
budget is the two added:

$$\text{budget} = \underbrace{\frac{1}{254} \sum_i |x_i| \max_{j \in b(i)} |w_j|}_{\text{quantization}} + \underbrace{\gamma(\text{depth}) \sum_i |q_i s_{b(i)} x_i|}_{\text{§7 accumulation}}$$

Composing them by addition is conservative and deliberately so: the two are not
independent, and a bound that assumed they were would be tighter and wrong.

**No ULP ceiling is stated for a quantized product**, and that is not an
omission. A ULP count measures distance from the correctly rounded result of the
*same* computation; a quantized dot product computes a different function, and
the question worth asking is how far its answer is from the unquantized one.
That is what the bound above states.

## 4. The kernels

Two, matching [009](009-sequencing.md)'s "quantized Rows/GEMM":

| Kernel | Shape |
| --- | --- |
| `QuantMatMul` | `a` f16 activations `[M,K]`, `bq` i8 quants `[K,N]`, `bs` f16 scales `[K*N/32]`, out f32 `[M,N]` |
| `QuantRows` | `table` i8 `[vocab,width]`, `scales` f16, `ids` u32, out f32 |

**The scale layout follows the quant layout, not the logical matrix.** For a
`[K,N]` weight matrix stored row-major, block $b$ covers 32 consecutive elements
of the flattened array, which is 32 consecutive *columns* of one row. A caller
quantizing along a different axis is quantizing a different matrix, and this
spec does not silently transpose for them.

## 5. Done

- quantize/dequantize round-trips within the per-weight bound, including the
  all-zero block and the clamp at ±127;
- `QuantMatMul` matches an f64 reference over the *dequantized* weights within
  §3's composed budget, at shapes with tails on every axis;
- `QuantRows` gathers what the table holds;
- both kernels agree between the CPU backend and Metal; and
- the tensor layer exposes them, and a plan mixing quantized and unquantized
  matrices compiles and runs.

## 6. Not in this spec

Named so the M8 item is not read as finished: sub-byte formats (int4, symmetric
and asymmetric), a quantized KV cache, activation quantization, and
`CapI8DotProduct` — the capability exists in [002](002-compute-model.md)'s table
and nothing here requires it, because the products are widened to f32 before
accumulating.
