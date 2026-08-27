---
title: "Linear attention: a matrix state, and a scan over a segmented extent"
status: in progress
layer: tensor
depends_on:
  - 007-tensor-layer.md
  - 008-numerics.md
  - 010-kernel-corpus.md
  - 026-tensor-decode.md
  - 043-per-row-values.md
  - 046-segmented-extents.md
---

# Linear attention

**One thing:** a layer whose state is a matrix per sequence rather than a cache
per position.

[#17](https://github.com/golang-design/accel/issues/17). Three of every four
layers in the hybrid models the open-weights frontier has moved to are not
softmax attention. Without this, that whole class of model is not expressible —
not slow, **inexpressible** — which is the only open report of that kind.

## 1. The recurrence, and where its dimensions come from

A gated delta layer carries $S$ per head and steps it per token:

$$
S_t = S_{t-1}\big(\alpha_t I - \beta_t k_t k_t^{\top}\big) + \beta_t v_t k_t^{\top},
\qquad o_t = S_t q_t
$$

The shapes follow from that expression rather than from a convention, and
getting them backwards is the kind of mistake that still runs:

| | is | because |
| --- | --- | --- |
| $S$ | $[\text{valueDim}, \text{keyDim}]$ | $S k k^{\top}$ needs $k$ on the right, $v k^{\top}$ needs $v$ on the left |
| $k$, $q$ | $[\text{keyDim}]$ | both contract against $S$'s second axis |
| $v$, $o$ | $[\text{valueDim}]$ | $o = Sq$ leaves $S$'s first axis |

**Expanded, it is three passes and not a matrix product.** With
$u = S_{t-1} k_t$, a valueDim vector:

$$
u_a = \sum_b S_{a b} k_b,
\qquad
S_{a b} \leftarrow \alpha S_{a b} + \beta\, k_b\,(v_a - u_a),
\qquad
o_a = \sum_b S_{a b} q_b
$$

```
per token, per head:      u = S k          read  K*V
                          S = aS + b k(v-u)^T   read+write K*V
                          o = S q          read  K*V
```

Three passes over $K \times V$. At $K = V = 128$ that is about 49k multiply-adds
per token per head — comparable to attending over 49k cached positions, except
that it **does not grow with context**, which is the entire appeal.

## 2. Why it is a scan over a segmented extent, and not a new concept

The recurrence is sequential in $t$: token $t$ needs $S_{t-1}$. A prefill of $T$
tokens is therefore a **scan**, not a batch of independent rows — which is the
one structural difference from softmax attention, where every query is
independent.

But the shape of the input is the one
[046](046-segmented-extents.md) just built. A step contributes $n_r$ tokens for
sequence $r$, flat, with a count per row:

```
offsets = [0, 3, 4, 6]        sequences contribute 3, 1, 2 tokens
q,k,v   = [6, heads, dim]     flat, one row per token
S       = [slots, heads, V, K]   one matrix per sequence per head
```

**So a decode step and a prefill are the same kernel.** A decode is $n_r = 1$; a
prefill is $n_r = T$; a mixed step is both at once. That falls out of the extent
rather than being designed for, and it is the reason this spec is short: the
hard part — expressing a per-row extent — is already built and tested.

### 2.1 One workgroup per (sequence, head), walking its own tokens

Softmax attention parallelises over tokens because they are independent. Here
they are not, so the parallelism is over **sequences and heads**, and each
workgroup walks its own tokens in order carrying $S$.

A workgroup has one lane per valueDim row of $S$. Lane $a$ owns row $a$ — $K$
floats — and loops over $b$. Nothing is shared between lanes except the two
reductions the recurrence needs, and those are over $b$ within a row, which one
lane does alone.

$S$ lives in the state buffer rather than in registers or shared memory: at
$K = V = 128$ one head's matrix is 64 KiB, which is past what a workgroup can
hold. So each token reads and writes it, which is what makes the arithmetic
above three passes rather than one.

## 3. The state is a State, and 043's correction is why that is enough

[043](043-per-row-values.md) §9 first recorded that `State` conflates two things
and that a recurrent state needed a type of its own **before** any kernel here
could be built. That was withdrawn on 2026-08-25 and the withdrawal has a test:
a recurrent state is a `State` whose leading axis is the **sequence slot**, and
`ScatterRows` indexes it as such.

So this spec needs no new type. $S$ is
`[slots, heads, valueDim, keyDim]` f32, caller-owned, carried across submissions
exactly as a KV cache is — and a hybrid model holding both kinds in one graph is
two `State`s of different shapes.

**What it does need and does not have is snapshot and restore**, which
[043](043-per-row-values.md) §9 records as a general `State` operation rather
than this layer's: prefix caching for a recurrent layer is copying a slot, where
for a KV cache it is a page table. It is not built here and nothing in §5
depends on it.

## 4. What is deliberately not built

**The chunked parallel form.** The recurrence can be reassociated so that a
block of tokens is processed with matrix products and only the block boundaries
are sequential. That is what makes linear attention *fast* rather than merely
correct, and it is a different kernel with its own numerics. The sequential form
here is the one that makes the layer **expressible**, which is what the report
asks for, and it is the reference the chunked form would be checked against.

**The depthwise causal convolution.** The consumer verified element by element
that it composes from `Slice`, `Contiguous`, `Broadcast` and a multiply-add per
tap, with a left pad making causality structural. It costs $K$ dispatches and
$K-1$ packing copies per layer where a kernel would take one, so it is *one less
kernel to unblock* rather than one less to want — recorded in
[010](010-kernel-corpus.md) and folded into the chunked scan if that is built.

**Fusing the gates.** $\alpha$ and $\beta$ arrive as per-token tensors, which is
[043](043-per-row-values.md) §2 applied without argument: they differ per row.

## 5. Done

Each assertion names the mutation it catches.

- **A single token matches the recurrence written out in f64.** The accepting
  half: three passes in the right order against one expression evaluated
  directly.
- **Two tokens in one dispatch equal two dispatches of one token each**, with
  the state carried between them. This is what says the scan is a scan — a
  kernel that recomputed from the initial state passes every single-token test.
- **$\alpha = 1, \beta = 0$ leaves the state exactly unchanged** and makes $o$
  the same for every token. The identity case, and it catches a sign error in
  the update that a random gate hides.
- **A sequence's tokens do not disturb another sequence's state.** Two sequences
  in one step, and each slot equals what that sequence alone produced — the same
  oracle [046](046-segmented-extents.md) §5 uses, for the same reason.
- **The state's shape is `[slots, heads, V, K]` and a step writes only its own
  slot**, asserted by leaving another slot filled with a sentinel.
- **CPU and Metal agree** within [008](008-numerics.md) §6's bound for a sum of
  products, which is what all three passes are.

## 5.1 Built — 2026-08-26

§1's recurrence, §2's scan, and §3's use of an ordinary `State`. Every assertion
in §5 is built except the CPU/Metal one, which is the corpus differential and
runs as a case there.

**What the extent bought.** This spec is short and its kernel is 80 lines
because [046](046-segmented-extents.md) had already expressed "how many tokens
does this sequence contribute". A decode step and a prefill are the same code
path — the loop runs `offsets[seq]` to `offsets[seq+1]` — and nothing in the
kernel or the operator tests which it is. That was the argument for building the
primitive first and it held.

**Two mutations that proved nothing, and what they cost.** Twice in one day a
mutation failed to reach the code it was meant to break and the passing test was
read as evidence. Once a `perl` substitution did not match; once a mutation used
a form outside the kernel subset, `go generate` failed, and its error was hidden
by a redirect. **A mutation that does not reach the code is indistinguishable
from a test that does not check it**, so a reinstatement now asserts that it
applied before the test is run — and where the property is about plumbing rather
than arithmetic, the test proves its own discriminating power instead. The
persistence test does that: it zeroes the state between two identical steps and
requires them to match, which is what says an equal result is what a reset
produces.

## 6. Open

- **The chunked form needs a second representation, not just a reassociation —
  derived 2026-08-26.** Splitting each output into a prior-state part and a
  within-chunk part gets one of the two into a shape a device likes:

  $$o_t = \underbrace{d_t\,(S_0 q_t)}_{\text{a GEMM over the chunk}} \;+\; \underbrace{\sum_{j\le t} \Big(\textstyle\prod_{i=j+1}^{t}\alpha_i\Big)\beta_j (v_j - u_j)(k_j \cdot q_t)}_{\text{still sequential in } u_j}$$

  where $d_t = \prod_{j \le t}\alpha_j$. The first term is $[C, K] \times [K, V]$
  and parallelises over the whole chunk. **The second still needs
  $u_j = S_{j-1}k_j$, which is the recurrence again**, so a chunked kernel is not
  this one with a bigger step: it needs the WY / UT-transform representation that
  makes the $u_j$ computable together, and that is where the summation order
  genuinely changes and the bound between the two forms has to be derived.

  So the work is a derivation before it is a kernel, and this kernel is what the
  derivation would be checked against. **The derivation is done — §6.1.**
- **Whether $\alpha$ and $\beta$ should be one tensor.** They are two per-token
  scalars and every model that has one has the other. Two bindings is two things
  to bind wrongly; one interleaved tensor is a layout a caller has to know.
  Left as two until a caller says.

## 6.1 The UT transform, derived — 2026-08-27

§6 said the chunked form needs a second representation because
$u_j = S_{j-1}k_j$ is the recurrence again. It does, and this is it: **the
sequential dependency becomes a triangular solve whose coefficients are one
Gram matrix.**

### The substitution

Write the recurrence's write vector as $w_t = \beta_t(v_t - u_t)$, so that
$S_t = \alpha_t S_{t-1} + w_t k_t^\top$. Let $d_t = \prod_{i\le t}\alpha_i$ be
the decay accumulated through token $t$ of the chunk, with $d_{-1} = 1$.
Unrolling from the chunk's entry state $S_0$:

$$S_t = d_t S_0 + \sum_{j\le t} \frac{d_t}{d_j}\, w_j k_j^\top$$

Substituting that into $u_t = S_{t-1}k_t$ and then into $w_t$'s own definition
gives an equation in the $w$ alone — the $u_j$ never appear:

$$w_t \;+\; \beta_t\sum_{j<t}\frac{d_{t-1}}{d_j}(k_j\cdot k_t)\,w_j
\;=\; \beta_t\big(v_t - d_{t-1}S_0k_t\big)$$

which is $(I + A)W = B$ with $A$ **strictly lower triangular**:

$$A_{tj} = \beta_t\,\frac{d_{t-1}}{d_j}\,(k_j\cdot k_t)\ \ (j<t),
\qquad B_t = \beta_t\big(v_t - d_{t-1}S_0k_t\big)$$

$A$ is one $C\times C$ Gram matrix of the chunk's keys, scaled per entry. That
is the whole content of the transform: what was $C$ dependent steps over a
$K\times V$ state becomes one forward substitution over $C$ rows.

```
sequential                    chunked
  t=0 ─ S ─ t=1 ─ S ─ t=2       KKᵀ  ──▶  A   (one GEMM)
      each step touches                    │
      the whole K×V state       B ──▶ (I+A)W = B   (forward substitution,
                                            │       C rows, no state)
                                            ▼
                                S₀ read once, S_C written once
```

### What comes out

With $W$ solved, the outputs and the exit state are both plain contractions:

$$o_t = d_t\,S_0q_t \;+\; \sum_{j\le t}\frac{d_t}{d_j}(k_j\cdot q_t)\,w_j
\qquad
S_C = d_{C-1}S_0 \;+\; \sum_j \frac{d_{C-1}}{d_j}\,w_jk_j^\top$$

### Why it is worth a kernel

Not arithmetic — the operation counts are close. **Global traffic.** The
sequential kernel reads and writes the $K\times V$ state three times per token
(§1's three passes). The chunked form reads $S_0$ once and writes $S_C$ once per
chunk, so the state crosses memory $2$ times per $C$ tokens instead of $3C$.
That is the reason the form exists, and it is a memory claim rather than a FLOP
one.

### The ratio is the numerical hazard

$d_{t-1}/d_j$ is $\prod_{i=j+1}^{t-1}\alpha_i$, computed as a quotient of two
cumulative products. Both shrink geometrically, so at $\alpha = 0.5$ and
$C = 64$ the denominator is near $10^{-20}$ and the quotient is computed from
two numbers that have each lost most of their significance. **A chunk size is
therefore part of the representation, not a tuning knob**, and a kernel that
takes $C$ from the caller has to refuse the sizes where the quotient is not
trustworthy — or form the product directly, which costs $O(C)$ per entry and
removes the hazard.

### Checked, and against what

`TestTheChunkedFormEqualsTheSequentialKernel` runs the transform against
`LinearAttention` itself at chunk sizes 1, 2, 3, 4, 5, 8 and 16 over sequences
of 7 and 5 tokens — sizes that do not divide the lengths, because a chunk that
always divides evenly hides the index arithmetic where an off-by-one lives.
Chunk 1 is the degenerate case and must reproduce the recurrence exactly.

Two mutations confirm it discriminates: dropping $d_{t-1}$ from $A_{tj}$, and
dropping it from $B_t$. Each fails five or six of the seven sizes — and chunk 1
correctly survives the first, because there is no $j<t$ term for it to get
wrong.

### What is still not built: the kernel

The derivation is no longer the blocker; the kernel is ordinary work with an
oracle. What it needs that this reference does not is a residency plan: lane $a$
wants $S_0$'s row $a$ held across the chunk, which is $K$ floats per lane and
spills at the widths a real model uses, so the kernel tiles over $K$ as well and
that tiling is what has not been written. [010](010-kernel-corpus.md) carries
`linear_attention_chunked` as not registered.
