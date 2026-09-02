---
title: "Per-row sampling: a batch where every row carries its own policy"
status: implemented
layer: tensor
depends_on:
  - 028-sampling.md
  - 039-sampling-policy.md
  - 063-uniform-loads.md
---

# Per-row sampling

**One thing:** sample a batch of logits where each row carries its own
temperature, truncation, bias and penalties, in one dispatch chain.

A child of [028](028-sampling.md) and [039](039-sampling-policy.md). Those own
the shared policy -- one `SamplingOptions` per plan -- and this owns the row
axis on every parameter of it.

## 1. The problem

`Sample` shares one policy across the batch because its parameters are uniform
fields: a temperature, a k, a p, three penalty strengths, and a greedy policy
is a different graph from a stochastic one. A server admits requests with their
own policies into one batch, so the consumer (tgo, its spec 020 §6) either
grouped requests by policy at admission, which makes admission depend on a
request's temperature, or read the logits back and sampled on the host. Its
named asks were per-row k and p, a row axis on the three penalty kernels, a
logit bias, and a greedy row beside a stochastic one.

What stood in the way of per-row k and p specifically was
[002](002-compute-model.md) §3.3: the extraction loops hold barriers, so their
trip counts must be workgroup-uniform, and a count loaded from a table was not
until [063](063-uniform-loads.md) let the table be declared uniform.

## 2. The operator

```go
type RowSampling struct {
    Factor    *Tensor       // [rows] f32 inverse temperature; 0 marks a greedy row
    TopK      *Tensor       // [rows] u32; 0 or >= vocab keeps every entry
    TopP      *Tensor       // [rows] f32; outside (0,1) keeps every entry
    Bias      *RowBias      // IDs, Values [rows, slots]; an id >= vocab is empty
    Penalties *RowPenalties // History [rows, cap], Counts [rows, vocab], Filled and three strengths [rows]
}

func SampleRows(b *Builder, logits, draws *Tensor, o RowSampling, prefix string) *Tensor
```

The chain is `Sample`'s: bias, penalties, scale, softmax, top-k, top-p, draw.
Each stage reads its parameter per row from a binding rather than from a
uniform, so the plan is one plan for any mix of policies, and the plan cache
hits on every step of a batch whose policies change at admission. A greedy row
is a factor of zero: `ScaleRows` scales it by one, so its order survives the
softmax, and `SampleRows` takes its argmax rather than its draw; the draw is
ignored, not refused, since a batch mixes the two.

### 2.1 The kernels

| kernel | per-row parameter | why it is its own kernel |
| --- | --- | --- |
| `ScaleRows` | factor | a zero scales by one |
| `TopKMaskRows` | `ks`, declared uniform | the extraction rounds are bounded by the load |
| `TopPMaskRows` | `ps`, declared uniform | the nucleus target is the load; the rounds stay a literal and an inactive row never advances its frontier |
| `SampleRows` | factor | argmax where zero, categorical otherwise |
| `LogitBias` | ids, values | one invocation per logit scanning the row's slots, which are few |
| `PenaltyClearRows`, `PenaltyCountRows`, `PenaltyApplyRows` | filled, three strengths | `PenaltyCount`'s and `PenaltyApply`'s bodies with a row index |

The shared-policy kernels stay; `Sample` is unchanged.

## 3. Testing

- Each row of a batch of three, under its own factor, k and p, chooses the
  token `Sample` chooses for that row alone under a `SamplingOptions` holding
  the same values and the same draw; the greedy row chooses what `Argmax`
  does.
- A bias on one row moves that row's argmax and no other's.
- The per-row penalties reproduce the host's reading of the shared penalty on
  each row.
- Every kernel has a differential case comparing its two lowerings, with two
  rows whose parameters differ so a lowering reading one row's parameter for
  both would show; the parity matrix has a case for the operator.

## 4. Outcome — 2026-09-02

Built as specified in one pass, the day [063](063-uniform-loads.md) made the
per-row truncation possible. Mixed greedy and stochastic rows, which 039 refused
in one batch, are the factor-of-zero rule.
