# 9. A decode step

**One thing:** state that stays on the device between submissions, and a
sampling policy that runs there too.

[Tutorial 7](07-tensors.md) compiled a plan and submitted it once. A decode step
submits one plan once per token. What makes it a *step* and not a fresh
computation is a KV cache the plan writes and reads without you uploading it
again.

## The cache is a State, not an Input

```go
kc := tensor.NewState(b, tensor.StateDesc{
	Name: "kcache", DType: accel.F32, Shape: tensor.Shape{capacity, kvHeads, headDim},
})
```

An `Input` is a value you supply each step. A `State` is storage you own and the
plan mutates. `Shape` is the whole extent, sequence capacity first, so you size
the cache once for the longest context you will serve.

Writing it returns the **next version**:

```go
att := tensor.Attention(b, q,
	tensor.ScatterRows(b, kc, k, slot),
	tensor.ScatterRows(b, vc, v, slot),
	tensor.AttentionOptions{Lengths: lens, ScaleName: "scale"})
```

`ScatterRows` writes this token's key and value at a runtime slot and returns a
new `*State`. Passing that to `Attention` says *attend over the cache including
what I just wrote*. Passing `kc` would name the version before the write, and
the token would be invisible to itself. The two nodes overlap in bytes, so the
graph orders them and you write no barrier.

`Lengths` says how much of the cache is real: a `u32` tensor, one entry per
sequence. One sequence binds one element — the same path a batch takes, not a
special case. [Tutorial 11](11-batching-sequences.md) is the batch.

## Positions are a tensor

```go
q := tensor.Reshape(b, tensor.MatMul(b, x, wq), tensor.Shape{qHeads, headDim})
q = tensor.RoPE(b, q, headDim, "ropebase", qpos)
k := tensor.RoPE(b, tensor.MatMul(b, x, wk), headDim, "ropebase", kpos)
```

`RoPE` takes one position per row of what it rotates, as device data. The
frequency base is a named scalar, because it belongs to the model and every row
shares it. A position does not: it belongs to the sequence.

Here `q` has one row per query head, so `qpos` holds this step's position twice.
That looks like ceremony at a batch of one. It is not — in tutorial 11 those
rows belong to different sequences and hold different numbers.

`RoPE` rotates its buffer in place, so the plan copies first, and
`plan.Selections()` reports the copy rather than hiding it.

## A whole sampling policy composes on the device

```go
logits := tensor.MatMul(b, tensor.Reshape(b, att, tensor.Shape{1, dim}), wout)

// The penalties need a ring of the tokens generated so far and a scratch
// buffer of counts. Both are storage you own, like the caches above.
history := tensor.NewState(b, tensor.StateDesc{
	Name: "history", DType: accel.U32, Shape: tensor.Shape{historyCap},
})
counts := tensor.NewState(b, tensor.StateDesc{
	Name: "counts", DType: accel.U32, Shape: tensor.Shape{vocab},
})

policy := tensor.SamplingOptions{
	Temperature: 0.8,
	TopK:        40,
	TopP:        0.95,
	Frequency:   0.1, // penalise tokens this sequence already used
}
tensor.DeclareSamplingScalars(b, policy, "sample")
tensor.Output(b, "next",
	tensor.Sample(b, logits, draw, history, counts, policy, "sample"))
```

**So a step reads back one token, not a vocabulary of logits.** At a real
vocabulary that is 128k floats you do not move per token per sequence — the
difference between a loop bound by the model and one bound by the bus.

`Sample` records the whole policy as one subgraph:

```
logits ─▶ penalties ─▶ ×1/T ─▶ softmax ─▶ top-k ─▶ top-p ─▶ draw ─▶ token
           (optional)                     (optional)(optional)
       └─▶ argmax ─▶ token   when Temperature is 0
```

**"Off" means the node is absent.** `TopK: 0` and `TopP: 0` remove those
operators from the graph. `TopK` equal to your vocabulary does *not* turn
truncation off — the mask keeps a bounded number of entries, so you would
silently get the top 128.

**`Temperature: 0` is greedy, and it is a different graph rather than a small
number.** Asking for `1e-6` instead gives you argmax almost always and the
second-best token whenever the top two logits are close — reproducibly, so it
reads as a model quirk rather than a bug. `Validate` refuses anything under
`MinTemperature` and names `0` as the way to ask for greedy.

Three things to hold:

- **Penalties act before temperature.** Subtracting a penalty *after* dividing
  by `T` is subtracting `penalty × T` before it, so a penalty tuned at one
  temperature would change strength at another. Two knobs you turn
  independently must not multiply.
- **Nothing renormalizes between the masks and the draw.** A kept entry carries
  its value and a dropped one carries zero. A second `Softmax` there would undo
  the mask, because `exp(0)` is 1 for every entry you just dropped.
- **The penalties need two buffers you own**: a `[cap]u32` ring of the tokens
  generated so far, and a `[vocab]u32` scratch for the counts. Pass `nil` for
  both when no penalty is configured — passing storage nothing reads is refused
  rather than ignored.

## Draws that reproduce

The draw is one `float32` per row per step. Where it comes from decides whether
a sequence you generated yesterday generates again today.

```go
stream := tensor.Derive(seed, sequenceIndex)
// ...
write(drawBuf, []float32{stream.Draw(uint64(pos))})
```

A `Stream` is a seed and nothing else, and `Draw` is a pure function of it and
the token index — **the index you already hold for the KV cache**. Three things
follow, and each is a bug you do not have:

- **Copying a `Stream` copies a number.** Put a `*rand.Rand` in a config struct
  and Go copies the pointer on assignment, so two sequences share one generator
  and neither reproduces. There is nothing here to share, and nothing to race.
- **Resuming is free**, because there is no position to advance. Turning
  temperature off for one step does not shift every later token.
- **`Draw` never returns 1.0.** `float32(rng.Float64())` rounds up to exactly
  1.0 about once in 2^24. The sampler clamps that rather than failing, so the
  last token in your vocabulary quietly receives the extra mass and every test
  still passes.

Use `tensor.Derive(root, i)` to give each sequence of a batch its own stream
from one root seed.

## The loop

The caches are bound once. Everything else is a small write.

```go
queue := dev.Queue()
write := func(buf *accel.Buffer, data any) {
	if err := queue.WriteBuffer(buf, 0, data); err != nil {
		log.Fatal(err)
	}
}

stream := tensor.Stream{Seed: 1}
token := uint32(7)
out := []uint32{token}
ring := make([]uint32, historyCap)
for pos := range 12 {
	write(tokBuf, []uint32{token})
	write(qposBuf, []uint32{uint32(pos), uint32(pos)})
	write(kposBuf, []uint32{uint32(pos)})
	write(slotBuf, []uint32{uint32(pos)})
	write(lenBuf, []uint32{uint32(pos + 1)})
	write(drawBuf, []float32{stream.Draw(uint64(pos))})

	// The penalty window: this sequence's tokens so far, in a fixed-capacity
	// ring. n is how much of it is filled.
	ring[pos%historyCap] = token
	write(historyBuf, ring)
	n := uint32(min(pos+1, historyCap))
	scalars, err := policy.Scalars("sample", n, uint32(historyCap))
	if err != nil {
		log.Fatal(err)
	}
	bindings.Scalars = scalars

	if err := plan.Submit(queue, bindings).Wait(); err != nil {
		log.Fatal(err)
	}
	got := make([]uint32, 1)
	if err := queue.ReadBuffer(nextBuf, 0, got); err != nil {
		log.Fatal(err)
	}
	token = got[0]
	out = append(out, token)
}
fmt.Println(out)
```

Twelve tokens from one plan. Nothing is rebuilt and the cache is never
re-uploaded.

**The ring is bound at its full capacity, every step.** Binding only the filled
part would change the input shape on every token, which is a new plan on every
token: the plan cache keys on operand shapes, so it would grow without bound.
The symptom is decode getting slower as the sequence lengthens and device memory
climbing — which reads as a memory leak rather than as a sampler mistake. `n` is
what says how much of the ring is real, and `Scalars` refuses an `n` past the
capacity rather than clamping it.

**Rebinding scalars costs nothing.** `Submit` rewrites every uniform on every
submission, so a temperature or a penalty coefficient that changes per step is
free. What is *structural* — whether there are penalties at all, whether top-k
is in the graph, and the k and p themselves — is not, and changing one of those
is a different plan.

## Narrower storage costs two lines

A KV cache is often the largest live allocation in a serving process. Halve it
by declaring the state `accel.F16` and casting the rows on the way in:

```go
kc := tensor.NewState(b, tensor.StateDesc{
	Name: "kcache", DType: accel.F16, Shape: tensor.Shape{capacity, kvHeads, headDim},
})
tensor.ScatterRows(b, kc, tensor.Cast(b, k, accel.F16), slot)
```

`ScatterRows` refuses f32 rows into an f16 state and names `Cast` in the
refusal. The cast is a device pass, and `Selections()` then reports
`CastF32ToF16`, `ScatterRowsF16` and `AttentionDecodeF16`.
[Tutorial 10](10-quantized-weights.md) is the same idea applied to weights.

## Try it

- Pass `kc` to `Attention` instead of what `ScatterRows` returned. It compiles,
  and the newest token is missing from every score.
- Set `TopK` and `TopP` to 0. The same draws now reach the tail.
- Set `Temperature` to `1e-6` and read the refusal, then set it to `0`.
- Configure `Frequency` and pass `nil` for the history. The refusal names which
  buffer is missing and why the step would penalise nothing.
- Write `SoftmaxOptions{}` and read the refusal.

---

Next: [quantized weights](10-quantized-weights.md) — storage width is not
compute width.
