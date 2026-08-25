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
probs := tensor.Softmax(b, tensor.Scale(b, logits, "invtemp"),
	tensor.SoftmaxOptions{Axis: -1})
kept := tensor.TopPMask(b, tensor.TopKMask(b, probs, 8), 0.9)
tensor.Output(b, "next", tensor.SampleCategorical(b, kept, draw))
```

Temperature, softmax, top-k, top-p and the draw are five operators in one plan.
**So a step reads back one token, not a vocabulary of logits.** At a real
vocabulary that is 128k floats you do not move per token per sequence — the
difference between a loop bound by the model and one bound by the bus.

Three things to hold:

- **`SoftmaxOptions{Axis: -1}`.** Logits are `[rows, vocab]` and each row is
  normalized on its own. The zero value means axis 0, which is refused rather
  than computed: *"axis 0 is not the last"*.
- **The masks feed the draw directly.** A kept entry carries its value and a
  dropped one carries zero, so no renormalizing pass belongs between them. A
  second `Softmax` there would undo the mask, because `exp(0)` is 1.
- **The draw is yours.** One `float32` per row per step, from your host RNG. The
  policy runs on the device; the randomness stays where you can seed it.

## The loop

The caches are bound once. Everything else is a small write.

```go
queue := dev.Queue()
write := func(buf *accel.Buffer, data any) {
	if err := queue.WriteBuffer(buf, 0, data); err != nil {
		log.Fatal(err)
	}
}

rng := rand.New(rand.NewPCG(1, 2))
token := uint32(7)
out := []uint32{token}
for pos := range 12 {
	write(tokBuf, []uint32{token})
	write(qposBuf, []uint32{uint32(pos), uint32(pos)})
	write(kposBuf, []uint32{uint32(pos)})
	write(slotBuf, []uint32{uint32(pos)})
	write(lenBuf, []uint32{uint32(pos + 1)})
	write(drawBuf, []float32{rng.Float32()})

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
fmt.Println(out) // [7 19 18 18 19 19 16 19 0 19 0 3 18]
```

Twelve tokens from one plan. Nothing is rebuilt and the cache is never
re-uploaded. The weights here are untrained, so those numbers are not words.

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
- Drop `TopKMask` and `TopPMask`. The same draws now reach the tail.
- Write `SoftmaxOptions{}` and read the refusal.

---

Next: [quantized weights](10-quantized-weights.md) — storage width is not
compute width.
