# 11. Batching sequences

**One thing:** a value every row of a dispatch shares is a scalar; a value that
differs per row is a tensor. Once you hold that, batching is not a second API.

[Tutorial 9](09-a-decode-step.md) stepped one sequence. Three sequences arrive
at different times, hold different amounts of context, and want to step
together. Nothing about them is padded to the longest.

## Every per-sequence value is already a tensor

```go
tok := u32in("tok", tensor.Shape{batch})            // one token each
qpos := u32in("qpos", tensor.Shape{batch * qHeads}) // one position per row of q
kpos := u32in("kpos", tensor.Shape{batch})
slots := u32in("slots", tensor.Shape{batch})        // where this token lands
lens := u32in("len", tensor.Shape{batch})           // how much context each holds
pages := u32in("pages", tensor.Shape{batch, maxPages})
draws := tensor.Input(b, tensor.ValueDesc{
	Name: "draws", DType: accel.F32, Shape: tensor.Shape{batch},
})
```

Compare that with tutorial 9's list. The shapes changed from `{1}` to `{batch}`
and nothing else did. `Lengths` was always per sequence, `RoPE` always took
positions per row, and `SampleCategorical` always drew per row. A batch of one
was the same path all along.

## The cache is a pool of blocks

```go
kc := tensor.NewState(b, tensor.StateDesc{
	Name: "kpool", DType: accel.F32,
	Shape: tensor.Shape{poolBlocks * block, kvHeads, headDim},
})
```

One state for every sequence, cut into fixed blocks. `pages[s][i]` is the
physical block holding sequence `s`'s `i`-th logical block, so a sequence's
context is scattered through the pool and grows a block at a time.

This is why the batch needs a page table rather than a contiguous cache. Three
sequences of 10, 3 and 7 positions in one contiguous state would each have to be
sized for the longest the server will ever accept. On a 36-layer model that is
gigabytes per sequence, reserved whether or not anything is that long.

A paged state is not a second kind of state. It is the same `State`; what
differs is the binding.

## The query carries a batch axis

```go
q = tensor.Reshape(b, q, tensor.Shape{batch, 1, qHeads, headDim})
att := tensor.Attention(b, q,
	tensor.ScatterRows(b, kc, k, slots),
	tensor.ScatterRows(b, vc, v, slots),
	tensor.AttentionOptions{
		Lengths: lens, Pages: pages, Block: block, ScaleName: "scale",
	})
```

`q`'s rank says which computation this is:

| `q` | meaning |
| --- | --- |
| `[qHeads, headDim]` | one sequence, one token — a decode |
| `[qSeq, qHeads, headDim]` | one sequence, many tokens — a prefill |
| `[batch, qSeq, qHeads, headDim]` | several sequences stepping together |

The batch axis is rank 4 and not rank 3 because rank 3 already means a prefill.
`Selections()` names `AttentionDecodeBatched`, one workgroup per (sequence,
head).

Four rules this shape carries, each reported by name if you miss it:

- `Pages` is required. No contiguous batched kernel is registered, for the
  reason above.
- `Pages` is exactly `[batch, maxPages]`, and `Block` is required with it.
- `Lengths` holds exactly `batch` entries.
- The cache dtype is f32. The f16 caches of tutorial 9 are for a single
  sequence; the batched kernel does not read them.

## Stepping

Each sequence's next slot is its own arithmetic — its length picks the logical
block, its page table row picks the physical one:

```go
for s := range batch {
	pos := lengths[s]
	kp[s] = pos
	for h := range qHeads {
		qp[s*qHeads+h] = pos
	}
	sl[s] = table[s*maxPages+int(pos)/block]*block + pos%block
	dr[s] = rng.Float32()
	lengths[s] = pos + 1
}
```

That is the whole scheduler for a fixed batch: six per-sequence values written
into six small buffers. Then one submission moves every sequence forward one
token.

```
sequence 0 (now 16 long): [7 27 25 25 4 26 27]
sequence 1 (now 9 long): [21 19 21 1 4 1 0]
sequence 2 (now 13 long): [3 30 22 30 24 12 8]
```

Three lengths, three page tables, one plan, six submissions. The sequences never
see each other's blocks, and none of them paid for the longest.

Those six ran one after another. A plan binds its buffers when you submit it
and runs when the queue reaches it, so a second submission before the first
completes would rebind the first one's slots, and by default the plan refuses
it. A server that keeps several requests in flight compiles with
`CompileOptions{MaxInFlight: n}`: the plan then holds up to `n` graph
instances, each with its own transient memory, and runs that many submissions
at once. The plan cache keys on the option, so a plan compiled for one and one
for four are different plans.

## Prefilling several sequences at once

The rectangular `[batch, 1, heads, dim]` query above is one token per sequence.
A batched **prefill** is the ragged form of the same operator: lay every
sequence's prompt tokens end to end as `[tokens, heads, dim]`, and pass how
many belong to each sequence as `QueryExtents`, next to the `Lengths` and
`Pages` the step already carries.

```go
// Three prompts of 5, 2 and 7 tokens, concatenated: q is [14, heads, dim].
extents := tensor.Input(b, tensor.ValueDesc{Name: "extents", DType: accel.U32, Shape: tensor.Shape{3}})
out := tensor.Attention(b, q, kcache, vcache, tensor.AttentionOptions{
	Lengths: lengths, Pages: pages, Block: block,
	QueryExtents: extents, ScaleName: "scale",
})
```

Each sequence attends causally within its own extent and against its own
cache, and a test holds the batched result to the three prompts prefilled one
at a time. What the rectangular form does not take is `qSeq` above one, and
its refusal says to use this form.

## Try it

- Reverse one row of the page table. The answer stays finite and becomes wrong,
  which is why the table is checked before the kernel is chosen.
- Run the same three sequences one at a time, with a batch of one, and compare
  the tokens.
- Give `Lengths` `batch+1` entries and read the refusal.

---

That is the tour. The [package documentation](https://pkg.go.dev/golang.design/x/accel)
is the reference; [`specs/`](../../specs/) is why things are shaped as they are.
