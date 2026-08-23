# 3. Memory

**One thing:** where device memory comes from, and the three properties of it
that will surprise you.

In [tutorial 1](01-hello-gpu.md) you called `dev.NewBuffer` and never thought
about it. That is fine for one buffer. A model has thousands, and drivers cap
how many allocations you may hold — so real allocation goes through a **pool**.

```go
weights, err := dev.NewPool(accel.MemoryDevice, 1<<30) // one 1 GiB allocation
if err != nil {
	log.Fatal(err)
}
defer weights.Close()

w, err := weights.Alloc(accel.BufferDescriptor{
	DType: accel.F16,
	Count: 4096 * 4096,
	Usage: accel.UsageStorage | accel.UsageCopyDst,
	Label: "blk.0.attn_q.weight", // labels appear in every error about it
})
defer w.Close()
```

A pool is **one** device allocation that you suballocate. `dev.NewBuffer` is a
convenience that uses an implicit pool behind your back.

## Views are ranges, not resources

```go
head, err := w.View(0, 128) // first 128 elements, no copy
```

A `BufferView` is a range measured in **elements**, not bytes. It holds no
reference: the view is re-validated against the live buffer every time it is
used, so a view that outlives its buffer is *reported*, not undefined.

## Three things that will surprise you

**A pool never grows.** No backend can resize an allocation in place, and
growing by reallocating would invalidate every address already handed out — a
device address is baked into recorded commands by the time anything runs. So
size a pool for your peak. The one thing that does grow is the implicit pool
behind `NewBuffer`, and it grows the only way a device allocation can: by adding
another one.

**Nothing compacts.** Fragmentation is permanent for a pool's life. That is what
a non-compacting allocator *is*, not a defect to be fixed. `PoolStats` reports
`LargestFree` beside `Free` so you can see an allocation failure coming:

```go
s := weights.Stats()
if s.LargestFree < need {
	// fragmented, even though Free may be plenty
}
```

The mitigation is separating pools by lifetime — weights in one, per-frame
scratch in another — which is why a pool takes a policy and a label instead of
accel guessing.

**Closing is ordered, not recursive.** A pool with live buffers refuses to
close, and so does a device with live pools. Both report and free nothing, and
the children keep working:

```go
err := weights.Close() // returns a LifetimeError naming the live children
```

Close children first. Closing them out from under you would turn your bug into a
silent success and make the next use undefined instead of reported.

## Choosing a memory kind

| Kind | For |
| --- | --- |
| `MemoryDevice` | weights, activations — fast for the GPU, not host-visible |
| `MemoryUpload` | staging data on its way to the device |
| `MemoryReadback` | results you will read on the host |

`Queue.WriteBuffer` and `ReadBuffer` stage through the right kind for you. You
choose explicitly when you want to skip a copy.

## Try it

- Allocate from a pool until `Alloc` fails, then print `Stats()`. Compare `Free`
  with `LargestFree`.
- Close a pool while a buffer is live and read the error.

---

Next: [graphs](04-graphs.md) — doing all of this once instead of every frame.
