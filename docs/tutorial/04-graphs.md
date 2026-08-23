# 4. Graphs

**One thing:** record work once and replay it, instead of re-issuing it every
step.

`Queue.Run` from [tutorial 1](01-hello-gpu.md) builds a graph, submits it, and
throws it away. In a loop that is most of your cost: validation, memory
planning, and barrier placement, every iteration.

So build it once and keep it.

```go
r := dev.NewRecorder()
r.Dispatch(pipe, []accel.Binding{
	{Index: 0, Buffer: state}, {Index: 1, Buffer: delta}, {Index: 2, Buffer: next},
}, accel.WorkgroupCount{X: n / 64})
r.CopyBuffer(state, next) // next becomes the state for the following step

g, err := r.Build() // validates, plans memory, computes barriers
if err != nil {
	log.Fatal(err)
}
defer g.Close()

for range steps {
	if err := dev.Queue().Submit(g).Wait(); err != nil {
		log.Fatal(err)
	}
}
```

`Build` does the analysis. Every `Submit` after it is a replay.

## You do not write barriers

The dispatch writes `next`; the copy reads it. That is a hazard, and the builder
finds it by comparing what each node **declared it touches** — which came from
the kernel body, not from you.

It compares byte ranges rather than whole resources, so two nodes writing
disjoint halves of one buffer are not serialised.

You can ask what it decided:

```go
fmt.Println(len(g.Edges()), g.Barriers(), g.Memory().TransientBytes)
```

That matters when a graph is slower than you expect: the answer is in the plan,
not in a profiler.

## Intermediates are the builder's

A value that only exists between two nodes should be a **transient**:

```go
mid := r.Transient(accel.BufferDescriptor{
	DType: accel.F32, Count: n,
	Usage: accel.UsageStorage | accel.UsageCopySrc | accel.UsageCopyDst,
	Label: "mid",
})
```

You do not allocate it, and you do not close it — closing the graph does. Two
transients whose uses cannot overlap share the same bytes, so a long chain costs
far less than the sum of its parts.

## What it costs you

**Errors move.** Under an immediate API a bad call fails at the call. Here it
fails at `Build`, possibly far from where you wrote it. Every build error names
the node, the binding slot and the numbers involved.

**A graph is immutable once built.** Four things vary between submissions:
buffer contents, bound resources through [slots](#rebinding-inputs), dispatch
and draw counts, and by-value parameters ([tutorial 6](06-uniforms.md)). Shape
changes need another graph.

## Rebinding inputs

To point the same graph at different buffers, record a **slot** instead of a
resource and bind it before submitting:

```go
in := r.Slot(accel.SlotDescriptor{
	Name: "in", Kind: accel.BindingStorageBuffer,
	DType: accel.F32, Access: accel.AccessRead, MinCount: n,
})
// ...
g.Rebind([]accel.Binding{{Slot: in, Buffer: thisFrame}})
```

## Try it

- Print `g.Barriers()` and then remove the `CopyBuffer`. The count changes,
  because the hazard was real.
- Submit the same graph twice without waiting. The second is refused: one
  submission in flight per graph, because its transients are reused.

---

Next: [cooperation](05-cooperation.md) — when invocations need to talk.
