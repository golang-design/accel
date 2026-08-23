# 6. Values that are not buffers

**One thing:** passing a scalar to a kernel, and changing it between submissions
without re-recording.

`Brighten` from [tutorial 2](02-writing-a-kernel.md) takes an amount:

```go
type Adjust struct{ Amount float32 }

//accel:kernel workgroup=64
func Brighten(t accel.Thread, p Adjust, in []float32, out []float32) { /* ... */ }
```

A by-value struct, not a one-element buffer. That distinction is not style: a
buffer is memory some invocation might write, so the graph builder has to treat
it as a hazard and order around it. A uniform is one value for the whole
dispatch, and nothing can write it.

## Binding one

```go
r := dev.NewRecorder()
node := r.Dispatch(pipe, []accel.Binding{
	{Index: 0, Uniform: kernels.Adjust{Amount: 0.25}},
	{Index: 0, Buffer: inView},
	{Index: 1, Buffer: outView},
}, accel.WorkgroupCount{X: n / 64})
g, _ := r.Build()
```

Two entries both say `Index: 0` and mean different things. `Uniform` names a
position in the kernel's **by-value** list; `Buffer` names one in its **binding**
list. They are separate spaces.

> This is a rough edge. `Binding.Index` meaning two things depending on which
> other field is set is on the list to change — see
> [`specs/036`](../../specs/036-documentation.md) §5.3. The `Uniform` field
> itself is staying.

## You never write a padding offset

GPU uniform blocks use std140 layout, whose alignment rules are not Go's — a
`[3]float32` occupies twelve bytes but aligns to sixteen. accel generates the
encoder from your struct, so the layout is derived from the type rather than
maintained by hand in two places.

## Changing it without re-recording

A uniform is compiled into the plan at `Build`, which makes it fast and fixed.
When you want a different value, ask the graph:

```go
if err := g.SetUniform(node, 0, kernels.Adjust{Amount: 0.5}); err != nil {
	log.Fatal(err)
}
dev.Queue().Submit(g).Wait()
```

`node` is what `Dispatch` returned; `0` is the by-value index. Same graph, no
rebuild.

The line to hold: **change a value, keep the graph; change the shape of the
work, build another graph.** A brightness amount is a value. A different array
length is a shape — the barriers and the transient layout were computed from it.

## Try it

- Submit, read, `SetUniform`, submit again. Two answers, one graph.
- Pass `Adjust` as `[]Adjust` instead and read the rejection.

---

Next: [tensors](07-tensors.md) — the layer where none of this is your problem.
