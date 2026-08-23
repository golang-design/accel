# 5. Cooperation

**One thing:** a kernel where invocations share data, and the barrier that makes
that safe.

Every kernel so far computed one output from its own inputs. A reduction cannot:
summing 64 values means combining across invocations.

```go
//accel:kernel workgroup=64
func GroupSum(t accel.Thread, in []float32, out []float32, sh *[64]float32) {
	lid := t.LocalID().X
	gid := t.GlobalID().X

	sh[lid] = in[gid]
	t.Barrier()

	stride := uint32(32)
	for stride > 0 {
		if lid < stride {
			sh[lid] = sh[lid] + sh[lid+stride]
		}
		t.Barrier()
		stride = stride / 2
	}
	if lid == 0 {
		out[t.GroupID().X] = sh[0]
	}
}
```

Dispatching four workgroups over 256 ones gives `[64 64 64 64]`.

## The two new things

**`*[64]float32` is workgroup-shared memory.** A pointer to a fixed-size array,
because the size is fixed at pipeline creation on every backend — it appears in
the shader's layout, so it cannot be a runtime length. One copy per workgroup,
shared by its invocations. A pointer rather than a value, because passing the
array by value would give every invocation its own and quietly compute something
else.

**`t.Barrier()` waits for every invocation in the workgroup.** Before it, some
invocations may not have written their `sh[lid]` yet. After it, all have.

`t.LocalID()` is the position within the workgroup; `t.GlobalID()` is across the
whole dispatch; `t.GroupID()` is which workgroup this is.

## Two rules the compiler enforces

**A barrier must not be in divergent control flow.** Every invocation has to
reach the same barrier, or the ones that do wait forever. Notice the barrier
above is *outside* the `if lid < stride` — inside, half the workgroup would skip
it. The compiler rejects the divergent form by name.

**Every invocation must write its shared memory before anything reads it.** On
real hardware, reading uninitialised shared memory gives you whatever was there,
which is often zero, which often looks right.

## Why this is worth running on the CPU

The CPU backend runs cooperative kernels through a scheduler that advances every
invocation to its next suspension point before releasing the epoch. That is not
for speed. It is so the schedule is **deterministic**, and so:

- a read of shared memory nothing wrote is *reported*, with a line number, on
  the first offending run;
- invocations reaching different barriers are *reported* rather than hanging;
- two invocations touching one location with nothing ordering them are
  *reported*.

A real GPU gives you a plausible number instead, on one machine, until it does
not.

If you suspect a kernel depends on which invocation runs first — which is wrong
on hardware — set `CPUOptions{ShuffleSeed: n}` and sweep `n`. A fixed order
makes an order-dependent kernel wrong *consistently*, which reads as correct.

## Try it

- Move a `t.Barrier()` inside the `if`. The compiler names the line.
- Delete the first barrier and run under `CPUOptions{}`. The diagnostic tells
  you which invocation read what nothing wrote.

---

Next: [values that are not buffers](06-uniforms.md).
