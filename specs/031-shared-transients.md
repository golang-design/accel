---
title: "One transient pool, many graphs"
status: implemented
layer: device
depends_on:
  - 003-command-graph.md
  - 017-graph-aliasing.md
  - 029-plan-cache.md
---

# One transient pool, many graphs

[009](009-sequencing.md)'s M8 item "additional transient sets", which
[003](003-command-graph.md) leaves as two open questions and
[007](007-tensor-layer.md) puts a number on:

> five prefill buckets at 200 MiB is a gigabyte of transients

[029](029-plan-cache.md) made that number real. A bucket set is several plans
over one model, each with its own transients, and **only one of them ever runs
at a time** — so the other four are holding device memory to be idle in.

## 1. The shape 003 already sketched

> The obvious shape is a pool object the caller creates and several graphs plan
> into, with a rule that graphs sharing a pool are mutually exclusive in flight,
> generalizing the one-in-flight rule from a graph to a pool.

That is what this builds, and the sketch is right for a reason worth stating:
**the rule that makes it safe is one a graph already has.** A graph may have one
submission in flight because its transients are reused across submissions; a
pool extends the same fact to a set of graphs. Nothing new has to be understood
— the scope of an existing rule widens.

```
without a pool          with a pool
┌────────┐              ┌──────────────────┐
│ bucket │ 200 MiB      │                  │
├────────┤              │   one pool       │  200 MiB
│ bucket │ 200 MiB      │   sized to the   │
├────────┤              │   largest graph  │
│ bucket │ 200 MiB      │                  │
├────────┤              └──────────────────┘
│ bucket │ 200 MiB       one submission in flight
├────────┤               across every graph sharing it
│ bucket │ 200 MiB
└────────┘
  1 GiB
```

## 2. What it costs, stated plainly

**Concurrency.** Two graphs sharing a pool cannot be in flight together. For a
bucket set that is free, because a request runs in one bucket. For two graphs a
caller wanted to overlap it is the wrong tool, and the refusal says so rather
than serializing quietly — a pool that queued would turn a design mistake into a
latency mystery.

The refusal reaches the caller **through the fence**, which is how every other
submission failure arrives, and the claim is taken inside the function that
executes a graph rather than inside `Queue.Submit`. Two reasons, both found
while building this:

1. `Queue.SubmitAfter` reaches the same graph by another road. A claim taken in
   `Queue.Submit` guards one road and leaves the other open, and the hole is
   invisible until two queues exist. One rule inside the executing function
   covers every road, now and for whatever road is added next.
2. A claim taken at call time also spans the wait for everything already queued
   in front of the graph, which is not a time the pool's bytes are live. Two
   submissions a serial queue could never overlap would then be refused, or not,
   depending on goroutine scheduling: the same program with two outcomes.

So the claim covers execution and nothing else:

```
Submit ──▶ │ queued, waiting on the stream │ executing │ ──▶ fence
                                            └── claim ──┘
```

**A pool grows and never shrinks.** It sizes itself to the largest graph built
into it, and building is the only moment it may resize: a submission holds
device addresses into it, so reallocating under one would be a use-after-free
whose symptom is a wrong answer rather than a crash. Building is refused while
anything is in flight, for exactly that reason.

$$
\text{pool bytes} \;=\; \max_{g \in \text{graphs}} \text{transientBytes}(g)
\qquad\text{against}\qquad
\sum_{g} \text{transientBytes}(g)
$$

## 3. Growing without invalidating what was built

A graph captures device addresses at `Build`: its transients, its plan
operands, and its compiled executable all hold the pool's block. Growing the
pool for a *later*, larger graph must not move the memory out from under them.

So a graph never holds the device allocation. It holds a **handle** to it, and
growing swaps what is inside the handle:

```
   graph A operands ──┐
   graph B operands ──┼──▶ handle ──▶ allocation (200 MiB)
   graph C operands ──┘                    │  build a larger graph
                                           ▼
                      handle ──▶ allocation (400 MiB)   old one freed after
```

The first version freed and replaced the allocation directly, and the first
test found it: a graph that had been correct reported *"the block has been
freed"* once a larger graph was built beside it.

A backend that type-asserts a block to its own concrete type therefore has to
resolve the handle first, so the seam carries `driver.Unwrap` and both backends
call it. Forgetting to is not subtle — the assertion fails and the error names
the wrapper — which is why it is a free function rather than a method a backend
could forget was there.

Nothing is copied across a growth. A transient holds no data between
submissions: `Build` already refuses a graph that reads one before writing it.

## 4. What it does not change

**Not the planner.** [017](017-graph-aliasing.md)'s aliasing runs per graph,
unchanged: each graph packs its own transients into offsets, and the pool is
the memory those offsets are relative to. Two graphs sharing a pool overlap
completely — which is sound *because* they never run together, and is the whole
saving.

**Not `Graph.Memory`.** It keeps reporting what that graph's transients need.
The pool reports what it allocated, which is the maximum over its graphs, and
those are different questions: one is "what does this plan cost" and the other
is "what did we reserve".

## 5. The surface

```go
// Device
func (d *Device) NewTransientPool(label string) (*TransientPool, error)

// Recorder — every graph built after this call plans into p.
func (r *Recorder) UseTransientPool(p *TransientPool)

// TransientPool
func (p *TransientPool) Bytes() int   // what the pool reserved: the maximum
func (p *TransientPool) Graphs() int  // how many built graphs share it
func (p *TransientPool) Close() error // refused while any of them is open
```

Four methods and one recorder call, which is the whole feature. `Bytes` and
`Graphs` exist because the saving is the reason to use this, and a caller who
cannot measure it has to take the saving on trust; `Graphs` is also what makes
the close refusal legible after the fact.

`UseTransientPool` is on the recorder rather than a `Build` argument so that a
plan builder one layer up — [029](029-plan-cache.md)'s bucket set — sets it once
where it already configures the recorder.

## 6. What is not reachable at v0

Both refusals — an overlapping submission, and a build while something
executes — need two graphs of one pool to be live at once, and at v0 every
backend reports a **single queue**, which runs its submissions in turn. Neither
refusal can be reached through the public API today.

They are built and tested anyway, because the rule they enforce is what makes
sharing sound, and a rule nothing checks is a rule that decays. The tests hold
the claim directly, which is what a second queue would do. When a backend
reports a second queue, the tests to add are the same two through `Submit`.

## 7. Done

- graphs sharing a pool allocate once, sized to the largest;
- a second submission against a pool already executing is refused through its
  fence, with an error naming the rule rather than blocking, and by both
  submission entry points;
- a growth after graphs are built keeps those graphs working, rather than
  freeing memory they captured;
- a graph built into a pool computes the same results as one with its own,
  which is what makes the sharing invisible to a caller;
- **a graph's results are unaffected by another graph having used the pool
  first**, which is the property that would break if the offsets were not
  per-graph; and
- closing a pool with graphs still open is refused, and closing a graph built
  into a pool does not free the pool.
