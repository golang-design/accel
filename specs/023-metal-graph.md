---
title: "Metal graph execution, lifetime, and device loss"
status: drafted
layer: device
depends_on:
  - 001-device-resources.md
  - 003-command-graph.md
  - 006-backends.md
  - 021-metal-bringup.md
---

# Metal graph execution

The third of [009](009-sequencing.md)'s three M6 children: everything about
*submission* that [021](021-metal-bringup.md) deferred, and the M6 end-to-end.

## 1. Re-encoding, and the question 021 left open

[006](006-backends.md) §4.3 settles the default: a `MTLCommandBuffer` is
single-submit, so a graph is re-encoded per submission and replay's category-2
saving is zero on Metal. [021](021-metal-bringup.md) implements that already.

What it left open is **how much of a barrier an encoder boundary needs to be**.
021 ends the current encoder at every `BarrierBefore`, which is correct and
conservative. Whether a `memory_barrier` within one encoder would serve, and
what that costs, is measured here — and it is measured rather than assumed,
because the conservative version is already correct and an optimisation with no
measurement behind it is a regression waiting to be attributed to something
else.

## 2. Indirect dispatch

`dispatchThreadgroupsWithIndirectBuffer:` with the device-supplied count, and
[003](003-command-graph.md)'s clamp against the build-time maximum. **Every
build mode clamps**: correctness does not depend on a flag, and no backend may
submit an out-of-range count. `Plan.CollectStats` reports what the count turned
out to be, at the cost of a readback.

## 3. Completion handlers, and the lifetime rule under pressure

[021](021-metal-bringup.md) §2 states the strongest form of the rule by having
no completion handler at all: the fence polls `-status` and blocks on
`-waitUntilCompleted`. That is enough for one submission at a time and not for
much else.

Adding a handler is where [`conventions.md`](../docs/conventions.md)'s
divergence bites:

> A Metal command buffer completion handler runs *after* the enclosing
> autorelease pool has drained. Releasing an autoreleased object from inside the
> handler is a use-after-free.

So the handler releases nothing it did not retain, and the test is repeated
early closes racing asynchronous completion — a graph closed while its
submission is in flight, many times, under the race detector.

## 4. Device loss

[021](021-metal-bringup.md) answers `Lost()` with nil always, which is a real
answer for a device that has not lost anything and not the whole contract.
[001](001-device-resources.md) §7.4 makes loss **sticky and terminal**: Metal
reports it through a command buffer's error, and turning that into a device-level
answer that every subsequent submit and every outstanding fence reports is this
child's.

## 5. Done

1. multi-node graphs re-encode correctly, including the worked graph of
   [003](003-command-graph.md);
2. indirect dispatch clamps in every mode and reports its actual count;
3. completion-handler lifetime survives repeated early closes under the race
   detector;
4. device loss is sticky, reported by every subsequent call, and signals every
   outstanding fence; and
5. E2E: the same public upload → GEMM → readback scenario as the CPU, selected
   by enumerating a Metal `AdapterID` and calling `OpenDevice`.

`MTLIndirectCommandBuffer` is **not** in this child. [006](006-backends.md) §4.3
puts it behind the same optional interface as everything else, off by default,
and says it ships only with a measurement against re-encode.

## Testing

Against the CPU backend, as everywhere in M6. The graph tests already exist and
already assert edge sets and barrier positions that were written down before the
code; what this child adds is running them on a second backend.
