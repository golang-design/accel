<h1 align="center">accel</h1>

<p align="center">
  <strong>GPU compute and graphics for Go. Backend-selectable, and free of cgo.</strong>
</p>

<p align="center">
  <a href="https://pkg.go.dev/golang.design/x/accel"><img src="https://pkg.go.dev/badge/golang.design/x/accel.svg" alt="Go Reference"></a>
  <a href="https://github.com/golang-design/accel/actions/workflows/ci.yml"><img src="https://github.com/golang-design/accel/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-BSD--3--Clause-blue.svg" alt="License: BSD-3-Clause"></a>
  <img src="https://img.shields.io/badge/go-1.27+-00ADD8.svg" alt="Go 1.27+">
  <img src="https://img.shields.io/badge/cgo-free-success.svg" alt="cgo-free">
  <img src="https://img.shields.io/badge/status-early-orange.svg" alt="Status: early">
</p>

---

> [!IMPORTANT]
> **Early, and still substantially a design.** The CPU backend can open a
> device, move memory, and compile a kernel written in the Go subset into a
> lowering it then runs. Command graphs, cooperative execution, and every GPU
> backend are specified and unimplemented, and calling into them reports
> `ErrNotImplemented`. The API will change. Feedback on the design is still the
> most useful thing you can give it.

## What it is

One Go API for running work on a GPU, over whichever backend the machine
actually has, with no cgo anywhere.

```go
dev, err := accel.OpenBest(accel.Policy{Prefer: []accel.Backend{accel.BackendMetal}})
if err != nil {
    log.Fatal(err)
}
defer dev.Close()

// Record work once.
rec := dev.NewRecorder()
rec.Dispatch(pipeline, bindings, accel.WorkgroupCount{X: 1024})
g, err := rec.Build() // validates, plans memory, computes barriers

// Replay it as often as you like.
for range steps {
    g.Rebind(nextInputs)
    dev.Queue().Submit(g).Wait()
}
```

That is the destination. **What runs today** is the memory half of it, on the
CPU backend:

```go
dev, err := accel.OpenCPU(accel.CPUOptions{})
if err != nil {
    log.Fatal(err)
}
defer dev.Close()

// Memory comes from pools, because a model has thousands of tensors and
// drivers cap how many allocations you may hold.
weights, err := dev.NewPool(accel.MemoryDevice, 1<<30)
defer weights.Close()

w, err := weights.Alloc(accel.BufferDescriptor{
    DType: accel.F16,
    Count: 4096 * 4096,
    Usage: accel.UsageStorage | accel.UsageCopyDst,
    Label: "blk.0.attn_q.weight", // labels show up in every error
})
defer w.Close()

dev.Queue().WriteBuffer(w, 0, hostData) // returns once your slice is free
head, err := w.View(0, 128)             // a slice, in elements, with no copy
```

Kernels are written in a subset of Go and compiled by a generator, not by a
driver at runtime:

```go
//accel:kernel workgroup=64
func Scale(t accel.Thread, in []float32, out []float32) {
    i := t.GlobalID().X
    if i < uint32(len(out)) {
        out[i] = in[i] * 2
    }
}
```

`go generate` turns that into a lowering with an explicit rounding point at
every arithmetic operation, plus a record carrying the workgroup extent and the
bindings with the read and write accesses **inferred from the body**. Anything
outside the subset is rejected with a source position and a reason.

The subset has loops, helpers, the scalar math in
[`kmath`](https://pkg.go.dev/golang.design/x/accel/kmath), 16-bit storage types
that deliberately carry no arithmetic operators, and by-value uniform structs
whose std140 encoder is generated so a caller never writes a padding offset.
A narrow value has to widen before it can be used, which makes f32 accumulation
the only thing that compiles rather than a rule to remember.

Two layers. The **device layer** gives you buffers, kernels, and recorded command
graphs, for renderers and simulations. The **tensor layer** gives you dtypes,
shapes, operators, and a computation graph, for inference, and never asks you to
think about a bind group.

Kernels are written once in a subset of Go. One typed IR produces an instrumented
Go runner for the CPU backend and the GPU's shading language: MSL at v0, with
GLSL, SPIR-V, and HLSL designed and following their backends.

## Why you might want it

- **`CGO_ENABLED=0` and still on the GPU.** Cross-compile freely. No toolchain on
  the build machine. Fast builds.
- **Write once, run on the CPU or the GPU.** Backends are selected explicitly and
  never silently swapped underneath you. v0 is the CPU backend and Metal, and on
  a Mac both enumerate today; Vulkan, D3D12, OpenGL, and WebGPU are designed in
  the specs and not yet built.
- **Test without a GPU.** The CPU backend is a first-class implementation and the
  correctness oracle, so `go test ./...` works on any machine.
- **Kernels in Go**, type-checked by the Go compiler, not strings handed to a
  driver at runtime.

## Why you might not

We would rather you knew this from the README than found out in a month.

- **No training, and no autodiff.** The tensor layer targets inference. Training
  is not a serious conversation until a competitive GEMM exists here.
- **No CUDA backend planned for v0.** If your workload is training on NVIDIA
  hardware, this is the wrong tool today.
- **cgo-free rules out the C ecosystem.** No cuBLAS, no cuDNN, no GGML. Every
  kernel has to be written here, and it will not beat vendor libraries on raw
  throughput for a long time, possibly ever.
- **It is not a WebGPU implementation.** The submission model is deliberately
  different and the API does not aim to match `wgpu`.
- **Graphics is designed, not built.** Its parent design identifies the child
  specs still needed for the stage ABI, render API, surfaces, and CPU rasterizer.
  If you need rasterization today, this is not it yet.

The bet is that a pure-Go stack which cross-compiles and tests without a GPU is
worth more to some people than peak throughput. If that is not you, existing
bindings are the better choice.

## Documentation

| | |
| --- | --- |
| [Architecture](docs/architecture.md) | How it is put together and why. Start here. |
| [Backend conventions](docs/conventions.md) | Where GPU backends actually disagree. Useful even if you never use accel. |
| [Specs](specs/) | Internal design documents, full reasoning, open questions. |
| [Contributing](CONTRIBUTING.md) | What would help most right now. |

## Status

| Component | State |
| --- | --- |
| Architecture and decisions | Decision record locked; bounded specs drafted |
| Device open, capabilities, limits | **Built** on the CPU backend |
| Pools, suballocation, buffers, views, lifetime | **Built** on the CPU backend |
| Host and device transfers | **Built** on the CPU backend |
| Textures, formats, and row pitch | **Built** on the CPU backend |
| Kernel compiler: subset checking, IR, Go lowering, generator | **Built** |
| Kernel language: loops, helpers, narrow storage, scalar math | **Built** |
| Kernel uniforms: std140 codecs and typed binding | **Built** |
| Command graphs: recording, slots, validation, submission, fences | **Built** on the CPU backend |
| Command graphs: inferred edges, sub-range hazards, computed barriers | **Built** on the CPU backend |
| Compute dispatch in a graph | **Built** on the CPU backend |
| Command graphs: transient aliasing and the whole-plan fuzz | **Built** on the CPU backend |
| Cooperative kernels: barriers, shared memory, the resumable lowering | **Built** on the CPU backend |
| Cooperative diagnostics: undefined reads, arrival, conflicting access | **Built** on the CPU backend |
| Atomics, emulated subgroups, capability inference | **Built**; subgroup shuffles and scans are specified and unbuilt |
| Portable tiled GEMM | **Built** on both backends, each against a higher-precision reference |
| Kernel corpus: the unquantized v0 kernels | **Built** on both backends; the selection registry is not |
| Metal backend | **Built** on an Apple M2: all 32 corpus kernels compile on the device and agree with the CPU backend, 25 of them bit for bit. Graphs, indirect dispatch and device loss included |
| Tensor layer | **Built**: a tensor graph compiles once and runs on both backends, with views, broadcasting, normalization, matrix multiplication, a KV cache, and prefill and decode attention that agree |
| Quantized weights: int8 with a per-block scale | **Built** on both backends, with a derived error bound rather than a measured one |
| Sampling: argmax and categorical | **Built** on both backends; top-k and top-p are not |
| Vulkan, D3D12, OpenGL, WebGPU backends | Specified, not scheduled for v0 |
| Graphics | Parent design drafted, child APIs and implementation post-v0 |

Built means it has tests that fail without it, greater than 90% statement
coverage on its package, and an end-to-end case through the public API. Those
rows came from [M1 through M7](specs/009-sequencing.md), and the last two from
[M8](specs/009-sequencing.md#m8-and-later)'s independently scoped work.

A graph infers its own dependency edges from what each node declares it touches,
comparing byte ranges rather than whole resources, so two nodes writing disjoint
halves of one buffer are not serialized. Barriers come from those edges and are
batched, because a barrier is queue-wide: [spec 003's worked
graph](specs/003-command-graph.md) has nine hazards and emits seven barriers,
and the test asserts their positions rather than only their count.

A kernel that needs its invocations to cooperate — shared memory, a barrier, a
reduction across a subgroup — is compiled to a resumable form and run by a
scheduler that advances every invocation to its next suspension point before
releasing the epoch. The point of doing that on a CPU is not speed: it is that
the schedule is deterministic, so a kernel reading shared memory nothing wrote,
or whose invocations reach different barriers, is *reported* with a line number
on the first offending run rather than producing a plausible number on one
machine and a different one elsewhere.

Transients the builder owns share memory when every node touching one is ordered
before every node touching the other. That is reachability, not record-order
position, and the difference is not theoretical: an interval-based planner
aliases two transients on opposite arms of a diamond and corrupts one of them on
any backend that runs the arms at once.

The conservative plan that ran before edges were inferred is kept rather than
deleted. It is the oracle: every random graph is built twice, once optimized and
once naively, and the results compared. It found three real bugs in minutes.

**v0 is compute only, on the CPU backend and Metal.** The other backends and the
graphics half are designed and normative so their shape cannot break callers
later, and neither is scheduled. What gets built in what order, and what counts as
done for each step, is [`specs/009-sequencing.md`](specs/009-sequencing.md).

## Contributing

The design is still soft, so an argument against one of its decisions is worth
more than a patch right now. Every spec ends with the questions we have not
resolved. See [CONTRIBUTING.md](CONTRIBUTING.md).

```sh
CGO_ENABLED=0 go build ./...
go test -race ./...
```

No GPU required, which is deliberate and should stay true.

## Acknowledgements

The design draws on lessons from [polyred](https://github.com/polyred/polyred),
whose cgo-free GPU abstraction proved out the Go-to-shader approach and, just as
usefully, made the mistakes that [`docs/conventions.md`](docs/conventions.md) now
records. [ollama](https://github.com/ollama/ollama)'s `ml` package informed the
graph-based execution model.

## License

[BSD-3-Clause](LICENSE) &copy; The golang.design Initiative Authors
