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
  <img src="https://img.shields.io/badge/status-design-orange.svg" alt="Status: design">
</p>

---

> [!IMPORTANT]
> **This is a design, not a library.** Every function returns
> `ErrNotImplemented`. There is no working backend yet and the API will change.
> The repository holds the architecture, the specs, and an API surface that
> compiles. Feedback on the design is the most useful thing you can give it right
> now.

## What it is

One Go API for running work on a GPU, over whichever backend the machine
actually has, with no cgo anywhere.

```go
dev, err := accel.Open(accel.BackendMetal)
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

Two layers. The **device layer** gives you buffers, kernels, and recorded command
graphs, for renderers and simulations. The **tensor layer** gives you dtypes,
shapes, operators, and a computation graph, for inference, and never asks you to
think about a bind group.

Kernels are written in a subset of Go. The same source runs as ordinary Go on the
CPU backend and compiles to MSL, GLSL, SPIR-V, or HLSL on the GPU.

## Why you might want it

- **`CGO_ENABLED=0` and still on the GPU.** Cross-compile freely. No toolchain on
  the build machine. Fast builds.
- **Write once, run on Metal, Vulkan, D3D12, OpenGL, or the CPU.** Backends are
  selected explicitly, never silently swapped underneath you.
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
| Architecture and decisions | Drafted |
| Device layer specs | Drafted |
| Tensor layer spec | Drafted |
| Device layer API surface | Compiles, unimplemented |
| CPU backend | Not started |
| Metal backend | Not started |
| Kernel compiler | Not started |
| Tensor layer | Not started |
| Vulkan, D3D12, OpenGL, WebGPU backends | Specified, not scheduled for v0 |
| Graphics | Specified and frozen, not built at v0 |

**v0 is compute only, on the CPU backend and Metal.** The other backends and the
graphics half are designed and normative so their shape cannot break callers
later, and neither is scheduled. What gets built in what order, and what counts as
done for each step, is [`specs/009-sequencing.md`](specs/009-sequencing.md).

## Contributing

The design is still soft, so an argument against one of its decisions is worth
more than a patch right now. Every spec ends with the questions we have not
resolved. See [CONTRIBUTING.md](CONTRIBUTING.md).

## Acknowledgements

The design draws on lessons from [polyred](https://github.com/polyred/polyred),
whose cgo-free GPU abstraction proved out the Go-to-shader approach and, just as
usefully, made the mistakes that [`docs/conventions.md`](docs/conventions.md) now
records. [ollama](https://github.com/ollama/ollama)'s `ml` package informed the
graph-based execution model.

## License

[BSD-3-Clause](LICENSE) &copy; The golang.design Initiative Authors
