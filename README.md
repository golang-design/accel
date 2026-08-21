# accel

Backend-selectable, cgo-free GPU compute and graphics for Go.

**Status: design. Nothing is implemented.** This repository currently holds the
architecture, the specs, and an API surface that compiles and does nothing. There
is no working backend yet, and the API will change.

## What it is meant to be

Two layers with a hard boundary between them.

**The device layer** gives you memory, kernels, pipelines, recorded command
graphs, and presentation, over whichever backend the machine actually has. Its
vocabulary is buffers, textures, workgroups, and barriers. A renderer or a
simulation uses this directly.

**The tensor layer** gives you dtypes, shapes, an operator set, and a computation
graph, written entirely in terms of the device layer. An inference engine uses
this and never touches a bind group.

Kernels are written in a subset of Go. One kernel source runs as ordinary Go on
the CPU backend and compiles to native shader code on the GPU, which is what
makes the CPU backend an exact correctness oracle rather than an approximation.

Every backend is reached without cgo, through `purego` or raw syscalls, so
`CGO_ENABLED=0` still uses the GPU and cross-compilation keeps working.

## Why it might not be what you want

Worth knowing before you invest attention:

- There is **no autodiff and no training**. The tensor layer targets inference,
  and training is not a real conversation until a competitive GEMM exists.
- There is **no CUDA backend** planned for v0. If your workload is training on
  NVIDIA hardware, this is the wrong tool today.
- Being cgo-free rules out linking cuBLAS, cuDNN, or GGML, so **every kernel has
  to be written here**. That is a lot of work, and it will not beat vendor
  libraries on throughput for a long time, if ever.
- It is **not a WebGPU implementation**. The submission model is deliberately
  different and the API does not aim to match `wgpu`.

The bet is that a pure-Go, cgo-free stack that cross-compiles and tests without
a GPU is worth more to some people than peak throughput. If it is not worth that
to you, existing bindings are the better choice.

## Design

- [`docs/design.md`](docs/design.md), the locked decisions. Start here.
- [`docs/conventions.md`](docs/conventions.md), where GPU backends actually
  disagree, and what is guaranteed on top of them. Most entries cost someone
  hours to find.
- [`specs/`](specs/), the bounded designs.

## License

BSD-3-Clause. See [LICENSE](LICENSE).
