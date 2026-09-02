<h1 align="center">accel</h1>

<p align="center">
  <strong>Run compute on the GPU from Go. No cgo, no vendor SDK, no toolchain.</strong>
</p>

<p align="center">
  <a href="https://pkg.go.dev/golang.design/x/accel"><img src="https://pkg.go.dev/badge/golang.design/x/accel.svg" alt="Go Reference"></a>
  <a href="https://github.com/golang-design/accel/actions/workflows/ci.yml"><img src="https://github.com/golang-design/accel/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-BSD--3--Clause-blue.svg" alt="License: BSD-3-Clause"></a>
  <img src="https://img.shields.io/badge/go-1.26+-00ADD8.svg" alt="Go 1.26+">
  <img src="https://img.shields.io/badge/cgo-free-success.svg" alt="cgo-free">
  <img src="https://img.shields.io/badge/status-early-orange.svg" alt="Status: early">
</p>

---

You write a kernel in a subset of Go. accel compiles it ahead of time and runs
it on whichever backend the machine has — today the CPU or Metal — with
`CGO_ENABLED=0` the whole way.

> [!IMPORTANT]
> **The two halves of accel are at different stages, and one hedge for both
> would mislead you either way.**
>
> **Compute and tensors are settled.** The surface is frozen — names, shapes and
> behaviour are the contract — and an inference framework is built on it and
> files what it finds. Changing something here costs an argument, a deprecation
> and a release note. `specs/036-documentation.md` §5 is the record, symbol by
> symbol, including the few marked provisional and the event that moves each.
>
> **Graphics works and is younger.** It runs on the CPU backend and on Metal,
> compared pixel by pixel — render passes, depth, blending, indexed and indirect
> draws, and attachment formats — and it presents to a window you create; accel
> does not create windows.
>
> The one change that was going to break callers has landed: an attachment names
> a texture view rather than a buffer view, so it can carry a format, a mip
> level and an array layer. Mip levels above one allocate, bind and hazard-track
> per subresource. What is left is **additive**: a host copy of a level above
> the base, and rejecting a subresource used as an attachment and read by a
> stage at once. `specs/045-texture-attachments.md` §8 and §10 are the ledger. Treat the graphics API as
> settling rather than settled: it is outside the freeze record below, and a
> name here may still move where one in the compute half will not.
>
> Vulkan, D3D12, OpenGL and WebGPU are designed and not started. The
> [status table](#what-works-today) says which is which.

## Install

```sh
go get golang.design/x/accel
go get -tool golang.design/x/accel/cmd/accel-kernel
```

The second line registers the kernel generator as a [tool
dependency](https://go.dev/doc/modules/managing-dependencies#tools), which is
what lets `go generate` build it. Without it the generator cannot resolve its
own dependencies from your module and `go generate` fails — the generator is a
Go program that type-checks the package it compiles, so it needs the type
checker in your module graph.

## Run something

Kernels go in their own package. Write one:

```go
// kernels/scale.go
package kernels

import "golang.design/x/accel"

//go:generate go tool accel-kernel .

//accel:kernel workgroup=64
func Scale(t accel.Thread, in []float32, out []float32) {
	i := t.GlobalID().X
	if i < uint32(len(out)) {
		out[i] = in[i] * 2
	}
}
```

`go generate ./...` turns that into `kernels.ScaleKernel`: a compiled record
carrying the workgroup size and the bindings, with the read and write access of
each one *inferred from the body*, so you never declare them.

> [!TIP]
> Keep kernels in a package of their own, as above. The generator type-checks
> the package it compiles, so a package that already refers to `ScaleKernel`
> cannot be generated for the first time — the symbol does not exist yet.

Then use it:

```go
// main.go
package main

import (
	"fmt"
	"log"

	"example.com/hello/kernels"
	"golang.design/x/accel"
)

func main() {
	// The best device this machine has, falling back to the CPU backend
	// when there is no GPU. The same program runs either way.
	dev, err := accel.OpenBest(accel.Policy{AllowCPU: true})
	if err != nil {
		log.Fatal(err)
	}
	defer dev.Close()

	pipe, err := dev.NewComputePipeline(accel.ComputePipelineDescriptor{
		Kernel: &kernels.ScaleKernel,
	})
	if err != nil {
		log.Fatal(err)
	}
	defer pipe.Close()

	const n = 256
	usage := accel.BufferStorage | accel.BufferCopySrc | accel.BufferCopyDst
	in, _ := dev.NewBuffer(accel.BufferDescriptor{DType: accel.F32, Count: n, Usage: usage, Label: "in"})
	out, _ := dev.NewBuffer(accel.BufferDescriptor{DType: accel.F32, Count: n, Usage: usage, Label: "out"})
	defer in.Close()
	defer out.Close()

	src := make([]float32, n)
	for i := range src {
		src[i] = float32(i)
	}
	if err := dev.Queue().WriteBuffer(in, 0, src); err != nil {
		log.Fatal(err)
	}
	inView, _ := in.View(0, n)
	outView, _ := out.View(0, n)

	err = dev.Queue().Run(func(r *accel.Recorder) {
		r.Dispatch(pipe, []accel.Binding{
			{Index: 0, Buffer: inView},
			{Index: 1, Buffer: outView},
		}, nil, accel.WorkgroupCount{X: n / 64})
	})
	if err != nil {
		log.Fatal(err)
	}

	got := make([]float32, n)
	if err := dev.Queue().ReadBuffer(out, 0, got); err != nil {
		log.Fatal(err)
	}
	fmt.Println(got[:4]) // [0 2 4 6]
}
```

That program runs on a GPU where there is one and on the CPU backend where there
is not, and the kernel is the same either way.

`AllowCPU` is what makes the fallback legal, and leaving it out is the other
useful default: `accel.OpenBest(accel.Policy{})` **fails** on a machine with no
GPU rather than quietly running on the CPU. Use that when a CPU run would be a
misconfiguration you want to hear about — a benchmark, or a deployment that is
supposed to have a device. A device you asked for is never silently
substituted.

To pin the CPU backend for a test, `accel.OpenCPU(accel.CPUOptions{})` opens it
directly.

The [tutorials](docs/tutorial/) take this apart one idea at a time: what the
kernel subset allows, where memory comes from, how to record work once and
replay it, and how to run the same code on a device you do not own.

## Two layers, four packages

| Package | You get | Use it for |
| --- | --- | --- |
| [`accel`](https://pkg.go.dev/golang.design/x/accel) | buffers, kernels, command graphs, textures | simulation, image and signal processing, anything with custom kernels |
| [`accel/tensor`](https://pkg.go.dev/golang.design/x/accel/tensor) | dtypes, shapes, operators, a computation graph | inference — you never touch a bind group |
| [`accel/quant`](https://pkg.go.dev/golang.design/x/accel/quant) | int8 weights with a per-block scale | fitting a model in less memory |
| [`accel/kmath`](https://pkg.go.dev/golang.design/x/accel/kmath) | scalar math callable from a kernel | inside kernel bodies |

The tensor layer contains no backend-specific code. Everything it does, it does
by asking the device layer.

## Replay instead of re-issuing

One transformer layer is roughly a hundred operations, and a model has dozens of
layers. Re-issuing thousands of commands per token, and asking for every
intermediate allocation again, is most of the cost.

So you **record** work into a `Graph` and keep it:

```go
rec := dev.NewRecorder()
rec.Dispatch(pipeline, bindings, nil, accel.WorkgroupCount{X: 1024})
g, err := rec.Build() // validates, plans memory, computes barriers

for range steps {
	g.Bind(nextInputs...)
	dev.Queue().Submit(g).Wait()
}
```

`Build` does the analysis once; every submission after that is a replay. It also
works out where the barriers go, so you do not write one.

**What it costs you:** errors move. Under an immediate API a bad call fails at
the call; here it fails at `Build`, possibly far from where you wrote it. A
build error names the node, the binding slot and the numbers involved, so you
can find the recording call — but it does not yet carry that call's source
position.

## What works today

| You want to | Today |
| --- | --- |
| Run a compute kernel on the CPU | yes |
| Run the same kernel on a GPU | yes, on Metal |
| Cross-compile with `CGO_ENABLED=0` | yes, every `GOOS` |
| Test without a GPU | yes — the CPU backend is a full implementation, not a stub |
| Use every core on a machine with no GPU | yes — a dispatch runs its workgroups at once, about 7.5x on eight cores, and the answer does not change with the core count |
| Use shared memory, barriers and atomics | yes |
| Use subgroup reductions, votes, broadcasts, shuffles and scans | yes, on both backends |
| Multiply matrices (tiled GEMM) | yes, on both backends |
| Build a tensor graph and run inference | yes — decode and prefill, with a KV cache |
| Use int8 quantized weights | yes, against f32 or f16 activations |
| Mix widths in one product | yes — f32 activations against f16 or int8 weights, with no cast between. A transformer's two operands are never the same width, and casting them was four dispatches per layer |
| Keep the KV cache at f16 | yes — written, read, and paged. It halves the largest allocation a serving process has after the weights |
| Load a bf16 checkpoint | yes — `Cast` widens bf16 to f32 exactly, which is a shift. No GEMM reads bf16, so convert on load |
| Sample a token (argmax, categorical, top-k, top-p) | yes, batched — one row per sequence, with the random draw supplied so a token is reproducible |
| Run a whole sampling policy on device | yes — one `Sample` call records penalties, temperature, top-k, top-p and the draw, so a decode step reads back a token rather than a vocabulary of logits. At a 152k vocabulary that is 4 bytes instead of 608 KB per token |
| Reproduce a generated sequence from a seed | yes — the draw is a pure function of a seed and the token index, so there is no generator to copy, share or advance, and resuming a sequence costs nothing |
| Page a KV cache, and batch several sequences in one step | yes |
| Draw triangles | yes, on both backends: vertex and index buffers, per-vertex and per-instance attributes, uniforms, depth, blending, indexed and indirect draws |
| Render into a chosen pixel format | yes — an attachment names a texture view, so it carries its own format, and sRGB converts on write and on read |
| Read a texture from a shader stage | yes, integer texel fetch — so one pass can read what another drew, which is what deferred shading and shadow maps are. There is no sampler and there will not be one: a filtered sampler cannot be reproduced exactly by the CPU reference, so a stage that wants filtering builds it from fetches |
| Render a frame loop with acquire and present | yes, headless — the pixels come back in a buffer |
| Present to a window | yes on Metal, into a `CAMetalLayer` you own; accel does not create windows |
| Use Vulkan, D3D12, OpenGL or WebGPU | not yet |

Every "yes" has tests that fail without it and an end-to-end case through the
public API. Every kernel in the corpus that lowers to Metal runs on both
backends and the two are compared, most of them bit for bit; the two that do
not (the ballot and the float atomic) carry the emitter's refusal in the
generated record, and a test fails if a third loses its lowering silently.

That sentence is the standard this table is held to, and it has been wrong
twice: two rows here claimed a capability whose *kernels* existed while no
operator reached them, so nothing a caller could write would have exercised
either. A kernel is not a capability, and this table is about what a caller can
do.

## When it fits, and when it does not

**It fits** if you want GPU compute from Go without cgo: cross-compilation that
works, fast builds, no toolchain on the build machine, and a test suite that
runs anywhere.

**It does not fit** if:

- **You need training.** The tensor layer targets inference. There is no
  autodiff.
- **You are training on NVIDIA.** No CUDA backend is planned for v0.
- **You need peak throughput.** cgo-free rules out cuBLAS, cuDNN and GGML. Every
  kernel is written here, and it will not beat vendor libraries for a long time,
  possibly ever.
- **You want accel to open a window.** It will not. You create the window with
  whatever toolkit you like and hand accel the layer; it owns everything from
  the swapchain inward. Only Metal has an on-screen path so far.
- **You expected `wgpu`.** The submission model is deliberately different and the
  API does not aim to match it.

## Documentation

| | |
| --- | --- |
| [**Tutorials**](docs/tutorial/) | Eleven short pages, one idea each. Start here |
| [Architecture](docs/architecture.md) | How it fits together, and the decisions behind it |
| [Backend conventions](docs/conventions.md) | Where GPU backends actually disagree. Useful even if you never use accel |
| [Specs](specs/) | Internal design documents, full reasoning, open questions |
| [Contributing](CONTRIBUTING.md) | What would help most right now |

## Testing

```sh
CGO_ENABLED=0 go build ./...
go test -race ./...
```

No GPU required, which is deliberate and should stay true.

> [!NOTE]
> On a Mac, `go test ./...` **skips** the Metal tests when it finds no adapter,
> and says so only in the skip message. Set `ACCEL_REQUIRE_METAL=1` to turn that
> skip into a failure, which is what CI does.

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
