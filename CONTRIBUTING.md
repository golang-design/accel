# Contributing

Thanks for looking. Compute works end to end on the CPU and Metal backends:
memory, command graphs, cooperative kernels, a portable tiled GEMM, the tensor
layer, quantized weights, sampling and a paged KV cache. Graphics is being built
now. Vulkan, D3D12, OpenGL and WebGPU are specified and unscheduled.

The API will still change, so an argument against one of the decisions is worth
as much as a patch — and patches are welcome.

## What is most useful right now

**Working through the [tutorials](docs/tutorial/) and telling us where they
lose you.** They are new, and the place a newcomer gets stuck is the thing
nobody who wrote them can see.

**Tell us where the design is wrong.** Especially if you have shipped GPU code,
or run models in production, and something here looks naive. Open an issue.
[`docs/architecture.md`](docs/architecture.md) is the readable overview;
[`specs/`](specs/) has the full reasoning and, at the end of every spec, a list
of open questions we have not resolved.

**Backend conventions we have missed.** If you know a place where two GPU
backends disagree and it is not in [`docs/conventions.md`](docs/conventions.md),
that is directly valuable. Those entries cost hours each to discover.

**Prior art.** If another project solved one of the open questions well, saying
so saves us finding out the hard way.

**Code.** The public API is still changing, so for anything larger than a fix,
open an issue with the shape you have in mind before you write it. You will get
an answer about whether a design change is about to move the ground under it.

## Ground rules

### No cgo

Not in the library, not in tests, not behind a build tag. Every backend reaches
its driver through [purego](https://github.com/ebitengine/purego) or raw
syscalls.

CI greps for `import "C"` rather than relying on the build, because a file can
import C behind a tag the CI platform does not select and still break someone
else. This is the one rule with no exceptions: if a feature needs cgo, it does
not go in.

### The CPU backend is the oracle

Anything a GPU backend does, the CPU backend does too. Exact domains compare
bits; other floating-point domains compare against a higher-precision reference
under the derived bounds in `specs/008-numerics.md`. A GPU path with no CPU
equivalent has no way to be verified, so it does not merge.

### Capabilities are explicit

If a backend cannot do something, say so through the capability system and fail
with an error naming what is missing. Never silently substitute a different
result, and never silently fall back to another backend. A user whose code
quietly ran on the CPU for six months is worse off than one who got an error on
day one.

### Costs go in the docs

Every design document here states what its choice gives up, not just what it
buys. If you add a feature with a real tradeoff, write the tradeoff down. This is
a house style and reviewers will ask for it.

### Do not run `go fix` unreviewed

Go 1.27's `go fix` modernizers are on by default and two of them rewrite exactly
the code that is hardest to get right here.

`atomictypes` turns a raw `atomic.AddInt32(&x, 1)` into an `atomic.Int32`. That
is usually an improvement and is wrong for a packed allocator header:
`atomic.Int32` and `atomic.Int64` carry a `noCopy` marker and, on the 64-bit
form, alignment padding, so the rewrite changes a struct's size and makes it
non-copyable. `unsafefuncs` turns `unsafe.Pointer(uintptr(p) + n)` into
`unsafe.Add`, which is the correct idiom and is still a mechanical edit to the
one part of the code where a mistake is not a panic.

Neither fires on the current tree and neither runs under `go test`, so CI cannot
be surprised by them. Run `go fix -diff` and read the diff.

### Style

- `gofmt`, and CI checks it.
- Doc comments explain **why**, not just what. The why is the part a reader
  cannot reconstruct.
- No em dashes in prose. Commas, colons, and parentheses do the job.
- Commit messages explain the reasoning behind a change, not only its content.

## Getting started

```sh
git clone https://github.com/golang-design/accel
cd accel
CGO_ENABLED=0 go build ./...
CGO_ENABLED=0 go test ./...
```

Requires Go 1.27 or later. There is nothing to install and no GPU required,
which is deliberate and should stay true.

Metal code is `//go:build darwin`, so building on a Mac stops proving that the
other platforms still compile. Run the cross-target build before pushing, since
it is cheap and it is the only thing that catches a darwin-only file that leaked
a reference into shared code:

```sh
for os in linux windows darwin; do GOOS=$os CGO_ENABLED=0 go build ./... || break; done
```

M0 through M7 are built, which is the v0 proof
[000](specs/000-decisions.md#the-v0-milestone) names. A CPU device opens, pooled
memory allocates and suballocates, buffers slice into views and textures, bytes
move to the device and back, kernels written in ordinary Go compile through a
typed IR to a generated CPU lowering and to Metal Shading Language, graphs record
and plan their own barriers and transient aliasing, cooperative kernels run with
deterministic diagnostics, and a portable tiled GEMM matches a higher-precision
reference on both backends. Above that,
[`tensor`](https://pkg.go.dev/golang.design/x/accel/tensor) compiles a graph of
tensors into a plan and runs a transformer decode step whose output matches the
same tokens prefilled in one pass.

What is not built: graphics is designed and being written — the stage types are
in the public API and the CPU rasterizer is under way, but there is no render
pipeline, pass or surface — and the Vulkan, D3D12, OpenGL and WebGPU backends
are specified and unscheduled. Subgroup scans are specified and unbuilt. `Sampler` is the one declaration that exists only for its shape, and it
**panics** with `ErrNotImplemented` rather than returning it, because it has
nothing to sample until a render pass exists.
[`specs/009-sequencing.md`](specs/009-sequencing.md) is the order the rest
arrives in.

The coverage gate is per package rather than one repository average, and it
excludes design-stage stubs by a checked rule so the number means something
while most of the surface is still unbuilt:

```sh
go test -race ./...
go test -coverprofile=cover.out -coverpkg=./... ./...
go run ./internal/conformance/cover/covercheck -profile=cover.out
```

**On a Mac, that number is higher than the one CI computes.** The Metal
differential runs here and not on Linux, and it is what exercises many kernels'
generated lowerings — so a package can pass locally and fail the gate on CI, and
the first sign is a coverage percentage rather than a named failure. It has
happened twice.

Reproduce the Linux number before pushing a new kernel:

```sh
go test ./internal/testkernels/ -skip 'Metal|Darwin' \
  -coverprofile=cover.out -coverpkg=./internal/testkernels/
go tool cover -func=cover.out | tail -1
```

The usual cause is a kernel whose **authored** Go function nothing calls: the
generated lowering runs in every test, and the authored form runs only where a
test calls it directly. `TestAuthoredFormsAgreeWithTheirLowerings` is where that
comparison goes, and a new kernel needs an entry in it.

## How the repository is organised

| Path | What it is | Who it is for |
| --- | --- | --- |
| `docs/` | Documentation | People using or contributing to accel |
| `specs/` | Internal design specs and decisions | People building or reviewing accel |
| `*.go` | The device layer's public API and its policy | Callers, and everyone |
| `tensor/` | The tensor layer: builder, plans, operators, state | Callers writing a model |
| `quant/` | Turning weights into the form a quantized kernel reads | Callers with quantized weights |
| `internal/driver/` | The backend contract, and the plan a built graph lowers to | Anyone adding a backend |
| `internal/cpu/` | The pure-Go backend and oracle | Anyone adding a backend |
| `internal/metal/` | The Metal backend: adapters, memory, and plan execution | Anyone adding a backend |
| `internal/mtl/` | The Objective-C shim the Metal backend sits on, cgo-free through purego | Anyone touching the Metal shim |
| `internal/alloc/` | Suballocation inside a pool | Anyone changing suballocation; it is fuzzed and self-contained, which makes it a good first read |
| `internal/kernelc/` | The kernel compiler: loader, subset checker, IR, emitters | Anyone changing the kernel language |
| `internal/kernel/` | The vocabulary a kernel is written in, declared below accel | Anyone changing the kernel language |
| `internal/conformance/` | Test machinery: profiles, comparisons, coverage | People writing tests |

Policy lives in the public package and only what a backend alone can answer
crosses into `internal/driver`. accel links its backends in, so a backend cannot
import accel: anything crossing that seam is declared below both, and the two
declarations are kept in step by the compiler rather than by a test.

That is also why a *built graph* crosses the seam as a value rather than being
replayed by the layer above calling backend primitives one at a time. A Vulkan
primary command buffer, a D3D12 closed list and a Metal indirect command buffer
are each built from a whole plan, and none of them can be assembled from a
stream of unrelated calls. So `driver.Plan` is what a backend receives:
validated, ordered, barriered, and with offsets already assigned.

If you change behaviour that a spec describes, update the spec in the same
change. A spec that no longer matches reality is worse than no spec.

## Reporting a problem

For a design objection, quote the specific claim you disagree with and say what
you would do instead. For a bug, once there is code to have bugs, include the
backend, the platform, and ideally a case that behaves differently on the CPU
backend than on a GPU one, since that difference is usually the whole story.

## License

Contributions are under [BSD-3-Clause](LICENSE), the same as the project.
