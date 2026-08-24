# 1. Hello, GPU

**One thing:** get a number computed somewhere other than your CPU's main
thread, and read it back.

We will double every element of an array.

## Kernels live in their own package

```go
// kernels/kernels.go
package kernels

import "golang.design/x/accel"

//go:generate go run golang.design/x/accel/cmd/accel-kernel .

//accel:kernel workgroup=64
func Scale(t accel.Thread, in []float32, out []float32) {
	i := t.GlobalID().X
	if i < uint32(len(out)) {
		out[i] = in[i] * 2
	}
}
```

`go generate ./...` turns that into `kernels.ScaleKernel`.

Their own package, because the generator type-checks the code it compiles — so
a package that already mentions `ScaleKernel` cannot be generated the first
time. The symbol does not exist yet.

Three things to notice in the kernel itself:

- **`accel.Thread` is always the first parameter.** It tells an invocation which
  one it is. `t.GlobalID().X` is this invocation's index across the whole
  dispatch.
- **The bounds check is yours.** You will launch in whole workgroups of 64, so
  the last group runs past the end of a 100-element array. Without the `if`, it
  writes somewhere it should not.
- **Nothing says which parameters are read and which are written.** The compiler
  works that out from the body. `in` is read, `out` is written, and the graph
  layer uses that later to place barriers you never write.

## Running it

```go
dev, err := accel.OpenCPU(accel.CPUOptions{})
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

// 64 invocations per workgroup, so n elements need (n+63)/64 groups.
groups := (n + 63) / 64
err = dev.Queue().Run(func(r *accel.Recorder) {
	r.Dispatch(pipe, []accel.Binding{
		{Index: 0, Buffer: inView},
		{Index: 1, Buffer: outView},
	}, nil, accel.WorkgroupCount{X: groups})
})
if err != nil {
	log.Fatal(err)
}

got := make([]float32, n)
if err := dev.Queue().ReadBuffer(out, 0, got); err != nil {
	log.Fatal(err)
}
fmt.Println(got[:4]) // [0 2 4 6]
```

## The three ideas

**`Binding.Index` is a position in the kernel's binding list, not its parameter
list.** `Scale` takes `(t, in, out)`; `t` is not a binding, so `in` is index 0
and `out` is index 1. The `nil` is where by-value parameters go — none here, and
[tutorial 6](06-uniforms.md) is about those.

**You dispatch workgroups, not elements.** `workgroup=64` in the kernel and
`WorkgroupCount{X: groups}` at the call site multiply together to give the
invocation count. Get this wrong in one direction and you compute part of the
answer; in the other, you rely on the bounds check.

**`Queue.Run` records and submits in one call.** It builds a graph every time,
so it is the wrong choice in a loop — [tutorial 4](04-graphs.md) is about that.

## Try it

- Delete the `if i < uint32(len(out))` and run with `n = 100`. The bounds check
  is not ceremony.
- Swap `OpenCPU` for `accel.OpenBest(accel.Policy{})`. On a Mac you get Metal
  and the same answer. On a machine with no GPU you get an error, not a silent
  fall back to the CPU — see [tutorial 8](08-backends.md).

---

Next: [writing a kernel](02-writing-a-kernel.md) — what the Go subset lets you
say.
