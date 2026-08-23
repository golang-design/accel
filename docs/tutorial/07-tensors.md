# 7. Tensors

**One thing:** the same work in shapes and operators, with no bindings, no
workgroup counts and no barriers.

Everything so far was the device layer. If you are running a model, you want the
layer above it.

```go
rt, err := tensor.NewRuntime(dev)
if err != nil {
	log.Fatal(err)
}

const n = 256
b := rt.NewBuilder("ffn")
desc := func(name string) tensor.ValueDesc {
	return tensor.ValueDesc{Name: name, DType: accel.F32, Shape: tensor.Shape{n}}
}

x := tensor.Input(b, desc("x"))
w := tensor.Weight(b, desc("w"))
gate := tensor.Weight(b, desc("gate"))
tensor.Output(b, "y", tensor.Mul(b, tensor.Add(b, x, w), tensor.SiLU(b, gate)))

plan, err := b.Compile(rt, tensor.CompileOptions{Label: "ffn"})
if err != nil {
	log.Fatal(err)
}
defer plan.Close()
```

That is `y = (x + w) · SiLU(gate)`. Three operators, two of them consuming an
intermediate you never allocated.

## Submitting

```go
f := plan.Submit(dev.Queue(), tensor.Bindings{Buffers: map[string]accel.BufferView{
	"x": xView, "w": wView, "gate": gateView, "y": outView,
}})
if err := f.Wait(); err != nil {
	log.Fatal(err)
}
```

Inputs and outputs are named. A missing or wrong-shaped one is refused before
anything runs.

## Input, Weight, Output

- **`Input`** changes between submissions — a token, an activation.
- **`Weight`** does not. Marking it lets the plan cache reuse a compiled plan
  across steps that share weights.
- **`Output`** is what you read back.

## What the layer does for you

The intermediates became transients, and the ones whose lifetimes do not overlap
share bytes:

```go
fmt.Println(plan.Memory().TransientBytes)
```

Barriers were computed. Kernels were selected — `plan.Selections()` tells you
which and why, which is the first thing to read when a plan is slower than
expected.

**It contains no backend-specific code.** Everything it does, it does by asking
the device layer. That is why the same plan runs on Metal unchanged.

## Errors arrive at compile, and they accumulate

The builder collects mistakes rather than stopping at the first:

```go
if err := b.Err(); err != nil { /* every mismatch, each with its call site */ }
```

`Compile` returns them too. A shape mismatch names both shapes and the operator.

## What is here

Elementwise arithmetic and activations, `RMSNorm`, `Softmax`, `MatMul`,
`Linear`, `Rows`, `RoPE`, `Cast`, a KV cache, and prefill and decode attention.
[`quant`](https://pkg.go.dev/golang.design/x/accel/quant) turns weights into
int8 with a per-block scale.

The view operators — `Reshape`, `Permute`, `Transpose`, `Slice`, `Broadcast` —
exist, but their results only reach elementwise operators today. A strided view
into `MatMul` is refused rather than silently copied.

## Try it

- Give two operands mismatched shapes and read `b.Err()`.
- Print `plan.Selections()` for a plan with a `MatMul`.

---

Next: [backends](08-backends.md) — running this somewhere other than the CPU.
