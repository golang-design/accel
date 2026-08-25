# 10. Quantized weights

**One thing:** the width a weight is *stored* in is not the width the graph
*computes* in, and you do not write a cast to move between them.

A checkpoint arrives in f32 or bf16. What you load it as is a memory decision:
f16 halves it, int8 quarters it, and on a decode step that is bound by reading
weights, that is most of the run time. The activations flowing through the model
stay f32 either way.

## Choosing the width at load

```go
quants, scales := quant.Int8Quantize(w)

narrow := make([]accel.Float16, len(w))
for i, v := range w {
	narrow[i] = accel.ToFloat16(v)
}
```

A quantized matrix is **two** arrays, not one: an `i8` quant per weight, and an
f16 scale per `quant.Int8Block` (32) weights of the flattened matrix. The device
holds them as two buffers because a buffer is typed by one dtype, and
`tensor.Quantized` binds them together so a matrix's quants can never meet
another matrix's scales.

## Both against the same f32 activations

```go
x := tensor.Input(b, tensor.ValueDesc{
	Name: "x", DType: accel.F32, Shape: tensor.Shape{1, in},
})
wq := tensor.Weight(b, tensor.ValueDesc{
	Name: "wq", DType: accel.I8, Shape: tensor.Shape{in, out},
})
ws := tensor.Weight(b, tensor.ValueDesc{
	Name: "ws", DType: accel.F16, Shape: tensor.Shape{len(scales)},
})
wf := tensor.Weight(b, tensor.ValueDesc{
	Name: "wf", DType: accel.F16, Shape: tensor.Shape{in, out},
})
tensor.Output(b, "i8", tensor.QuantMatMul(b, x,
	tensor.Quantized{Quants: wq, Scales: ws}))
tensor.Output(b, "f16", tensor.MatMul(b, x, wf))
```

One f32 activation, two weight widths, no `Cast` anywhere. `plan.Selections()`
says which kernel each reached:

```
QuantMatMul -> QuantMatVecF32
MatMul      -> MatMulTiledF32F16
```

Both kernels widen the weight as they read it and accumulate in f32. Two rules
follow from that and are worth knowing before you hit them:

- **`MatMul` is asymmetric on purpose.** f32 activations times an f16 weight is
  registered; f16 activations times an f32 weight is refused. The second is the
  memory decision made in the expensive direction.
- **`QuantMatMul` is a separate operator, not `MatMul` with another argument
  type.** The cost differs, so the source says which one you wrote.

## What the int8 costs, and where that is stated

```go
terms := make([]accel.Float16, in)
for t := range in {
	terms[t] = scales[(t*out)/quant.Int8Block]
}
bound := quant.Int8ErrorBound(acts, terms)
```

```
exact   -0.202312
int8    -0.142821  off by 0.059491, bound 0.469747
f16     -0.200934  off by 0.001379
```

That bound is **derived, not measured**. Rounding to nearest puts each weight
within half a scale step of its original, a step is the block's largest
magnitude over 127, and a dot product weights each of those errors by the
activation it multiplies:

$$\left| \sum_i q_i s_{b(i)} x_i - \sum_i w_i x_i \right| \le \frac{1}{254} \sum_i |x_i| \cdot \max_{j \in b(i)} |w_j|$$

`specs/027-quantization.md` §3 states it and derives it. There is no tolerance
parameter anywhere in accel, and this is the reason: a tolerance is a number
somebody raised until a test passed, and a bound is a property of the
representation. If you compare against an exact reference, add
`specs/008-numerics.md` §7's f32 reduction bound on top — the products are
summed in f32 and that error is not this one.

**`Int8ErrorBound` takes one scale per term.** Not the array `Int8Quantize`
returned. A per-block scale is only a bound where the dot product's terms are
contiguous in the quantized array, which is true down a row and false down a
column — so the signature asks for what the arithmetic needs, and panics when
the counts disagree. The loop above builds a column's terms.

The bound is loose here because it charges every weight in a block at that
block's largest magnitude. That is what makes an outlier expensive: one large
weight represents its 31 neighbours worse.

## Uploading f16 from the host

`Queue.WriteBuffer` takes the bit patterns an f16 buffer holds, and
`accel.Float16` keeps its bits behind a method, so the load path converts:

```go
func f16bits(vals []accel.Float16) []uint16 {
	bits := make([]uint16, len(vals))
	for i, v := range vals {
		bits[i] = v.Bits()
	}
	return bits
}
```

Both the scales and an f16 weight matrix go up through it. This is a rough
edge and not a design: `WriteBuffer` accepts `[]uint16` and refuses
`[]accel.Float16`, which is what `Int8Quantize` returns. Write the helper once
per program.

## Try it

- Print `plan.Selections()` and look for a `Cast`. There is none; that is the
  point of the f32 variants.
- Give `MatMul` an f16 activation and an f32 weight, and read the refusal.
- Put one large outlier in `w`, requantize, and watch the bound move while the
  other 31 weights in its block get worse.

---

Next: [batching sequences](11-batching-sequences.md) — a value that differs per
row is a tensor.
