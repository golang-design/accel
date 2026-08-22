---
title: "Kernel uniforms: std140 codecs and typed binding"
status: drafted
layer: device
depends_on:
  - 001-device-resources.md
  - 013-kernel-subset.md
---

# Kernel uniforms

The third of [009](009-sequencing.md)'s three M2 children, and the one that
completes M2's definition of done. It gives a kernel by-value parameters and
gives callers a way to supply them without ever spelling a padding offset.

## 1. Why this comes last, and why it is not optional

**Last**, because the only honest test of it needs both of the children before
it. [001](001-device-resources.md) §11.2 requires std140 encoding to be checked
against the *device* rather than against the encoder: a kernel reads each field
of a uniform struct and writes it to a distinct storage element, and the host
asserts the values. That is the test that catches an encoder agreeing with
itself and disagreeing with the shader, and no host-side round trip can catch
it. It needs kernel execution, from [012](012-kernel-pipeline.md).

And the uniform that matters is a loop bound. A storage-buffer substitute would
make a uniform loop bound appear non-uniform to the barrier analysis, which is
the reason [004](004-kernel-authoring.md) calls this path required rather than
convenient. Testing that needs `for`, from [013](013-kernel-subset.md).

So the ordering is forced by what constitutes proof, not chosen for convenience.

**Not optional**, because [009](009-sequencing.md)'s dotted edge from M2 to M5
is exactly this: the portable tiled GEMM's uniform needs a generated codec, so
M5 is behind M2 rather than merely after it.

## 2. What it builds

- **The std140 layout algorithm** over the Go struct, walked with `go/types`,
  assigning each field the offset [001](001-device-resources.md) §3.3's table
  gives it. The Go struct's own field offsets are irrelevant to the device
  layout and are never assumed to match it.
- **A generated per-kernel encoder and decoder** for the exact Go struct type,
  plus its block size. The caller's struct declares no padding fields.
- **`UniformBuffer[T]`**, which owns an ordinary uniform buffer and whose
  `Write` encodes `T` into it through the queue, so values may change between
  submissions without changing graph structure.
- **Typed bindings**: a generated bindings struct whose `Bind` checks every
  field and returns ordinary resource bindings.
- **Validation against the device**: a struct whose encoded size exceeds
  `Limits.MaxUniformBlockBytes` is a pipeline-creation error naming the struct,
  the encoded size, and the device's limit.
- **Generate-time rejection** of every type [001](001-device-resources.md) §3.3
  forbids in a uniform struct, naming the struct, the field, and the reason.

An unsafe cast from a Go struct to uniform bytes stays rejected. It is silently
correct for a struct of four floats and silently wrong for the first one
containing a three-component vector, which is the worst possible failure
distribution: it works until the first person trusts it.

## 3. The uniform buffer's dtype

A std140 block has no scalar dtype and `BufferDescriptor` requires one, so a
uniform buffer is declared `DType: U8` with `Count` equal to the encoded block
size in bytes. On a uniform binding, dtype means bytes rather than elements.

This is one of exactly two exceptions in the design, the other being a vertex
buffer, and [001](001-device-resources.md) §3.3 states both. Recording it here
too because the generated code is where it becomes visible.

## 4. Testing

- **The device check, not the encoder check.** A kernel reads a uniform struct
  containing a scalar, a three-component vector, a scalar occupying that
  vector's tail, and a 4x4 matrix, writes each to a distinct storage buffer
  element, and the host asserts the values. This is the required case from
  [001](001-device-resources.md) §11.2 and it runs on every backend, because
  std140 agreement is exactly the kind of thing that holds on three backends and
  not the fourth.
- Host-side encode/decode round-trips structs containing scalar, vector, and
  array fields, which is necessary and not sufficient: it is the check that
  cannot see an encoder agreeing with itself.
- A uniform struct containing a forbidden type (`bool`, `int`, `float64`, an
  unexported field, an array of structs) fails at generate time naming the
  field.
- An array of 64 floats occupies 1024 bytes rather than 256, asserted directly,
  because that padding is the reason arrays belong in storage buffers and a
  reader who has not seen the number does not believe it.
- A kernel whose loop bound comes from a uniform lowers with the bound uniform,
  which is the property the storage-buffer substitute would lose.
- A struct exceeding `MaxUniformBlockBytes` is rejected naming the struct, the
  size, and the limit, checked against a mimicked profile with a small limit so
  the path does not wait for hardware that has one.

## 5. Open question

- **Whether uniform buffers should exist at all.** Carried from
  [001](001-device-resources.md) §10 rather than resolved here, because this
  child is where the cost becomes concrete: a second layout convention and a
  second alignment, for a payload usually under 256 bytes. The argument for
  keeping them is the constant cache, which is a real win for values every
  invocation reads. A measurement showing otherwise would simplify both specs
  considerably, and this is the first milestone that could take it.
