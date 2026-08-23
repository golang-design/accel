# 2. Writing a kernel

**One thing:** what you may write inside `//accel:kernel`, and why the subset
is small.

Say you are brightening an image and want the amount to be a runtime value.

```go
// kernels/kernels.go
package kernels

import "golang.design/x/accel"

type Adjust struct{ Amount float32 }

//accel:kernel workgroup=64
func Brighten(t accel.Thread, p Adjust, in []float32, out []float32) {
	i := t.GlobalID().X
	if i < uint32(len(out)) {
		v := in[i] + p.Amount
		if v > 1 {
			v = 1
		}
		out[i] = v
	}
}
```

That is ordinary Go. It is also checked by the Go compiler before accel sees it,
which is the point of writing kernels this way rather than in strings.

## What the parameter types mean

Four spellings, four meanings, no annotations:

| You write | It is |
| --- | --- |
| `accel.Thread` | the invocation's identity, always first |
| `[]T` | a storage buffer |
| `T` where `T` is a struct | a by-value uniform — see [tutorial 6](06-uniforms.md) |
| `*[N]T` | workgroup-shared memory — see [tutorial 5](05-cooperation.md) |

## What you may write in the body

Arithmetic, comparisons, `if`/`else`, `for`, local variables, indexing, calls to
`//accel:helper` functions in the same package, and the scalar maths in
[`kmath`](https://pkg.go.dev/golang.design/x/accel/kmath).

Not: slices of slices, maps, channels, goroutines, closures, recursion, `defer`,
`panic`, string operations, or allocation. A GPU has no heap and no stack to
unwind, so these are not restrictions accel invented.

Anything outside the subset is rejected **when you run `go generate`**, with the
line and a reason — never at pipeline creation, and never at dispatch. If you
ever see a kernel error that does not name a line, that is a bug worth filing.

## Narrow types carry no arithmetic

`accel.Float16` and `accel.BFloat16` are storage. They have no `+`:

```go
acc := f.F32() + 1     // widen, then work in f32
out[i] = accel.ToFloat16(acc)
```

This is deliberate and it is Go doing the enforcing, not a rule to remember.
Accumulating in f16 loses precision in a way that shows up as a slightly wrong
answer rather than an error, so the type makes it not compile.

## Regenerating is not optional

The generated file is checked in, and CI fails if it is stale. After editing a
kernel, run `go generate ./...` — the same command, every time.

## Try it

- Add `v := someMap[i]` and run `go generate`. Read the error; it names the
  construct and the line.
- Give the kernel a second uniform struct. The generator will tell you where the
  index goes.

---

Next: [memory](03-memory.md) — where buffers come from, and who closes them.
