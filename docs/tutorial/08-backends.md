# 8. Backends

**One thing:** choosing where work runs, and testing code for hardware you do
not have.

## Seeing what is there

```go
e := accel.Enumerate()
for _, info := range e.Devices {
	fmt.Println(info.Backend, info.Name)
}
for _, d := range e.Diagnostics {
	fmt.Println("rejected:", d) // why an adapter did not qualify
}
```

The diagnostics matter more than they look. An adapter that fails to qualify
says so, with the stage it failed at — rather than being absent and leaving you
to guess whether the machine has no GPU or accel could not open it.

## Opening one

```go
dev, err := accel.OpenBest(accel.Policy{
	Prefer: []accel.Backend{accel.BackendMetal},
})
```

**`OpenBest` never picks the CPU backend unless you set `Policy.AllowCPU`.** On
a machine with no GPU you get an error, not a slow success. A device you asked
for is never silently substituted — which is the behaviour you want the first
time a deployment box turns out to have no GPU.

When you want the CPU deliberately, say so: `accel.OpenCPU(accel.CPUOptions{})`.

## Requiring a capability

Some kernels need something not every device has. Ask before you dispatch:

```go
if !dev.Capabilities().Has(accel.CapAtomicFloatAddStorage) {
	// choose a different kernel, or a different device
}
```

Or make it a selection criterion, so a device that cannot run your kernel is
never chosen:

```go
dev, err := accel.OpenBest(accel.Policy{Require: accel.CapSubgroupArithmetic})
```

Absence is always explicit. A device lacking a capability produces an error
naming it, never a wrong answer.

## The CPU backend is not a fallback

It is a full implementation and the reference the others are checked against.
Which means:

- **`go test ./...` works on any machine.** No GPU, nothing to provision.
- **It catches what hardware hides.** Reading shared memory nothing wrote gets
  you a poison value and a diagnostic, not convenient zeroes.
- **It is the oracle.** Every GPU backend is verified against it, which is how a
  lowering bug is caught on a laptop instead of in production.

## Testing honestly

If you have a Mac and run `go test ./...`, the Metal tests **skip** when no
adapter is found, and say so only in the skip message. A green run is not
necessarily a GPU run.

```sh
ACCEL_REQUIRE_METAL=1 go test ./...   # turn that skip into a failure
```

Do this in CI. A test suite that quietly skips its GPU half misleads you about
how proven your code is.

## Cross-compiling

```sh
CGO_ENABLED=0 GOOS=linux GOARCH=arm64 go build ./...
```

No cgo anywhere, so this works for every `GOOS` without a toolchain on the build
machine. That is the point of the project.

## Try it

- Run with no GPU and `OpenBest(accel.Policy{})`. Read the error.
- Print `dev.Capabilities().Set()` on the CPU backend and on Metal.
- Build for `linux/arm64` from your laptop.

---

That is the tour. The [package documentation](https://pkg.go.dev/golang.design/x/accel)
is the reference; [`specs/`](../../specs/) is why things are shaped as they are.
