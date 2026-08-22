---
title: "Kernel pipeline: the generator, the IR, and one kernel end to end"
status: drafted
layer: device
depends_on:
  - 002-compute-model.md
  - 004-kernel-authoring.md
  - 011-conformance-harness.md
---

# Kernel pipeline

The first of [009](009-sequencing.md)'s three M2 children. It builds the whole
compiler pipeline for the smallest kernel that is still a real one, so that
every later child grows a working thing rather than assembling one.

[004](004-kernel-authoring.md) owns the decisions. This spec does not reopen the
node set, the intrinsic table, the tool placement, or the corpus boundary; it
says what it takes to make them run once, and what counts as proof.

## 1. Why the smallest kernel and not the front end

The obvious split is front end then back end: build `go/types` loading, subset
validation, and the IR, then build the lowering that consumes it. It was
rejected, and the reason generalizes.

**A milestone's evidence must be able to fail.** A front end delivered alone can
only be evidenced by a golden of the IR it produced, and a wrong IR passes its
own golden. That is the same argument [011](011-conformance-harness.md) §6 makes
for why the generated lowering is compared against the authored Go function:
when one artifact is derived from another, comparing the pair to itself proves
nothing about either.

The repair for a front-end-first split is to give it a second, independent
consumer of the IR to check the first against. That consumer is an interpreter
over the typed IR, which is the direct flat executor. Pulling it in makes the
split this one. So the vertical cut is not a scheduling preference; it is where
the horizontal cut lands once its evidence has to be real.

## 2. Scope: the kernel this child must compile and run

One entry function, whose body is straight-line arithmetic and `if`:

```go
//accel:kernel workgroup=64
func Scale(t accel.Thread, in []float32, out []float32) {
	i := t.GlobalID().X
	if i < uint32(len(out)) {
		out[i] = in[i] * 2
	}
}
```

The first parameter is `accel.Thread` and the ids come off it, per
[004](004-kernel-authoring.md)'s binding table. An earlier draft of this spec
wrote `accel.GlobalID()`, which is not the authored spelling.

Everything needed to take that from source to a checked result:

- **The tool.** `cmd/accel-kernel`, with the reusable compiler in
  `internal/kernelc`, invoked under `go generate`. `golang.org/x/tools/go/packages`
  becomes a build-tool dependency; neither the root package nor a deployed binary
  imports it, and a test asserts that.
- **The front end.** Package loading, type checking, and subset validation over
  the constructs above, with every rejection positioned.
- **The IR.** [004](004-kernel-authoring.md)'s closed node set, restricted to
  what the body above needs: constants, parameters, locals, indexing, binary
  operations, explicit conversion, block, local declaration, assignment, and
  `if`. The remaining nodes are declared and unreachable, because the set is
  closed by design and adding to it later is what this child exists to prevent.
- **The intrinsic table.** Keyed by the identity `go/types` resolves, which for
  a method is (package path, receiver type name, method name); see §3. It
  carries opcode, uniformity effect, capability requirement, numeric class, and
  target lowering. At this scope it holds the thread-id accessors and nothing
  else, and it rejects a same-named method on any other type and a same-named
  function in any other package.
- **The generated artifacts.** The flat Go lowering, the `Kernel` record with
  binding metadata and inferred access modes, registration, the source digest,
  and the generator/IR ABI version.

  The bound arguments are validated **once against the declared bindings before
  the invocation loop**, not per invocation. The signature is the binding
  layout, so a dtype mismatch is something generation already proved; checking
  it again inside the loop would reopen at runtime what the compiler settled,
  and would report it 64 times instead of once with the binding's name.

  The digest's preimage is versioned and line-oriented, and it carries a helpers
  section that is empty in this child. [013](013-kernel-subset.md) adds lines to
  it rather than changing its format, which it would otherwise have to do while
  also reissuing every committed generated file.
- **The direct flat executor.** Test infrastructure and compiler bring-up, not a
  public submission API: it invokes the generated adapter over independent
  invocations with no `Graph`. Its restricted descriptor rejects shared
  parameters and every cooperative intrinsic, so a kernel that should not run
  here cannot.
- **Freshness.** Editing a kernel without regenerating fails a check naming it.

### Explicitly not in this child

`for` in any form, `break`, `continue`, helper functions, and the full scalar
type set are [013](013-kernel-subset.md). Uniform structs and std140 are
[014](014-kernel-uniforms.md). Shared memory, barriers, atomics, and subgroups
are M4 and are rejected here with a position rather than ignored. No GPU target
is emitted: MSL arrives with Metal at M6, and emitting an artifact no compiler
consumes would be golden output nobody can falsify.

## 3. Where `Thread` lives, and what that does to the table

`accel.Thread` and `accel.ID3` are aliases of types in `internal/kernel`. This
is not a style choice, it is forced by who has to construct a `Thread`.

At M3 and after, the CPU backend executes generated kernels, and the CPU backend
is `internal/cpu`, which cannot import `accel` because `accel` links it in. A
`Thread` therefore has to be constructible from below `accel`, which means the
type lives below `accel` and the root package aliases it. The authored spelling
stays `accel.Thread`, a kernel author never names an internal package, and the
generated code in an author's package refers to `accel.Thread` while the
`Kernel` record's entry point is typed `kernel.Thread`. The alias makes those
one type, which is what lets the two compose at all.

It has one consequence that has to be decided rather than discovered.
`go/types` reports a method's package as the *resolved* one, so `t.GlobalID()`
resolves to `internal/kernel`, and [004](004-kernel-authoring.md)'s "functions
or methods in the root `accel` package" is no longer literally where the object
lives. So:

- **Resolution keys on the resolved identity**, which for a method is (package
  path, receiver type name, method name). The receiver type name is not
  optional: without it the key is (package path, method name), which is the
  predecessor's bare-name bug wearing a package prefix, and a user type with its
  own `GlobalID` method would lower to the builtin.
- **The digest records the authored spelling**, `accel.Thread.GlobalID`. Keeping
  the two independent is what stops a later relocation of `Thread` from
  invalidating every committed digest, and a relocation is exactly the kind of
  thing that happens when M4 grows the rendezvous state.

## 4. The intrinsic table is not separable, and why

It would be natural to defer the table until there are intrinsics worth
tabulating. It is here instead because the failure it prevents is
indistinguishable from working code.

The predecessor keyed its builtin table by bare name, so any user function
called `Dot` or `Mix` lowered to the GPU builtin. Nothing errors. The kernel
compiles, runs, and computes something else. Building the resolution correctly
once, when the table has one entry, costs nothing; retrofitting it after a
name-keyed table has shipped means auditing every kernel written against it.

The same reasoning is why this child carries the rule
[004](004-kernel-authoring.md) states about rejections being the checker's and
never the parser's: a front end that inherits an upstream tool's refusals has an
unstated dependency on that tool's release, and Go 1.27's generic methods are
the live example.

## 5. Testing

**Level 5 from [004](004-kernel-authoring.md) §Testing is mandatory here**, not
deferred. The authored `Scale` is called directly over the same buffers and
compared against the generated lowering, under [008](008-numerics.md) rather
than as bits, because the lowering emits an explicit rounding point at every
operation and the authored function does not. Without it a mistake in IR
construction produces a lowering that is wrong identically everywhere and agrees
with itself.

- generated flat `Scale` accepts slice parameters and matches an independent
  reference through the direct executor;
- the authored function and the generated lowering agree under the class
  contract;
- one positioned negative test per construct this child rejects, including a
  generic method, a `for` loop (out of scope for this child but in scope for the
  next, so the message says out of scope for now rather than unrepresentable),
  and a same-named function shadowing an intrinsic;
- editing a kernel without regenerating fails freshness, naming the kernel;
- the root package's import graph contains no `go/packages`; and
- E2E: source package → generator → registered adapter → direct CPU execution →
  independently checked output.

Harness increment from [011](011-conformance-harness.md): generated-kernel
discovery, source-position negative assertions, the exact and primitive-bounded
comparison contexts, generated-source freshness, and the flat direct-execution
adapter.

**The rejection corpus needs no second module and no `replace`.**
`packages.Config.Overlay` creates packages that do not exist on disk, inside
this module, resolving imports of the real `accel` and type-checking normally,
and one `packages.Load` takes every case at once. That matters because
[013](013-kernel-subset.md) makes the corpus the executable form of the subset:
a corpus that costs a module resolution per case is one nobody runs, and one
that needs a toolchain per case cannot run where the rest of the suite does.

## 6. Open questions

- **Whether the direct executor should survive M3.** [009](009-sequencing.md)
  says it disappears behind the common harness once graphs exist. Keeping it
  would give level 5 a path that does not depend on graph planning being
  correct, which is a real property when a planning bug and a lowering bug look
  alike. Leaning toward keeping it as harness-internal and never public.
- **Whether the IR carries the authored source position for constant-folded
  expressions.** Folding at IR construction loses the position a diagnostic
  would name; not folding leaves the target compilers to do it. Not decided,
  and it only becomes visible when a folded expression is the thing rejected.
