---
title: "The PTX target"
status: drafted
layer: device
depends_on:
  - 001-device-resources.md
  - 002-compute-model.md
  - 004-kernel-authoring.md
  - 006-backends.md
  - 008-numerics.md
  - 022-msl-target.md
---

# The PTX target

The analogue of [022](022-msl-target.md) and [038](038-spirv-target.md): a third
GPU emitter over the same [004](004-kernel-authoring.md) IR.

[000](000-decisions.md) named this as the reason CUDA was out of v0 — *"PTX or
cubin generation is a target [004](004-kernel-authoring.md) does not have"* — and
it is worth saying which of the two, because they are not the same spec. **PTX,
never cubin.** A cubin is `ptxas` output, so shipping cubins puts a toolkit
binary in the runtime story, which [006](006-backends.md) §2.3 already rejected
for glslang: *"fine for a proof and not shippable"*. PTX is JIT compiled by the
driver, measured at 10.1 ms cold and 63 µs warm in
[060](060-cuda-bringup.md) §1, so the target needs nothing on the machine that
the NVIDIA driver did not already install.

**This is the easier of the two remaining emitters, and 038's difficulty table
is the reason.** SPIR-V is 004's only target row with *Source level: no*, and
every check it lost, PTX keeps:

| Lost by SPIR-V with the binary format | PTX |
| --- | --- |
| read the emitted artifact in review | **it is the golden**, §8 |
| a regex over emitted text in a test | works |
| the driver compiler rejects bad source | the JIT rejects it, with a log |
| a hand-written probe kernel | hand-written, §7 |
| a Go-side round-trip to catch encoding bugs | there is no encoding |
| a vendored `spirv.core.grammar.json` and generated opcode tables | there are no opcode numbers |

Two more properties cut the same way. **PTX has unbounded virtual registers**, so
there is no allocator and no spilling decision in the emitter — `ptxas` does
that. And **PTX has no structured-control-flow requirement**, so 038 §2's
two-phase assembly and 038 §3's merge instructions have no analogue: the emitter
walks the IR once and writes labels.

What is *harder* than MSL is exactly one thing, and it is §7.

## 1. The three header directives

| Directive | Value | Why |
| --- | --- | --- |
| `.version` | **6.2**, raised only by an instruction that needs more | Measured, not read off a table: at 6.0 and 6.1 the JIT rejects the baseline warp kernel with *"Feature 'activemask' requires PTX ISA .version 6.2 or later"*, and 6.2 through 8.0 all accept it. A newer driver accepts an older `.version`; the reverse is a JIT error, so the floor is what the corpus needs and never what the developer's driver has. |
| `.target` | **`sm_70`**, derived, §6 | Independent thread scheduling, and therefore the point below which 002 §5.3's active-set semantics are not expressible. |
| `.address_size` | **64** | `CUdeviceptr` is 64-bit and [060](060-cuda-bringup.md) §3 gates the backend to 64-bit hosts. |

`.target` is a **virtual** architecture as far as this emitter is concerned: the
driver JIT compiles it for the physical device, and
[060](060-cuda-bringup.md) §1 measured `sm_70` PTX running on `sm_121`. So there
is no per-architecture artifact matrix. What `.target` still decides is which
instructions are legal, which is §6.

The entry point is `.visible .entry <kernel name>`, and the name is the
**kernel's**, not a convention. [038](038-spirv-target.md) §1 records what the
alternative costs: glslang always emits `main`, the predecessor hardcoded it, and
every pipeline-creation error was nameless.

`.reqntid <x>, <y>, <z>` is emitted from `ir.Func.Workgroup`, not `.maxntid`.
`.maxntid` is a hint that lets `ptxas` size register allocation; `.reqntid` is
the same hint plus a launch-time refusal of any other block shape. **Measured**:
a kernel declaring `.reqntid 64, 1, 1` launches at 64 threads and is refused at
32 with `CUDA_ERROR_INVALID_VALUE`. That refusal is
[038](038-spirv-target.md)'s `LocalSize` failure caught by the hardware — 004
cites the predecessor's own `polyred/gpu/shader/compile.go:652` for it — and it
costs nothing. [060](060-cuda-bringup.md) §7 still checks at `Compile`,
because `cuFuncGetAttribute` reports the product and not the triple.

## 2. The emitter is single-pass

038's opening problem does not exist here. PTX is assembled top to bottom, a
label may be referenced before it is defined, and there is no annotations section
that must precede the instruction it decorates. So the emitter walks `ir.Func`
once and writes text, the way `emit/msl.go` does.

**Determinism is still the constraint the golden depends on.** Virtual register
numbers and label numbers come from a monotonic counter in first-seen order over
a deterministic IR walk, because a counter driven by Go map iteration emits a
different, equally valid module every run and destroys the golden. This is
[038](038-spirv-target.md) §2's rule with the result ids renamed.

One structural difference from MSL: PTX declares its register *counts* up front
(`.reg .b32 %r<N>;`), and `N` is known only after the body is lowered. The body
is therefore built into a buffer and the declarations are prepended — a
formatting concern, not 038's phase problem, and stated so nobody reintroduces
a two-pass walk to solve it.

**Register pressure is a cost, not a correctness question.** The emitter can
declare a million registers and `ptxas` will allocate physical ones and spill the
rest to local memory. The symptom is occupancy, which is a
[010](010-kernel-corpus.md) tuning matter and never a wrong answer. Said plainly
so that no one adds a reuse pass to the emitter for a problem `ptxas` owns.

## 3. Control flow is labels, and `continue` still has one trap

| IR node | PTX |
| --- | --- |
| `ir.If` | `setp.<cmp>` into `%p`, `@!%p bra <else-or-join>`, `bra <join>` |
| `ir.For` | `<head>:` test, `@!%p bra <merge>`, body, `<post>:` `Post`, `bra <head>`, `<merge>:` |
| `ir.Break` | `bra <merge>` |
| `ir.Continue` | `bra <post>` |

**`continue` branches to the post block, never to the head.** Branching to the
head skips `Post`, so `for i := 0; i < n; i++ { … continue … }` spins forever on
a device with no watchdog. This is [038](038-spirv-target.md) §3's bug in a
target that will not catch it: `spirv-val` at least demands a structured CFG, and
`ptxas` accepts either branch without comment. The check is §9's golden and a
device test with a timeout.

**The cooperative lowering is not re-emitted.** CUDA has real barriers, so a CUDA
kernel is the *authored* structure, never
[018](018-cooperative-lowering.md)'s resumable state machine. Both come from one
IR: the CPU runs a program counter and CUDA runs `bar.sync`, and they must still
agree.

## 4. Locals are registers, and they are initialised

Unlike [038](038-spirv-target.md) §4, where every `ir.Local` is a Function-class
`OpVariable` in memory, a scalar `ir.Local` is a virtual register and assignment
is `mov`. PTX registers are freely assignable, so there is no `OpPhi`, no
dominance frontier and no hoisting rule.

Two locals do not fit a register and go to `.local` memory: an array, and
anything whose address is taken. `ptxas` places `.local` in per-thread global
memory, so this is a performance cliff worth naming rather than a correctness
one.

**Every local carries an initialiser**, for the reason 002 §2.2 gives against
SPIR-V by name and which applies unchanged: an unwritten PTX register holds
whatever was there, so the Go path reads 0, the GPU path reads garbage, and the
oracle silently stops being an oracle. The emitter writes the Go zero value.

**`.shared` arrays carry none.** 002 §2.2 requires shared memory to be undefined
at workgroup start, and zeroing it makes a missing barrier produce plausible
results here and garbage elsewhere. A `.shared` declaration with an initialiser
is §9's most valuable injected fault, because it *passes*.

## 5. Parameters, bindings, barriers, atomics, warp operations

### Parameters

[038](038-spirv-target.md) §5's slot numbering is preserved, and
[060](060-cuda-bringup.md) §7 maps slot *k* to launch parameter *k*:

| Slot | `.param` declaration | Read by |
| --- | --- | --- |
| `0 … n-1` | `.param .u64` | `ld.param.u64` then **`cvta.to.global.u64`** |
| `n` | `.param .align 4 .b8 lengths[4n]` | `ld.param.u32 [lengths+4k]` |
| `n+1+i` | `.param .align 16 .b8 uniform<i>[size]` | `ld.param.<t> [uniform<i>+off]` |

`emit.MSLLengthsIndex(n)` and `emit.MSLUniformIndex(n, i)` carry the numbering
(`msl.go:54-58`). They are MSL-prefixed because MSL was the only target when they
were written; [038](038-spirv-target.md) §5 proposes the rename that makes them
shared and this target is the second caller that makes it worth doing. **Three
copies of a numbering is two too many.**

**`cvta.to.global` is not an optimisation.** A `.param` pointer is a generic
address; `ld.global` applied to a generic address is undefined, and it usually
works, which is how it survives. Every binding pointer is converted once at entry
and the converted register is what the body uses.

**The lengths block is scalar `u32`s at 4-byte offsets**, and here PTX is simpler
than both siblings: `.param` space has no std140, so there is no stride-16 array
trap and no padding to reproduce. The slot is reserved whether or not any body
calls `len`, and the host fills entry *k* as

$$\texttt{len}_k = \left\lfloor \frac{\texttt{bindings}[k].\texttt{Len}}{\texttt{sizeof}(\texttt{dtype}_k)} \right\rfloor$$

**Uniform blocks reuse `internal/kernelc/std140` offsets even though `.param`
space does not require them.** Natural packing would be tighter and would fork
the host fill three ways; one layout owner is worth the padding.

**The padding is not free, and the budget is smaller than either documented
number.** Measured on this driver by bisection, the whole parameter space of an
entry function is **4352 bytes** (`ptxas` reports `0x1100 max`), and it does not
move with `.target` — sm_70, sm_80 and sm_90 all report the same ceiling. So the
sum

$$8n \;+\; 4n \;+\; \sum_i \texttt{std140size}(\texttt{uniform}_i) \;\le\; 4352$$

is a real constraint the emitter checks, not a formality: *n* pointers, *n*
lengths, and every uniform block with its std140 padding. Neither of the values
the documentation offers — 4 KB below CUDA 12.1, 32,764 above it — is this
number, which is why it is measured here and asserted in §10.

### Barriers

Transcribed from 002 §2.5 and [050](050-barrier-scopes.md), not chosen here.

| Call | PTX | Note |
| --- | --- | --- |
| `Barrier` | `bar.sync 0` | control plus a CTA-wide memory fence |
| `BarrierShared` | `bar.sync 0` | **the same instruction** |
| `BarrierStorage` | `bar.sync 0` | **the same instruction** |
| `SubgroupBarrier` | `bar.warp.sync <mask>` | mask per below |

050's three spellings collapse to one on this target, because PTX has no
scope-narrowed control barrier. That is legal — stronger than required is always
legal — and it is **recorded rather than silently enjoyed**, because it means
CUDA cannot catch a kernel that says `BarrierShared` and needs `BarrierStorage`.
Metal and the CPU oracle can.

### Atomics

**Scope is emitted explicitly and never defaulted.** `atom.global.gpu.add.u32`
for a storage atomic and `atom.shared.cta.add.u32` for a shared one, per 002
§4.2's storage-class rule. PTX's default scope happens to be `.gpu` today, and
[038](038-spirv-target.md) §5 states what relying on that class of default costs:
a storage atomic at workgroup scope returns the right total on one machine and
loses counts on another, while 002 §4.3 promises a dispatch-wide counter reaches
exactly *n*.

Semantics are relaxed, per 002 §4.2, which is `atom`'s own ordering.

`atom.cas.b32` is **strong**, so like SPIR-V and unlike MSL — which needs
`mslPrelude`'s `_accel_cas_*` helpers for the weak form Metal offers — this
target needs no CAS helper. Recorded, not silently enjoyed.

**Native f32 atomic add.** `atom.global.add.f32` since sm_20 makes CUDA the first
backend where [006](006-backends.md) §3's *Atomic float add* row is `yes` rather
than `cap`, in both storage and shared.

### Warp operations

`compute.go:126-135`'s capability split maps one to one, and the widths differ
from every other target:

| `Capability` | PTX | Width |
| --- | --- | --- |
| `CapSubgroupBasic` | `activemask.b32`, `%laneid` | — |
| `CapSubgroupVote` | `vote.sync.any.pred`, `vote.sync.all.pred` | — |
| `CapSubgroupBallot` | `vote.sync.ballot.b32` | **32 bits** |
| `CapSubgroupShuffle` | `shfl.sync.{idx,up,down,bfly}.b32` | — |
| `CapSubgroupArithmetic` | butterfly `shfl.sync.bfly` loop, §6 | — |

`accel.KernelMask` being opaque pays here for the second time. SPIR-V is the
target that would have broken a `uint64` upward at 128 bits; CUDA is the one that
would have made a fixed 64 look half-empty at 32.

> **The member mask is `activemask.b32`, never a literal `0xffffffff`.**

This is the sharpest single line in the target. 002 already names the mechanism
at line 418 — independent thread scheduling on Volta and later — and the
consequence is that on `sm_70`+ a `.sync` operation whose mask names lanes that
are not converged is **undefined**. A literal full mask is correct exactly while
control flow is uniform, which is every simple test, and 002 §5.3 explicitly
makes a subgroup operation in divergent control flow *legal* with semantics over
the active set. So the emitter computes the mask and the divergent case is the
one the corpus must cover.

The **portable subset does not widen** because CUDA exposes more than MSL, for
[038](038-spirv-target.md) §5's reason: what 002 §5.1 makes unportable is the
active set, not the control flow reaching it.

## 6. `.target` is derived, never listed

Hand-listing an architecture per kernel is two sources for one fact, which 002
§8.2 forbids. `.target` is `max(sm_70, what the body's instructions force)`, from
`ir.Func.Caps` plus what the types force:

| What the body reaches | Floor | Above the baseline? |
| --- | --- | --- |
| `.sync` warp primitives, independent thread scheduling | sm_70 | the baseline itself |
| f16 arithmetic (`add.rn.f16`, `fma.rn.f16x2`) | sm_53 | no |
| `dp4a` packed 8-bit dot product | sm_61 | no |
| bf16 storage conversions | none — integer shift/mask | no |
| `redux.sync` integer warp reductions | sm_80 | **yes** |
| bf16 arithmetic | sm_80 | refused by 006 regardless |

Only one row is above the baseline, and **this target does not take it**.
[059](059-subgroup-reductions.md)'s reductions lower to a
$\log_2(\text{warp})$ butterfly of `shfl.sync.bfly` on every architecture,
including the integer ones `redux.sync` would do in a single instruction at
sm_80. One artifact per kernel, at one `.target`, is the v0 shape; a second
artifact at a higher `.target` is a *variant*, which is
[010](010-kernel-corpus.md)'s mechanism and not a second emitter path bolted on
here.

**bf16 is emitted as storage, and the reasoning is 038's exactly.** 008 §4's
conversions are `u16` shift and mask with round-to-nearest-even, which is integer
arithmetic no `.target` gates. Arithmetic stays f32 and `CapBF16Arithmetic` is
still refused.

## 7. The numeric contract, which is the only place this target is harder than MSL

PTX splits every transcendental into an IEEE-rounded form and an `.approx` form,
and the split does not line up with 008 §6's ceilings. Two rows are met by
construction and are *better* than what Vulkan guarantees; the rest are not met
at all by the fast form.

| Primitive | PTX | 008 §6 ceiling | Verdict |
| --- | --- | --- | --- |
| `Sqrt` | `sqrt.rn.f32` | ≤ 1 representable step | **met — measured**, 0 of 1024 samples differ from `float32(math.Sqrt(float64(x)))` |
| division | `div.rn.f32`, correctly rounded by the ISA | 2.5 ULP | met by construction; unmeasured |
| `InverseSqrt` | `rsqrt.approx.f32` | 4 ULP | ? measure |
| `Exp` | `ex2.approx.f32` composed with a scale | 4 ULP | ? measure — the scale adds error the instruction's own bound does not cover |
| `Log` | `lg2.approx.f32` composed with a scale | 4 ULP | ? measure |
| `Tanh` | `tanh.approx.f32`, sm_75+ | 4 ULP | ? measure, and it is above the baseline `.target` |
| `Sin`, `Cos` | `sin.approx.f32` / `cos.approx.f32` | 2⁻²⁰ absolute over \|x\| ≤ 2¹⁶ | **not met, and not by range reduction alone** |

`sqrt` and division being correctly rounded is the one place where CUDA is
*stricter* than the ceiling and Vulkan is looser (038 §7 has `sqrt` at ~3 ULP).
Recorded because the natural instinct is to reuse 038's answer wholesale, and
`sqrt` is measured rather than taken from the ISA document for the same reason.

**`sin` and `cos` are generated, not called.** This is 038 §7's conclusion for
the same reason: reduction alone fixes only the *range*, and the hardware
approximation is short of 2⁻²⁰ even on a reduced argument. The emitter writes a
Cody–Waite reduction with a three-part π/2 split and then a minimax polynomial,
which meets the ceiling by construction. **A ceiling is never widened to match
what a device reported.**

The other four rows take 038 §7's three-answer structure — generate, measure and
record a profile, or refuse the primitive naming the measured ULP — and which
answer applies to which primitive is decided *after* the probe, not now.

### Contraction, which this target does not have to fight

MSL needs `precise::`; SPIR-V needs a `NoContraction` decoration per result id
and a test that counts decorations against decorable results. **PTX needs
neither, and the measurement is stronger than the argument that was written
here first.**

The argument was that an explicit rounding modifier forbids contraction, since
fusing would change the rounding the instruction names. That is true and it is
not what is doing the work. Measured, with `a = 1 + 2^{-12}` and
`c = -(1 + 2^{-11})`, chosen so that a separately-rounded multiply-then-add is
exactly 0 and a fused one is exactly $2^{-24}$:

| Emitted | Result | Fused? |
| --- | --- | --- |
| `mul.rn.f32` then `add.rn.f32` | 0 | no |
| **`mul.f32` then `add.f32`** | **0** | **no** |
| `fma.rn.f32` | 5.9604645e-08 | yes |

**`ptxas` does not fuse a separate multiply and add in PTX at all**, with or
without the modifier, at the JIT's default optimisation level. Contraction in
the CUDA toolchain is a *front-end* decision — `nvcc`'s `--fmad`, applied while
generating PTX — and for this target accel's emitter **is** the front end. So
008 §3's contraction condition is met because the IR decides, one instruction at
a time, and there is nothing downstream to override it.

The rounding modifiers stay, as the guard rather than the mechanism: they cost
nothing and they are what makes the property hold on a future `ptxas` that does
contract. But **the fault injection that would prove it does not exist**:
reinstating `mul.f32`/`add.f32` produces no difference to catch. The honest test
is the positive one — that `fma.rn.f32` appears exactly where the IR fuses and
nowhere else, asserted on the text, and a Tier 4 differential on an expression
where fusion is observable.

This retires [060](060-cuda-bringup.md)'s question about reaching an fmad
control through the driver JIT. There is no contraction to disable.

**Denormals are expressed declaratively.** `.ftz` flushes to zero and is
*omitted*, so f32 denormals are preserved by construction, which is 008 §3's
condition without a probe. It is still probed, because a modifier a driver
ignores is exactly the class of claim this project does not take on documentation.

**The probes are hand-written PTX**, following
`internal/metal/prober_darwin_test.go`. 038 §7 had to hand-*assemble* its probes
because SPIR-V has no source form; here the instrument is a text file a reviewer
reads, which removes the one circularity that section had to argue around. Probes
carry the same explicit rounding modifiers the emitter emits, and
`probe.Ops.MulAdd` evaluates `a*b+c` as one expression, per 022 §1.

## 8. Where the artifact lives

`PTX string` sits beside `MSL string` (`kernel.go:455`), which
[038](038-spirv-target.md) §8 already ratified as the shape, and
`kernel.MissingArtifact(name, target)` covers both.

**PTX is text, so generation writes one file and not three.** 038 needs a `.spv`
plus a `.spvasm` golden plus an embedded SHA-256, because *"a golden of an opaque
binary is a golden nobody reviews"*. PTX goes into `accel_kernels.go` the way MSL
does, subject to the same backquote guard `emit.go:317` already applies to
`mslArtifact`, and **the emitted text is the golden**. A reviewer reads the diff.

There is no vendored grammar, no generated opcode table and no recorded
SPIRV-Headers tag, because PTX has no opcode numbers to transcribe. 038 §8's
machinery exists to avoid the predecessor's hand-transcribed COM vtable indices
([006](006-backends.md) §2.4); this target has nothing of that shape.

## 9. How a malformed module is caught

**The emitter's own checks come first.** 004 requires every rejection to be the
emitter's, positioned in the author's source, and naming the target. `ptxas` and
the JIT report line numbers in generated text, which is the wrong coordinate
system for an author.

**The golden is the second layer, and on this target it is the strong one.** The
inversion of 038 §9's table: there, the rows that only decoded assertions catch
are the argument for having that layer. Here the artifact is readable, so a text
golden catches the same rows.

| Injected fault | golden diff | `ptxas` | on the device |
| --- | --- | --- | --- |
| `.reqntid` hardcoded 1,1,1 | **yes** | no | wrong results, plausible |
| `continue` branches to the loop head | **yes** | no | hang |
| storage atomic emitted without `.gpu` scope | **yes** | no | passes on one GPU, loses counts on another |
| warp op with a literal `0xffffffff` mask | **yes** | no | passes until control flow diverges |
| a `.shared` array given an initialiser | **yes** | no | **passes**, and hides a missing barrier |
| `mul.f32` instead of `mul.rn.f32` | **yes** | no | **nothing** — measured; `ptxas` does not fuse either spelling, §7 |
| missing `cvta.to.global` | **yes** | maybe | undefined; usually correct |
| a local read before its initialiser | maybe | no | garbage |
| an instruction above the emitted `.target` | no | **yes** | JIT error |
| a malformed operand or an undeclared register | no | **yes** | JIT error |

The last two rows are `ptxas`'s whole contribution, and they are the two the JIT
would also catch — which is why it is a Tier 2 convenience and not a gate.

**`ptxas` is Tier 2 only**, behind an `ACCEL_REQUIRE_PTXAS` promise that skips in
Tier 1 and *fails* in Tier 2 — the variable is the promise, not the capability,
as `.github/workflows/ci-metal.yml` does for Metal. It is never a step of
`go generate`, which Tier 1 runs on every commit with no provisioning, and it is
never *the* check. Provisioning is one pinned line of the CUDA toolkit's compiler
package; the *runtime* still needs nothing, which is the property this whole
target exists to keep.

**The device is the last layer and the only differential one.** Every kernel in
[010](010-kernel-corpus.md) runs against the CPU oracle at Tier 4 on the lab box.
That is where the `activemask` row and the contraction row are actually settled,
because both are correct-looking everywhere else.

## 10. Done

1. **The emitted PTX is byte-identical across two runs and across linux and
   darwin generation** — a register or label counter driven by map iteration,
   which destroys the golden without failing anything.
2. **`.version`, `.target` and `.reqntid` are derived from `ir.Func` and no
   kernel names an architecture** — 002 §8.2's two-sources rule.
3. **Every binding pointer is `cvta.to.global`-converted exactly once at entry**,
   asserted on the text — the undefined access that usually works.
4. **Every `.sync` warp operation's mask operand is a register produced by
   `activemask.b32`**, asserted on the text — the literal full mask, which is
   correct until control flow diverges.
5. **Every storage atomic carries `.gpu` and every shared atomic `.cta`**,
   asserted on the text — the counter that reaches *n* on one machine.
6. **Every `.shared` declaration is uninitialised and every scalar local is
   initialised**, asserted on the text — 002 §2.2 in both directions, and the
   fault that passes.
7. **`fma.rn.f32` appears exactly where the IR fuses and nowhere else, and
   every other f32 multiply and add carries `.rn`** — asserted on the text as
   counts against the IR's own counts, per 038 §7's argument that a
   per-expression test only catches what a corpus happens to reach. The positive
   half is the one that can fail: §7 measured that the negative half has no
   fault to inject.
8. **`continue` targets the post block**, asserted on the CFG the emitter built,
   plus a device test with a timeout.
9. **A kernel using every subgroup family produces the same result as the CPU
   oracle under divergent control flow**, at Tier 4 — item 4's real check.
10. **A contraction-sensitive expression matches the CPU oracle**, at Tier 4 —
    an expression where fusion is observable, which is what item 7's positive
    half is checked against.
11. **`sin` and `cos` meet 2⁻²⁰ absolute over |x| ≤ 2¹⁶** against a high-precision
    reference, at Tier 4 — the generated reduction and polynomial.
12. **A deliberately corrupted artifact is rejected with the JIT's own log
    reaching the caller** — [060](060-cuda-bringup.md) §14 item 9 from this side.
13. **A kernel whose pointers, lengths and std140 uniforms exceed 4352 bytes is
    refused by the emitter, naming the total and the ceiling** — §5's budget,
    which `ptxas` otherwise reports against generated line numbers the author
    cannot read.

## What this target does not build

- **cubins, and `ptxas` on the shipping path.** §0's opening; the runtime
  dependency is the driver and nothing else.
- **`mma` and `wmma`.** Cooperative matrix is deferred by
  [002](002-compute-model.md) and it is the single largest reason a cgo-free GEMM
  will lose to cuBLAS. It is a spec, not a section.
- **`redux.sync` at sm_80**, §6 — a variant, owned by
  [010](010-kernel-corpus.md).
- **`cp.async`, thread-block clusters and distributed shared memory**, all sm_80
  and above, and all of them changes to the compute model rather than to this
  emitter.
- **Dynamic parallelism**, which is the missing indirect-dispatch answer and
  belongs with [060](060-cuda-bringup.md)'s graph child.
- **f64 anything.** 002 has no f64 and this target does not introduce one.

## Open questions

1. **Which of §7's four `.approx` rows are met, generated, or refused.** The
   probe decides, and the probe needs the lab box.
2. **The rest of the `.version` floor.** 6.2 is measured as the floor for
   `activemask`, which is the baseline's binding constraint today. Every *other*
   instruction the emitter reaches has its own introducing version, and the JIT
   names it precisely when it is wrong — so this is a transcription exercise the
   corpus completes, not a risk. It fails on old drivers only.
3. **Whether §5's 4352-byte parameter space fits the widest kernel in**
   [010](010-kernel-corpus.md). The ceiling is now measured and the arithmetic is
   in §5; what is unknown is the widest kernel. If it does not fit, the fallback
   is a uniform *buffer* for the kernels that overflow, which forks the host fill
   exactly where §5 said one layout owner was worth avoiding.
