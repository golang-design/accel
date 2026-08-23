---
title: "The SPIR-V target"
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

# The SPIR-V target

The analogue of [022](022-msl-target.md): a second GPU emitter over the same
[004](004-kernel-authoring.md) IR. Harder for one reason, and the reason is in
004's target table — SPIR-V is the only row with **Source level: no**.

Every check the MSL target leaned on goes with that row, and
`vkCreateShaderModule` is permitted to validate nothing: drivers accept invalid
SPIR-V and then miscompile or hang the GPU. So this spec is the lowering, which
is mostly transcription because the IR was built for it, plus a named
replacement for each lost check.

| Lost with the binary format | Replacement |
| --- | --- |
| read the emitted source in review | committed **disassembly** golden (§8) |
| a regex over emitted text in a test | assertions on the **decoded module** (§9) |
| the driver compiler rejects bad source | `spirv-val`, Tier 2 only (§9) |
| a hand-written probe kernel | hand-**assembled** probe modules (§7) |
| "it compiled, so the encoding is right" | a Go-side **round-trip** (§9) |

## 1. The header words

Three operands sit in the header, and all three change every byte downstream.

| Choice | Value | Why |
| --- | --- | --- |
| version | **1.3** (Vulkan 1.1) | Lowest with the `StorageBuffer` class and `OpGroupNonUniform*` in core. At 1.4 `OpEntryPoint`'s interface list must name *every* statically used global, not only Input/Output; an emitter written to one rule with a header word for the other validates on one toolchain and is rejected on another. |
| memory model | **`Logical GLSL450`** | `Logical Vulkan` needs the `VulkanMemoryModel` capability and `MakeAvailable`/`MakeVisible` operands on exactly the accesses [002](002-compute-model.md) §2.1 simplifies away. accel *follows* the Vulkan model as an ancestor; it does not declare it. |
| word order | **little-endian, always** | 004 requires Linux and macOS generation to produce identical bytes. Host order is not a source of truth. |

A device below Vulkan 1.1 is **refused**, naming both versions — not downgraded,
because subgroup operations have no pre-1.3 core spelling and a downgrade is a
second emitter. The entry point name is the **kernel name**, not `"main"`: that
is a glslang convention the predecessor hardcoded
(`polyred/gpu/backend_vk.go:402`), and it makes every pipeline-creation error
nameless.

## 2. The emitter is two-phase

The obvious emitter writes instructions in walk order. SPIR-V forbids it: the
module has a fixed section layout, and a decoration — including
`NoContraction`, which is per result id — sits in the annotations section
*before* the instruction it decorates exists.

```
 phase 1: lower the body                phase 2: assemble in layout order
┌──────────────────────────┐          ┌───────────────────────────────┐
│ walk ir.Func             │          │ OpCapability      (from §6)   │
│  ├─ intern types      ───┼─────────▶│ OpExtension                   │
│  ├─ intern constants  ───┼─────────▶│ OpExtInstImport "GLSL.std.450"│
│  ├─ pending decorations ─┼─────────▶│ OpMemoryModel Logical GLSL450 │
│  ├─ globals           ───┼─────────▶│ OpEntryPoint GLCompute …      │
│  └─ body words ─────┐    │          │ OpExecutionMode LocalSize     │
└─────────────────────┼────┘          │ OpName / OpDecorate           │
                      │               │ types, constants, globals     │
                      └──────────────▶│ OpFunction … OpFunctionEnd    │
                                      └───────────────────────────────┘
```

**Types and constants are interned**: `OpTypeInt 32 1` declared twice is not two
types, it is a malformed module. **Result ids come from a monotonic counter in
first-seen order**, a deterministic IR walk, which is what makes the artifact
byte-stable as 004's testing level 1 requires. An allocator keyed on Go map
order emits a different valid module every run and destroys the golden.

## 3. Structured control flow is a transcription

This is where 004's rejection of `golang.org/x/tools/go/ssa` pays. go/ssa gives
a general CFG with phi nodes, and recovering structure needs a relooper — which
`spirv-val` then *demands* back ("Selection must be structured", "Loop must be
structured"), so it would exist only to reconstruct what the tree already
carries.

| IR node | SPIR-V |
| --- | --- |
| `ir.If` | `OpSelectionMerge <join> None` + `OpBranchConditional`; `<join>` is the join block the tree names |
| `ir.For` | `OpLoopMerge <merge> <continue> None` in the header; `<merge>` is where `break` goes, `<continue>` is where `Post` runs |
| `ir.Break` | `OpBranch <merge>` |
| `ir.Continue` | `OpBranch <continue>` |

**`continue` branches to the continue target, never to the header.** Branching
to the header skips `Post`, so `for i := 0; i < n; i++ { … continue … }` spins
forever on a device with no timeout — and it passes `spirv-val`, because the CFG
is structured either way.

**The cooperative lowering is not re-emitted.** Vulkan has real barriers, so a
Vulkan kernel is the *authored* structure, never
[018](018-cooperative-lowering.md)'s resumable state machine. Both come from one
IR: the CPU runs a program counter and Vulkan runs a barrier, and they must
still agree.

## 4. Locals live in memory, and they are initialised

Every `ir.Local` is a Function-class `OpVariable` with `OpLoad`/`OpStore`. No
`OpPhi`, no dominance frontier. 004 settles this and states the cost, carried
forward and not re-argued: **the emitted SPIR-V leans on the driver optimizer**.

1. **Every function-scope `OpVariable` is in the entry block.** The naive
   lowering emits one where the Go `var` appears; for a `var` inside an `if` or
   a loop body that is a hard validation error. All locals are hoisted, and
   assignment stays at the declaration point.
2. **Every one carries an initialiser.** 002 §2.2 names this failure against
   SPIR-V by name: the initialiser is optional, so if the emitter omits it the
   Go path reads 0, the GPU path reads whatever the register held, and the
   oracle silently stops being an oracle. **Workgroup-class variables carry
   none**, because 002 §2.2 requires shared memory to be undefined at workgroup
   start and zeroing it makes a missing barrier produce plausible results here
   and garbage elsewhere.

## 5. Bindings, barriers, atomics, subgroups

Each storage binding is `OpTypeStruct{ OpTypeRuntimeArray T }`, `Block`
decorated, member `Offset 0`, `ArrayStride` exactly `DType.Size()` — no padding
ever, per [001](001-device-resources.md) §3.2. Uniform member offsets come from
`ir.Uniform.Fields`, so **nothing here re-implements std140**. Numbering is
descriptor set 0 and matches MSL slot for slot:

| Slot | Descriptor type | Contents |
| --- | --- | --- |
| `0 … n-1` | storage buffer | bindings, in signature order |
| `n` | uniform buffer | the lengths block |
| `n+1+i` | uniform buffer | uniform block `i` |

`emit.MSLLengthsIndex(n)` and `emit.MSLUniformIndex(n, i)` carry the numbering today. Both are MSL-prefixed because MSL was the only target when they were written; sharing them means renaming them, which this spec proposes rather than assumes. They are shared by both emitters
(`msl.go:57-63` exports them), and the descriptor *type* per slot is exported
alongside so the backend builds its set layout without parsing the module back.

**`len(binding)` reads the lengths block, not `OpArrayLength`**, which needs no
slot but reports the *bound descriptor range* over the stride: a whole-buffer
bind to a kernel given a sub-range then answers a different number than MSL and
the CPU do. **The binding count also becomes checkable**, because Vulkan's
guaranteed `maxPerStageDescriptorStorageBuffers` is 4 while
`Limits.MaxBindingsPerKind` (`limits.go:68`) is declared and unread —
`Requirements` and `Device.Missing` each gain one.

Barriers are transcribed from 002 §2.5, not chosen here. Execution and memory
scope are both `Workgroup`; `Device` would be legal and slower, `Invocation`
silently wrong.

| Call | SPIR-V |
| --- | --- |
| `Barrier` | `OpControlBarrier Workgroup Workgroup (AcquireRelease\|WorkgroupMemory\|UniformMemory)` |
| `BarrierShared` | same, `WorkgroupMemory` only |
| `BarrierStorage` | same, `UniformMemory` only |
| `SubgroupBarrier` | `OpControlBarrier Subgroup Subgroup …` |

**Atomics take relaxed semantics (`None`)** per 002 §4.2, with scope following
the storage class: `Device` for a storage-buffer atomic, `Workgroup` for a
shared one. `Workgroup` for both is a one-token change giving per-workgroup
atomicity — on lavapipe the histogram total is still right, on real hardware
counts vanish, and 002 §4.3 promises a dispatch-wide counter reaches exactly
*n*. `OpAtomicCompareExchange` is **strong**, so unlike MSL — which needs
`mslPrelude`'s `_accel_cas_*` helpers for the weak form Metal offers — SPIR-V
needs no CAS helper. Recorded, not silently enjoyed.

Subgroups map one-to-one onto the capability split in `compute.go:126-135`,
which was designed for these families and is not collapsed. `accel.KernelMask` is
opaque precisely because Vulkan's ballot is 128 bits wide; SPIR-V is the target
that would have broken a `uint64`.

| `Capability` | SPIR-V capability |
| --- | --- |
| `CapSubgroupBasic` | `GroupNonUniform` |
| `CapSubgroupVote` | `GroupNonUniformVote` |
| `CapSubgroupBallot` | `GroupNonUniformBallot` (128-bit `uvec4` → `accel.KernelMask`) |
| `CapSubgroupShuffle` | `GroupNonUniformShuffle`, deferred by [020](020-cooperative-atomics.md) |
| `CapSubgroupArithmetic` | `GroupNonUniformArithmetic` |

The **portable subset does not widen** because SPIR-V exposes more than MSL. What
002 §5.1 makes unportable is the *active set*, not the control flow reaching it:
[002](002-compute-model.md) §5.3 is explicit that a subgroup operation in
divergent control flow is **legal**, and defines its semantics in terms of the
active set at that point. Barriers are the ones that are not. So this target
emits `GroupNonUniform*` wherever the IR reaches a subgroup operation, divergent
or not, and every kernel using subgroups still keeps a correct path that does
not — because the *result* depends on an active set no backend agrees about,
which is a different objection from the code being illegal.

## 6. Capabilities are inferred, never listed

`OpCapability` is derived from `ir.Func.Caps` **plus what the types force**;
hand-listing per kernel is two sources for one fact, which 002 §8.2 forbids.

The type-forced half has no row in accel's `Capability` bits: an 8-bit binding
needs `Int8` to declare the type *and* `StorageBuffer8BitAccess` to access it
through a Block; 16-bit is the same with `StorageBuffer16BitAccess`, plus
`Float16` for f16 arithmetic. Declaring the type and letting the driver sort it
out yields a module rejected at pipeline creation on an otherwise capable device.

**bf16 is emitted, and SPIR-V is the only target where it works.** There is no
core bfloat16 type, and Metal refuses the conversions because bfloat is a Metal
family capability. SPIR-V expresses [008](008-numerics.md) §4's conversions
exactly as `u16` shift/mask with round-to-nearest-even, which is integer
arithmetic no capability gates. bf16 stays a *storage* format; arithmetic is f32
and `CapBF16Arithmetic` is still refused.

## 7. The numeric contract

Contraction control is the `NoContraction` **decoration on a result id** — not a
module mode, not an execution mode, and there is no pragma to find. The emitter
decorates every arithmetic result that could participate in a fusion, and the
check belongs **on the module, not on a result**: a test decodes the module and
asserts the decoration count equals the count of decorable arithmetic results. A
result test only catches a miss for the expressions a corpus happens to reach,
which is 020 §6.1's argument about the `3/4` constant bug.

`OpFDiv` meets 008 §6's 2.5 ULP ceiling by construction — that ceiling was taken
from SPIR-V. **The rest are not met by Vulkan's guarantees**, and there is no
`precise::` to escape to. Transcribe Appendix A per primitive before writing the
lowering; every row below is looser than accel's ceiling:

| Primitive | Vulkan's guarantee | 008 §6 ceiling |
| --- | --- | --- |
| `Sqrt` | ~3 ULP | ≤ 1 representable step |
| `Sin`, `Cos` | ~2⁻¹¹ absolute, and only over roughly [-π, π] | 2⁻²⁰ absolute over \|x\| ≤ 2¹⁶ |
| `Exp`, `Log`, `Tanh`, `InverseSqrt` | looser than 4 ULP, each with its own shape | 4 ULP |

Three answers, and each primitive gets exactly one:

| Answer | Applies to | Cost |
| --- | --- | --- |
| emitter-generated implementation | `sin`, `cos` | throughput: reduction plus a polynomial on every call |
| measure the device, record the profile | `sqrt`, `rsqrt`, `exp`, `log`, `tanh` | the claim is per device and driver, not per API |
| refuse the primitive, naming the measured ULP | any device that misses | that kernel does not run there |

**`sin` and `cos` do not call `GLSL.std.450`.** Reduction alone fixes only the
*range*: inside [-π/4, π/4] the ext-inst is still guaranteed ~2⁻¹¹, nine bits
short of 2⁻²⁰. So the emitter generates both halves — a Cody–Waite reduction
with a three-part π/2 split, then a minimax polynomial — and the ceiling is met
by construction. **A ceiling is never widened to match what a device reported.**

**No float-control execution modes are declared at v0.** SPV_KHR_float_controls
answers 008 §3's four conditions declaratively, which beats measuring — but a
module declaring a mode the device lacks fails *pipeline creation*, and 32-bit
`DenormPreserve` is widely unsupported. The conditions are probed instead.

**The probe bootstrap is circular here, and the spec says so rather than hiding
it.** `internal/metal/prober_darwin_test.go` hand-writes MSL probes so the
instrument is not the subject. There is no hand-written SPIR-V source, so the
probe modules are **hand-assembled**: a second, tiny, auditable emitter of a few
dozen instructions sharing no code with the target emitter, because otherwise a
probe failure is ambiguous between the device and the emitter under test. Probes
carry the same `NoContraction` decorations the emitter emits, and
`probe.Ops.MulAdd` evaluates `a*b+c` as one expression, per 022 §1.

## 8. Where the artifact lives, and how a binary stays reviewable

[021](021-metal-bringup.md) §6 deferred `kernel.Kernel`'s artifact shape to
"when a third target arrives". It has arrived, and the flattening is
**ratified**: `SPIRV []byte` sits beside `MSL string`, since a `TargetArtifacts`
struct is the same field count with an extra hop. What wants to be uniform is
the *error*, so `kernel.MissingArtifact(name, target)` becomes one function, and
the no-fallback rule carries over — an empty artifact on a device the caller
selected is a build error naming kernel and target, never a quiet fall back to
the Go lowering.

Generation writes three files, because a binary cannot live in a Go raw string
literal — `emit.go:317`'s `mslArtifact` already guards MSL against a backquote:

| File | Role |
| --- | --- |
| `accel_kernels.go` | `//go:embed` of the `.spv`, plus the module's SHA-256 |
| `<kernel>.spv` | the module. Never read in review. |
| `<kernel>.spvasm` | the emitter's own disassembly. **This is the golden.** |

**A golden of an opaque binary is a golden nobody reviews.** A `.spv` diff is
unreadable, so a regression gets waved through with "regenerated". A `.spvasm`
diff is a text diff a reviewer reads, which is how the artifact regains the
source level 004 says it does not have. The recorded SHA-256 makes an edited or
truncated `.spv` fail by name at test time rather than at
`vkCreateShaderModule`.

Opcode and operand tables are **generated from a vendored
`spirv.core.grammar.json`** with its SPIRV-Headers tag recorded beside it,
because hand-transcribing opcode numbers is the failure shape of the
predecessor's hand-transcribed COM vtable indices ([006](006-backends.md) §2.4).
The grammar version participates in the kernel digest, as the intrinsic table's
ABI version already does.

## 9. How a malformed module is caught

**Round-trip.** A Go decoder reparses the emitted words and re-encodes them;
bytes must be identical. This catches encoding bugs with no other symptom: a
literal string missing its pad word (a string whose byte length is an exact
multiple of 4 still needs a *full extra word* of zeros), or a word count written
as `len(operands)` instead of `len(operands)+1` — off by one word and every
following instruction is misparsed. **What it does not catch is the point of
stating it**: a decoder sharing tables with the emitter proves self-consistency,
not correctness, since a wrong opcode number is spelled identically by both.
Hence §8's generated tables, and hence the `spirv-dis` cross-check.

**Decoded-module assertions.** This replaces reading the text: a test decodes
the module and asserts on operands — `LocalSize` equals `ir.Func.Workgroup`,
every storage atomic carries `Device` scope, the `OpLoopMerge` continue target
is the block holding `Post`. `spirv-val` cannot make these claims, which are
accel's semantics rather than SPIR-V's.

**`spirv-val` and `spirv-dis`.** External binaries, Tier 2 only, behind an
`ACCEL_REQUIRE_SPIRV_VAL` promise that skips in Tier 1 and *fails* in Tier 2 —
the variable is the promise, not the capability, as
`.github/workflows/ci-metal.yml` does for Metal. Neither is ever a step of
`go generate`, which Tier 1 runs on every commit with no provisioning, and
neither is ever *the* check: 004 requires every rejection to be the emitter's,
positioned, and naming the target, while `spirv-val` reports byte offsets. The
`spirv-dis` comparison is **canonical, not textual** — both disassemblies are
parsed to a stream of opcode plus operand ids and the streams must be equal,
because friendly names, id numbering and whitespace differ between two
disassemblers forever and a criterion that fails on formatting gets switched
off. Provisioning is one pinned line:
`apt-get install -y spirv-tools mesa-vulkan-drivers libvulkan1`.

| Injected fault | round-trip | decoded assertions | `spirv-val` | lavapipe |
| --- | --- | --- | --- | --- |
| `LocalSize` hardcoded 1,1,1 | no | **yes** | no | wrong results |
| word count off by one | **yes** | yes | yes | — |
| string missing its pad word | **yes** | yes | yes | — |
| `OpTypeInt 32 1` declared twice | no | yes | **yes** | — |
| `OpVariable` outside the entry block | no | yes | **yes** | — |
| 8-bit type without `StorageBuffer8BitAccess` | no | **yes** | yes | — |
| `continue` branches to the header | no | **yes** | no | hang |
| storage atomic at `Workgroup` scope | no | **yes** | no | **passes** |
| a missing `NoContraction` | no | **yes** | no | maybe |

The rows only the decoded assertions catch are the argument for that layer.
`LocalSize 1,1,1` is the predecessor's own fatal bug — 004 cites
`polyred/gpu/shader/compile.go:652` by line — and it crashes nothing and returns
plausible wrong numbers, which makes reinstating it the best fault injection
this spec has.

**The differential runs last, and that ordering is the answer to "you cannot
read the artifact".** The comparison is unchanged from
`internal/testkernels/differential_darwin_test.go`; what changes is that a
mismatch has a fourth suspect beside the lowering, the device and the test —
**the module might not encode what the emitter thinks it wrote** — and the three
layers above are what eliminate it.

**A lavapipe pass is a plumbing claim, never a memory-model claim.**
[006](006-backends.md) §7 states it, and it bites hardest here because on this
target the memory model lives in *operands* rather than in a language's built-in
call. A missing barrier and a `Workgroup`-scoped storage atomic both pass on a
software rasterizer. So does a kernel whose *answer* depends on the active set
at a subgroup operation — which is not a fault in the code, since
[002](002-compute-model.md) §5.3 makes divergent subgroup operations legal, but
is a portability bug all the same: lavapipe's active set is one vendor's, and
002 §5.1 says no two agree. No claim of the form "verified on Vulkan" comes out
of a lavapipe run; those are Tier 4.

## 10. What this costs

- **The emitted code leans on the driver optimizer** (§4); a weak mem2reg is
  slower than the MSL path for the same kernel.
- **Three committed files per package**, a `.spv` a reviewer takes on trust from
  its disassembly and hash, and a vendored Khronos grammar to keep fresh behind
  `git diff --exit-code`.
- **A second, hand-assembled emitter** for probe modules (§7), and `sin`/`cos`
  become a reduction plus a polynomial paid on every call, including the calls
  whose argument was already small.
- **One descriptor slot spent on lengths** that `OpArrayLength` would not need,
  bought against a `len()` meaning the same thing on all three backends.

## 11. What lands here, and what waits for the backend

009's rule is that an emitter with no consumer is code nobody ran. So the
criteria split: validation and disassembly cross-checks need one apt install,
while *execution* needs the purego Vulkan backend
([037](037-vulkan-bringup.md)) — roughly forty structs marshalling correctly
before anything runs.

| Provable here | Waiting on [037](037-vulkan-bringup.md) |
| --- | --- |
| round-trip, byte-stability, golden disassembly | the corpus execution differential |
| `spirv-val` clean, `spirv-dis` agrees canonically | 006 §7's entry gate, in order |
| decoded-module assertions, every §9 fault | the tiled GEMM, the terminal gate |
| capability inference from `Caps` and binding dtypes | the measured Vulkan numeric profile (§7) |

004's target policy settles the ordering: **the generator does not admit the
`vulkan` token until the emitter and its conformance gate both exist.** So this
spec builds and validates the emitter while `-targets=vulkan` still fails
generation — an artifact nothing can run is not offered to callers as one. The
token is `vulkan`, not `spirv`, because the target set names backends.

## 12. Done

- a module from any corpus kernel **round-trips byte for byte**, and
  regenerating twice on Linux and macOS gives identical bytes — catching a
  missing string pad word, an off-by-one word count, and an id allocator that
  depends on Go map order;
- the decoded `LocalSize` operands equal `ir.Func.Workgroup` — catching the
  predecessor's `local_size_x = 1`, which crashes nothing and returns plausible
  wrong numbers;
- every `OpLoopMerge` continue target is the block holding `Post` and every
  `ir.Continue` branches to it — catching the infinite loop `spirv-val` accepts;
- every storage atomic carries `Device` scope and every shared atomic
  `Workgroup` — catching the one-token change lavapipe cannot see;
- the `NoContraction` count equals the decorable arithmetic result count,
  asserted on the module rather than on a result; every function-scope
  `OpVariable` is in the entry block and carries an initialiser, and no
  Workgroup-class variable carries one — catching the uninitialised read 002
  §2.2 names against SPIR-V by name;
- the `OpCapability` list is exactly what `Caps` and the binding dtypes imply,
  `StorageBuffer8BitAccess` included, and a kernel over
  `Limits.MaxBindingsPerKind` is refused by name rather than inside
  `vkCreateDescriptorSetLayout`;
- `sin` and `cos` reach no `GLSL.std.450` instruction, and the generated
  lowering meets 008 §6's 2⁻²⁰ bound against a higher-precision reference over
  \|x\| ≤ 2¹⁶;
- `spirv-val` accepts every corpus module and the emitter's `.spvasm` matches
  `spirv-dis` as an instruction stream, in a Tier 2 job that skips without its
  promise and fails with it, and never as a step of `go generate`;
- every rejection is the emitter's, carries a source position, and names
  `vulkan` plus the capability or limit that caused it; and
- each of the nine faults in §9's table is reinstated in a test and caught by
  the layer that table claims catches it.

## 13. Open questions

- **`GlobalIndex`, `LocalIndex`, `GroupIndex`.** SPIR-V has
  `LocalInvocationIndex` natively and computes the other two:
  $$
  \text{GroupIndex} = g_z N_y N_x + g_y N_x + g_x
  \qquad
  \text{GlobalIndex} = \text{GroupIndex}\cdot S_xS_yS_z + \text{LocalIndex}
  $$
  `GlobalInvocationId.x` is right for a 1-D grid and a plausible wrong number
  everywhere else, because 002 §1.3 makes `GlobalIndex` workgroup-contiguous.
  The MSL emitter refuses all three and no corpus kernel uses them: lowering
  them makes MSL the odd one out, refusing them leaves 002 §1.3's normative
  table describing accessors no GPU target supports. Either way a corpus kernel
  must exercise the choice.
- **Whether `SignedZeroInfNanPreserve` is worth declaring** once a device
  reports it. It is the one of the four float controls whose absence changes
  results rather than precision.
- **Where the Vulkan numeric profile is recorded.** 008 §10 holds Metal's table,
  and a per-device-and-driver profile does not fit a per-API shape.
- **Whether one module per kernel is right.** §8 assumes it; sharing helpers
  across kernels in one module would shrink artifacts and complicate the digest.
