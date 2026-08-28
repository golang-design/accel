# Outstanding work, per spec

Every spec whose status is **in progress**, and exactly what is unbuilt in it.

This file exists because the status field alone could not be trusted. A spec
said `implemented` and a reader had no way to check it without reading the
code themselves, which is what an audit on 2026-08-27 found: fifteen specs
claimed `implemented` while each owned at least one unbuilt section.

**The rule this applies** ([009](009-sequencing.md), and `.claude/skills/land`):
`implemented` only when *no* section the spec owns is unbuilt. A spec whose
current milestone is finished but which owns other unbuilt sections stays
`in progress` and names them here.

**How this was produced.** Each spec was read whole, and every normative
statement in it -- a MUST, a refusal, a guarantee, a Done bullet, a registered
kernel, a named API -- was traced to the code implementing it and to a test
exercising it. Items below are what had neither, or had code and no test. Two
findings were re-verified by hand before the statuses were changed, because a
fifteen-spec downgrade on an audit's say-so is not evidence.

**Two kinds of item appear here and they are not the same.** *Unbuilt* means
the section has no implementation. *Unbacked* means the code exists and
nothing checks it -- the failure this project keeps finding, one level up from
operator refusals.

As of 2026-08-27: **31 in progress**, 9 implemented, 4 drafted,
with **266 outstanding items** across the in-progress set.

All 50 specs are audited. 003, 004, 005, 009, 010 and 011 were re-audited after
the first pass failed on them and appear in their own sections at the end.

## Implemented — nothing outstanding

- [021-metal-bringup.md](021-metal-bringup.md)
- [024-tensor-bringup.md](024-tensor-bringup.md)
- [026-tensor-decode.md](026-tensor-decode.md)
- [027-quantization.md](027-quantization.md)
- [028-sampling.md](028-sampling.md)
- [031-shared-transients.md](031-shared-transients.md)
- [043-per-row-values.md](043-per-row-values.md)
- [048-int4.md](048-int4.md)
- [049-grouped-gemm.md](049-grouped-gemm.md)

## Drafted — nothing built yet, by intent

- [037-vulkan-bringup.md](037-vulkan-bringup.md) — 11 sections specified, none built
- [038-spirv-target.md](038-spirv-target.md) — 10 sections specified, none built
- [040-batch-scheduler.md](040-batch-scheduler.md) — 8 sections specified, none built
- [041-msaa.md](041-msaa.md) — 12 sections specified, none built

## In progress — what is outstanding in each

### [000-decisions.md](000-decisions.md)

**Unbuilt:**

- **Layering rules (rules 1, 2 and 4)** — No test enforces "layer 1 never imports layer 2", "backends implement an unexported interface", or "the CPU backend is never build-tagged away". The facts hold today (go list -deps golang.design/x/accel shows no tensor import; internal/driver is internal; internal/cpu has no build tag) but nothing h
- **The v0 milestone — "v0 inference is deliberately unquantized"** — Superseded rather than unbuilt: quantization shipped (quant/, tensor/int4.go, specs/027 marked Implemented, specs/048 in progress). The section states it moves as milestones land, and it has not moved.

**Unbacked (code exists, nothing checks it):**

- "**v0 inference is deliberately unquantized.** It proves one f16/f32 transformer decode path ... Quantized weights, a quantized KV cache, and the kernel variant — Contradicted by shipped code: quant/quant.go (Int8Quantize, Error), quant/int4.go, tensor/int4.go, tensor/grouped.go, and testkernels/int4.go / int4tiled.go. specs/027-quantization.md is Implemented a
- "**v0 is compute only.** ... No graphics public API was promised until its stage ABI, render API, surface/present contract, and CPU rasterizer had their own imp — A full public graphics API now exists and runs: render.go (RenderPipeline, RenderPass, draws), surface.go (Surface, acquire/present), texture.go, textureview.go, internal/raster/. Exercised by render_
- Layering rule 3: "No backend-specific type appears in a public signature." — Violated by the public surface: device.go:103 `func OpenCPU(opts CPUOptions) (*Device, error)`, plus the exported CPU-only types CPUOptions (accel.go:298), CPUMode with CPUDeveloper/CPUStrict/CPUMimic
- Layering rules 1, 2 and 4 ("Layer 2 imports layer 1. Layer 1 never imports layer 2", "Backends implement an unexported interface", "The CPU backend ... is never — no test. The rules are true of the tree today but nothing checks them; importgraph_test.go only checks that the root package does not reach golang.org/x/tools or go/types.

### [001-device-resources.md](001-device-resources.md)

**Unbuilt:**

- **§3.5 Byte order** — The normative refusal is at the wrong entry point and the wrong text. OpenDevice (device.go:84) performs no endianness check at all. The only check is transfer.go:312, inside hostBytes, so a big-endian host would open a device, allocate, build and submit graphs, and only fail at the first host trans
- **§4.2 Row pitch — the plan-time reporting half** — CopyStats is declared (limits.go:124) and NodeStats carries `Copy *CopyStats` (graph.go:485), but graph.go:426-435 NodeStats never assigns it and no assignment to that field exists anywhere in the tree. A recorded texture copy therefore reports nothing about its pitch or its repack. The immediate ha

**Unbacked (code exists, nothing checks it):**

- §11: "**§4 in full (textures, formats, row pitch) is the one section still unbuilt** ... this spec therefore stays *in progress* rather than implemented, and §4 — The spec asserts its own status twice with two different answers, and both are wrong. The frontmatter says `implemented`; §11 says `in progress` because of §4; and §4 is in fact built and tested (form
- §3.5: "A big-endian `GOARCH` is not supported: `OpenDevice` fails naming the architecture, rather than producing silently byte-swapped results." — No code: device.go:84 OpenDevice has no endianness check. The only guard is transfer.go:312 in hostBytes, which fires on the first host transfer rather than at open, and its message does not name the 
- §4.2: "the backend knows at build whether a copy's pitch needs padding, so a recorded copy carries this from `Graph.NodeStats` as soon as `Build` returns" (Copy — The field exists (graph.go:485) and is never written. graph.go:426-435 constructs NodeStats with Node, Kind, Label and BarriersBefore only. Grepping the tree finds no assignment to a CopyStats value. 
- §8.1: the recorded transfer entry points are "`Recorder.CopyToBuffer`, `CopyBuffer`, `CopyTextureToBuffer`, `CopyBufferToTexture`", and "`Recorder.CopyToBuffer` — There is no Recorder.CopyToBuffer. The record-time host write is spelled Recorder.UploadToBuffer (and Recorder.UploadToSlot) in record.go. The spec names an API that does not exist, twice in one secti
- §4.1: "**this format is not host-copyable**: `Queue.ReadTexture` on it is an error, and buffer copies of it are an error, each naming the format and pointing at — format.go:207 sets `info.HostCopyable = !e.depth`, so Depth32Float is non-host-copyable too, and texturealloc.go:159 refuses Queue.ReadTexture for every depth format. The message names neither Depth32

### [002-compute-model.md](002-compute-model.md)  · **too large, see the split plan**

**Unbuilt:**

- **§2.3 and §2.5 — the barrier call set** — Only Thread.Barrier exists (internal/kernel/kernel.go:136, intrin.go:398). BarrierShared, BarrierStorage and SubgroupBarrier appear nowhere in the tree — no Thread method, no IR op, no intrinsic table entry, no emitter case. §2.5's four-row table, the §2.3 Subgroup-scope barrier row, and §2.4's stor
- **§2.5 — what Barrier makes visible, on Metal** — internal/kernelc/emit/msl.go:1042 emits `threadgroup_barrier(mem_flags::mem_threadgroup)` unconditionally. `mem_device` occurs nowhere in the repository. So the one barrier that does exist orders shared memory only, while §2.5 makes it normative that Barrier covers shared *and* storage. specs/022-ms
- **§1.3 — three built-in id accessors** — t.WorkgroupSize(), t.NumGroups() and t.GlobalSize() are in §1.3's table and in §3.3's workgroup-uniform seed set, and none of the three exists on internal/kernel.Thread or in intrin.go's table. §1.3's t.SubgroupID() and t.SubgroupInvocationID() shipped under different names (SubgroupIndex at subgrou
- **§6.2 — float-to-integer conversion** — accel.ToI32, accel.ToU32, accel.ToI8 and accel.ToU8 do not exist anywhere. internal/kernelc/front/build.go:1024 conversion() accepts any convertible pair and emits ir.NewConvert with no float-source check, so `int32(f)` compiles. The whole saturating/NaN-to-zero contract, and the bit-identical-on-ar
- **§5.2 — the ballot value type** — There is no accel.KernelMask. The mask type is internal/kernel.Mask re-exported only as kernelabi.Mask (kernelabi/kernelabi.go:75), so a caller cannot name it from the public package. Thread.SubgroupBallot (subgroup.go:327) exists on the scheduler side with no intrinsic, which the header notes; the 

**Unbacked (code exists, nothing checks it):**

- Header: "§§1–5 are implemented on the CPU backend and on Metal: the execution hierarchy and built-in ids, workgroup-shared memory, barriers ... This spec is the — The "what remains" list omits at least five things §§1-5 own: BarrierShared, BarrierStorage and SubgroupBarrier (no code at all), t.WorkgroupSize/NumGroups/GlobalSize (no code at all), and the Metal B
- §2.5's lowering table: `Barrier` lowers to `threadgroup_barrier(mem_threadgroup|mem_device)` on MSL, and "a memory barrier makes prior writes both available and — internal/kernelc/emit/msl.go:1042 emits `threadgroup_barrier(mem_flags::mem_threadgroup)` for every barrier; `mem_device` is absent from the whole repository. A kernel that writes a storage buffer, ca
- §6.2: "**the kernel subset forbids a bare conversion from a float type to an integer type**, and provides `accel.ToI32(x float32) int32`, `accel.ToU32`, `accel. — No code and no test. The four intrinsics are absent from internal/kernelc/intrin/intrin.go. internal/kernelc/front/build.go:1023-1039 builds any explicit conversion into ir.NewConvert with no source-k
- §7.2's worked GEMM listing — `func GEMMf16(t accel.Thread, d Dims, a []accel.F16, ..., tileA *[256]float32)` calling `t.BarrierShared()` and `accel.ToF16(acc0)` — Three of the names in the spec's own proof-of-sufficiency kernel do not exist: accel.F16 is spelled accel.Float16 (kernelabi.go:67), accel.ToF16 is spelled accel.ToFloat16 (intrin.go:266), and t.Barri
- Testing: "Generated GLSL places `memoryBarrierShared` and `memoryBarrierBuffer` before `barrier`; a golden lowering test rejects the reversed sequence." — no code, no test. internal/kernelc/emit contains msl.go and mslstage.go only — there is no GLSL emitter, and 000's v0 section says SPIR-V and the other targets are post-v0. This obligation cannot be m
- §2.3 and §5.3: `t.SubgroupBarrier()` exists as a capability-gated subgroup-scope barrier obeying §3.1 at subgroup scope, and §3.3's "subgroup barriers require S — no code. SubgroupBarrier is not a Thread method, not an IR op, and not in intrin.go. The uniformity analysis's SubgroupUniform lattice level therefore gates nothing that can be spelled. The Testing se

### [006-backends.md](006-backends.md)  · **too large, see the split plan**

**Unbuilt:**

- **§2.3 Vulkan** — No Vulkan backend package exists. specs/037-vulkan-bringup.md and 038-spirv-target.md own this and are both status: drafted.
- **§2.4 D3D12** — No D3D12 backend, no BackendD3D12 adapter, and open question 4 (SM6/DXIL) is still open. No child spec exists.
- **§2.5 OpenGL ES 3.1** — No GL backend package, no context-thread replay loop. No child spec exists.
- **§2.6 WebGPU and WASM** — No GOOS=js path, no Backend constant, and none of the reserved Pending[T]/EnumerateAsync/OpenDeviceAsync/MapAsync shape the section states is 'a constraint, not an open choice'.
- **§3 packed narrow-dtype emulation** — No packed compare-exchange store path and no conformance test writing every lane of one backing word. Nothing in the tree emulates f16/i8 by packing; CPU and Metal both store natively.
- **§4.1, §4.2, §4.4, §4.5 (WebGPU half)** — Reusable Vulkan primary command buffer, closed D3D12 command list, GL context-thread replay, and WebGPU encoder replay all require backends that do not exist.
- **§4.3 indirect command buffers** — ICB lowering is not written; Metal reports NativeGraphReplay false (internal/metal/metal_darwin.go). The spec states this itself.
- **§6.1 isolated probe helper** — No helper subprocess, no versioned wire record, no abnormal-exit/signal/malformed-output/timeout classification. Probing is in-process (adapters_darwin.go:39).
- **§6.3 environment override** — ACCEL_BACKEND is not read anywhere, and SelectionReport carries no field for the applied restriction.
- **§7 CI tiers 2 (non-Metal), 3, and 4** — No lavapipe, llvmpipe, WARP, ANGLE, headless-browser, or hardware jobs in .github/workflows/. Only tier 1 plus Metal exist.

**Unbacked (code exists, nothing checks it):**

- R1 and §6.1: 'native-driver probes run in an isolated helper process so an abort becomes a diagnostic rather than taking down enumeration' / 'Native probes run  — No code. grep for os/exec across all non-test .go files returns nothing. adapters_darwin.go:39 calls metal.Adapters() directly in-process, so a Metal abort takes down enumeration exactly as the requir
- §6.2's SelectionReport declares a field EnvironmentBackend string. — accel.go:254-258 declares only Selected and Rejected. The field does not exist, so §6.3's 'the applied environment restriction appear[s] in the selection report' has no carrier.
- §6.3: 'ACCEL_BACKEND may restrict the candidate set for automatic selection only.' — No code. The string ACCEL_BACKEND appears in no .go file; OpenBest (device.go:135-178) reads no environment. The whole section is unimplemented, and the header's 'What is not' paragraph does not name 
- Testing: 'ACCEL_BACKEND cannot change the result of an explicit open.' — no test — and nothing to test, since the variable is never read.
- Testing: 'The same class-A f32 kernel within 008's proved exact domain is bit-identical on arm64 and amd64.' — no test. The only GOARCH reference in any test is internal/conformance/probe/probe_test.go:25, which asserts the probe records the current arch. Nothing compares results produced on two architectures.
- §3: 'A conformance test concurrently writes every lane of one packed word and requires all lane values to survive.' — no test, and no packed-emulation store path for it to exercise.
- §3 matrix: 'f16 arithmetic | Metal | yes', where yes means 'Architecturally guaranteed. Every device on this backend has it.' — internal/metal/metal_darwin.go:347-400 never assigns F16Arithmetic, so every Metal device reports it false and Capabilities.Set() omits CapF16Arithmetic. The backend's own comment on Subgroups states 
- §6.1: 'OpenDevice rejects an ID from an earlier process or a removed adapter with an error containing the last known identity.' — device.go:96 returns 'accel: no enumerated adapter has that ID; call Enumerate first'. No identity is carried, and nothing distinguishes a stale ID from a removed adapter.
- §3 Amendment and matrix: the Metal column reads '?' for denormals f32/f16, Inf/NaN produced, and FP contraction control, with '?' defined as 'Resolved by measur — They were measured and recorded elsewhere and the matrix was never corrected. 008 §10 records Metal subnormals as no; docs/conventions.md:255-296 records both the subnormal flush and the contraction p
- Header: 'What remains is every backend in §2 other than the CPU and Metal, and §4's lowering strategies for those.' — The status rule requires an in-progress spec to name which sections are outstanding, and this sentence names only the backend set. It omits §6.1's probe subprocess, §6.3's environment override, §3's p

### [007-tensor-layer.md](007-tensor-layer.md)

**Unbuilt:**

- **§'Persistent state and the KV cache' — the sizing helper** — KVCacheDesc, KVCacheSizeInfo, and KVCacheSize do not exist. Nothing checks 'every dimension, dtype, integer multiplication, and layer-1 buffer limit' before a caller allocates. This section is not listed in the spec's own 'Absent, with what each waits on' table.
- **§'Persistent state and the KV cache' — capacity checking** — 'An index or length beyond capacity fails before submission on the host; strict CPU execution also checks it in the kernel.' Neither half exists: the write index is now a device tensor (ScatterRows takes ids *Tensor), and internal/testkernels/elementwise.go:177,218 record that an out-of-range scatte
- **§'v0 operator contracts' — Softmax mask and causal** — SoftmaxOptions carries only Axis. No mask binding and no causal attribute. The spec's own audit table names this as waiting on a kernel with a mask binding, but the normative operator block below still declares both fields.
- **§'Layouts, views, and broadcasting' — Squeeze and Unsqueeze** — Neither operator exists; Reshape is the only path. The spec's audit table records this as a deliberate absence, but the operations table above still lists both as producing views.

**Unbacked (code exists, nothing checks it):**

- §'Creating graph values': func Persistent(b *Builder, d StateDesc) *State — Built as NewState(b, d) (tensor/state.go). The function named in the spec does not exist, and the rename is not in the 'What is built' audit section — the built function's own doc comment still opens 
- §'v0 operator contracts': func Rows(b *Builder, table, ids *Tensor) *Tensor — Built as GatherRows (tensor/ops.go). The doc comment still opens 'Rows gathers whole rows of a table by index'. The rename is not recorded anywhere in the spec.
- §'Creating graph values': PortDesc carries Access accel.Access, and StateDesc carries Capacity int. — Neither field exists. PortDesc is {Name, DType, Shape, Kind}; StateDesc is {Name, DType, Shape}, with capacity folded into Shape's leading axis. A caller reading Plan.Ports() cannot learn the access t
- §'Persistent state and the KV cache': func ScatterRows(b *Builder, state *State, rows *Tensor, indexName string) *State, with 'the runtime index names the first — Built as ScatterRows(b, s, rows, ids *Tensor). The spec's own audit table 200 lines later records the correction, but this normative code block and its surrounding prose were never updated, so the spe
- §'v0 operator contracts': SoftmaxOptions{Axis, ScaleName, Mask, Causal} and RoPE(b, x, positions *Tensor, rotaryDim int, baseName string). — Same contradiction. Built: SoftmaxOptions{Axis} only, and RoPE(b, x, rotaryDim, baseName, positions) (tensor/ops.go:208). The spec's audit table records both corrections and the normative API block st
- §'Shape and indexing': 'Rows ... Out-of-range ids fail in strict mode and are a caller error otherwise.' — There is no strict-mode failure. internal/testkernels/elementwise.go:114 states the kernel writes zeros for an out-of-range id, and internal/testkernels/elementwise_test.go:137-148 asserts exactly tha
- §'Ownership and core types': 'There is no automatic plan cache in v0', repeated in the audit table as 'a plan cache | post-v0 by this spec's own §Ownership'. — tensor.PlanCache, NewPlanCache, PlanCache.Compile/Close/Len are shipped and documented, keyed by Builder.Identity. Owned by specs/029-plan-cache.md — so this is a stale scope claim contradicted by shi
- §'Prefill and decode plans': 'Production bucketing is not part of v0.' — tensor.Buckets, NewBuckets, Buckets.For and Buckets.Sizes are shipped, with padding-without-a-mask reasoning in their doc. Owned by specs/040-batch-scheduler.md.
- §'Post-v0 scope': quantized dtypes and quantized GEMM/Rows, sampling operators and policy, plan caches, paged KV caches, and multi-sequence scheduling are post- — All shipped in the same package: Quantized/QuantMatMul/QuantGatherRows and Int4/Int4MatMul/Int4MatVec (027, 048), Sample/SampleCategorical/Argmax/TopKMask/TopPMask/SamplingOptions/DeclareSamplingScala
- §'Persistent state and the KV cache': 'The v0 KV cache is for one sequence (batch=1)', and §'RoPE and attention': Attention 'accepts q shaped [1, qSeq, qHeads,  — AttentionOptions.Lengths is a per-sequence tensor precisely because 'cache lengths are genuinely independent across a batch', and tensor/decode_test.go:1105 TestABatchOfSequencesStepsTogether runs a b
- §'What is built' audit table: 'Contiguous | a gather kernel with strides in a uniform block; 010 registers none'. — Contiguous is fully built (tensor/materialize.go, exported at tensor/views.go) with its own multi-section doc block explaining the cost model, and tensor/pack_test.go:123 tests that an already-contigu
- §'Fusion': 'v0 fuses by authorship only: RMSNorm, SwiGLU, Linear's bias epilogue, Softmax's scale/mask, and eligible Attention variants.' — Softmax has neither a scale nor a mask to fuse — SoftmaxOptions is {Axis}. The other four are real.
- Header/status obligation: the 'Absent, with what each waits on' table is this spec's statement of what is outstanding. — It fails in both directions at once: it lists Contiguous and the LayerState binding as absent when both are built, and it omits KVCacheSize/KVCacheDesc/KVCacheSizeInfo and the host-side capacity check

### [008-numerics.md](008-numerics.md)  · **too large, see the split plan**

**Unbuilt:**

- **§6 — the committed high-precision corpus** — 'The committed conformance corpus contains input bits and reference bits produced at at least 256-bit precision... the corpus generator records its oracle version and is reproducible.' No corpus file, no generator, no oracle-version record. TestPrimitiveCeilings computes its reference live as float6
- **§6 — the sin/cos domain gate** — 'Compile rejects a larger declared capacity/domain' for RoPE angles beyond 2^16. tensor/ops.go:219 validates only rotaryDim's positivity, evenness and width. Nothing bounds the generated maximum angle, so a capacity that leaves the bounded domain compiles.
- **§8 — composed operator budgets** — No forward sensitivity propagation, no budget trace, no refusal when an interval crosses a singularity, and no per-operator attribution for RMSNorm, Softmax, MatMul/Linear, composed Attention, or the golden model. Only §7's reduction/dot-product form and §8.1's interpolation bound exist. The spec na
- **§9 — the profile-aware comparison API** — Context, OracleID, Special, Division, Primitive, Reduction and Operator are all absent. The spec's own built table concedes this, but §9's normative prose above it does not.
- **§9 — the static tolerance check** — 'A static check rejects direct approximate-comparison helpers and numeric tolerance arguments in conformance tests.' No such analyzer, vet pass, or test exists anywhere in the module or in CI.
- **§5 — contraction control on non-Go targets** — SPIR-V NoContraction and HLSL precise require targets that do not exist. Only the Go lowering and the Metal pragma are built.

**Unbacked (code exists, nothing checks it):**

- §9: 'numeq.Exact fails immediately if the backend profile has not proved that class/domain exact.' — Not expressible. The built signature is Exact[T comparable](got, want []T) Report — no Context, no class, no domain, no profile. There is no path by which the function could consult a profile, so this
- §9: 'Every failure reports backend/device, class, input index, got/reference bits, absolute and ULP error, allowed budget, and the budget trace.' — numeq.Report carries Equal, FirstDiff, Got and Want as formatted strings, Diffs, Len and WantLen. No backend, no class, no ULP error, no budget, no trace. SumReport adds Error/Budget/Depth/Magnitude b
- §9 and §11: 'A static check rejects direct approximate-comparison helpers and numeric tolerance arguments in conformance tests' / 'Mechanically reject ad hoc to — No such check exists. The only match for 'tolerance' outside comments and prose is tensor/sample.go:212, a doc reference. CI (.github/workflows/ci.yml) runs build, vet, test, fuzz, generated-file fres
- §6: 'The committed conformance corpus contains input bits and reference bits produced at at least 256-bit precision. Runtime CI therefore does not depend on MPF — No corpus and no generator. internal/metal/primitives_darwin_test.go:29-32 states the opposite in its own comment: the reference is float64 rounded once to f32, 'not correctly rounded by construction'
- §6: 'Compile rejects a larger declared capacity/domain' once the generated maximum RoPE angle would leave the 2^16 sin/cos domain. — no code. tensor/ops.go:208-259 checks only rotaryDim; no path derives a maximum angle from the declared capacity or refuses one. A model configured past the bounded domain compiles and runs on kernels
- §11: 'For positive normal-reference sqrt cases, assert that r and each finite adjacent f32 value pass and that the second representable value on either side fai — no test. TestPrimitiveCeilings runs sqrt over eighteen positive inputs at 1 ULP but never probes the adjacency boundary, never asserts sqrt(+0) is +0, and carries no adversarial rsqrt case. The §6 dis
- §11: 'Force a backend profile without class-A proof and assert numeq.Exact refuses.' — no test, and not testable: Exact takes no profile.
- §11: 'Test budgets monotonically: a synthetic result at the budget passes and the next representable value beyond it fails' and 'A deliberately wrong reduction  — no test. internal/testkernels/reducesum_test.go covers Gamma's refusal cases, tree-versus-sequential ordering, and magnitude scaling, but no test drives a synthetic value to the budget edge or injects
- Header 'What is not': the outstanding work is '§6's normative primitive ceilings ... §5's contraction control on targets other than Go; and §8's composed budget — The status rule requires an in-progress spec to name what is outstanding, and this list is wrong in both directions. It omits §6's committed corpus and generator, §6's Compile-time domain gate, §9's s
- Header 'What is built': '§6's normative primitive ceilings are stated and unmeasured on any GPU' (paired with the later claim that they were measured on Metal 2 — They are measured, but only where the test can run: internal/metal/primitives_darwin_test.go carries //go:build darwin. That gate also means the CPU oracle's own ceilings — the check the test says mus

### [012-kernel-pipeline.md](012-kernel-pipeline.md)

**Unbuilt:**

- **§4 The generated-kernel ABI is public (the root-package name block)** — KernelABIVersion, KernelBinding, KernelArgs, KernelDType, KernelAccess, KernelF32/F16/BF16/I32/U32/I8/U8, KernelRead, KernelWrite and func KernelSlice exist nowhere in the module. The surface moved to package kernelabi with unprefixed names (kernelabi/kernelabi.go:45-153, const Version at :131, func
- **§7 negative test for a generic method** — No positioned rejection case for a generic method. front_test.go's TestRejections covers "generic kernel" (:296), "generic helper" (:697) and "a generic stage" (:190). The only generic-method-adjacent test is internal/kernelc/intrin/key_internal_test.go:38, a white-box synthesized types.Signature th

**Unbacked (code exists, nothing checks it):**

- §4: the root package carries `KernelABIVersion`, `KernelBinding`, `KernelArgs`, `KernelDType`, `KernelAccess`, the Kernel* dtype/access constants, and `func Ker — None of these identifiers exists. grep over all .go files finds KernelSlice only inside a comment at internal/kernelc/emit/msl.go:65. Generated files name kernelabi.Binding / kernelabi.Args / kernelab
- §4: `type Kernel = kernel.Kernel` on the root package. — compute.go:349 declares `type Kernel = kernelabi.Kernel`. The alias chain still resolves to one type, but the spec's stated layer is wrong, and a reader following §4 imports internal/kernel, which is 
- §7: "one positioned negative test per construct this child rejects, including a generic method". — no test — there is no source-level generic-method case in TestRejections; the only coverage is a synthesized-signature unit test with no position.
- §7: "editing a kernel without regenerating fails freshness, naming it". — The naming half has no test. cmd/accel-kernel/main_test.go:39 TestCheckFailsOnAnEditedKernel asserts a non-zero exit and that stderr contains "go generate"; it never asserts the kernel's name appears,
- §2: the direct executor has a "restricted descriptor [that] rejects shared parameters and every cooperative intrinsic". — There is no descriptor. internal/conformance/direct/direct.go:43 Run is a one-line delegation to kernel.Dispatch, and the refusal is a nil-Flat check at internal/kernel/dispatch.go:40. The property ho

### [013-kernel-subset.md](013-kernel-subset.md)

**Unbuilt:**

- **§3 rejection corpus — rows 2 and 3 (interfaces, channels, strings, slices of slices; panic)** — No positioned source case exists for any of them. Interfaces, channels, strings and maps are only rejected through white-box unit tests that feed synthesized go/types values and assert no position: internal/kernelc/front/internal_test.go:153-173 (TestIRTypeRejections) and :81-98 (TestUnhandledExpres
- **§3 rejection corpus — generic methods** — No case. front_test.go covers generic kernels (:296), generic helpers (:697) and generic stages (:190). Same gap as 012 §7.
- **§4 — break and continue inside nested loops** — No kernel in internal/testkernels puts a break or continue inside an inner loop, and front_test.go's accept case "nested loops" (front_test.go:957 region) contains neither and asserts only inferred accesses, not level-5 agreement. linear.go:79's continue sits in the outer loop.
- **§4 — helper freshness naming both** — No test edits a helper and asserts the freshness failure names the helper and its caller. cmd/accel-kernel/main_test.go:39 edits a kernel body ("* 2" -> "* 3") and asserts only that stderr says "go generate". emit_test.go:224 checks the preimage carries helper lines, which is the input to the check,

**Unbacked (code exists, nothing checks it):**

- §3 table, last row: "shared parameters, barriers, atomics, subgroups | sequencing | cooperative kernels arrive at M4". — Inverted since M4 landed. front_test.go:1312 asserts a kernel calling t.Barrier() is *admitted* and marked cooperative, and :1341 asserts the same for shared memory. The row now describes a refusal no
- §4: "break and continue inside nested loops match the authored function". — no test — no corpus kernel and no front-end accept case exercises a branch inside an inner loop.
- §4: "editing a helper without regenerating its callers fails the dependency digest, naming both". — no test — the digest carries the helper lines (digest.go:91-93) but nothing exercises the failure, and nothing asserts the message names the caller.
- §3: "Every message carries a source position, and a test asserts the position and not only the text." — True of the ~70 cases in TestRejections, false of the interface/channel/string/map family, which is checked only through synthesized types in internal_test.go where there is no position to assert.

### [014-kernel-uniforms.md](014-kernel-uniforms.md)

**Unbuilt:**

- **§2 — "A generated per-kernel encoder and decoder"** — There is no decoder. accel.UniformCodec[T] (uniform.go:20-28) declares only EncodedSize and Encode, no generated file contains a Decode method, and internal/kernelc/std140 has no decode path. §4's "Host-side encode/decode round-trips structs" therefore cannot exist: scaled_test.go:41 TestCodecMatche
- **§2 — "Typed bindings: a generated bindings struct whose Bind checks every field and returns ordinary resource bindings"** — No generated bindings struct and no generated Bind exist. internal/testkernels/accel_kernels.go contains only codecs, Kernel records and lowerings. A caller builds []accel.Binding and []accel.UniformValue by hand (pipeline.go:199-230).
- **§2 — "Validation against the device: a struct whose encoded size exceeds Limits.MaxUniformBlockBytes is a pipeline-creation error naming the struct, the encoded size, and the device's limit"** — pipeline.go:25-100 newComputePipeline checks the ABI version, the entry points, the workgroup extents and V10/V11/V17, and never looks at uniform block size. The only check is at uniform.go:139, in NewUniformBuffer, and its message names the size and the limit but not the struct. A kernel whose by-v
- **§4 — the device check in its stated form** — No kernel writes each uniform field to a distinct storage element. The closest is TransformKernel, which accumulates the fields into one output.

**Unbacked (code exists, nothing checks it):**

- §2: `UniformBuffer[T]` exists "so values may change between submissions without changing graph structure". — Nothing consumes a uniform buffer binding. uniform.go:100-116 says so in the code: "UniformBuffer.View returns a binding no draw and no dispatch is parameterised by today", the mechanism having been r
- §6: "§4's device-side check is deferred, and this is the one gap ... it lands with Metal". — Stale in both directions. It effectively landed: TransformKernel with the exact worked-example Params value is in the Metal differential list at internal/testkernels/differential_darwin_test.go:303-31
- §4: "Host-side encode/decode round-trips structs containing scalar, vector, and array fields". — There is no decode, so no round trip. The encoder is checked against a literal byte expectation only.
- §5: the root package carries `accel.KernelUniform` and `accel.KernelUniformValue`. — Neither identifier exists. The record's by-value declaration is internal/kernel.Uniform re-exported as kernelabi.Uniform (kernelabi/kernelabi.go:62), and the generated entry point recovers one through
- §2/§4: the oversize refusal names "the struct". — uniform.go:140-144 formats only the encoded size and the device limit, and scaled_test.go:240 asserts only that "MaxUniformBlockBytes" and "std140 pads" appear. No code names the Go type.

### [015-graph-recording.md](015-graph-recording.md)

**Unbuilt:**

- **§4 validation table — row V23 assigned to 015** — V23 was withdrawn, not implemented. build.go:809 calls it "the withdrawn V23" and render_feedback_test.go:150 names it as one of "two rules in this project [that] were withdrawn". No concrete same-node overlap check exists and none should be built; §4 still lists V23 under "015" with no note.
- **§4 — V4 enforced at Bind and Rebind** — checkBinding (submit.go:21-72) labels V2, V3, V19, V5 and V6 and performs no access check, although the function's own doc comment at submit.go:16 claims V4. It is not merely missing: a BufferView carries no access mode, so the check as §4 specifies it is unenforceable at Bind. The access half is en
- **§5 — "Attempting [a dispatch] is ErrNotImplemented naming M3's second child"** — Dead. Dispatch shipped (graph.go:35), and device_test.go:532 TestErrNotImplementedHasNoReachableUser asserts that accel.ErrNotImplemented has no caller anywhere in the module.

**Unbacked (code exists, nothing checks it):**

- §2 and §7 name `Graph.Rebind`, `Graph.Submit`, `CopyBufferToBuffer`, `CopyToBuffer` / `CopyFromBuffer`, and `CopyToBufferSlot`. — None of these exists. The real surface is Graph.Bind (graph.go:392), Queue.Submit(g *Graph) (graph.go:597), Recorder.CopyBuffer (graph.go:120), Recorder.UploadToBuffer (graph.go:62), Recorder.UploadTo
- §3 and §6: `GraphStats.Barriers` equals the node count. — There is no GraphStats type anywhere in the module. The figure is Graph.Barriers() (graph.go:452), which §7 of the same spec names correctly — the spec disagrees with itself. The equality also holds o
- §4: "this child implements V2–V4 at Bind and Rebind". — V4 is not implemented at Bind, and cannot be: submit.go:16's comment asserts it while submit.go:21-72 checks kind, dtype, openness, size and usage only. No test asserts a V4 rejection at Bind — graphv
- §4: V23 (concrete same-node overlap) is enforced by this child. — The rule was withdrawn after this spec was written (build.go:809). The spec assigns a check no code performs and no test covers.
- §5: recording a dispatch yields ErrNotImplemented naming M3's second child. — Removed by 016. device_test.go:532 exists specifically to assert the sentinel has no reachable user.
- §3: "The plan is one sentence: execute nodes in record order, with a full barrier between consecutive nodes." — True only of Recorder.BuildNaive (build.go:40). Recorder.Build (build.go:59) runs the inferred plan from 016. §1's diagram and §3's prose read as the current default and are not marked as the retained

### [016-graph-execution.md](016-graph-execution.md)

**Unbuilt:**

- **§5 (Testing) — the V22 bullet** — No test reaches the internal acyclicity assertion. infer.go:311 assertAcyclic is only called from build.go:95 on a graph built in record order, where the check is vacuous by construction. §5 says V22 "is asserted by a test that reaches the internal assertion through a deliberately corrupted edge set

**Unbacked (code exists, nothing checks it):**

- §5: "V22 is asserted by a test that reaches the internal assertion through a deliberately corrupted edge set." — no test — assertAcyclic (infer.go:311) has exactly one caller, build.go:95, and no test constructs a corrupted edge set to fire it.
- Correction (2026-08-24): "V11 stands as stated and unenforced" and "requirementsOf never sets Requirements.SharedBytes, so Device.Missing always compares 0 agai — stale in the other direction — requirementsOf now sets SharedBytes from k.SharedBytes (pipeline.go:115) and TestAKernelOverTheSharedBudgetIsRefused (sharedbudget_test.go:24) drives the refusal through
- §6: "requirementsOf returns a zero capability set, and that is a fact about the v0 subset." — stale — pipeline.go:112 returns Caps: Capability(k.Caps), the set 020's inference computes from the body; TestAKernelRequiringAnAbsentCapabilityIsRefused (graphdispatch_test.go:449) depends on it bein
- §8: "No indirect dispatch. V9 exists for it and is written; the payload is not." — stale — Recorder.DispatchIndirect exists at graph.go:48 with the payload path at pipeline.go:379 and tests in graphindirect_test.go. The bullet names no successor spec, so a reader is left believing t

### [017-graph-aliasing.md](017-graph-aliasing.md)

**Unbuilt:**

- **§7 (Testing) — the interval-planner regression** — No executable interval planner exists anywhere in the repo, so nothing drives one over the diamond and asserts the unsound layout. TestTheDiamondDoesNotAliasUnorderedTransients (graphalias_test.go:26) asserts the real planner's placement; it cannot detect a future "simplification" of reachability ba
- **§7 (Testing) — the worked graph's offset table** — TestWorkedGraphMemoryNumbers (graphworked_test.go:284) asserts each transient's user set and the three byte totals only. No test reads TransientPlacement.Offset for the worked graph; grep for .Offset in graphworked_test.go/graphalias_test.go finds only pairwise-overlap comparisons (graphalias_test.g
- **§7 (Testing) — V24's transient term** — The promised test ("a rebind of a slot onto a range overlapping a transient placement") does not exist and cannot: TestATransientCannotReachASlot (graphvalidate_test.go:472) pins BufferView.check refusing the transient at bind time instead. The two V24 rows that do exist (graphvalidate_test.go:176,2

**Unbacked (code exists, nothing checks it):**

- §7: "A test drives the interval planner over the same diamond and asserts it produces the unsound layout, so the counterexample is executable and not only descr — no code and no test — there is no interval planner in the repo; every "interval" hit is prose in a comment.
- §7: "The worked graph ... asserts the offset table row by row." — no test — TestWorkedGraphMemoryNumbers asserts user sets and the three totals; no per-transient offset is asserted anywhere for that graph.
- §7: "V20 and V24's transient term have focused negative tests; V24's is a rebind of a slot onto a range overlapping a transient placement." — no test, and self-contradicting — §5 of the same spec says the transient term "is not added" because it is unreachable. V20 has its test; V24's transient term has none.

### [018-cooperative-lowering.md](018-cooperative-lowering.md)

**Unbuilt:**

- **§4 (Testing) — the comparative benchmark** — No benchmark reports the cooperative lowering's cost against the flat one. The spec's own closing note admits this and explains why it is hard (selection is derived from the body, so no kernel has both forms), but §5 still declares "§4's cases pass" while excluding only this bullet in a note the fro
- **§4 (Testing) — the state-count assertion** — "A kernel with two barriers generates three states" is not asserted. What is asserted is Kernel.Suspensions (coop_test.go:31,316-321; norm_test.go:271; attention_test.go:313), and countSuspensions (emit/coop.go:84) counts segments where suspend is true — barriers, not states. grep over internal/kern

**Unbacked (code exists, nothing checks it):**

- §5: "§2 is built and §4's cases pass" (with status: implemented in the frontmatter). — no test — §4's last bullet has no benchmark, which the spec's own appended note concedes; the frontmatter status was never moved back.
- §4: "a kernel with two barriers generates three states" is asserted on the generated code. — no test: the golden is committed, but only Kernel.Suspensions is asserted and it counts barriers, not states.

### [019-cooperative-diagnostics.md](019-cooperative-diagnostics.md)

**Unbuilt:**

- **§2 (Shared-memory definition tracking)** — The source position. §2 says the read "is reported with the element index, the source position, and the invocation". SharedTracker.Read (diag.go:252) receives only (array, index) and the emitted Diagnostic (diag.go:257) carries no position field. Only barrier arrival has a position, via BarrierID.Po
- **§4 (Conflicting access)** — §1's table says the pair is "reported with both accesses". diag.go:276 reports both invocation ids and the element; neither access's source position is recorded, for the same structural reason as §2.
- **§5 (Testing) — position assertions** — "Every row of §1's table has a negative test asserting the message, the source position, the workgroup, and the invocation." Two of the three rows have no position to assert, so the tests assert message/workgroup/invocation/element only.
- **§5 (Testing) — the shuffled-order sweep** — No diagnostic test runs under a shuffled order. The spec's own correction says this "has not been done"; verified by grep — ShuffleSeed never appears in diag_test.go.
- **§5 (Testing) — the instrumentation benchmark** — No benchmark reports the cost of the instrumentation, so the developer/strict gap the spec says "justifies having two modes" is unmeasured.

**Unbacked (code exists, nothing checks it):**

- §2: "A read of an element whose bit is clear is reported with the element index, the source position, and the invocation." — no code — SharedTracker.Read (diag.go:252) has no position parameter and Diagnostic (diag.go:18) has no position field; the generated lowering cannot pass one.
- §5: "Every row of §1's table has a negative test asserting the message, the source position, the workgroup, and the invocation." — no test for the position on two of three rows; there is no position to assert on the undefined-read and conflict reports.
- §5: "Each diagnostic is asserted to fire on the first run, by running the offending kernel repeatedly with different scheduler orders — including the shuffled o — no test — no diagnostic test passes kernel.Options.ShuffleSeed; TestArrivalReportsAreDeterministic (diag_test.go:390) repeats under one fixed order.
- §5: "A benchmark reports the instrumentation's cost, since it is on by default in developer mode and off in strict mode, and the gap is what justifies having tw — no benchmark exists that measures diagnostics on versus off.
- §7: "No atomics or subgroups, so no diagnostics for them." — stale — DiagUndefinedLane (diag.go:52-56) is a fourth diagnostic kind covering subgroup lane reads and non-uniform broadcast operands, reported from schedule.go:382 laneRead and :472 checkUniformLane 
- §6: "All three diagnostics of §1's table are built and §5's cases pass" (with status: implemented). — three §5 bullets have no test or no benchmark, and the spec's own correction already retracts one of them.

### [020-cooperative-atomics.md](020-cooperative-atomics.md)

**Unbuilt:**

- **§1 — "the rule that subgroup operations do not require uniform control flow while barriers do"** — Not built. A subgroup rendezvous is a suspension point like a barrier (emit/coop.go:247-256 splits the state at subgroupRendezvous and sets suspend: true, and :280 refuses one nested inside an expression), so a subgroup operation is subject to the same structural placement a barrier is and cannot ap
- **§6.5 — the remaining reduction and scan operators** — internal/kernel/subgroup.go:43-66 defines Add/Min/Max reductions over f32 and Add scans over f32 only. Mul, And, Or, Xor for the reduction, Min/Max/Mul/And/Or/Xor for the scan, and the i32 and u32 dtypes for both are absent from the opcode set, the intrinsic table, and the Metal spellings.

**Unbacked (code exists, nothing checks it):**

- §1: "Emulated subgroups ... and the rule that subgroup operations do not require uniform control flow while barriers do." — no code — the rendezvous shares the barrier's state-split machinery (emit/coop.go:247), so the placement restriction is identical; §6.3 of the same spec concedes it.
- §6: "§1 is built except the CPU mode wiring noted below." — stale and self-contradicting — §6.3 records the wiring as shipped, and internal/cpu/profile.go:292-311 resolve implements developer, strict (with intersect(targets)) and mimic. The exception named in 
- §1: "The arm64 and amd64 numeric probes of 008, establishing the available exact domain." — partially verified only — internal/conformance/probe carries an Arch field and an arm64 case (probe_test.go:132, probe.go:223), but I found no amd64-specific probe or test case; confidence low on this

### [022-msl-target.md](022-msl-target.md)

**Unbuilt:**

- **§5, second bullet — the Metal family query behind atomic<float>** — internal/metal/metal_darwin.go:371-372 hardcodes AtomicFloatAddStorage:false and AtomicFloatAddShared:false. The code comment at :368-370 concedes it: "it needs the family query this table does not make yet". grep for supportsFamily / MTLGPUFamily over internal/mtl and internal/metal returns nothing
- **§5, third bullet — an array member of a uniform block** — No std140 array-member support. internal/kernelc/std140 and the generator at internal/kernelc/emit/emit.go:664-668 emit scalar/vector/struct members only, and the spec's stated blocker (a 16-byte stride forcing the index expression to be rewritten) has no code addressing it. The spec justifies defer

**Unbacked (code exists, nothing checks it):**

- §3's illustrative generated code: `return accel.EncodeKernelUniform(dst, v, ScaleParamsCodec{}.Encode)`. — No symbol named accel.EncodeKernelUniform exists; grep over the tree returns zero hits outside the spec. The generator emits kernelabi.EncodeUniform (internal/kernelc/emit/emit.go:293 and :577), and t
- §2's table row: "workgroup-shared memory | `threadgroup T *name [[threadgroup(k)]]`, extent fixed at pipeline creation". — The emitter does not emit a [[threadgroup(k)]] parameter and no host code sets a threadgroup length. internal/kernelc/emit/msl.go:277-283 declares shared storage as a body-local fixed-size array (`thr

### [023-metal-graph.md](023-metal-graph.md)

**Unbuilt:**

- **§1 — how much of a barrier an encoder boundary needs to be** — The measurement the section is built around does not exist. grep for memory_barrier / mem_device over internal/metal returns nothing; the only barrier spelling in the tree is threadgroup_barrier inside the emitter (internal/kernelc/emit/msl.go:1042), which is intra-kernel and unrelated. No benchmark
- **§5 item 4 — device loss signals every outstanding fence** — Only the sticky and report-on-next-call halves exist (internal/metal/metal_darwin.go:159-190, exec_darwin.go:287, :719-720). A fence created before the loss learns nothing until someone waits on it; there is no path that walks outstanding fences, because the executable tracks a single current fence 
- **§5 — MTLIndirectCommandBuffer** — Nothing in internal/mtl or internal/metal binds it; grep for IndirectCommandBuffer over the Go tree returns zero hits, and the only mentions are in specs/006, 009, 021 and 023. The spec states this deliberately and defers the gating rule to 006 §4.3, so it is scoped-out rather than forgotten — but i

**Unbacked (code exists, nothing checks it):**

- Frontmatter `status: implemented`. — The spec's own §5 closes with "What is not built: the measurement of whether a memory barrier inside one encoder would serve... and MTLIndirectCommandBuffer", and marks Done item 4 "partly met". Both 

### [025-tensor-operators.md](025-tensor-operators.md)

**Unbuilt:**

- **§6 Done bullet 1 — "every operator above builds, infers, lowers, and runs on both backends"** — The operators are built and lower on CPU, and their kernels agree on Metal through the corpus differential (internal/testkernels/differential_darwin_test.go:108,111,221,281,353). What has no both-backends test is the tensor-layer lowering of Mul, Scale, Softmax, RoPE and Linear: no *_darwin_test.go 

**Unbacked (code exists, nothing checks it):**

- §4.1: "an f32 GEMM is a corpus kernel that does not exist" and "MatMul, Linear and MatVec take f16 operands and produce f32". — Half false and contradicted by code. MatMulTiledF32Kernel and MatMulTiledF32F16Kernel are registered (internal/testkernels/accel_kernels.go; documented at specs/010-kernel-corpus.md:216) and tensor/ma
- §3 operator table names the operator `Rows`. — No such exported function. The operator is tensor.GatherRows (tensor/ops.go:50). A reader following the spec's API name finds nothing.
- §3 table: `Cast` — "the identity is a no-op rather than a copy". — no test. Code is tensor/ops.go:288-292 (returns x unchanged when dtype matches), but no test in tensor/ constructs a same-dtype Cast and asserts that no node is recorded; grep for Cast across tensor/*
- §4.4 and §5: RoPE's copy into scratch "is reported". — no test. tensor/compile.go:421-425 appends a KernelSelection{Kernel: "copy"} for the in-place path, but tensor/ops_test.go:345's RoPE subtest never reads plan.Selections(), and no other test asserts t
- §2: "`MatMul` operands are refused" rather than materialized. — no test for MatMul specifically. The shared lowering refusal at tensor/compile.go:370 is exercised only through RMSNorm (tensor/opsrefusal_test.go:169). No test feeds a strided or broadcast operand to

### [029-plan-cache.md](029-plan-cache.md)

**Unbuilt:**

- **2. The cache, and the key (the compile-options row of the six-component table)** — The digest hashes a constant string, not any option: tensor/cache.go:78 writeString(h, "opts v1"). CompileOptions carries only Label and Label is excluded by design, so there is no compile option that can change the key. The row is a placeholder rather than a component.

**Unbacked (code exists, nothing checks it):**

- §4 Done: "two builders recording the same DAG produce the same key, and every one of §2's six components changes it" — Only four of the six are exercised: DAG structure, ports/scalars, and kernel digests are covered by tensor/cache_test.go:66 and :101. The device adapter token is in the key (tensor/cache.go:73) but no

### [030-paged-kv.md](030-paged-kv.md)

**Unbuilt:**

- **4.1 Outcome (the API name) and §2 Pages, as a caller-facing mechanism** — There is no tensor.BlockPool. The pool is tensor/internal/pagetable.BlockPool, and grep over the repo shows its only importer is its own test (tensor/internal/pagetable/pagetable_test.go:11). Its package doc (pagetable.go:5-12) states it stays internal until an operator accepts a page table and that

**Unbacked (code exists, nothing checks it):**

- §4.1: "Built as AttentionDecodePaged and tensor.BlockPool" — tensor.BlockPool does not exist. The type is tensor/internal/pagetable.BlockPool, unexported from the tensor package; grep for "BlockPool" across all non-worktree Go files returns only tensor/internal
- §5 Done: "a pool hands out blocks, takes them back, and refuses when empty rather than overwriting" - as a property of this spec's shipped surface — The behaviour is implemented and tested, but only inside an internal package with no consumer. Nothing that a caller can reach hands out a block, so the paging mechanism §2 designs is not usable throu

### [032-stage-abi.md](032-stage-abi.md)  · **too large, see the split plan**

**Unbuilt:**

- **3.1 Interpolation** — No interpolation qualifiers exist. There is no `flat` or `noperspective` tag handling in the front end or either emitter, and no rejection of an integer varying. Worse than absent: emit/emit.go:416-430 stageFlatten appends every varyings field straight into a []float32, so an authored integer varyin
- **3.2 The varyings slot limit** — No slot computation and no limit. Nothing computes ceil(components/4) per field, no device limit is consulted at generation, and no error names the field list, the count and the limit.
- **3.3 One struct type, both stages** — The check is not at generation and not by resolved object. The compiler stores only the struct's *name* (emit/emit.go:279 `Varyings: %q`), and the pairing is compared as two strings at pipeline creation (render.go:328). The degenerate case §3.3 exists for - two unrelated empty structs - pairs whenev
- **4.2 Discard** — There is no accel.Discard intrinsic, no IR node, and no lowering. ir.Func.Discards is never set, so kernel.Stage.Discards is always false and no backend decision depends on it.
- **4.3 Not in the baseline: fragment storage writes** — A slice parameter in a fragment stage is accepted as a storage binding (front/front.go:788). No compile error, no mention of ROA. Since a draw has no way to bind a stage storage buffer, such a stage compiles to a binding nothing can supply.
- **6 What the IR gains (the Builtins row)** — ir.Func has no Builtins field. ir.Func.Intrinsics []string (ir.go:829) carries roughly equivalent information, so this is a record-shape divergence rather than missing capability, but the table as written does not describe the IR.
- **6.1 Binding indices are per stage** — kernel.Stage carries no bindings list at all, and ir.Binding.Index is set to the *parameter* index (front/front.go:793) - exactly what this section says it must not be. Only Attributes, Uniforms and Textures are dense. The gap is currently unreachable because a stage cannot bind a buffer, which is t
- **9 Errors, all at generation** — Four of the eight listed generation errors do not exist (integer varying without flat, varyings slot limit, written slice naming ROA, no object-identity varyings check), and two more exist only at pipeline creation rather than at generation (attachment count at render.go:305, attribute format at ren

**Unbacked (code exists, nothing checks it):**

- §10 Done: "an integer varying without the flat tag is refused, and the refusal names the tag" — No code and no test. No tag parsing exists anywhere in the repo; front/stage.go:262-288 records an integer field as an ordinary varying and emit/emit.go:420 then packs it into []float32, which is a ge
- §10 Done: "a fragment stage whose varyings type is merely structurally identical is refused, confirmed by making the two structs match field for field and check — No such test exists. The only varyings-mismatch test is render_test.go:98, which swaps in a stage whose varyings are named differently (NoVaryings vs Varyings). The mechanism is a string comparison at
- §10 Done: "accel.Discard writes neither colour nor depth, checked against a target pre-cleared to a value the stage never writes" — No code and no test. accel.Discard does not exist; the only Discard symbols in the repo are Surface.Discard and StoreDiscard, which are unrelated.
- §10 Done: "a fragment stage with a written slice parameter is refused by name" — No code and no test. front/front.go:788 classifies a slice parameter as a storage binding for a stage exactly as it does for a compute kernel, and nothing inspects whether a stage writes it.
- §10 Done: "a texel fetch returns the same value on both backends for every in-range coordinate, and zero for every out-of-range one" — The Metal backend never binds a stage texture. driver.RenderDraw.VertexTextures/FragmentTextures (internal/driver/plan.go:738) are consumed only by internal/cpu/exec.go:654-663; grep for setFragmentTe
- §5.2: "three things are outstanding" and §12.2: "Not built: the texel fetch of §5 ... That is also what blocks 033's feedback rejection" — Stale in the opposite direction - these say work is missing that has shipped, which invites a contributor to rebuild it. kernel.VertexFn/FragmentFn already carry `textures []Texture2D` (internal/kerne
- §9: "a varyings struct exceeding the interpolation slot limit" and "a fragment output struct with more fields than the target profile's attachment limit", both  — The first has no implementation at all. The second exists only as a device-limit check at pipeline creation (render.go:305-307), which is precisely the placement §3.2 and §9 argue against because it a

### [033-render-api.md](033-render-api.md)  · **too large, see the split plan**

**Unbuilt:**

- **§2 and §2.1 — polygon fill mode** — PrimitiveState (render.go:98-102) is {Topology, FrontFace, Cull}. There is no Fill field and no FillSolid constant, so §2's own code example does not compile.
- **§2.1, §2.2 — stencil** — DepthStencilState (render.go:109-114) is {Format, Test, Write, Compare}. No stencil ops, read/write masks, per-face state or dynamic reference value is exposed. internal/raster/pipeline.go:64-137 implements all of it, but nothing in the accel API can reach it. §2.2's creation-time refusal for stenci
- **§2.3 — depth bias** — Constant, slope and clamp do not exist. grep for DepthBias over the whole tree returns nothing. RenderPipelineDescriptor carries no bias fields and no backend applies one.
- **§3.4 — viewport and scissor per pass** — RenderPassDescriptor (render.go:425-441) carries Width/Height only. internal/cpu/render.go:169 hardcodes raster.Viewport{W: rp.Width, H: rp.Height, MinDepth: 0, MaxDepth: 1}. internal/raster has a Rect scissor (raster.go:145) that no public call can set, so viewport and scissor are pinned to the are
- **§4 — DrawIndexedIndirect and DrawIndirectCount** — RenderPass exposes Draw (render.go:907), DrawIndexed (render.go:817) and DrawIndirect (render.go:866). The other two rows of §4's table have no method. There is no count buffer, so §4.2's n_drawn = min(n_device, n_max) is realised only against the arg buffer's own instance count (TestAnIndirectDrawC
- **§4.1 and §6 — the recorded uniform offset channel** — No call binds a BufferView to a stage uniform. render_feedback_test.go:221 TestADrawCanBeParameterisedByAUniformBuffer t.Skip()s naming exactly this. The N-object replay frame does not exist. (The spec's Outstanding table admits this one.)
- **§6 — the `Clear` without `LoadClear` refusal** — Recorder.RenderPass (render.go:545-622) never inspects ColorAttachment.Clear or DepthAttachment.Clear against the load op; build.go:448 appends c.Clear unconditionally. A clear value set with LoadKeep or LoadDontCare is silently ignored, which is the exact hazard §3.1 says must be an error.

**Unbacked (code exists, nothing checks it):**

- §2.2: "each target's format admits the field's type" is checked at pipeline creation. — no code. kernel.StageOutput (internal/kernel/stagerecord.go:162-165) is {Name string; Index int} — it carries no type at all, so the check is not expressible. NewRenderPipeline (render.go:296-320) che
- §2.2: "the varyings slot count fits the device limit (032 §3.2)". — no code. render.go:328 compares Vertex.Varyings != Fragment.Varyings, two strings. No slot count is computed and no limit is consulted; grep for MaxVaryings over the tree returns nothing.
- §2.2: "A pipeline with a stencil state but no stencil aspect in its depth format is an error at creation, naming both." — no code. There is no stencil state on DepthStencilState to carry, and NewRenderPipeline's depth branch (render.go:316-322) only checks info.IsDepth.
- §2.3: the depth-bias formula, and "[035] implements this formula". — no code, no test. Neither accel nor internal/raster nor internal/cpu nor internal/mtl mentions depth bias. 035 cannot implement a formula 033 never exposes.
- §6 error table: "`Clear` set without `LoadClear` — graph build". — no code, no test. render.go:545-622 and build.go:440-470 never compare a clear value against a load op.
- §7 Done bullet: "feedback is rejected for an overlapping subresource and **accepted** for a disjoint mip and a disjoint array layer, which is the half a handle  — no test — the accepting half skips. render_feedback_test.go:154 TestADisjointSubresourceIsNotFeedback calls t.Skipf as soon as NewTexture with MipLevels: 2 is refused (045 §8.3 still refuses it), so t
- §2 and §3's code examples name accel.FillSolid, accel.Rect{W,H}, accel.ClearColor{}, accel.ClearDepth{Depth: 1.0}, and pass.SetBindings(bindings). — no code. None of those five identifiers exists. The real shapes are RenderPassDescriptor.Width/Height (render.go:437), ColorAttachment.Clear [4]float32 (render.go:397), DepthAttachment.Clear float32 (
- §4's call table lists DrawIndexedIndirect and DrawIndirectCount as calls the API offers. — no code, no test. Only Draw, DrawIndexed and DrawIndirect are methods on RenderPass.
- §2.1: "Sample count, fixed to 1" is a compiled pipeline property. — no code. Nothing in the tree names a sample count (grep SampleCount/MSAA returns nothing outside specs/041). Nothing records or enforces the value, so the row states a property no object holds.

### [034-surface-present.md](034-surface-present.md)

**Unbuilt:**

- **§2 — three of the six rows in the present-slot table** — presentSlot (surface.go:501-506) records surface, generation, width and height. It does not record format, does not record render-target usage, and does not record a final state of Present. The device check happens in PresentSlot (surface.go:479) rather than in the slot record, so five of §2's six r
- **§2 — the present transition as the pass's store** — "recording that the slot's final state is Present makes the present transition representable to the graph planner, so the transition to present-ready is emitted as the pass's store". BindPresent (surface.go:514-542) validates and then calls g.Bind with an ordinary SlotBinding. Nothing in build.go, g
- **§6 — the constraint-reporting half of the boundary** — "accel reports its constraints — required visual or pixel format, supported present modes, supported extents". No API returns any of the three. §6 argues the boundary needs traffic in both directions and cites the predecessor's WindowVisualID; only the inbound direction (NativeHandle) exists.
- **§6 — every non-Metal native handle** — X11 Display*+XID, Wayland wl_display+wl_surface, and Win32 HWND appear in §6's table. NativeHandleKind has NativeMetalLayer and NativeNSView only, and NativeNSView is itself a refusal (surface.go:589, internal/metal/present_darwin.go:126). Self-disclaimed in §8.2.
- **§8.2 — the compositor handoff** — A drawable reaching a screen with the bounded pool, the blocking acquire and vsync. Self-disclaimed, and present_darwin_test.go:19-36 records the same three-claim split in code. Kept here because §9's last Done bullet claims it anyway.

**Unbacked (code exists, nothing checks it):**

- §2: "PresentSlot records, internally: ... format and extent ... render-target usage ... final state `Present`" and "`BindPresent` takes a `Frame`, never a naked — no code. presentSlot (surface.go:501-506) has four fields — surface, gen, width, height. There is no format field, no usage field and no final-state field, so BindPresent (surface.go:514-542) verifies
- §2 and §9 bullet 3: "the graph carries the transition as the pass's store rather than as a node" / "a frame ... binds and reaches the present state". — no code, no test. BindPresent terminates in g.Bind(SlotBinding{Slot: slot, Buffer: f.view}) — an ordinary slot bind. No present state, no store transition, and no test asserts either the presence of t
- §6: "accel reports its constraints — required visual or pixel format, supported present modes, supported extents — and accepts a platform-tagged native handle". — no code. Only the second half exists. grep over the whole tree finds no present-mode, visual-ID or supported-extent query on Device, DeviceInfo, Capabilities or Surface.
- §9 Done bullet: "Metal's `CAMetalLayer` drawable path presents on a machine with a display, tracked as its own claim." — no test. present_darwin_test.go:38 TestTheFrameLoopAgainstAMetalLayer runs against mtl.NewOffscreenLayer — a layer attached to no window — and its own doc comment (lines 30-36) states "Claim 3 needs a
- §9 Done bullet: "the headless frame loop runs several frames with double buffering and a resize in the middle, verified by readback, on every backend and with n — no test on the second backend. TestTheHeadlessFrameLoop (surface_test.go:70) opens through openDevice, which is accel.OpenCPU (graph_test.go:20-28). No headless-surface frame loop runs on Metal; the o

### [035-cpu-rasterizer.md](035-cpu-rasterizer.md)  · **too large, see the split plan**

**Unbuilt:**

- **§7 corpus row: "Origin agreement, discriminating form"** — The two-path test (path A: compute texel fetch of (x,0) then a buffer read; path B: host texture read of row 0; assert A == B and that both hold the top-row value) does not exist. internal/raster/raster_test.go:283 TestRowZeroIsTheTopRow checks the rasterizer's own window mapping only — one path, no
- **§7 corpus row: "Handoff stays on device"** — No test inspects a deferred graph's nodes for the absence of a host transfer between a geometry pass and a tonemap. No such graph is built anywhere in the tree.
- **§7 corpus row: "Depth readback through a transfer node"** — No test reads a depth attachment back through a transfer node. The macOS private-depth constraint this row exists to cover is unverified.
- **§7 corpus row: "Per-object replay"** — N objects at recorded uniform offsets, submitted twice. Blocked by 033 §4.1's unbuilt channel; render_feedback_test.go:221 skips with that reason.
- **§7 corpus row: "Determinism"** — No test submits the same render graph twice on one backend and compares the images. TestPlansAreDeterministic (internal/conformance) is about plan-cache identity, not pixels.
- **§6 bounded row: "depth-bias results"** — Depth bias does not exist in any package (033 §2.3 is unbuilt), so the row declares a side for a comparison that cannot run.
- **§7.1 — the differential's coverage** — "Every corpus entry that produces pixels runs on the CPU backend **and** on Metal". The MRT, per-attachment blend, write-mask, stencil and discard entries live only in internal/raster/pipeline_test.go and internal/cpu/render_test.go, both CPU-side. The Metal differential (render_darwin_test.go) cove

**Unbacked (code exists, nothing checks it):**

- §9 Done bullet 1: "every corpus entry in section 7 exists, declares its side, and runs on the CPU backend". — no code, no test, for five of eighteen entries: Origin agreement, Handoff stays on device, Depth readback through a transfer node, Per-object replay, and Determinism. Verified by enumerating every `fu
- §9 Done bullet 2: "every pixel-producing entry also runs on Metal and is compared on its declared side". — no test on Metal for the MRT, blend, write-mask, stencil and discard entries. Those assertions exist only in internal/raster/pipeline_test.go and internal/cpu/render_test.go, neither of which touches 
- §9 Done bullet 6: "the origin test asserts both equality and the top-row value, confirmed by mirroring both paths and checking the test still fails". — no test. There is no origin test at all, so nothing was confirmed by mirroring. This is the entry §7 singles out as catching the predecessor's actual bug, and docs/conventions.md:33 says explicitly th
- §9 Done bullet 7: "`conventions.md`'s graphics entries — clip depth range, winding, readback origin, depth storage mode — each name the corpus test that verifie — no code. docs/conventions.md lines 31-33 name *backends* ("both backends", "the CPU rasterizer only"), not corpus tests. No test name appears in that table, and the readback-origin row states the oppo
- §7 corpus row "Feedback validation ... disjoint mip and disjoint layer accepted", declared side exact. — no test — the accepting half skips. render_feedback_test.go:154 TestADisjointSubresourceIsNotFeedback t.Skipf()s when NewTexture with MipLevels: 2 is refused, which it still is. Only the rejecting hal
- §7.1: "When a comparison fails, `conventions.md`'s diagnostic order applies ... **Equal pixel counts with roughly half overlap is the flip fingerprint**". — no code. No differential in internal/testkernels computes coverage counts or overlap between competing interpretations on failure; the comparisons report per-pixel deltas. The diagnostic procedure is 

### [036-documentation.md](036-documentation.md)

**Unbuilt:**

- **§3.1 — "Every example compiles, because `go test` runs it"** — Zero `func Example` functions exist anywhere in the repository. Tutorial code is fenced blocks. All three of §3.1's stated reasons (go test compiles them, pkg.go.dev renders them, no examples/ directory to widen the coverage gate for) are unrealised. The trailing 2026-08-25 correction admits this; t
- **§3 — the automated extraction check** — "every kernel is extracted from the prose and run through the real generator, every host program is extracted and executed from a clean module, every Go block is parsed". Only the fourth of the four (symbol existence) is automated, in docdrift_test.go. The spec's own inline correction says "Automati
- **§5.6 — the per-tutorial gate table** — The table is stale in both directions rather than unbuilt. Rows 8, 9 and 10 list preconditions and §3 says tutorials 8 and 9 are "not written" — but docs/tutorial/09-a-decode-step.md and 10-quantized-weights.md are on disk. There is also an eleventh page, 11-batching-sequences.md, that the table has
- **§7 — the three open questions** — docs/architecture.md still sits in docs/ with the contributor-facing audience §1's rule assigns to specs/. Neither the README status-table question nor the conventions.md placement question is resolved. Open questions are legitimately open, listed so the spec's status is not read as finished.

**Unbacked (code exists, nothing checks it):**

- §6 Done bullet 5: "the ten tutorials exist, each teaching one concept, each ending with the reader having run something". — the count is wrong in both places. docs/tutorial/ holds eleven pages. §3 says the deck "shipped as eight" and that a decode step and quantized weights "are not written" — 09-a-decode-step.md and 10-qu
- §6 Done bullet 6: "every tutorial example is an `Example` function that `go test` runs". — no code. grep for `func Example` over every *_test.go in the tree returns nothing. docdrift_test.go's own header comment names this requirement as the one it replaced with something narrower.
- §3: "Graphics tutorials are **not** in this deck. [033] and [034] specify a public API that does not exist in code, and a tutorial for it would be fiction." — contradicted by code. render.go (1103 lines) and surface.go (637 lines) implement the API, Metal runs it, and README.md:32 already tells readers "Graphics works". §5's own inline correction says the r
- §5.5: "the stage types and the `//accel:vertex`/`//accel:fragment` directives (nothing runs them and the directives are silently ignored)". — contradicted by code. internal/kernelc/front/stage.go:120-250 compiles both directives, kernel.Stage carries RunVertex/RunFragment/MSL (internal/kernel/stagerecord.go:68-86), accel.Stage is exported (
- §3 and §5.6: no tutorial is written before its gate row is clear, and a decode step and quantized weights wait on Contiguous, PlanCache ownership documentation, — contradicted by the filesystem. Both pages exist. Nothing in the spec records that the four preconditions cleared, so either the gate was bypassed or the spec was never updated — and a reader cannot t

### [039-sampling-policy.md](039-sampling-policy.md)

**Unbuilt:**

- **§6 The public type** — `Shape(vocab int) SamplingShape` and the `SamplingShape` type do not exist (the spec's State block admits this). `SamplingOptions.HistoryCap int` is also absent from tensor/policy.go:36-57 and is not listed among the four recorded deviations. `Sample`'s declared signature `(b, logits, history *Tenso
- **§8 What this costs — "Single sequence only"** — Half stale. SampleCategorical now takes a per-row draws tensor (tensor/sample.go:213) and a batch works end to end (tensor/sample_test.go:202 TestCategoricalDrawsEachRowAgainstItsOwnUniform, :542 TestABatchOfOneIsTheSamePath), so the "writes out[0] from a uniform draw" premise no longer holds. Still
- **§9 Done — the history-shape bullet** — No test asserts "a history at capacity keeps binding one shape ... one compile across 4×HistoryCap steps". The only compile-count assertions in tensor/ are tensor/cache_test.go:134 and tensor/ragged_test.go:490, neither about the history ring.

**Unbacked (code exists, nothing checks it):**

- §6's API block declares `HistoryCap int` on `SamplingOptions`. — No such field. tensor/policy.go:36-57 declares Temperature, TopK, TopP, Repetition, Presence, Frequency only; the capacity is a parameter of Scalars (policy.go:148) and a property of the caller's hist
- Done: "A history at capacity keeps binding one shape, asserted as one compile across 4×HistoryCap steps; binding the history at its current length compiles ever — no test. Nothing in tensor/policy_test.go or tensor/sample_test.go compiles a plan repeatedly across a filling history ring, and the State block claims §9's assertions are built.
- Done: "`TopK = vocab` is refused, not silently top-128; the mutation is clamping instead of refusing." — no test for the stated condition. Validate (tensor/policy.go:110-116) refuses only TopK > TopMaxRounds, and the only test case is TopK = TopMaxRounds+1 (tensor/policy_test.go:183). For a vocabulary at
- Done: "Changing `Temperature` at step k does not change the draws at steps > k, which fails for any design where the stream position depends on the policy." — no test. Stream.Draw (tensor/stream.go:85) takes only a step and cannot see a policy, so the property holds by construction, but no test varies Temperature across steps and re-reads the draw stream; t

### [042-surface-completion.md](042-surface-completion.md)  · **too large, see the split plan**

**Unbuilt:**

- **§6 The documentation guard** — No `Example` functions exist anywhere (`grep -c '^func Example'` = 0). docdrift_test.go checks that names used in docs still exist; it does not compile tutorial code, which is the guard 036 §3.1 and this section specify.
- **§5 / §7 "the freeze record covers graphics"** — specs/036-documentation.md §5 still reads "Graphics is still outside the freeze" with the event named as 045. No graphics tier list (§5.1/§5.2 equivalent) exists for render pipelines, passes, attachments or draws.
- **§5.3 "Metal's ABI is refusing callers on devices that are not Metal"** — Limits has no MaxVertexBuffers field (limits.go). vertexlayout.go:132 still refuses >16 vertex buffers on every device from the mslabi.StageVertexBufferLimit constant, and the error does not name a device number.
- **§5.3 "The CPU oracle is setting the public API's ceiling"** — No normalized integer vertex attribute formats. AttrFormat is {AttrFloat32, x2, x3, x4} only (vertexlayout.go:38-44) and the doc comment still gives the oracle-parity reason this section rejects. No unorm8/snorm16 conversion is stated anywhere.
- **§3.1 Uniform buffers: keep, and connect** — No draw or dispatch takes a BufferView as a stage uniform. UniformBuffer.View (uniform.go:177) has no consumer; render_feedback_test.go:238 skips with that reason. §7's bullet 2 additionally still says "a dispatch reads a UniformBuffer at a recorded offset", which §3.1's 2026-08-24 correction replac
- **§2.1 Line and point rasterization** — Not built (render.go:284 and internal/raster/draw.go:103-108 accept triangles only). §2.1 moved this out pending a hardware measurement, but §7's bullet 5 still lists "lines and points rasterize" as Done criteria — the two sections disagree.
- **§2.1 Texture attachments / texel fetch (the Metal half)** — The binding half exists only on the CPU backend. build.go:846-900 fills driver.RenderDraw.VertexTextures/FragmentTextures; only internal/cpu/exec.go:654-658 reads them. `grep -rn Textures internal/metal internal/mtl` returns nothing, and internal/kernelc/emit/mslstage.go:162 still emits `texture2d<f

**Unbacked (code exists, nothing checks it):**

- §7: "no exported declaration is reachable-and-unusable without a refusal naming what is missing" — the spec's whole thesis. — Falsified by RenderPass.SetTexture on a Metal device. build.go:860-900 accepts the binding and lowers it into driver.RenderDraw.FragmentTextures; internal/metal reads that field nowhere and refuses no
- §7: "every tutorial's code is an Example that go test compiles" — No code implements it. There are zero Example functions in the repository. The eight tutorials under docs/tutorial are still fenced blocks that nothing compiles; docdrift_test.go only greps names out 
- §7: "the freeze record covers graphics" — No code or document implements it. specs/036-documentation.md:197 still states the freeze covers the compute half and that graphics is outside it, with the event named as 045.
- §5.2: "Renderable and Blendable acquire consumers" once the attachment change lands — Blendable is populated at format.go:198 and read nowhere in the repository; only Renderable/Sampleable/StorageRead/StorageWrite gained a consumer (texturealloc.go:98-100). The CPU half of 045 has land

### [044-unbounded-context.md](044-unbounded-context.md)

**Unbuilt:**

- **§7, final bullet — "Selections() names the kernel and the tile count"** — The reason strings carry the kernel and the cached-position count (tensor/attention.go:561) or the page-block size (:338,:397,:469,:527). Nothing computes or reports how many 128-wide tiles the loop will run. No test asserts a tile count appears in Selections().

**Unbacked (code exists, nothing checks it):**

- Frontmatter `status: implemented` — The spec's own §8 outcome table records "`Selections()` names the kernel and the tile count | **not done**" and §8's "What this did not close" repeats it. Verified in code: no selection reason contain
- §8 "What this did not close": "Issue 9, the LayerState view, is untouched: `Attention` still refuses a non-zero offset." — Overtaken by shipped code. tensor/attention.go contains no offset refusal, and Attention is called on LayerState views in passing tests — tensor/decode_test.go:903 and tensor/bindbench_test.go:65-66. 

### [045-texture-attachments.md](045-texture-attachments.md)  · **too large, see the split plan**

**Unbuilt:**

- **§4 stages row / §7 bullet 4 — texel fetch on Metal** — internal/metal never binds a stage texture (no setFragmentTexture:/setVertexTexture: selector in internal/mtl/render_darwin.go, no read of driver.RenderDraw.FragmentTextures anywhere in internal/metal), and nothing refuses the case. The MSL emitter does declare the argument (internal/kernelc/emit/ms
- **§4 Metal row / §7 bullet 7 — staging copies and the present conversion draw** — internal/metal/render_darwin.go:67 still blits the attachment buffer into the pass texture and :103/:110 blit it back, once per pass. internal/metal/present_darwin.go:37 still carries presentSource, "the built-in stage pair that converts a rendered frame". No test counts a frame's transfers.
- **§4 resources row / §7 bullet 1 — MipLevels and ArrayLayers** — texturealloc.go:76-85 still refuses both above one. §8.3 records the reason (textureBytes sizes one level, texture-buffer copies move one, and a recorded access covers the whole byte range), and none of those three layers has changed.
- **§7 bullet 5 / §9 — the accepting half of the disjoint-subresource rule** — TestADisjointSubresourceIsNotFeedback (render_feedback_test.go:154) skips because a two-mip texture cannot be created. The permission is asserted by nothing; only the refusal (render_feedback_test.go:25) is exercised.
- **§1 — the FormatInfo table's unread fields** — Blendable is populated at format.go:198 and read nowhere in the repository. §1's symptom "Blendable is read nowhere" still holds after the CPU half landed. Only Renderable/Sampleable/StorageRead/StorageWrite gained a consumer (texturealloc.go:98-100).
- **§8.1 — the four rules the pass owns** — There is no rule that a texture bound with RenderPass.SetTexture and fetched by a stage declares TextureSampled. textureOperands (build.go:846-900) checks binding presence and feedback but never the usage flag, and TextureSampled is checked only at allocation (texturealloc.go:99).

**Unbacked (code exists, nothing checks it):**

- §8.4: "Only the CPU backend lowers a texture attachment. A Metal device refuses one at build, naming the pass, the attachment and this spec." — Overtaken by the spec's own §6.1 "Closed" note and contradicted by the code: build.go:759 documents that the gate was deleted — "Both backends lower a texture attachment. The gate that stood here refu
- §7 bullet 1: "an attachment names a TextureView, and MipLevels/ArrayLayers above one are accepted rather than refused" — Half of it is unimplemented: texturealloc.go:76-85 refuses both, TestMipsAboveTheBaseLevelAreRefused (texture_test.go:613) asserts the refusal. §8.3 records the deviation but the §7 Done list was left
- §8.1: "texel fetch, the binding half — **done** — RenderPass.SetTexture, a texture channel in the flat stage form ... and a refusal for a stage that fetches a s — True only on the CPU backend, and the spec does not say so. On Metal the binding is accepted at build (build.go:860-900) and dropped at execution — internal/metal reads neither VertexTextures nor Frag
- §8.5's owed list, first two bullets — Internally inconsistent with §8.1 and §9, which mark the texel-fetch binding half and feedback rejection **done**. §8.5 says it is "Unchanged from §7, minus what §8.1 marks done" and then repeats both

### [046-segmented-extents.md](046-segmented-extents.md)

**Unbuilt:**

- **§4 Refusals, row 1 ("QueryExtents whose element count is not the batch")** — No such refusal can exist: tensor/attention.go:211 DEFINES batch as opts.QueryExtents.shape.Elements(), so there is nothing to compare it against. The real batch-consistency checks are Lengths count vs batch (attention.go:272) and Pages.shape[0] vs batch (attention.go:288). The row should be rewritt
- **§4 Refusals, row 2 ("Σn_r ≠ q.shape[0]")** — No code and no test. §6's own Correction establishes it can never be built. Delete the row.
- **§6 Open** — Deferred by design (binary-search segment lookup; deriving Lengths from QueryExtents). §5.1 records this explicitly, so it is not a broken promise.

**Unbacked (code exists, nothing checks it):**

- §4 Refusals: "Σn_r ≠ q.shape[0] | §1 property 3: the host can check it and the kernel cannot" — listed as a host-side refusal. — No code and no test. tensor.Attention has no host value to compare q.shape[0] against, because QueryExtents is a tensor. The spec's OWN §6 Correction retracts this exact claim and states "the host can
- §4 Refusals: "QueryExtents whose element count is not the batch | one count per row is what the primitive is". — No code implements it and no test checks it. batch is derived from QueryExtents itself (tensor/attention.go:211), so the refusal is unwritable as stated. The only refusal on that element count is the 
- §4 header: "Every one is host-side, because 043 §2's rule cuts both ways: a value that reaches the device as data cannot be checked there." — Two of the five rows are not host-side refusals at all — one is enforced in the kernel (property 3's padding write, internal/testkernels/ragged.go:110) and one does not exist. The header is what makes

### [047-linear-attention.md](047-linear-attention.md)

**Unbuilt:**

- **§6.1 "What is still not built: the kernel" (linear_attention_chunked)** — The chunked UT-transform kernel itself. The derivation and the Go-side oracle exist (internal/testkernels/linear_test.go:424), but no kernel is authored or registered — confirmed against internal/kernelc/kernelc_test.go:51 and specs/010-kernel-corpus.md:129. Also missing: the refusal of chunk sizes 
- **§2 / §2.1, the over-sum direction of 046 §1 property 3** — No clamp of `last` to the token count in LinearAttention, no host refusal, no test. Counts summing past q.shape[0] read q/k/alpha/beta and write out past their ends.

**Unbacked (code exists, nothing checks it):**

- §5: "A sequence's tokens do not disturb another sequence's state. Two sequences in one step, and each slot equals what that sequence alone produced — the same o — No test runs two NON-EMPTY sequences and compares each slot against that sequence run alone. Both candidate tests use a second sequence with count zero: internal/testkernels/linear_test.go:235 TestALi
- §2 states the input is 046's segmented extent unchanged, which carries 046 §1 property 3. — No code honours property 3's over-sum direction in this kernel and no test checks it. `last` comes straight from the offsets with no clamp in internal/testkernels/linear.go:75, in the flat lowering, o
- §5: "alpha = 1, beta = 0 ... makes o the same for every token." — No assertion. internal/testkernels/linear_test.go:209 TestALinearStepWithNoWriteLeavesTheStateAlone checks only that the state did not move, then closes with `if len(out) == 0 { t.Fatal("no output") }
- §5: "The state's shape is [slots, heads, V, K] and a step writes only its own slot, asserted by leaving another slot filled with a sentinel." — No sentinel exists. The untouched slot starts at zero in both tests (tensor/linear_test.go:197 reads it back and asserts it is still 0), so "untouched" is indistinguishable from "written with zeros". 


## The split plan

Eight specs are **too large to finish in one go**, and that is a property of
how they were written rather than of the work left. Each owns several unbuilt
chunks that touch disjoint files, block none of the others, and could each
ship alone. A spec in that shape cannot be driven to `implemented`, because
there is no single change that completes it — which is exactly why they have
sat `in progress` while their dependents were marked done.

Each successor below is scoped to be completable in one pass. They are
proposals, not yet written as spec files.

### [002-compute-model.md](002-compute-model.md) → 5 successors

It is in progress and owns at least five independent unbuilt chunks, each with its own IR ops, emitter cases, capability wiring and test corpus, and none blocking the others: the barrier storage-class masks, the saturating float-to-integer conversions, the three dispatch-shape accessors, a spellable ballot with a public mask type, and the remaining subgroup reductions. Bundling them keeps the spec

- **Barrier storage-class masks and the Metal device-memory barrier** — §2.3, §2.4, §2.5 and §3.1 at subgroup scope. Add BarrierShared, BarrierStorage and SubgroupBarrier as Thread methods, IR ops and intrinsics; give the CPU scheduler a per-class rendezvous so a masked barrier orders only the class it names; fix emit/msl.go:1042 to emit mem_threadgr
- **Saturating float-to-integer conversion intrinsics** — §6.2's last three rows and the API requirement it adds. Add accel.ToI32, ToU32, ToI8, ToU8 with toward-zero saturating, NaN-to-zero semantics in the IR, the CPU lowering and MSL; make front/build.go:1024 reject a bare float-source conversion with a message naming the intrinsic. T
- ~~**Dispatch-shape accessors as uniformity seeds**~~ — **written as [052](052-dispatch-shape.md) and built 2026-08-28.** §1.3's WorkgroupSize/NumGroups/GlobalSize rows and their §3.3 workgroup-uniform seed entries. Add the three Thread methods, the three IR ops and their intrinsic entries, seed them WorkgroupUniform, and lower them on both backends. Also rename §1.3's SubgroupID/SubgroupInvocationI
- **A ballot a kernel can spell** — §5.2's Ballot row and its KernelMask value type. Add the Ballot intrinsic and IR op, export the mask as accel.KernelMask with the full Count/Bit/LowestSet/CountLower/Any method set, and lower it where the backend can (Metal cannot, per 022 §5, so this is also the first case where
- **The remaining subgroup reductions** — §5.2's reduction row beyond Add/Min/Max over f32: Mul, the i32 and u32 reductions, and the And/Or/Xor family, plus the inclusive and exclusive scans for the types they gain. Tests: each against its no-subgroup fallback, exact for integers and within §5.5's tolerance for f32, swep

### [006-backends.md](006-backends.md) → 4 successors

In progress with several independent unbuilt chunks that ship separately. Vulkan and SPIR-V are already owned by drafted child specs 037 and 038, so they need no new spec. What has no owner is the rest: §6.1's isolated probe process and §6.3's environment override (one selection/enumeration slice), §3's packed narrow-dtype emulation contract plus its conformance test, the cross-architecture oracle

- **Isolated adapter probing and the environment override** — §6.1's helper subprocess with a versioned wire record, mapping abnormal exit, signal, malformed output and timeout to stage-specific ProbeDiagnostics; §6.3's ACCEL_BACKEND restriction of automatic selection only, plus SelectionReport.EnvironmentBackend and the test that an explic
- **Packed narrow-dtype storage emulation** — §3's rule that an emulated 16- or 8-bit store uses a word-sized compare-exchange loop with matching atomic word loads, the backend-facing helper it lowers through, and the conformance test that concurrently writes every lane of one backing word. Definable and testable on the CPU 
- **Cross-architecture oracle determinism** — The CPU target's obligation to emit an explicit float32 rounding point at every f32 rounding site, and the conformance run that compares the same class-A kernels built on arm64 and amd64 for bit identity, wired into tier-1 CI so the oracle cannot differ between a laptop and a run
- **The remaining backend set and the D3D12 shader-model decision** — §2.4, §2.5 and §2.6 as a set: resolve open question 4 (DXC dependency versus emitting DXIL) jointly with 004, decide whether GLES is still worth its column now that Windows GLES is non-blocking, and fix WebGPU's Pending[T] asynchronous surface before any of the three is dispatche

### [008-numerics.md](008-numerics.md) → 4 successors

In progress with four independent unbuilt chunks that each ship on their own and none of which blocks the others: the committed high-precision corpus and its generator, the profile-aware comparison API, §8's composed budgets with attribution, and the static tolerance check together with §11's missing edge tests. §5's non-Go contraction control is excluded from the split because it is blocked on ba

- **The committed high-precision primitive corpus** — §6's corpus: an offline generator producing input bits and correctly rounded reference bits at 256-bit precision, recording its oracle version and reproducible; the committed data files; and a runtime check that reads them instead of computing a float64 reference live. Includes §
- **The profile-aware comparison surface** — §9's Context and OracleID, the per-primitive entry points (Special, Division, Primitive, Reduction), and the failure report carrying backend/device, class, input index, got and reference bits, absolute and ULP error and allowed budget. Makes §3's 'exactness is a property of (clas
- **Composed operator budgets and the budget trace** — §8's forward sensitivity propagation, the recorded per-term trace, outward interval rounding at casts, and the refusal when an interval crosses a singularity or a derivative bound is missing; applied to RMSNorm, Softmax, MatMul/Linear, composed Attention and the golden model, so 
- **Mechanical rejection of tolerances, and budget monotonicity** — §9's static check over the conformance corpus rejecting approximate-comparison helpers and numeric tolerance arguments while allowing golden bit patterns and exact constants, wired into CI; plus §11's monotonicity tests — a synthetic result at the budget passes, the next represen

### [032-stage-abi.md](032-stage-abi.md) → 5 successors

In progress, and the outstanding work is five independent chunks with no ordering dependency between them: interpolation qualifiers, generation-time limits, discard, the fragment storage-write refusal, and Metal stage-texture binding. Each is a separate front-end or backend change that can land and be tested on its own, and §5.1/§5.2/§12.2 already contradict each other about what is done - which i

- **Varying interpolation qualifiers and the integer-varying refusal** — Parse the accel:"flat" and noperspective struct tags in front/stage.go's namedStructType, carry the qualifier in ir.Type.Field and kernel.Stage, refuse an untagged integer varying with an error naming the field and the tag, and make the generated flat form and the MSL stage agree
- **Generation-time limits: varying slots and attachment count** — Compute ceil(components/4) per varyings field against the target profile's interpolation budget, and the fragment output field count against its attachment limit, both in the generator with a source position. Move or duplicate render.go:305's pipeline-creation attachment check so
- **Discard: intrinsic, IR, both lowerings, and the early-Z record** — Add accel.Discard as an intrinsic, set ir.Func.Discards from the body, lower it to an early return in the generated Go rasterizer path and to discard_fragment() in MSL, and make the CPU rasterizer write neither colour nor depth for a discarded fragment. Test against a target pre-
- **The fragment storage-write refusal, naming ROA** — Classify a slice parameter in a graphics stage and refuse a written one at generation with an error naming rasterizer-ordered access and stating that no backend reports it. Decide in the same change whether a read-only stage slice is admitted at all, given that no draw path can b
- **Metal stage-texture binding, and the cross-backend fetch differential** — Consume driver.RenderDraw.VertexTextures/FragmentTextures in the Metal render encoder at mslabi.StageTextureIndex, then run a texel fetch over every in-range coordinate and a set of out-of-range ones on both backends and compare. This is the only remaining part of section 5.2, an

### [033-render-api.md](033-render-api.md) → 5 successors

in progress with at least five mutually independent unbuilt chunks that ship separately and touch different code: fixed-function pipeline state (fill mode, depth bias), the whole stencil half of the public API, the two missing indirect-draw entry points and their count buffer, per-pass viewport/scissor plus the missing Clear/LoadClear refusal, and the recorded-uniform-offset channel. None blocks a

- **Fixed-function pipeline state: fill mode and depth bias** — Add PrimitiveState.Fill with its constants and DepthStencilState/pipeline depth-bias constant, slope and clamp; lower both on the CPU rasterizer and Metal; implement §2.3's z' = z + slope*m + constant*r with the clamp; add the bounded-comparison corpus entry 035 §6 already declar
- **Stencil through the public render API** — Expose per-face compare, read/write masks, fail/depth-fail/pass ops and the dynamic reference value on DepthStencilState and the pass, wire them to the existing internal/raster StencilState (pipeline.go:64-137), add the creation-time refusal for stencil state without a stencil as
- **DrawIndexedIndirect, DrawIndirectCount, and the device count buffer** — Add the two missing draw entry points, a separate count-buffer argument with its build-time maximum, the on-device min(n_device, n_max) clamp, the indirect-read access kind for the count buffer, and the run-time-statistics report when the clamp fires. Capability-gate per decision
- **Per-pass viewport and scissor, and the clear/load-action refusal** — Add per-pass viewport and scissor to RenderPassDescriptor defaulting to the render area, validated against every attachment extent, plumbed to raster.State's existing Rect scissor and to Metal; and add §6's missing refusal of a Clear value set without LoadClear.
- **The recorded uniform offset channel for an N-object frame** — Bind a BufferView to a stage uniform at a recorded byte offset with the stride derived from minUniformBufferOffsetAlignment; land the N-object replay corpus entry (035 §7 "Per-object replay") submitted twice with different transform contents and no re-recording; un-skip render_fe

### [035-cpu-rasterizer.md](035-cpu-rasterizer.md) → 4 successors

in progress with five absent corpus entries that are mutually independent and each shippable alone, plus a differential-coverage gap that is its own piece of work. The rasterizer core (§1-§5, §8 steps 1-3) is genuinely finished and well tested; what remains is corpus breadth, and bundling it under one spec is why §9 could claim completeness while a third of §7 has no test. Per-object replay additi

- **The origin-agreement corpus entry** — Build §7's discriminating three-path test: a target encoding row position, path A a compute texel fetch of (x,0) then a buffer read, path B a host texture read of row 0; assert A == B and that both hold the top-row value; confirm by mirroring both paths and watching it fail. Then
- **Structural corpus: deferred handoff and depth readback** — Build the deferred graph (geometry pass to tonemap compute) and assert on its nodes that no host transfer sits between them; and read a depth attachment back through a transfer node, covering the macOS private-depth constraint. Both are structural or transfer-shaped, neither need
- **Render determinism and the Metal fixed-function differential** — Submit the same render graph twice on one backend and assert identical images (003's guarantee); and extend the Metal differential to the MRT, per-attachment blend, write-mask, stencil and discard entries that today assert only on the CPU side, so §7.1's "every pixel-producing en
- **The failure-diagnostic harness §7.1 describes** — Implement the coverage-count and overlap report the differential is supposed to emit on failure — the flip fingerprint of equal pixel counts with roughly half overlap — so a red comparison says which interpretation differed instead of dumping deltas.

### [042-surface-completion.md](042-surface-completion.md) → 5 successors

In progress with five mutually independent unbuilt chunks that touch disjoint files and could each ship alone: the tutorial Example guard (docs/tutorial/*), the graphics freeze record (specs/036), Limits.MaxVertexBuffers (limits.go + vertexlayout.go + internal/mslabi), normalized integer vertex formats (vertexlayout.go + internal/raster), and the draw-time uniform-buffer channel (render.go + inter

- **Tutorials compile: every fenced block becomes an Example** — Convert the eight docs/tutorial code blocks into Example functions in the root and tensor packages so `go test` compiles them; keep docdrift_test.go as the prose-name check. Closes §6 and §7's Example bullet.
- **A device reports its vertex-buffer limit** — Add Limits.MaxVertexBuffers, populate it per backend (internal/metal, internal/cpu/profile.go), and make vertexlayout.go:132 refuse against the device's number instead of mslabi.StageVertexBufferLimit. Closes §5.3's second force.
- **Normalized integer vertex attribute formats** — Add unorm8/snorm8/unorm16/snorm16 AttrFormat values with the conversion stated the way the fill rule is, implement the fetch in internal/raster and the Metal vertex descriptor, and refuse only what has no portable definition. Closes §5.3's first force.
- **The graphics freeze record** — Extend specs/036 §5 with frozen/provisional/fix-before tiers for the render surface, each provisional entry naming its event. Closes §5 and §7's freeze bullet.
- **A draw parameterised by a uniform-buffer offset** — Restore a draw-time uniform channel taking a BufferView plus a byte offset (033 §4.1's N-object frame), lower it on both backends, and un-skip TestADrawCanBeParameterisedByAUniformBuffer. Closes §3.1.

### [045-texture-attachments.md](045-texture-attachments.md) → 4 successors

In progress with three independent unbuilt chunks in disjoint files, each shippable alone: the Metal shader-visible texture path (internal/metal, internal/mtl), the Metal attachment/present rework (internal/metal/render_darwin.go, present_darwin.go), and subresource addressing for mips and layers (texturealloc.go, texture copy, barrier range). §5's own ordering already says the backends and the fe

- **Metal binds a stage texture** — Add setVertexTexture:/setFragmentTexture: to internal/mtl, consume driver.RenderDraw.VertexTextures/FragmentTextures in internal/metal/render_darwin.go, and until then refuse a draw carrying textures on a non-CPU device by name. Run TestAPassReadsWhatAnEarlierPassDrew on both bac
- **Metal renders into MTLRenderPassDescriptor attachments directly** — Delete the per-pass CopyBufferToTexture/CopyTextureToBuffer staging (internal/metal/render_darwin.go:67,103,110) and the presentSource conversion draw (present_darwin.go:37), and assert a frame's transfer count in a test. Closes §4's Metal row and §7 bullet 7.
- **Subresources: mip chains and array layers end to end** — Size a mip chain in textureBytes, compute a subresource byte offset for texture-buffer copies, and narrow the recorded access range so the barrier plan reasons per subresource; then lift the texturealloc.go:76-85 refusals. Un-skips TestADisjointSubresourceIsNotFeedback. Closes §7
- **The format table's remaining consumers** — Give Blendable a consumer — validate a blend state against the attachment format at pipeline build — and add the missing TextureSampled usage check for a texture bound through RenderPass.SetTexture. Closes §1's "built and thrown away" symptom.

### [010-kernel-corpus.md](010-kernel-corpus.md) → 4 successors

In progress with four independent unbuilt chunks that ship separately.

- **Kernel registry and identity** — `KernelID`/`KernelMeta`, generated
  per-variant records, duplicate and equal-priority rejection at init, and the
  package-location decision: either build §1's `tensor/internal/kernels/` tree
  or formally supersede it.
- **Layout classes and materialization** — §2's four classes as compile-time
  entities, the writable-layout rejection, and `Contiguous`'s
  explicit-materialization boundary now that `Pack` exists.
- **Deterministic selection** — §4's rules 1–7, the plan recording selected ids
  alongside rejected candidates and reasons, and §3's fused-attention
  present/absent case driven by removing a record from the registry.
- **Numeric recipes and the RoPE domain proof** — §5's per-family recipe
  records, and the compile-time proof that a declared maximum position and base
  stay inside [008](008-numerics.md)'s bounded sin/cos domain.

### [005-graphics.md](005-graphics.md) → 1 successor

Almost every unbuilt item in 005 already has an owning in-progress child —
stencil, depth bias and the blend constant to [033](033-render-api.md) and
[035](035-cpu-rasterizer.md), lines and points to 035, MSAA to
[041](041-msaa.md), the flat tag to [032](032-stage-abi.md). Exactly one chunk
has no owner, and it is the one the spec is organised around.

- **Compute-visible textures and the render-to-compute handoff** — give the
  compute argument set a texture channel, lift the `//accel:kernel` texture
  refusal, accept `Binding.Texture` on a dispatch, declare the access as a
  sampled read in a shader stage so the attachment-to-compute barrier is
  inferred, and land the two tests 005 names: pixel-origin agreement in its
  discriminating three-path form, and "the handoff stays on device", asserting
  no host transfer node between the geometry pass and the tonemap.

### [003-command-graph.md](003-command-graph.md) → 1 successor

- **Graph build diagnostics** — `BuildError`/`NodeError`, the thirteen missing
  sentinels, a `runtime.Caller` capture per node, the `file:line:col` message
  format, and the test that matches against the test's own file name.

**42 successor specs** across the eleven, replacing prose that cannot be
finished with units that can.

The first is written: [050-barrier-scopes.md](050-barrier-scopes.md), the
highest-priority piece of 002's split, because it is the one gap that is a
latent wrong answer rather than missing surface.


## 009, 010 and 011 — audited separately

These three were re-audited after the first pass failed on them. They are the
process specs, and two of the findings are about the register itself.

### [009-sequencing.md](009-sequencing.md)

The build history. The status vocabulary fits it badly — a milestone log can
never be `implemented` — and what matters is whether its recorded outcomes are
true. Mostly they are. Three are stale enough to mislead:

- **"An unverified commit range"** tells a reader the matrix has not been run
  over it. Six consecutive green `ci` and `ci-metal` runs on main through
  2026-08-27 say otherwise. The section outlived its cause.
- **M4's outcome contradicts M4's own child table.** The outcome says subgroup
  shuffles, scans and the strict-mode narrowing were deferred or not built; all
  three shipped (`internal/testkernels/subgroup.go:100,146,183`,
  `internal/cpu/profile.go:302,349`). All six of M4's done criteria hold, and M4
  is the only milestone in M0–M7 carrying **no completion marker**.
- **M8's status row** says the sampling policy integration remains. It is built
  and reachable (`tensor/policy.go:350,358,375`).

Not splittable: a build history has no shippable chunks. The fix is three edits.

### [010-kernel-corpus.md](010-kernel-corpus.md)  · **too large, see the split plan**

**Unbuilt:**

- **§1's package layout** — `tensor/internal/kernels/{elementwise,layout,…}` does
  not exist and everything lives in one flat `internal/testkernels`. §3.1
  mentions the real location but never records §1 as superseded. This is the
  section the naming requirement misses.
- **§1's record types** — `KernelID`, `KernelMeta`, `LayoutClass`,
  `NumericRecipe`, `Priority`: zero matches repo-wide.
- **§2's layout classes** — no compile-time `contiguous` / `row_contiguous` /
  `broadcast_read` / `general_read`; only the informal `bcast` and `strided`
  node flags.
- **§4's selection rules 1–7**, and **§5's numeric recipe records**.
- **`rows` with i32 ids** — `GatherRows`/`ScatterRows` take `[]uint32` only,
  while §3 promises i32/u32 and §7 requires the invalid-id cases for both.

**Unbacked:**

- **010's own rule, applied to 010.** "A corpus entry with no operator reaching
  it is recorded as unreachable rather than as done." **50 of 72 kernels are
  reachable from a `tensor/` operator; 22 are not**, and four of those carry
  rows that do not say so — `ReduceSum`, `ElemBias`, `AtomicOpsI32`,
  `AtomicAddF32`.
- **`PackKernel` is reachable and has no row**, while §3.1 implies the base
  `contiguous_copy` is missing.
- §4 quotes `var Kernels = []*accel.Kernel{…}`; the real type is
  `[]*kernelabi.Kernel` since those names left the root package.
- The scope line still excludes quantized and sampling kernels while §3 now
  carries rows for both.

### [011-conformance-harness.md](011-conformance-harness.md)

**Unbuilt:**

- **§2's central rule, "every test receives a profile explicitly"** — enforced
  nowhere. `internal/conformance/device` is imported by exactly one file while
  **26 test files call `accel.OpenCPU`/`OpenDevice` directly**.
- **§2's hardware enumeration** — `device.All()` returns three CPU profiles and
  no Metal profile, though §13 promises them at M6 and 009 records M6 complete.
- **§3's manifest** of present/absent cases — no form of it exists.
- **§4's static conformance check**, which would scan comparison call sites and
  reject ad hoc helpers and tolerance parameters. No code.
- **§6's kernel runner and generated manifest**, **§10's branch manifests**, and
  **§12's artifact retention** (CI uploads `cover.out` and nothing else).

**Unbacked:**

- **§8's E2E table and §13 disagree with 009 about M3, M4, M5 and M7.** The
  tests exist — `graphdispatch_test.go:520`, `tensor/model_test.go:42`,
  `tensor/parity_test.go:38`, `graphoracle_test.go` — 009 records them complete,
  and 011 carries no marker. **Two specs, one question, two answers**, which is
  the failure this register exists to end. 011 is the stale one.
- **§13's M6 counts are stale in every number**: "all 29 kernels … 22 agreeing
  bit for bit and 7 within a ceiling" against a corpus of 72, with 61
  differential cases over 58 kernels and 21 carrying a ceiling.
- **§13 claims the comparison package "has no tolerance parameter and will not
  grow one"**, while `WithinULP(got, want, ceiling uint64)` takes a
  caller-supplied ceiling. The argument that a ceiling is not a tolerance rests
  on it coming from 008's table, and the §4 check that would enforce that is
  itself unbuilt.

**Not splittable.** Its three big unbuilt chunks all need 010's per-variant
records first, so they cannot ship independently of that successor. The rest is
accuracy work.

## 003, 004 and 005 — audited separately

The last three. 005's status was wrong and is corrected; the other two carry the
right word and the wrong outstanding list.

### [003-command-graph.md](003-command-graph.md)

Status correct, and it **names no outstanding sections anywhere**, which the
rule requires. Nearly all of it is built and tested — recording, edge inference,
memory planning, barrier insertion, validation, fences, statistics, and the
worked example's goldens verified exactly (22/12/16 MiB, 7 barriers, 9 hazards).

**Unbuilt:**

- **The error taxonomy and call sites.** `BuildError`, `NodeError` and 13 of the
  16 sentinels do not exist; build errors are an `errors.Join` of formatted
  strings, and `runtime.Caller` appears nowhere in the root package. The whole
  `model/attn.go:118:0: node 7 …` format is unbuilt, and no other spec re-owns it.
- **`AccessMode` as an 11-bit mask.** The code has a three-value enum
  `Access{Read,Write,ReadWrite}`. The consequence is the finding: **the
  `AtomicRMW` rule is unbuilt** — "two nodes that both only AtomicRMW one range
  get no edge" has no implementation and no test, so atomics take ordinary write
  edges.
- **Texture layout derivation** — deferred behind Vulkan and D3D12, which do not
  exist.

**Unbacked:**

- "Three of the ten `StageMask` names are not produced by any declaration yet …
  `StageVertex` and `StageFragment` arrive with 033" — both exist and are
  exercised. Only `StageHost` remains.
- "**There is no timing observability at all**" — closed by
  `Recorder.CollectTimings` and `SubmissionStats.Elapsed`.
- "Transient identities and pool offsets are intentionally not a stable user
  API" — contradicted by the public, documented, tested
  `Graph.TransientPlacement()`.
- "Plan structure is identical across backends for one recording" — no test
  compares plans across CPU and Metal, only results.

### [004-kernel-authoring.md](004-kernel-authoring.md)

Status correct, **outstanding list wrong**: the header says the
`//accel:vertex` and `//accel:fragment` directives are unbuilt. They are built,
with six vertex and six fragment artifacts in the corpus.

**Unbuilt:** the GLSL ES 3.1 / GLSL 4.3 and HLSL SM 6 emitters, both correctly
gated on backends [006](006-backends.md) lists as unbuilt; and the **target
generation policy** — `-targets=cpu,metal` does not exist, the record carries a
single `MSL string` rather than an ordered target set, and "requesting a
not-yet-admitted target fails generation" is unreachable.

**Unbacked:** "**helpers take values, never storage** … both cases produce one
error naming the restriction." The compiler *accepts* slice helper parameters by
design, and the corpus uses one. The rule was a GLSL ES 3.1 constraint, GLSL is
unbuilt, and MSL passes a device pointer to a function without complaint — so
this is a rule the code deliberately does not enforce while the spec asserts it
does.

### [005-graphics.md](005-graphics.md)  ·  status corrected `drafted` → `in progress`

`drafted` means nothing it specifies has shipped. Most of it has: both backends
render, present, and are compared pixel by pixel — `render.go` (1103 lines),
`surface.go` (637), `internal/raster/` (2553), the Metal `CAMetalLayer` drawable
path, the top-left fill rule, perspective-correct interpolation.

**Unbuilt, and one of these has no owner:**

- **The render-to-compute handoff, variant 1** — "compute reads the attachments
  as textures, the normal path, no copy at all". A texture in a compute kernel
  is *refused by name*: `Texture2D`/`Fetch` reach vertex and fragment stages
  only, and `Binding.Texture` is refused on every dispatch path.
  [032](032-stage-abi.md) §5.1 explicitly declines to own this. **So the
  deferred-renderer graph this spec is organised around cannot be recorded**,
  and the discriminating pixel-origin test it names is unbuildable for the same
  reason. This is the one chunk with no owning child.
- Depth bias, fill mode and the blend constant; the whole of stencil; the sample
  count field; line and point topologies — each already owned by an in-progress
  child (033, 035, 041), not by 005.
- **Integer varyings must be flat-interpolated** — no `accel:"flat"` tag
  handling and no rejection. The rasterizer carries a `Flat []bool` mask that no
  front end populates. Owned by 032 §3.1.

**Unbacked:**

- "**Status: normative parent design** … does not freeze a public graphics API
  before any implementation has tested that shape." The API is public and frozen.
- "**Rasterizer-ordered access**: the CPU rasterizer must emulate the same order
  before it can report ROA support." The CPU profile reports
  `RasterizerOrderedAccess: true` and nothing consumes it — no stage
  requirement, no build-time gate.

  **Corrected 2026-08-27, and the first version of this entry was too strong.**
  It said "a capability advertised and not emulated". The ordering *is*
  provided: `internal/raster` contains no goroutine and processes primitives
  sequentially, so primitive-ordered access holds by construction and the bit is
  honest. What is true is narrower — the capability is **unreachable**, because
  no fragment stage can bind a written slice, so nothing can observe the
  ordering it promises. `TestTheCPUReportsOrderedAccessOnlyWhileItRasterizesInOrder`
  now ties the bit to the nearest observable, so parallelising the rasterizer
  fails a test instead of quietly making the bit false.
- "The CPU rasterizer: triangles, lines, and points" — triangles only.
- "The four child specs exist as of 2026-08-23" — there are five.

## What to do first

266 items is a backlog, not a plan. What makes one is knowing which kind of
wrong each item is, because they are not comparable: a latent wrong answer and a
missing convenience both read as "outstanding" and only one of them can cost a
consumer their afternoon.

### 1. Latent wrong answers — a caller who writes the obvious thing gets a wrong result

Nothing is broken today. Each of these is a guarantee the specs make that the
implementation does not, so the failure arrives the first time somebody writes
the code the spec says is legal. Verified mechanically that no current code
relies on any of them; that is a fact with a shelf life, not a reprieve.

| | Spec | The gap | Successor |
| --- | --- | --- | --- |
| 1 | [002](002-compute-model.md) §2.5 | ~~`Barrier` is normative over shared **and** storage; Metal emits `mem_threadgroup` only~~ — **closed 2026-08-27**, the lowering now matches §2.5's table. [050](050-barrier-scopes.md)'s masked variants remain | [050](050-barrier-scopes.md) |
| 2 | [002](002-compute-model.md) §6.2 | ~~a saturating float-to-int contract that does not exist~~ — **the hazard is closed 2026-08-27**: `kmath.ToI32`/`ToU32` on both backends, the three stages migrated, the bare conversion refused by name. 051 stays in progress for the CPU-versus-Metal boundary differential, which is the only assertion that would catch a real divergence | [051](051-float-to-int.md) |
| 3 | [029](029-plan-cache.md) §2 | the plan-cache key claims to cover every compile option that affects lowering, and hashes a constant string | — |

029 is third rather than first only because `CompileOptions` currently carries
nothing but `Label`, which is excluded by design. **It becomes first the day
that struct grows a field**, and it will grow one silently.

### 2. Advertised and absent — a caller queries a capability, or reads an API, that is not there

| | Spec | The gap |
| --- | --- | --- |
| 4 | [005](005-graphics.md) | `RasterizerOrderedAccess` is reported and **unreachable**: the ordering holds (the rasterizer is sequential) and nothing can observe it, because no fragment stage binds a written slice |
| 5 | [030](030-paged-kv.md) | ~~`BlockPool` is presented as caller-facing and lives in `tensor/internal/pagetable`~~ — **closed 2026-08-27**: it is `tensor.BlockPool`, and its own doc's condition for re-exporting ("until an operator accepts a page table") had been met by `Attention` and gone unnoticed |
| 6 | [014](014-kernel-uniforms.md) §2 | ~~no uniform-size validation against the device limit~~ — **closed 2026-08-27** at pipeline creation, and §6's claim that §4's test for it passed is corrected: no test referenced `MaxUniformBlockBytes` at all. The generated decoder and typed bindings struct remain, and neither has a consumer |
| 7 | [003](003-command-graph.md) | the error taxonomy's **structure** — `BuildError`, `NodeError`, `callSite` and the `file:line:col` format, plus seven unwrapped sentinels. The six binding sentinels and their checks landed 2026-08-27, so `errors.Is` works for what a caller binds |

4 is the one to take first, and taking it revised it — see 005's entry above.
The bit is not false; it is unreachable, which is a different problem with a
different fix. That is worth recording as a method note: **the audit's framing
was checked against the code before it was acted on, and it changed.** An audit
finding is a lead, not a verdict.

### 3. Conservative, not wrong — correct today and slower or narrower than specified

| | Spec | The gap |
| --- | --- | --- |
| 8 | [003](003-command-graph.md) | `AccessMode` is an 11-bit mask in the spec and a three-value enum in code, so the `AtomicRMW` no-edge rule is unbuilt and atomics take ordinary write edges. **Investigated 2026-08-27 and deliberately not built**: the rule as written is unsound. Exchange and compare-exchange are reachable from the corpus and are not commutative, so dropping the edge makes the result depend on which dispatch finished last — visible on Metal, hidden on the CPU. 003 records what a sound version needs (`OrderIndependent`, which the compiler already infers) and that the mask is a breaking public API change needing a decision |
| 9 | [044](044-unbounded-context.md) §7 | ~~`Selections()` does not report the tile count~~ — **closed 2026-08-27** for the contiguous decode. The paged variants' loop is bounded by the page table's reach rather than a cache shape, so their count is a different quantity and 044 §7 records why it is not printed as the same one |
| 10 | [025](025-tensor-operators.md) §6 | ~~no test runs the operators themselves on both backends~~ — **closed 2026-08-27**: `GatherRows`, `RoPE`, `Softmax` and the views now agree across backends through a composed plan, not through the kernel differential |

### Tier 3 outcome — 2026-08-27

Items 9 and 10 are closed. **Item 8 is closed as a decision rather than as
code**, and it is the one place in this plan where executing the spec would have
been the wrong move: 003's `AtomicRMW` no-edge rule is unsound for the
non-commutative atomics the corpus already reaches, so the conservative
behaviour it complains about is the correct one until the rule is narrowed. The
finding and the sounder alternative are in 003.

### 4. Discipline erosion — the checks that were supposed to stop all of the above

These are last by urgency and first by leverage. Every item above exists because
one of these is unbuilt.

| | Spec | The gap |
| --- | --- | --- |
| 11 | [011](011-conformance-harness.md) §2 | **the rule's purpose is met 2026-08-27** — a failing test now reports its backend, capabilities and limits, from the two shared helpers rather than 34 call sites. The literal rule ("every test receives a profile") is *not* enforced and wants narrowing instead: a graph-validation test has no numeric context to receive. That narrowing is a decision, recorded in 011 §2 |
| 12 | [011](011-conformance-harness.md) §4 | ~~the static check that rejects ad hoc comparison helpers and tolerance arguments~~ — **built 2026-08-27 as a ratchet**: 75 existing sites are pinned and a new one fails. Converting the rest is per-site work, and the first conversion found a tolerance three orders of magnitude looser than the derived bound |
| 13 | [010](010-kernel-corpus.md) | ~~010's own rule, unapplied to 010~~ — **closed 2026-08-27**: the unreached set is enumerated in 010 with the spec that needs each, and pinned by a test so a new one cannot join silently |
| 14 | [000](000-decisions.md) | layering rule 3, "no backend-specific type appears in a public signature", violated by `OpenCPU`, `CPUOptions` and `CPUMode`. **Rules 1, 2 and 4 are now tested** (2026-08-27); rule 3 is left untested on purpose, because asserting a rule the tree breaks means weakening it, and a weakened rule reads as enforced. It needs a decision: change the surface or amend the rule |

14 needs a decision before work: either the public surface changes or the rule
does. It is a normative document contradicted by the API it governs, and leaving
both standing is what made the rest of this list possible.

### The shape of the whole finding

Two-thirds of the 266 items are **unbacked** rather than **unbuilt** — the code
exists and nothing checks it. That is the same defect this project spent the
week removing from operator refusals, one level up, and §4 above is where it
stops recurring. Fixing §1 makes the library correct; fixing §4 makes it stay
correct.
