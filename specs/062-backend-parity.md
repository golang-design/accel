---
title: "Backend parity: an enumerated CPU/Metal agreement matrix, gated on every platform"
status: drafted
layer: process
depends_on:
  - 006-backends.md
  - 008-numerics.md
  - 011-conformance-harness.md
---

# Backend parity

**One thing:** make "the CPU backend and Metal agree" a *checked enumeration*
rather than a collection of individually written tests.

A child of [011](011-conformance-harness.md). 011 owns the harness; this owns
one thing the harness never grew: a registry of parity cases, and a gate that
fails when a member of a public enumeration or operator set has no case in it.

## 1. What exists, and why it is not enough

Thirty CPU/Metal agreement tests already run. They are not the problem.

The problem is that **exactly one surface knows what it is missing.**
`internal/testkernels/differential_darwin_test.go` checks the case table
against the generated corpus:

```go
for _, k := range testkernels.Kernels {
    if k.MSL != "" && !listed[k.Name] {
        t.Errorf("%s lowers to MSL and is in no differential case", k.Name)
    }
}
```

Every other surface is covered by hand. Nothing anywhere says how many
`accel.Format` values exist, or how many `tensor` operators, or which of them a
parity case has ever run. So the honest answer to "is `CompareGreaterEqual` the
same on both backends" is *nobody knows*, and adding a twelfth format or a
forty-sixth operator produces a green run.

**The gate is in the wrong file, too.** It sits behind `//go:build darwin`, so
it compiles on one of the three Tier 1 platforms. An unlisted kernel added on
Linux is invisible until someone runs the suite on a Mac.

### 1.1 The measured gap

| Surface | Universe | Parity cases today |
| --- | --- | --- |
| Kernel corpus | generated | **gated** (darwin-only gate) |
| `tensor` operators | 45 exported | 8 named, rest incidental |
| `accel.Format` | 12 | Metal-only sweep, no CPU comparison |
| `accel.DType` | 7 | Metal-only round-trip |
| `CompareFunc` | 8 | 1 |
| `StencilOp` | 8 | 1 |
| `BlendFactor` × `BlendOp` | 10 × 5 | 2 |
| `Topology` | 5 | 1 |
| `AttrFormat` | 5 | 1 |
| `IndexFormat` | 2 | 1 |
| `CullMode` / `FrontFace` | 3 / 2 | 1 / 1 |
| `LoadOp` / `StoreOp` | 3 / 2 | 2 / 2 |
| `ColorWriteMask` | 4 bits | 1 |

"Metal-only sweep" is the failure mode worth naming: `TestEveryDTypeRoundTrips`
`OnMetal` and `TestMetalRendersEachAttachmentFormat` both enumerate their whole
universe and then check *one* backend. They read as parity tests and are not.

## 2. The shape

Three pieces, and the split between them is the point.

```
                      universe              registry
                 (derived from source)   (declared by tests)
                          |                     |
                          +----------+----------+
                                     |
                              completeness gate        no build tag
                                     |                 runs on linux, windows, darwin
                          -----------+-----------
                                     |
                                run and compare        //go:build darwin
                              CPU  <-- equal -->  Metal
```

- **Universe** — every member of a public enumeration, or every operator in a
  package, extracted from the source by `go/ast`. Not a hand-written list: a
  hand-written list goes stale the same way the tests did.
- **Registry** — parity cases, declared as data in a file with no build tag, so
  the gate can see them everywhere.
- **Runner** — opens both devices and compares. Darwin only, because Metal is.

The gate needs no device. That is what lets it run on every platform, and it is
why the two halves must not live in one file.

### 2.1 Package layout

```
internal/conformance/parity/
  parity.go     Case, Set, coverage gate, failure text   [no build tag]
  universe.go   go/ast extraction of enums and operators [no build tag]

<domain>/parity_matrix_test.go   the registry, plus the gate  [no build tag]
<domain>/parity_darwin_test.go   the CPU/Metal runner          [darwin]
```

`parity/` joins the reserved names in 011 §1.

## 3. The universe, derived

Two extractors, both over `go/ast` with no type checking, because the shapes
this needs are syntactic.

**`Enum(dir, typeName)`** returns the identifiers of a package-level `const`
block typed `typeName`, including the `= pkg.X` alias form the render enums use
(`BlendFactor = driver.BlendFactor`, whose constants are `FactorZero =
driver.FactorZero`). It returns declaration order, so a failure names members
the way the source does.

**`Funcs(dir, firstParamType)`** returns exported function names whose first
parameter has the given type — `*Builder` for the tensor operator set. This is
the shape of a `tensor` operator and nothing else in that package has it.

Neither imports `go/types` or `golang.org/x/tools`: `importgraph_test.go`
already refuses those in the root package's non-test graph, and there is no
reason for a test helper to need more than the syntax.

### 3.1 Exclusions are declared, not implied

A universe member with no parity case is a failure. A member that *cannot* have
one states why, in the registry, next to the name:

```go
parity.Excluded{Name: "LineList", Why: "refused on both backends; " +
    "035 §10 leaves the rule open. Covered by the refusal-agreement case."}
```

An exclusion with an empty `Why` is itself a failure. The rule this enforces:
the reason a thing is untested is written down where the next reader looks,
rather than reconstructed from a missing row.

## 4. What agreement means

Reuse [008](008-numerics.md)'s comparison vocabulary through `numeq`. A case
declares its ceiling and where the ceiling comes from, exactly as the corpus
differential does:

- **Exact** is the default. Two lowerings of one IR, over integer or copy-only
  paths, must agree bit for bit.
- **`WithinULP(n)`** where a bounded primitive is reached, with the primitive
  named.
- **`WithinAbs(e)`** where a value crosses zero and a ULP bound is meaningless.

A case that states no ceiling and no reason does not compile: `Case.Ceiling` is
required and `Why` is required whenever the ceiling is not exact.

### 4.1 The oracle's configuration

Two constraints the corpus differential already discovered, made general:

1. The CPU device opens at the Metal device's `MinSubgroupSize`. The CPU
   backend emulates subgroups at a caller-chosen width; its default is 4 and
   the device executes 32, so a reduction over 64 elements would be comparing
   two different computations.
2. A device reporting `MinSubgroupSize != MaxSubgroupSize` fails rather than
   skips. The oracle emulates one width.

**The CPU oracle is permissive, not strict.** `OpenCPU{}` and not
`Strict(BackendMetal)`, because strict mode forces capability *absence* to
model a portable intersection — a case comparing against Metal wants the CPU's
natural behaviour on the capabilities Metal actually reports, not the
intersection with a backend that is not running. Strict mode's own agreement
obligation is 006's and stays where it is.

## 5. Absence

Per [006](006-backends.md) §7 and the CI header: a job that promises Metal and
finds no adapter fails; one that does not, skips. The runner keeps the existing
`ACCEL_REQUIRE_METAL` rule rather than inventing a second one.

The gate never skips. It needs no device, so there is no condition under which
it is allowed to be silent.

## 6. Sections and what each covers

| § | Surface | Universe source |
| --- | --- | --- |
| 6.1 | `accel.DType` | `Enum(".", "DType")` |
| 6.2 | `accel.Format` | `Enum(".", "Format")` |
| 6.3 | Render fixed-function state | `Enum` over `Topology`, `FrontFace`, `CullMode`, `CompareFunc`, `StencilOp`, `IndexFormat`, `AttrFormat`, `ColorWriteMask` |
| 6.4 | Blend | `Enum` over `BlendFactor`, `BlendOp` |
| 6.5 | Attachment lifecycle | `Enum` over `LoadOp`, `StoreOp` |
| 6.6 | `tensor` operators | `Funcs("tensor", "*Builder")` |
| 6.7 | Kernel corpus | the generated `Kernels` slice, gate moved out of the darwin file |
| 6.8 | Transfers and textures | `Enum(".", "Format")` crossed with the copy entry points |

Each section's case runs the same recorded work on both devices and compares
the readback. A case is one graph, not one API call: an API call whose result
never reaches a buffer cannot be compared, and a case that cannot be compared
is not a parity case.

## 6.9 Existing cases that are not parity cases

Four tests read as parity and are not, and they are part of this spec rather
than a separate cleanup: a test whose name claims a comparison it does not make
is worse than an absent one, because it answers "is this covered" with a yes.

| Test | What it actually does | Fix |
| --- | --- | --- |
| `TestEveryDTypeRoundTripsOnMetal` | enumerates 7 dtypes, checks Metal alone | §6.1 compares both |
| `TestMetalRendersEachAttachmentFormat` | compares both backends over **2** of 9 colour formats, while named "each" | §6.2 covers the enumeration |
| `TestIndirectDispatchOnMetal` | clamp behaviour on Metal alone | compare the clamped count on both |
| `TestContiguousRunsOnMetal` | Metal against a host reference | folded into §6.6 |

## 7. Cost

The gate makes every uncovered member a red build the moment it lands. That is
the point and it is also the cost: adding an enumeration member now means
adding a parity case in the same change, or writing down why not. Sections 6.3
and 6.4 are the expensive ones — 8 compare functions, 8 stencil ops and 50
blend combinations are 66 render passes on two devices.

The blend matrix is run as a **product, not a sweep**: one pass per factor
against a fixed op, one per op against a fixed factor, and the full 50 behind
`-parity.full`. A full product on every run buys little over the two sweeps and
costs pipeline compilation per combination.

## 8. Testing

- The gate is tested by feeding it a universe with a member the registry omits,
  and asserting the failure names that member. A gate nobody has seen fail is
  a gate nobody knows works.
- `Enum` is tested against a fixture package holding all three const forms:
  `iota`, explicit values, and the `= pkg.X` alias.
- An exclusion with an empty `Why` fails.
- Every case runs on the CPU alone on non-darwin platforms, so a case that
  panics or produces a degenerate all-zero result is caught without a GPU.
- Each case asserts its own output is non-degenerate before comparing. Two
  backends agreeing on a buffer of zeros is the failure this project keeps
  finding.

## 9. Open questions

- Whether `Vulkan` joins as a third column here or gets its own spec. The
  registry is written with a device list rather than a pair, so the answer does
  not change the shape.
