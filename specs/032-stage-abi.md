---
title: "The vertex and fragment stage ABI"
status: in progress
layer: device
depends_on:
  - 002-compute-model.md
  - 004-kernel-authoring.md
  - 005-graphics.md
  - 008-numerics.md
---

# The vertex and fragment stage ABI

[005](005-graphics.md)'s first child spec, and the one the other three wait on:
a render pipeline cannot be described until the shape of the code it compiles is
fixed. [004](004-kernel-authoring.md) reserved `//accel:vertex` and
`//accel:fragment` for exactly this document and deferred them; this spec
un-defers them and says what they mean, what the IR gains, and what each target
emits.

Nothing here changes the compute subset. A graphics stage is the same authored
Go, the same `go/types` resolution, the same typed IR, and the same two
lowerings — a generated Go runner and a shading-language emission. What is new
is the **shape of the signature** and the **interface between stages**, which is
fixed-function hardware and therefore not something a Go signature describes by
accident.

## 1. The pipeline this spec is one third of

```
authored Go                    one typed IR                  two targets
┌──────────────────┐          ┌──────────────┐          ┌────────────────┐
│ //accel:vertex   │──front──▶│ ir.Func      │──emit───▶│ generated Go   │
│ func VS(...)     │          │  Stage=Vertex│          │ (CPU raster)   │
├──────────────────┤          ├──────────────┤          ├────────────────┤
│ //accel:fragment │──front──▶│ ir.Func      │──emit───▶│ MSL / GLSL /   │
│ func FS(...)     │          │  Stage=Frag  │          │ SPIR-V / HLSL  │
└──────────────────┘          └──────────────┘          └────────────────┘
                                     │
                              stage record: varyings, attachments,
                              built-ins used, bindings, digest
```

The stage record is the part a pipeline descriptor consumes, and
[033](033-render-api.md) validates a pipeline against it. It is generated, never
authored, for the reason [004](004-kernel-authoring.md) gives for bindings: a
declaration a caller writes is a second source of truth, and a second source of
truth can be wrong while nothing fails to build.

## 2. The vertex stage

```go
//accel:vertex
func Geometry(v accel.Vertex, xf Transforms,
	pos accel.Vec3, uv accel.Vec2) (accel.Clip, Varyings)
```

**Signature rule.** A vertex stage takes an `accel.Vertex` first, then its
bindings, and returns exactly two values: a clip position and a varyings struct.
The varyings struct may be empty, and a stage that returns only a position
returns `accel.NoVaryings`.

Two returns rather than a position field inside the varyings struct, because the
position is not a varying: it is consumed by primitive assembly and the
rasterizer and never reaches the fragment stage as itself. Making it a struct
field would let a caller reorder it, drop it, or interpolate it, and each of
those is a different wrong thing.

### 2.1 `accel.Vertex`

The graphics analogue of `accel.Thread`, and deliberately a different type: a
vertex stage has no workgroup, no barrier, no shared memory, and no subgroup, so
handing it `Thread` would make three quarters of that type's methods a
compile-time trap rather than an unavailable one.

| Method | Meaning |
| --- | --- |
| `VertexIndex() uint32` | The index of this vertex. For a non-indexed draw it is `firstVertex + i`; for an indexed draw it is the value read from the index buffer. |
| `InstanceIndex() uint32` | `firstInstance + n`. |

`VertexIndex` is what makes a vertex-buffer-less pipeline work — 005's
full-screen triangle computes its position from the index alone — so it is not
optional and not capability-gated.

**Not offered: a base-vertex accessor.** `DrawIndexed`'s vertex offset is applied
to the *attribute fetch*, not to the index the stage sees, and backends disagree
about whether a built-in reports the pre-offset or post-offset value. The value
the caller can act on is already in `VertexIndex`; a second nearly-identical
number is a portability bug waiting for its first user.

### 2.2 Attributes are parameters, not fetched

A vertex buffer reaches the stage as an ordinary slice parameter, indexed by the
compiler at `VertexIndex` — the same way a compute kernel reaches a storage
buffer, except the index is implicit. The stage body sees `pos[0]`-style access
nowhere: it sees `pos` already resolved to this vertex's element.

That is stated as a rule because it is the one place this ABI could have gone
two ways:

| Option | Why not |
| --- | --- |
| Slice parameter, caller indexes by `v.VertexIndex()` | Legal-looking code can index any vertex, which no backend can express: an attribute fetch is fixed function and reads exactly one element. |
| **Element parameter, compiler fetches** | Chosen. The signature says what the hardware does. |

So the authored form is:

```go
//accel:vertex
func Geometry(v accel.Vertex, xf Transforms, pos accel.Vec3, uv accel.Vec2) (accel.Clip, Varyings)
```

`pos` is *one vertex's* position. `xf` is an ordinary uniform, indexed by nothing
and shared by every vertex, and the difference between the two is the parameter's
type, not a tag. The compute subset already classifies parameters this way, and
the graphics stages add exactly one row:

| Go spelling | Category | Since |
| --- | --- | --- |
| `[]T` | storage buffer | the compute subset |
| `T` where T is a struct | by-value uniform, std140-encoded | the compute subset |
| `*[N]T` | workgroup-shared memory | the compute subset, and illegal in a graphics stage |
| **`[N]float32`** | **vertex attribute** | **here** |

Four categories, four Go spellings, no annotations. The new row is unambiguous
against the existing three because shared memory is a *pointer* to an array where
an attribute is an array by value, and a by-value array is not a legal compute
parameter at all today — the front end's classifier rejects it, so nothing has to
be re-decided to admit it here.

Attribute formats are their own enumeration, per 005: `unorm8x4` and `u8x4` have
the same width and different meanings, and the conversion to the Go parameter
type happens in the fetch. The legal pairs are in [033](033-render-api.md), which
owns the descriptor; this spec owns only the rule that the pair is **checked**,
at generation, against the declared layout.

### 2.3 Vector types are Go arrays, which the compiler already understands

Before the clip convention, one thing the examples above would otherwise imply
wrongly. `accel.Vec2`, `accel.Vec3`, `accel.Vec4` and `accel.Clip` are **aliases
for `[2]float32`, `[3]float32` and `[4]float32`**, not new named types.

The kernel language already spells a vector that way: `std140` maps `[3]float32`
to a three-component vector consuming twelve bytes aligned to sixteen, and the
uniform encoder is generated from exactly those Go array types. Introducing a
parallel set of named vector types for the graphics stages would give the
compiler two spellings for one thing, and the second one is the one nobody
teaches the layout code about.

The names exist because `accel.Clip` at a signature's return says what the value
*is* where `[4]float32` says only how wide it is. An alias buys the reading
without a second type.

### 2.4 The one clip convention

`accel.Clip` is the four-component clip-space position a vertex stage returns. The convention is 005's and
is restated here because this is where it is enforced:

$$
z_{\text{ndc}} = \frac{z_{\text{clip}}}{w_{\text{clip}}} \in [-1, 1]
\qquad\text{and}\qquad
z_{\text{window}} = \frac{z_{\text{ndc}} + 1}{2} \in [0, 1]
$$

A kernel emits `[-1, 1]`. Every backend whose native NDC range is `[0, 1]` —
Metal, Vulkan, D3D12 — has the emitter fold the second equation into the
position it writes, so the correction is one multiply-add in generated code and
never in a caller's projection matrix. The depth attachment stores window depth,
so clears and compares are in `[0, 1]` and the two ranges are never the same
number in the same place.

### 2.5 Decision: reverse-Z needs no API change

005 leaves this open and works out its own answer; this spec adopts it.

Reverse-Z places the near plane at the far end of the float depth range so that
float precision, which is dense near zero, lands where depth precision is
scarce. Under the convention above:

| | clip `z` | window `z` |
| --- | --- | --- |
| near | `+1` | `1.0` |
| far | `-1` | `0.0` |

so reverse-Z is spelled **clear depth to `0.0`, compare `Greater`, use a float
depth format**, with no API change and no second convention. A caller builds the
reversed projection matrix, which they were going to do anyway.

The alternative — presenting `[0, 1]` clip to kernels — was rejected because it
is a breaking change to every authored vertex stage and to
[`conventions.md`](../docs/conventions.md)'s fragment-side recovery
`position.z * 2 - 1`, in exchange for saving a multiply-add that generated code
performs. **This closes 005's first open question.**

## 3. Varyings

A varyings struct is an ordinary Go struct of value types. Its fields are
interpolated across the primitive and delivered to the fragment stage.

```go
type Varyings struct {
	Normal   accel.Vec3
	UV       accel.Vec2
	MaterialID uint32 `accel:"flat"`
}
```

### 3.1 Interpolation

| Field type | Interpolation | Enforced how |
| --- | --- | --- |
| float scalar or vector | perspective-correct, the default | — |
| float scalar or vector, tagged `noperspective` | screen-linear | tag |
| integer scalar or vector | **flat only** | compiler error without the tag |

Integer varyings must be flat, and that is not an accel choice: no target
backend interpolates an integer. The compiler rejects an untagged integer
varying with an error naming the field, the restriction, and the tag to add,
rather than emitting something a driver will reinterpret.

Perspective correction is the interpolation the hardware performs, and the CPU
rasterizer performs the same one — [035](035-cpu-rasterizer.md) states it as a
formula rather than an intention:

$$
a(x,y) \;=\; \frac{\displaystyle\sum_i \lambda_i \frac{a_i}{w_i}}
                  {\displaystyle\sum_i \lambda_i \frac{1}{w_i}}
$$

where $\lambda_i$ are the screen-space barycentric weights. Results are compared
against the oracle within [008](008-numerics.md)'s interpolation budget, never
exactly — see section 8.

### 3.2 The varyings struct is a layout, so it has a limit

Varyings occupy a hardware-limited interpolation budget, reported as a device
limit. It is checked at **generation**, not at pipeline creation, which §9
requires of every stage error and which this section said otherwise until the
two were reconciled: the generator knows the struct and the target profile's
limit, and an error that waits for pipeline creation arrives without the source
position that makes it actionable. Per
[`000-decisions.md`](000-decisions.md) decision 6. The count is in
**four-component slots**, because that is the unit every backend limits:

$$
\text{slots} \;=\; \sum_{f \in \text{fields}} \left\lceil \frac{\text{components}(f)}{4} \right\rceil
$$

A struct exceeding the limit is a generation error naming the field list, the
computed slot count and the limit. Packing two `Vec2`s into one slot is a later
optimization and is deliberately absent: the count above is one a caller can
compute by reading their own struct, and an optimizing packer would make the
number depend on the compiler's mood.

### 3.3 One struct type, both stages

The fragment stage's varyings parameter must be **the same Go type** as the
vertex stage's second result. Not a structurally identical one: the same named
type, checked by `go/types` identity, exactly as
[004](004-kernel-authoring.md) resolves intrinsics by identity rather than by
name.

Structural matching was rejected for the reason 004 gives for its own rule. Two
structs that happen to line up today drift apart in a later edit, and the
symptom is a fragment stage reading a normal out of the field that now holds a
UV — silently, on every backend, with no build error anywhere.

The check compares the `types.Object` the two names resolve to, which matters
most in the degenerate case: `accel.NoVaryings` is an empty struct, and so is
any empty struct a caller writes. Structural comparison would pair a vertex
stage returning `NoVaryings` with a fragment stage taking an unrelated empty
`type Lighting struct{}`, and the pairing would look deliberate.

## 4. The fragment stage

```go
//accel:fragment
func Shade(f accel.Fragment, in Varyings, mat Material) Targets
```

A fragment stage takes an `accel.Fragment`, then the varyings struct, then its
bindings, and returns one struct whose fields map **in declaration order** onto
the pipeline's colour attachments.

**Parameter order is load-bearing, not a convention.** The front end classifies
a parameter by its type, and after §2.3 a varyings struct and a uniform struct
are both `*types.Struct` — indistinguishable. So position decides, and only
position can:

| Stage | Parameter 0 | Parameter 1 | Parameters 2+ |
| --- | --- | --- | --- |
| vertex | `accel.Vertex` | *classified by type* | *classified by type* |
| fragment | `accel.Fragment` | **the varyings struct** | *classified by type* |

A vertex stage needs no such rule because its two categories — a by-value array
attribute and a by-value struct uniform — are distinguishable by type alone. A
fragment stage's do not, so its second parameter is the varyings struct by
position, whatever its type, and a fragment stage that omits it is refused
rather than having its first uniform silently interpreted as varyings.

```go
type Targets struct {
	Albedo   accel.Vec4 // → attachment 0
	Normal   accel.Vec4 // → attachment 1
	WorldPos accel.Vec4 // → attachment 2
}
```

One field per attachment is how MRT is expressed, and 005 records that the
predecessor proved this shape on Metal. A stage with a single attachment returns
a one-field struct rather than a bare `Vec4`, because the one-attachment case
being a different shape is how a caller ends up writing the two-attachment case
twice.

### 4.1 `accel.Fragment`

| Method | Meaning |
| --- | --- |
| `Coord() accel.Vec4` | Window coordinate. `xy` is the pixel centre, `z` is window depth in `[0, 1]`, `w` is `1/w_clip`. |
| `FrontFacing() bool` | Whether this fragment came from a front-facing primitive, under the pipeline's declared winding. |

**`Coord().xy` is top-origin**, which is 005's guarantee restated at the point it
binds: row 0 is the top row in the fragment stage, in an on-device texel fetch,
and in host readback. Which of those three a backend corrects is the backend's
choice; that they agree is not.

`Coord().z` is window depth, matching the depth attachment's range and matching
what a depth compare uses. `conventions.md`'s recovery to NDC is
`Coord().z * 2 - 1`, and it type-checks here because both sides of that
expression are this ABI's own numbers.

### 4.2 Discard

`accel.Discard()` ends the invocation and writes nothing — no attachment, no
depth.

It is an intrinsic rather than a `return`, because a Go `return` needs a value
and any value it returned would be a lie. Discard interacts with early depth
testing on every backend: a stage that discards cannot have its depth test
hoisted before the shader, and the compiler records that in the stage record so
[033](033-render-api.md) does not promise an early-Z the pipeline cannot have.

### 4.3 Not in the baseline: fragment storage writes

A baseline fragment stage writes attachments. It does not take a slice parameter
it writes to.

`conventions.md` records why, and it is the sharpest reason in this document:
ordinary fragment buffer writes are **unordered** between fragments covering the
same pixel, so a G-buffer written that way holds a nondeterministic winner
wherever geometry overlaps — while the depth buffer beside it looks perfect. The
failure reads as a shading bug, reproduces differently per driver, and has no
oracle.

Rasterizer-ordered access is the capability-gated extension that makes it legal,
and it is out of scope here: it needs its own ordering rule in the CPU
rasterizer before the CPU backend may report it. A slice parameter written by a
fragment stage is a **compile error** in this spec, naming ROA as the thing that
would make it legal and stating that no backend reports it yet. Rejected at
generation, not at pipeline creation, per 004.

## 5. Texel fetch, and why it is not sampling

005's flagship handoff has a compute pass reading a G-buffer attachment as a
texture. 004 defers **sampled textures** on evidence: a CPU sampler and a
hardware sampler could not be reconciled bit-exactly by the predecessor, and
admitting one admits a permanent tolerance into the oracle.

Those are two different operations, and this spec admits exactly one of them:

| | Admitted | Why |
| --- | --- | --- |
| **Texel fetch** — integer coordinate, explicit mip level, no filter, no addressing mode | **Yes** | It is an indexed load. Given the same texture and the same integer coordinate, every backend returns the same texel. There is nothing to reconcile, so there is no tolerance. |
| **Sample** — normalized coordinate, filter, LOD selection, addressing mode | **No** | 004's evidence: half-texel addressing, an off-by-one in LOD, and uint8-truncating lerps at every tap. Each is a per-vendor difference the oracle would have to absorb as tolerance. |

So the kernel language gains one intrinsic and one binding kind:

```go
// In a compute kernel or a fragment stage.
texel := accel.Fetch(gbuf, accel.IVec2{X: x, Y: y}, 0) // level 0
```

- `accel.Texture2D` and friends are new binding types, distinct from slices, so
  the compiler can tell an image binding from a storage buffer.
- Out-of-range coordinates return zero rather than being undefined. Every
  backend can guarantee that (a clamp-to-border-zero fetch, or an explicit
  bounds test in emitted code) and it is the one addressing question a fetch
  still has. Undefined here would be the same unreproducible class the sampler
  refusal exists to avoid.
- The returned type comes from the texture's format through the same
  format-to-dtype table [001](001-device-resources.md) already owns. A format
  whose fetch type does not match the destination is a generation error.

**Sequencing note.** Texel fetch is new front-end, IR, and emitter work, and
005's handoff has a second variant that needs none of it: a transfer node
copying the attachment into a buffer, which shipped at M1 and is device-to-device,
so it still satisfies 005's "handoff stays on device" test. [035](035-cpu-rasterizer.md)'s
corpus therefore builds the deferred graph on variant 2 first, and the origin
agreement test — which genuinely requires an on-device texel read — lands with
this section rather than gating the rasterizer.

## 6. What the IR gains

`ir.Func` carried `Kernel bool`. Four values replaced it — **built** — because a
boolean cannot answer "which stage" and every consumer would end up inferring it
from the presence of a workgroup extent, which is wrong for a graphics stage
since it has no extent at all:

```go
type Stage uint8

const (
	StageHelper Stage = iota
	StageCompute
	StageVertex
	StageFragment
)
```

and `ir.Func` gains, all derived from the body or the signature and none
declared:

| Field | Derived from |
| --- | --- |
| `Stage` | the directive |
| `Attributes []*Attribute` | the by-value array parameters of a vertex stage |
| `Varyings *Type` | the second result, or the varyings parameter |
| `Outputs []*Target` | the fields of a fragment stage's result struct |
| `Builtins uint32` | which `Vertex`/`Fragment` methods the body reaches |
| `Discards bool` | whether the body reaches `accel.Discard` |

`Workgroup`, `Shared`, and `Cooperative` stay zero for a graphics stage, and the
front end refuses a barrier, shared memory, or a subgroup operation inside one
with an error naming the stage. That is a compiler guarantee rather than a
convention, and this repository has already recorded that a guarantee in the
compiler is worth more than the same guarantee in a test.

### 6.1 Binding indices are per stage

A compute dispatch has one binding space. A draw has two, because vertex and
fragment stages are bound separately on every target backend. The stage record
therefore carries its own dense binding indices, and
[033](033-render-api.md)'s descriptor names the stage a binding belongs to.

The rule that made the compute path correct applies unchanged: **the index is
the dense position within the stage's own bindings**, not the parameter index.
Using the parameter index was a real bug in the compute path — the `Vertex` or
`Fragment` receiver occupies parameter 0 and is not a binding — and it would
recur here with a second offset to get wrong.

## 7. What each target emits

| Target | Vertex stage | Fragment stage |
| --- | --- | --- |
| Generated Go | a function taking a fetched attribute set and returning position plus varyings, called by [035](035-cpu-rasterizer.md)'s rasterizer | a function taking interpolated varyings and returning the target struct |
| MSL | `vertex` function, `[[stage_in]]` attributes, `[[position]]` on the returned position | `fragment` function, `[[color(n)]]` on each output field |
| GLSL / SPIR-V / HLSL | designed with their backends; the stage record carries everything they need | as vertex |

The generated Go form matters more than it looks. It is what makes the CPU
rasterizer an *oracle* rather than a second implementation: the same IR produces
both, so a disagreement is a lowering difference, not two people's arithmetic.
[004](004-kernel-authoring.md)'s explicit rounding point at each arithmetic
operation applies unchanged, and so does the rule that the authored function is
still run by a test that compares it against the generated one.

**Contraction is off**, as it is for compute. `#pragma METAL fp contract(off)`
goes in every emitted fragment and vertex function for the reason the compute
path found: `MTLMathMode.safe` does not disable contraction, and a fused
multiply-add changes results by more than the ULP ceilings
[008](008-numerics.md) states.

## 8. What is proved exactly and what is proved within a bound

005 makes this split normative and this spec inherits it, because a corpus where
everything is "within tolerance" proves nothing:

**Exact.** Which attachment a field lands in. Whether a built-in reports what it
says. Whether an integer varying arrives unchanged. Whether a discard wrote
nothing. Whether an out-of-range fetch returned zero. Whether the compiler
rejected what it says it rejects.

**Within a stated bound from [008](008-numerics.md).** Interpolated varying
values, and anything computed from them. The bound is derived from the
interpolation formula in section 3.1 and the stage's own arithmetic, never
measured and then written down.

## 9. Errors, all at generation

Per 004, a stage error arrives when the generator runs, with a source position,
never at pipeline creation:

- an integer varying without the flat tag, naming the field and the tag;
- a fragment stage whose varyings parameter is not the identical named type the
  vertex stage returns, naming both types — compared by resolved object, so two
  unrelated empty structs do not pair;
- a fragment stage with no varyings parameter at all, since position rather than
  type is what identifies it;
- a barrier, shared memory, or subgroup operation in a graphics stage;
- a slice parameter written by a fragment stage, naming ROA;
- an attribute parameter whose Go type does not match the declared attribute
  format;
- a varyings struct exceeding the interpolation slot limit, naming the computed
  count and the limit;
- a fragment output struct with more fields than the target profile's attachment
  limit.

## 10. Done

- `//accel:vertex` and `//accel:fragment` compile, and 004's out-of-scope entry
  for graphics stages is replaced by a pointer here;
- a vertex stage's attributes are fetched by the compiler, and a body cannot
  index another vertex because the signature does not offer one;
- the clip convention is enforced in emitted code, and a reverse-Z pipeline is
  written with no API beyond a clear value and a compare function — with a test
  that a reversed projection keeps near geometry;
- an integer varying without the flat tag is refused, and the refusal names the
  tag;
- a fragment stage whose varyings type is merely structurally identical is
  refused, confirmed by making the two structs match field for field and
  checking it still fails, and again with two *empty* structs, which is the case
  a structural comparison gets wrong most quietly;
- a fragment stage's second parameter is its varyings whatever its type, so a
  stage that omits it is refused rather than reading its first uniform as
  varyings;
- attachment routing is exact: a stage writing a distinct constant per field is
  read back per attachment, each holding its own value;
- `accel.Discard` writes neither colour nor depth, checked against a target
  pre-cleared to a value the stage never writes;
- a texel fetch returns the same value on both backends for every in-range
  coordinate, and zero for every out-of-range one;
- a fragment stage with a written slice parameter is refused by name; and
- the stage record's binding indices are dense per stage, confirmed by a stage
  whose parameter index and binding index differ.

## 11. Open questions

- **Whether a vertex stage may write a storage buffer.** Vertex-stage writes have
  the same ordering problem as fragment-stage writes and a different shape (no
  ROA equivalent applies). Excluded for now by the same rule, but the case that
  wants it — a GPU-driven culling pass writing compacted output — is real and is
  better served by a compute pass today.
- **Whether the varyings slot count should pack.** Section 3.2 chooses not to.
  If a real caller hits the limit with fields that would pack, this is worth
  revisiting as an explicit, caller-visible packing rather than an automatic one.
- **Point size.** A point-list topology needs one, every backend spells it as a
  vertex-stage built-in output, and no test in 005's corpus needs it. Deferred
  until the corpus does, rather than designed against no use.
