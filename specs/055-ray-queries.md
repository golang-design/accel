---
title: "Ray queries: traversal as a call, and the order nobody may depend on"
status: drafted
layer: device
depends_on:
  - 002-compute-model.md
  - 004-kernel-authoring.md
  - 008-numerics.md
  - 053-ray-tracing.md
---

# Ray queries

[053](053-ray-tracing.md)'s second child. What a kernel writes, what it gets
back, and which parts of the answer are portable.

## 1. Two calls, and why the closest one is not sugar

```go
//accel:kernel workgroup=64
func Shade(t accel.Thread, scene accel.AccelStruct, org []float32, dir []float32,
    out []float32) {

    i := t.GlobalID().X
    r := accel.Ray{
        Origin: accel.Vec3{org[3*i], org[3*i+1], org[3*i+2]},
        Dir:    accel.Vec3{dir[3*i], dir[3*i+1], dir[3*i+2]},
        TMin:   1e-4, TMax: 1e30, Mask: 0xFF,
    }
    if h := accel.TraceClosest(scene, r); h.Hit {
        out[i] = h.T
    }
}
```

`TraceClosest` returns the nearest hit. `TraceAny` returns whether *any* hit
exists and stops at the first one found, which is what a shadow ray wants and is
**not** the same call with a flag: it is allowed to stop early, so it visits a
different set of candidates and its `T` is meaningless. Two calls rather than one
with a mode, because a caller who reads `T` off an any-hit result has written a
bug that returns a number.

## 2. The hit record

| field | is | portable? |
| --- | --- | --- |
| `Hit` | whether anything was hit in $[t_{\min}, t_{\max}]$ | **yes** |
| `T` | the distance along the ray | within a derived bound |
| `Instance` | the TLAS instance index | yes, where the hit is not coincident |
| `Primitive` | the index within the BLAS geometry | yes, same caveat |
| `Bary` | barycentric $(u,v)$ on the triangle | within a derived bound |
| `FrontFacing` | winding relative to the ray | yes |

`Hit` is a boolean over a closed interval and is exact. Everything else divides
into an integer that is right or wrong, and a float that carries a bound —
[008](008-numerics.md)'s division, not a new one.

**There is no "hit position" field.** It would be $\text{org} + T\cdot\text{dir}$,
computed identically by the caller, and a stored one is a second copy that can
disagree with $T$. The same argument [046](046-segmented-extents.md) §1.1 makes
for deriving offsets rather than taking them.

## 3. `TMin` is exclusive, and this is the decision

$$\text{a hit counts} \iff t_{\min} < t \le t_{\max}$$

[053](053-ray-tracing.md) §9 raised it and this settles it. The self-hit at
$t=0$ — a secondary ray leaving the surface it just hit and immediately hitting
it again — is the most common bug in the domain, and an inclusive $t_{\min}$
makes the natural spelling wrong. Exclusive makes `TMin: 0` mean "everything
strictly in front", which is what a reader assumes.

The upper bound is inclusive so that `TMax` set to a light's distance includes
geometry exactly at the light.

**Targets differ here**, which is why it is stated once and enforced by the
CPU intersector's own comparison rather than inherited from whichever backend a
caller tried first.

## 4. The order rule

**A kernel may not depend on the order candidates are visited.**

`TraceClosest` is a minimum and is order-independent by construction — that is
the point of preferring it. `TraceAny` returns a boolean and is
order-independent for the same reason: "does any hit exist" does not care which
one was found.

What is **not** available in v0 is a traversal a caller can step, visiting each
candidate. That is what would make an order-dependent kernel expressible, and
[053](053-ray-tracing.md) decision 1 already declines the callback model it
implies. So the rule costs nothing today and is stated now because it is what
a later `TraceStepwise` would have to obey — and because the CPU intersector
must be built to shuffle from the start ([056](056-cpu-intersector.md)), which
is not a thing that can be retrofitted honestly.

## 5. Where a query may appear

In a compute kernel. Not in a vertex or fragment stage, and the reason is
[005](005-graphics.md)'s open handoff rather than anything about traversal:
a stage cannot bind a resource of this kind today because
[032](032-stage-abi.md) §5.1 declines to own the compute-visible-resource
question. When that is answered, a query in a fragment stage is additive.

A query is **not** allowed in a helper called from a stage, for the same reason
and checked the same way — the front end refuses by name, as it does for a
texture reaching a compute kernel.

## 6. Uniformity, and why traversal is not a barrier

A hit is a load from a storage-class resource, so under
[002](002-compute-model.md) §3.3's lattice its result is **non-uniform** — like
any buffer read. A `Hit` used as a barrier predicate is therefore rejected,
which is correct and worth stating because "every lane traced the same ray" is a
thing a caller will believe and the analysis cannot prove.

Traversal itself contains loops and is lowered as a call, not inlined into the
state machine [018](018-cooperative-lowering.md) builds. **So a query is legal in
a cooperative kernel and does not split a state**, which keeps it out of the
resumable transform entirely.

## 7. Done

- **A ray that hits nothing returns `Hit == false`** and leaves every other
  field alone, checked by seeding the record with a sentinel.
- **A ray through two triangles returns the nearer**, and returns the farther
  when `TMin` is raised past the first — the discriminating pair, since a
  traversal returning "some hit" passes the first test alone.
- **`TMin` is exclusive**: a ray originating exactly on a surface with
  `TMin: 0` does not hit it, and hits it with `TMin: -1`.
- **`TraceAny` finds the same existence answer as `TraceClosest().Hit`** over a
  scene where they must agree, which is the only property both share.
- **A barycentric coordinate reconstructs the hit position** to within
  [008](008-numerics.md)'s bound for the interpolation, tying `Bary` to `T`
  rather than testing it alone.
- **A query used as a barrier predicate is refused**, naming the uniformity rule.
- **CPU and Metal agree** on `Hit`, `Instance` and `Primitive` exactly, and on
  `T` and `Bary` within the derived bound, over rays generated to avoid
  coincident surfaces — and the generator's avoidance is itself asserted, since
  a differential over degenerate rays would be measuring the wrong thing.
