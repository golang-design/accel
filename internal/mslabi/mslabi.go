// Copyright 2026 The golang.design Initiative Authors.
// All rights reserved. Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

// Package mslabi is the buffer numbering the MSL emitter writes and the Metal
// backend binds against.
//
// # Why this is its own package
//
// It is three declarations, and they were in internal/kernelc/emit. The Metal
// backend needs exactly these three and nothing else from the emitter, but
// importing emit reaches intrin and ir, which import go/types — so every darwin
// binary that linked accel also linked the Go type checker, for one string
// constant and two additions.
//
// specs/012-kernel-pipeline.md's guarantee is that the root package does not
// depend on the kernel compiler's toolchain, because kernels are compiled ahead
// of time by cmd/accel-kernel and a deployed binary needs neither the go tool
// nor a type checker. That held for the loader and leaked here.
//
// The numbering is shared rather than restated because two copies of a layout
// are one too many: the emitter writes it into the shader and the backend binds
// against it, and a disagreement is a kernel reading another kernel's argument.
package mslabi

// ContractOff is the pragma every emitted kernel carries.
//
// Metal's default is -ffp-contract=fast, so without this a*b+c becomes
// fma(a,b,c) and differs from the CPU backend in the last bit — and
// specs/006-backends.md makes the CPU backend the oracle, which turns that
// difference into a failure rather than a tolerance to widen. MTLMathMode.safe
// does not disable it; this was measured on an M2 rather than assumed.
//
// specs/008-numerics.md section 6 requires contraction to be controlled rather
// than observed. This is where it is controlled.
const ContractOff = "#pragma METAL fp contract(off)"

// LengthsIndex reports the buffer index the generated slice lengths occupy for a
// kernel with n bindings.
//
// The slot is reserved whether or not the body calls len, because a layout that
// depends on the body is a layout the host has to be told about, and one unused
// argument slot is cheaper than a second source of truth.
func LengthsIndex(n int) int { return n }

// UniformIndex reports the buffer index uniform i occupies for a kernel with n
// bindings.
func UniformIndex(n, i int) int { return n + 1 + i }

// StageVertexBufferLimit is how many vertex buffers a render pipeline may bind.
//
// The limit exists because a vertex stage's uniforms and its vertex buffers
// share one Metal buffer index space, so the two have to be separated by a rule
// both the emitter and the backend follow. The emitter does not know how many
// buffers a pipeline will bind -- that is pipeline state and this is stage state
// -- so the split is a constant rather than a computation.
const StageVertexBufferLimit = 16

// StageUniformIndex reports the buffer index a vertex stage's uniform i occupies.
//
// Above every vertex buffer, for the reason above. Getting this wrong is not a
// compile error on either side: the uniform lands on top of vertex buffer zero,
// the stage reads geometry as a transform, and the picture is wrong in a way
// that looks like a bad matrix.
func StageUniformIndex(i int) int { return StageVertexBufferLimit + i }

// StageFragmentUniformIndex reports the buffer index a fragment stage's uniform
// i occupies.
//
// A fragment stage binds no vertex buffers, so its uniforms start at zero and
// no reservation is needed. Named anyway, so the two sides of the ABI are two
// calls to one place rather than one call and a literal.
func StageFragmentUniformIndex(i int) int { return i }

// StageTextureIndex reports the texture index a stage's texture binding i
// occupies.
//
// Metal's texture argument space is separate from its buffer argument space, so
// this needs no reservation the way StageUniformIndex does: a texture at index
// zero and a vertex buffer at index zero do not collide. It is named rather
// than written as a literal on both sides for the same reason the buffer
// numbering is -- the emitter writes it into the shader and the backend binds
// against it, and a disagreement is a stage reading the wrong texture.
func StageTextureIndex(i int) int { return i }
