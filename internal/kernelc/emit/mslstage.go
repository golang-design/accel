// Copyright 2026 The golang.design Initiative Authors.
// All rights reserved. Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

package emit

import (
	"fmt"

	"golang.design/x/accel/internal/mslabi"

	"golang.design/x/accel/internal/kernelc/ir"
)

// The MSL target for a graphics stage: specs/032-stage-abi.md section 12.1.
//
// # What differs from a compute kernel, and why
//
// A compute kernel writes through bindings and returns nothing, so its MSL
// signature is a function of its binding layout alone. A stage returns values,
// and MSL expresses that as one struct per direction with attribute qualifiers
// on the fields — `[[attribute(n)]]` going in, `[[position]]` and the varyings
// coming out, `[[color(n)]]` for a fragment stage's attachments. So this target
// emits three declarations where the compute one emits a signature: the input
// struct, the output struct, and the function that maps between them.
//
// The bodies are the same IR. That is what makes the differential meaningful:
// the generated Go lowering and this text come from one source, and a
// disagreement is a bug in one of the two lowerings rather than in the kernel.

// stageEmit lowers one graphics stage.
func (m *msl) stageEmit(k *ir.Func) {
	for _, u := range k.Uniforms {
		m.uniformStruct(u)
	}
	for _, h := range k.Helpers {
		m.helper(h)
	}

	if k.Varyings == nil {
		m.fail("stage %s has no varyings type", k.Name)
		return
	}
	if k.Stage == ir.StageVertex {
		m.vertexStage(k)
		return
	}
	m.fragmentStage(k)
}

// varyingsStruct emits the struct the two stages exchange.
//
// The vertex stage's position is a field of it rather than a separate result,
// because MSL has no way to return two things: `[[position]]` on a field is how
// the rasterizer is told which one it is. The Go lowering keeps them separate
// and the flat adapter packs the varyings alone, which is why the two agree on
// the varyings and never have to agree on where the position sits.
func (m *msl) varyingsStruct(k *ir.Func, name string, withPosition bool) {
	m.printf("struct %s {\n", name)
	if withPosition {
		m.printf("    float4 _pos [[position]];\n")
	}
	for i, f := range k.Varyings.Fields {
		// [[flat]] is an interpolation qualifier on the member, and it has to
		// appear on *both* structs: MSL matches a vertex output to a fragment
		// input by attribute, and a mismatched qualifier is a pipeline the
		// driver rejects. Emitting it here rather than at each call site is
		// what makes that automatic.
		interp := ""
		if f.Flat {
			interp = " [[flat]]"
		}
		m.printf("    %s %s [[user(v%d)]]%s;\n", m.stageType(f.Type), f.Name, i, interp)
	}
	m.printf("};\n\n")
}

// vertexStage emits a vertex function.
func (m *msl) vertexStage(k *ir.Func) {
	out := k.Name + "_out"
	m.varyingsStruct(k, out, true)

	in := k.Name + "_in"
	if len(k.Attributes) > 0 {
		m.printf("struct %s {\n", in)
		for _, a := range k.Attributes {
			m.printf("    %s %s [[attribute(%d)]];\n", m.stageType(a.Type), a.Name, a.Index)
		}
		m.printf("};\n\n")
	}

	m.printf("vertex %s %s(", out, k.Name)
	var params []string
	if len(k.Attributes) > 0 {
		params = append(params, fmt.Sprintf("%s _in [[stage_in]]", in))
	}
	for i, u := range k.Uniforms {
		params = append(params, fmt.Sprintf("constant %s &%s [[buffer(%d)]]",
			u.TypeName, u.Name, mslabi.StageUniformIndex(i)))
	}
	params = append(params, m.textureParams(k)...)
	// The two ids are always declared, for the reason a compute kernel's three
	// are: the signature stays a function of the declared inputs alone, and MSL
	// does not object to a parameter nothing reads.
	params = append(params,
		"uint _vid [[vertex_id]]",
		"uint _iid [[instance_id]]")
	m.paramList(params)
	m.printf(") {\n")

	// Attributes arrive inside one stage_in struct and the body names them as
	// parameters, so they are unpacked into locals of the same name. A rename
	// in the body would work too and would make the emitted text stop looking
	// like the source a reader is comparing it against.
	for _, a := range k.Attributes {
		m.printf("    %s %s = _in.%s;\n", m.stageType(a.Type), a.Name, a.Name)
	}
	m.printf("    %s _out;\n", out)
	m.stageBody(k)
	m.printf("}\n")
}

// fragmentStage emits a fragment function.
func (m *msl) fragmentStage(k *ir.Func) {
	in := k.Name + "_in"
	m.varyingsStruct(k, in, true)

	out := k.Name + "_out"
	m.printf("struct %s {\n", out)
	for _, t := range k.Outputs {
		m.printf("    float4 %s [[color(%d)]];\n", t.Name, t.Index)
	}
	m.printf("};\n\n")

	m.printf("fragment %s %s(", out, k.Name)
	params := []string{fmt.Sprintf("%s _in [[stage_in]]", in)}
	for i, u := range k.Uniforms {
		params = append(params, fmt.Sprintf("constant %s &%s [[buffer(%d)]]",
			u.TypeName, u.Name, mslabi.StageFragmentUniformIndex(i)))
	}
	params = append(params, m.textureParams(k)...)
	// No separate [[position]] parameter: the varyings struct already carries
	// one, and MSL rejects a signature declaring the attribute twice. The
	// interpolated window position arrives in that field, which is what
	// accel.Fragment.Coord reports.
	params = append(params, "bool _front [[front_facing]]")
	m.paramList(params)
	m.printf(") {\n")

	// The body names the varyings by the authored parameter name, so it is
	// bound here the way a vertex stage's attributes are.
	if len(k.Params) > 1 {
		m.printf("    %s %s = _in;\n", in, k.Params[1].Name)
	}
	m.printf("    %s _out;\n", out)
	m.stageBody(k)
	m.printf("}\n")
}

// textureParams spells a stage's texture bindings.
//
// texture2d<float> whatever the bound format is, which is the same decision the
// Go lowering makes by holding four float32 per texel: a float-typed texture
// decodes its format in fixed function and hands the shader four floats, so one
// spelling covers every format in the table. An integer-typed texture would be
// a second binding type with a second intrinsic, and nothing asks for one.
//
// The default access is sample, which permits read. Declaring access::read
// instead would be narrower and would also stop the same view being sampled if
// a later spec ever admits it, for no gain here.
func (m *msl) textureParams(k *ir.Func) []string {
	out := make([]string, 0, len(k.Textures))
	for _, t := range k.Textures {
		out = append(out, fmt.Sprintf("texture2d<float> %s [[texture(%d)]]",
			t.Name, mslabi.StageTextureIndex(t.Index)))
	}
	return out
}

// paramList prints a parameter list one per line, which is what the compute
// signature does and what keeps a long one readable.
func (m *msl) paramList(params []string) {
	for i, p := range params {
		if i > 0 {
			m.printf(",")
		}
		m.printf("\n    %s", p)
	}
}

// stageBody emits the body with returns rewritten into the output struct.
//
// The rewrite is here rather than in the IR because it is a property of this
// target: MSL returns one struct, Go returns two values, and the IR carries
// what the author wrote. A pass that rewrote the IR would make the two
// lowerings come from different trees, which is exactly what the differential
// test exists to rule out.
func (m *msl) stageBody(k *ir.Func) {
	// The variable, not the type. They differ by one underscore and confusing
	// them emits `Type.field = ...`, which MSL rejects -- so it is caught, but
	// at the point furthest from the mistake.
	m.stageOut = "_out"
	m.stageKind = k.Stage
	m.stageVaryings = k.Varyings
	m.stageOutputs = k.Outputs
	defer func() { m.stageOut = "" }()
	m.block(k.Body, 1)
}

// stageReturn emits one `return` inside a stage, assembling the output struct.
func (m *msl) stageReturn(s *ir.Return, depth int) {
	ind := mslIndent(depth)
	if m.stageKind == ir.StageVertex {
		if len(s.Values) != 2 {
			m.fail("a vertex stage returns a position and its varyings, and this "+
				"returns %d values", len(s.Values))
			return
		}
		m.printf("%s%s._pos = ", ind, m.stageOut)
		m.vectorized(s.Values[0])
		m.printf(";\n")
		m.stageAssign(s.Values[1], depth, func(i int) string {
			return m.stageVaryings.Fields[i].Name
		}, len(m.stageVaryings.Fields))

		// The depth convention, converted here and nowhere else.
		//
		// specs/032-stage-abi.md section 2.3 fixes clip depth as -w <= z <= w,
		// which is OpenGL's and what the CPU rasterizer implements. Metal's
		// clip space is 0 <= z <= w. Emitting the author's z unchanged puts
		// every near-half vertex behind the near plane, so geometry straddling
		// it loses its near half and reads as a broken projection rather than
		// as a convention mismatch -- which is the symptom
		// docs/conventions.md names.
		m.printf("%s%s._pos.z = (%s._pos.z + %s._pos.w) * 0.5;\n",
			ind, m.stageOut, m.stageOut, m.stageOut)
		m.printf("%sreturn %s;\n", ind, m.stageOut)
		return
	}

	if len(s.Values) != 1 {
		m.fail("a fragment stage returns one attachment struct, and this returns %d "+
			"values", len(s.Values))
		return
	}
	m.stageAssign(s.Values[0], depth, func(i int) string {
		return m.stageOutputs[i].Name
	}, len(m.stageOutputs))
	m.printf("%sreturn %s;\n", ind, m.stageOut)
}

// stageAssign writes one returned struct into the output struct, field by
// field.
//
// Field by field rather than as one assignment, because the returned value is
// the authored struct type and the output is this target's own: they hold the
// same values in the same order and are not the same type, since the output
// carries MSL's attribute qualifiers. Assigning a whole struct would need a
// conversion that does not exist.
func (m *msl) stageAssign(v ir.Value, depth int, name func(int) string, n int) {
	ind := mslIndent(depth)
	c, ok := v.(*ir.Composite)
	if !ok {
		// Not a literal: a local holding the struct, so its fields are read.
		for i := range n {
			m.printf("%s%s.%s = ", ind, m.stageOut, name(i))
			m.value(v)
			m.printf(".%s;\n", name(i))
		}
		return
	}
	if len(c.Elems) != n {
		m.fail("a returned struct has %d fields and the stage declares %d",
			len(c.Elems), n)
		return
	}
	for i, e := range c.Elems {
		m.printf("%s%s.%s = ", ind, m.stageOut, name(i))
		m.vectorized(e)
		m.printf(";\n")
	}
}

// vectorized spells a value where a vector is wanted, converting a std140
// member that is spelled as a C array.
//
// One IR type has two MSL spellings, and this is where they meet. A vec4 local
// is float4 and a vec4 *inside a uniform block* is float[4], because std140's
// vec3 occupies twelve bytes where MSL's float3 occupies sixteen — so the block
// cannot use vectors without the padding arithmetic going wrong. MSL will not
// convert an array to a vector, and a stage returning a uniform member straight
// into a varying is exactly where that shows up.
func (m *msl) vectorized(v ir.Value) {
	t := v.Type()
	if !m.arraySpelled(v) || t == nil || t.Kind != ir.Array {
		m.value(v)
		return
	}
	m.printf("%s(", m.stageType(t))
	for i := range t.Len {
		if i > 0 {
			m.printf(", ")
		}
		m.value(v)
		m.printf("[%d]", i)
	}
	m.printf(")")
}

// arraySpelled reports whether a value's MSL spelling is a C array rather than
// a vector, which is true of exactly one thing: a member of a std140 uniform
// block.
func (m *msl) arraySpelled(v ir.Value) bool {
	f, ok := v.(*ir.FieldSel)
	if !ok {
		return false
	}
	p, ok := f.X.(*ir.Param)
	if !ok {
		return false
	}
	for _, u := range m.fn.Uniforms {
		if u.Name == p.Name {
			return true
		}
	}
	return false
}

// stageType spells one attribute or varying type.
//
// Only the float32 vector widths, which is what specs/033-render-api.md's
// AttrFormat admits and what a varyings struct can hold: a stage's attribute
// parameter is [N]float32 and nothing else can receive a fetch.
func (m *msl) stageType(t *ir.Type) string {
	if t == nil {
		m.fail("a stage member with no type")
		return "float"
	}
	if t.Kind != ir.Array {
		if scalar := mslScalar(t.Kind); scalar != "" {
			return scalar
		}
		m.fail("a stage member is %v, and a stage exchanges floats and integers", t)
		return "float"
	}
	if t.Elem == nil {
		m.fail("a stage member is %v, and an array member needs an element type", t)
		return "float"
	}
	scalar := mslScalar(t.Elem.Kind)
	if scalar == "" {
		m.fail("a stage member is %v, and a stage exchanges floats and integers", t)
		return "float"
	}
	switch t.Len {
	case 1:
		return scalar
	case 2, 3, 4:
		return fmt.Sprintf("%s%d", scalar, t.Len)
	}
	m.fail("a stage member is %v, and MSL has no vector of that width", t)
	return "float"
}

// mslScalar is MSL's spelling of a scalar a stage may exchange, or "" for one
// it may not.
//
// Integers are here because specs/032-stage-abi.md section 3.1 admits an
// integer varying, tagged flat. They are *not* interpolated, and the tag on the
// member is what says so -- an integer member without one is a pipeline the
// driver rejects, which is why the front end refuses it before this is reached
// rather than leaving it to a driver's error text.
func mslScalar(k ir.Kind) string {
	switch k {
	case ir.F32:
		return "float"
	case ir.I32:
		return "int"
	case ir.U32:
		return "uint"
	}
	return ""
}
