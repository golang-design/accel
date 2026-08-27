// Copyright 2026 The golang.design Initiative Authors.
// All rights reserved. Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

package front_test

import (
	"strings"
	"testing"

	"golang.design/x/accel/internal/kernelc/front"
)

// The shader-visible texture binding of specs/032-stage-abi.md section 5, at
// the front end: which signatures may declare one, what index it takes, and
// whether the body is recorded as reading it.

// A texture parameter is a stage binding, and a fetch from it is recorded as a
// read.
//
// The accepting half. The refusals below would all be satisfied by a compiler
// that rejected every texture, which is why the accepting case is a test rather
// than an assumption. And a texture whose read went unrecorded is a subresource
// the graph draws no edge to, which is the barrier
// specs/045-texture-attachments.md section 3 puts between the pass that writes
// an attachment and the pass that fetches it.
func TestATextureIsAStageBindingAndAFetchReadsIt(t *testing.T) {
	for _, c := range []struct {
		name  string
		body  string
		want  string
		reads bool
	}{
		{
			// The coordinate is an integer varying rather than a converted
			// float. A float-to-integer conversion is refused since
			// specs/051-float-to-int.md, and this case is about a texture being
			// a stage binding that a fetch reads -- converting here would make
			// it about the conversion, and textureSource carries one import so
			// it cannot reach kmath without giving the other cases an unused one.
			name: "a fragment stage that fetches",
			body: "type In struct{ Texel accel.Vec2 }\n" +
				"type Out struct{ C accel.Vec4 }\n\n" +
				"//accel:fragment\n" +
				"func F(f accel.Fragment, in In, src accel.Texture2D) Out {\n" +
				"\treturn Out{accel.Fetch(src, 0, 0)}\n" +
				"}",
			want: "src", reads: true,
		},
		{
			name: "a vertex stage that fetches",
			body: "//accel:vertex\n" +
				"func V(v accel.Vertex, h accel.Texture2D) (accel.Clip, accel.NoVaryings) {\n" +
				"\tc := accel.Fetch(h, int32(v.VertexIndex()), 0)\n" +
				"\treturn accel.Clip{c[0], c[1], c[2], 1}, accel.NoVaryings{}\n" +
				"}",
			want: "h", reads: true,
		},
		{
			name: "a texture nothing fetches from",
			body: "type Out struct{ C accel.Vec4 }\n\n" +
				"//accel:fragment\n" +
				"func F(f accel.Fragment, in accel.NoVaryings, unused accel.Texture2D) Out {\n" +
				"\treturn Out{accel.Vec4{1, 0, 0, 1}}\n" +
				"}",
			want: "unused", reads: false,
		},
	} {
		t.Run(c.name, func(t *testing.T) {
			pkg := checkSource(t, textureSource(c.body))
			if pkg == nil {
				t.Fatal("the source did not type-check")
			}
			fns, diags := front.Check(pkg)
			if len(diags) > 0 {
				t.Fatalf("a texture binding should be admitted in a stage: %v", diags)
			}
			if len(fns) != 1 {
				t.Fatalf("got %d stages, want 1", len(fns))
			}
			tex := fns[0].Textures
			if len(tex) != 1 {
				t.Fatalf("got %d textures, want 1", len(tex))
			}
			if tex[0].Name != c.want {
				t.Errorf("texture is named %q, want %q", tex[0].Name, c.want)
			}
			if tex[0].Index != 0 {
				t.Errorf("texture index is %d, want 0: the index is dense among the "+
					"textures rather than the parameter position", tex[0].Index)
			}
			if tex[0].Reads != c.reads {
				t.Errorf("Reads = %v, want %v", tex[0].Reads, c.reads)
			}
			// A texture is its own resource kind. Counting it as a storage
			// binding would make a caller supply a buffer for it.
			if n := len(fns[0].Bindings); n != 0 {
				t.Errorf("got %d slice bindings, want 0", n)
			}
			if n := len(fns[0].Uniforms); n != 0 {
				t.Errorf("got %d uniforms, want 0: a texture must not be placed as a "+
					"std140 block just because its underlying type is a struct", n)
			}
		})
	}
}

// Two textures take consecutive dense indices whatever sits between them in the
// signature.
//
// The index is what a backend binds against. A texture that took its parameter
// position instead would be bound at a slot the emitter never wrote, and
// nothing would fail to compile on either side.
func TestTextureIndicesAreDenseAmongTextures(t *testing.T) {
	pkg := checkSource(t, textureSource(
		"type Tint struct{ C accel.Vec4 }\n"+
			"type Out struct{ C accel.Vec4 }\n\n"+
			"//accel:fragment\n"+
			"func F(f accel.Fragment, in accel.NoVaryings, a accel.Texture2D, "+
			"tint Tint, b accel.Texture2D) Out {\n"+
			"\tx := accel.Fetch(a, 0, 0)\n"+
			"\ty := accel.Fetch(b, 1, 1)\n"+
			"\treturn Out{accel.Vec4{x[0], y[1], tint.C[2], 1}}\n"+
			"}"))
	if pkg == nil {
		t.Fatal("the source did not type-check")
	}
	fns, diags := front.Check(pkg)
	if len(diags) > 0 {
		t.Fatalf("two textures and a uniform should be admitted: %v", diags)
	}
	tex := fns[0].Textures
	if len(tex) != 2 {
		t.Fatalf("got %d textures, want 2", len(tex))
	}
	for i, want := range []string{"a", "b"} {
		if tex[i].Name != want || tex[i].Index != i {
			t.Errorf("texture %d is %q at index %d, want %q at index %d",
				i, tex[i].Name, tex[i].Index, want, i)
		}
		if !tex[i].Reads {
			t.Errorf("texture %q is fetched from and Reads is false", tex[i].Name)
		}
	}
	if tex[1].Param <= tex[0].Param {
		t.Errorf("parameter positions %d and %d are not in signature order",
			tex[0].Param, tex[1].Param)
	}
	if len(fns[0].Uniforms) != 1 {
		t.Errorf("got %d uniforms, want 1", len(fns[0].Uniforms))
	}
}

// What a texture is not.
//
// Each of these type-checks as Go and means nothing on a texture, so each has
// to be refused by name rather than lowered into something.
func TestTextureRefusals(t *testing.T) {
	for _, c := range []struct {
		name, body, want string
	}{
		{
			name: "a texture in a compute kernel",
			body: "//accel:kernel workgroup=64\n" +
				"func K(t accel.Thread, src accel.Texture2D, out []float32) {\n" +
				"\tout[0] = 1\n" +
				"}",
			want: "reaches a vertex or fragment stage only",
		},
		{
			name: "a texture indexed like a slice",
			body: "type Out struct{ C accel.Vec4 }\n\n" +
				"//accel:fragment\n" +
				"func F(f accel.Fragment, in accel.NoVaryings, src accel.Texture2D) Out {\n" +
				"\treturn Out{accel.Vec4{src[0], 0, 0, 1}}\n" +
				"}",
			want: "texture2d is not something a kernel indexes",
		},
		{
			name: "a fetch with a level operand",
			body: "type Out struct{ C accel.Vec4 }\n\n" +
				"//accel:fragment\n" +
				"func F(f accel.Fragment, in accel.NoVaryings, src accel.Texture2D) Out {\n" +
				"\treturn Out{accel.Fetch(src, 0, 0, 0)}\n" +
				"}",
			// specs/032-stage-abi.md section 5 wrote a level operand and
			// specs/045-texture-attachments.md section 2 then put the mip on
			// the view. There is no level to pass, and a stage written against
			// the older spelling is told so rather than silently ignored.
			want: "accel.Fetch takes 3 arguments and got 4",
		},
	} {
		t.Run(c.name, func(t *testing.T) {
			pkg := checkSource(t, textureSource(c.body))
			if pkg == nil {
				// Go's own type checker refuses it, which is the strongest
				// refusal available and needs no diagnostic of ours.
				if c.want != "did not type-check" {
					t.Fatalf("the source did not type-check, and the refusal expected "+
						"was %q", c.want)
				}
				return
			}
			_, diags := front.Check(pkg)
			if len(diags) == 0 {
				t.Fatal("accepted, and it should be refused by name")
			}
			var found bool
			for _, d := range diags {
				if strings.Contains(d.Error(), c.want) {
					found = true
				}
			}
			if !found {
				t.Errorf("diagnostics %v, want one containing %q", diags, c.want)
			}
		})
	}
}

// textureSource wraps a body in the package header these cases share.
func textureSource(body string) string {
	return "package k\n\nimport \"golang.design/x/accel\"\n\n" + body + "\n"
}
