// Copyright 2026 The golang.design Initiative Authors.
// All rights reserved. Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

package kernels

import (
	"testing"

	"golang.design/x/accel"
)

// Every generated uniform codec's decoder is its encoder's inverse.
//
// The two are generated from the same std140 offsets, so this is the property
// that says they still are: a field added to one and not the other, or an
// offset computed differently, shows up as a block that does not round-trip.
//
// # Enumerated rather than listed
//
// It walks the corpus's own records rather than naming codecs, for
// specs/062-backend-parity.md's reason one level down: a list somebody
// maintains by hand reports full coverage of whatever it happens to contain,
// and a new uniform block would join the corpus with nothing checking it.
//
// # And it is portable on purpose
//
// The Metal differential is what exercises a lowering on a Mac and it runs
// nowhere else, so a generated function with no portable caller reads as
// covered on darwin and drops the Linux gate. specs/009-sequencing.md records
// that failure three times; this is the fourth avoided rather than repeated.
func TestEveryGeneratedUniformCodecRoundTrips(t *testing.T) {
	var checked int

	check := func(t *testing.T, owner, name string, size int,
		enc func([]byte, any) error, dec func([]byte) (any, error)) {
		t.Helper()
		if enc == nil || dec == nil {
			t.Errorf("%s's uniform %q carries encode=%t decode=%t; a block a backend "+
				"can write and not read is one the CPU rasterizer cannot receive",
				owner, name, enc != nil, dec != nil)
			return
		}
		// The zero value, which is the one every block has and the one a
		// decoder returning its own zero would agree with by accident -- so the
		// assertion below is that the *bytes* round-trip, not the value.
		raw := make([]byte, size)
		v, err := dec(raw)
		if err != nil {
			t.Errorf("%s's uniform %q: decoding a zero block: %v", owner, name, err)
			return
		}
		again := make([]byte, size)
		if err := enc(again, v); err != nil {
			t.Errorf("%s's uniform %q: re-encoding: %v", owner, name, err)
			return
		}
		for i := range raw {
			if raw[i] != again[i] {
				t.Errorf("%s's uniform %q: byte %d is %d after a round trip and %d "+
					"before; the generated encoder and decoder disagree about the layout",
					owner, name, i, again[i], raw[i])
				return
			}
		}
		checked++
	}

	for _, s := range Stages {
		for _, u := range s.Uniforms {
			check(t, s.Name, u.Name, u.Size, u.Encode, u.Decode)
		}
	}
	// A compute kernel's record carries no decoder, and no decoder is generated
	// for a block only a dispatch takes: a by-value parameter is written and
	// never read back, so one would be generated code nothing can call. What is
	// checked for those is that the encoder is there at all.
	for _, k := range Kernels {
		for _, u := range k.Uniforms {
			if u.Encode == nil {
				t.Errorf("%s's uniform %q carries no encoder", k.Name, u.Name)
			}
		}
	}
	if checked == 0 {
		t.Fatal("no uniform block was checked, so the corpus registries are not being read")
	}
	t.Logf("%d uniform blocks round-tripped", checked)
}

// The stages whose authored form had no portable caller.
//
// specs/010-kernel-corpus.md section 6 puts an authored-versus-generated
// comparison on every corpus entry, and these had it only through the Metal
// differential -- which runs on a Mac and nowhere else, so they read as covered
// on darwin and dropped the Linux gate.
//
// Each is compared over inputs that make its own content visible: a constant
// stage at one point would agree with any other constant.
func TestTheRemainingStagesAgreeWithTheirSource(t *testing.T) {
	tex := checker3x2()
	frag := func(x, y float32) accel.Fragment {
		return accel.NewFragmentForTest(accel.Vec4{x, y, 0.25, 1}, true)
	}

	t.Run("SolidFS", func(t *testing.T) {
		f := frag(0.5, 0.5)
		if got, want := solidFSFlat(f, accel.NoVaryings{}), SolidFS(f, accel.NoVaryings{}); got != want {
			t.Errorf("generated %+v, authored %+v", got, want)
		}
	})

	t.Run("TintedFS", func(t *testing.T) {
		// Two tints, because a stage ignoring its uniform agrees with itself.
		for _, c := range []accel.Vec4{{1, 0, 0, 1}, {0, 0.25, 0.5, 1}} {
			f := frag(0.5, 0.5)
			in := StageTint{Colour: c}
			if got, want := tintedFSFlat(f, accel.NoVaryings{}, in),
				TintedFS(f, accel.NoVaryings{}, in); got != want {
				t.Errorf("tint %v: generated %+v, authored %+v", c, got, want)
			}
		}
	})

	t.Run("BlitFS", func(t *testing.T) {
		for y := range 3 {
			for x := range 4 {
				f := frag(float32(x)+0.5, float32(y)+0.5)
				if got, want := blitFSFlat(f, accel.NoVaryings{}, tex),
					BlitFS(f, accel.NoVaryings{}, tex); got != want {
					t.Errorf("(%d,%d): generated %+v, authored %+v", x, y, got, want)
				}
			}
		}
	})

	t.Run("HalfTriangleVS", func(t *testing.T) {
		// All three vertices: the stage's whole content is which index moves
		// which axis, so one vertex proves nothing.
		for i := range uint32(3) {
			v := accel.NewVertexForTest(i, 0)
			gotPos, gotVary := halfTriangleVSFlat(v)
			wantPos, wantVary := HalfTriangleVS(v)
			if gotPos != wantPos || gotVary != wantVary {
				t.Errorf("vertex %d: generated (%v, %+v), authored (%v, %+v)",
					i, gotPos, gotVary, wantPos, wantVary)
			}
		}
	})

	t.Run("PerspectiveVS", func(t *testing.T) {
		for i := range uint32(3) {
			v := accel.NewVertexForTest(i, 0)
			gotPos, gotVary := perspectiveVSFlat(v)
			wantPos, wantVary := PerspectiveVS(v)
			if gotPos != wantPos || gotVary != wantVary {
				t.Errorf("vertex %d: generated (%v, %+v), authored (%v, %+v)",
					i, gotPos, gotVary, wantPos, wantVary)
			}
			// The three w values differ, which is what makes the two
			// interpolation qualifiers distinguishable at all.
			if i > 0 && wantPos[3] == 1 {
				t.Errorf("vertex %d has w = 1; a triangle at constant w compares "+
					"noperspective against itself", i)
			}
		}
	})

	t.Run("PerspectiveFS", func(t *testing.T) {
		for _, in := range []PerspectiveVaryings{
			{Smooth: accel.Vec2{0, 1}, Linear: accel.Vec2{2, 3}},
			{Smooth: accel.Vec2{-1, 0.5}, Linear: accel.Vec2{0.25, 4}},
		} {
			f := frag(0.5, 0.5)
			if got, want := perspectiveFSFlat(f, in), PerspectiveFS(f, in); got != want {
				t.Errorf("%+v: generated %+v, authored %+v", in, got, want)
			}
		}
	})
}

// Every generated codec encodes, and the list of them is guarded.
//
// specs/010-kernel-corpus.md section 6's obligation reaches the codecs too: a
// generated Encode that nothing calls is covered on a Mac by the Metal
// differential and by nothing anywhere else, which is how this package's Linux
// gate goes red on a change that touched none of it.
//
// The list is by hand and the guard is what keeps it honest -- the shape
// specs/062-backend-parity.md argues for: a codec added to the corpus and not
// added here fails, rather than reporting full coverage of whatever the list
// happens to contain.
func TestEveryGeneratedCodecEncodes(t *testing.T) {
	codecs := []struct {
		name string
		run  func() (int, error)
	}{
		{"AttnDims", func() (int, error) {
			c := AttnDimsCodec{}
			return c.EncodedSize(), c.Encode(make([]byte, c.EncodedSize()), AttnDims{})
		}},
		{"BallotDims", func() (int, error) {
			c := BallotDimsCodec{}
			return c.EncodedSize(), c.Encode(make([]byte, c.EncodedSize()), BallotDims{})
		}},
		{"BatchedDims", func() (int, error) {
			c := BatchedDimsCodec{}
			return c.EncodedSize(), c.Encode(make([]byte, c.EncodedSize()), BatchedDims{})
		}},
		{"ShapeDims", func() (int, error) {
			c := ShapeDimsCodec{}
			return c.EncodedSize(), c.Encode(make([]byte, c.EncodedSize()), ShapeDims{})
		}},
		{"ScaleParams", func() (int, error) {
			c := ScaleParamsCodec{}
			return c.EncodedSize(), c.Encode(make([]byte, c.EncodedSize()), ScaleParams{})
		}},
		{"RowParams", func() (int, error) {
			c := RowParamsCodec{}
			return c.EncodedSize(), c.Encode(make([]byte, c.EncodedSize()), RowParams{})
		}},
		{"RoPEParams", func() (int, error) {
			c := RoPEParamsCodec{}
			return c.EncodedSize(), c.Encode(make([]byte, c.EncodedSize()), RoPEParams{})
		}},
		{"BiasParams", func() (int, error) {
			c := BiasParamsCodec{}
			return c.EncodedSize(), c.Encode(make([]byte, c.EncodedSize()), BiasParams{})
		}},
		{"GEMMDims", func() (int, error) {
			c := GEMMDimsCodec{}
			return c.EncodedSize(), c.Encode(make([]byte, c.EncodedSize()), GEMMDims{})
		}},
		{"GroupedDims", func() (int, error) {
			c := GroupedDimsCodec{}
			return c.EncodedSize(), c.Encode(make([]byte, c.EncodedSize()), GroupedDims{})
		}},
		{"GroupedTiledDims", func() (int, error) {
			c := GroupedTiledDimsCodec{}
			return c.EncodedSize(), c.Encode(make([]byte, c.EncodedSize()), GroupedTiledDims{})
		}},
		{"LinearDims", func() (int, error) {
			c := LinearDimsCodec{}
			return c.EncodedSize(), c.Encode(make([]byte, c.EncodedSize()), LinearDims{})
		}},
		{"RowDims", func() (int, error) {
			c := RowDimsCodec{}
			return c.EncodedSize(), c.Encode(make([]byte, c.EncodedSize()), RowDims{})
		}},
		{"PackParams", func() (int, error) {
			c := PackParamsCodec{}
			return c.EncodedSize(), c.Encode(make([]byte, c.EncodedSize()), PackParams{})
		}},
		{"PagedDims", func() (int, error) {
			c := PagedDimsCodec{}
			return c.EncodedSize(), c.Encode(make([]byte, c.EncodedSize()), PagedDims{})
		}},
		{"PagedPrefillDims", func() (int, error) {
			c := PagedPrefillDimsCodec{}
			return c.EncodedSize(), c.Encode(make([]byte, c.EncodedSize()), PagedPrefillDims{})
		}},
		{"PenaltyDims", func() (int, error) {
			c := PenaltyDimsCodec{}
			return c.EncodedSize(), c.Encode(make([]byte, c.EncodedSize()), PenaltyDims{})
		}},
		{"PrefillDims", func() (int, error) {
			c := PrefillDimsCodec{}
			return c.EncodedSize(), c.Encode(make([]byte, c.EncodedSize()), PrefillDims{})
		}},
		{"RaggedDims", func() (int, error) {
			c := RaggedDimsCodec{}
			return c.EncodedSize(), c.Encode(make([]byte, c.EncodedSize()), RaggedDims{})
		}},
		{"SampleDims", func() (int, error) {
			c := SampleDimsCodec{}
			return c.EncodedSize(), c.Encode(make([]byte, c.EncodedSize()), SampleDims{})
		}},
		{"Params", func() (int, error) {
			c := ParamsCodec{}
			return c.EncodedSize(), c.Encode(make([]byte, c.EncodedSize()), Params{})
		}},
		{"SegmentDims", func() (int, error) {
			c := SegmentDimsCodec{}
			return c.EncodedSize(), c.Encode(make([]byte, c.EncodedSize()), SegmentDims{})
		}},
		{"StageTransform", func() (int, error) {
			c := StageTransformCodec{}
			return c.EncodedSize(), c.Encode(make([]byte, c.EncodedSize()), StageTransform{})
		}},
		{"StageTint", func() (int, error) {
			c := StageTintCodec{}
			return c.EncodedSize(), c.Encode(make([]byte, c.EncodedSize()), StageTint{})
		}},
		{"TopDims", func() (int, error) {
			c := TopDimsCodec{}
			return c.EncodedSize(), c.Encode(make([]byte, c.EncodedSize()), TopDims{})
		}},
		{"MatrixParams", func() (int, error) {
			c := MatrixParamsCodec{}
			return c.EncodedSize(), c.Encode(make([]byte, c.EncodedSize()), MatrixParams{})
		}},
	}

	listed := map[string]bool{}
	for _, c := range codecs {
		t.Run(c.name, func(t *testing.T) {
			listed[c.name] = true
			size, err := c.run()
			if err != nil {
				t.Errorf("encoding a zero %s: %v", c.name, err)
			}
			if size <= 0 || size%16 != 0 {
				t.Errorf("%s's block is %d bytes, and a std140 block is a positive "+
					"multiple of sixteen", c.name, size)
			}
		})
	}

	// The guard. Every uniform type the corpus's own records name has to be in
	// the list above.
	for _, k := range Kernels {
		for _, u := range k.Uniforms {
			if !listed[u.Type] {
				t.Errorf("kernel %s takes a %s block and no codec case names it; add "+
					"one, or its encoder is generated and never run", k.Name, u.Type)
			}
		}
	}
	for _, s := range Stages {
		for _, u := range s.Uniforms {
			if !listed[u.Type] {
				t.Errorf("stage %s takes a %s block and no codec case names it",
					s.Name, u.Type)
			}
		}
	}
}
