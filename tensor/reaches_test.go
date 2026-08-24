// Copyright 2026 The golang.design Initiative Authors.
// All rights reserved. Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

package tensor_test

import (
	"reflect"
	"testing"

	"golang.design/x/accel"
	"golang.design/x/accel/tensor"
)

// Every field of an options struct must reach the kernel or be refused.
//
// # The class this exists for
//
// Six times now a field has been accepted and reached nothing: a shuffle seed,
// a uniform's index, a store op, a base vertex, a shared-byte requirement, and
// AttentionOptions.Pages on a prefill (accel issue 10). Only the last was
// caught by a consumer, and only because they compared a value against an
// oracle -- the others were found by reading code. A refusal-based probe cannot
// see this class at all: the call compiles, so it looks supported.
//
// The test is a comparison of [Builder.Identity], which covers every operator's
// operands but not the values a caller binds. Setting a field must therefore
// change the digest, and it can only do that by reaching a node.
//
// # Why it is driven by reflection
//
// A field that is set and ignored is wrong exactly once, silently, when
// somebody adds the next one. Enumerating the struct's fields and requiring a
// row for each means a new field fails this test until its author says which it
// is -- reaching a kernel, or refused with a reason. That is the whole point;
// a hand-written list of the fields that exist today would have passed on the
// day Pages was added.
func TestEveryAttentionOptionReachesTheKernelOrIsRefused(t *testing.T) {
	rt := newRuntime(t)

	// Ports are declared identically in every case, so a digest that differs
	// differs because an *operator* consumed the value and not because the
	// graph declared one more input.
	type setup struct {
		b     *tensor.Builder
		q     *tensor.Tensor
		k, v  *tensor.State
		opts  tensor.AttentionOptions
		pages *tensor.Tensor
		lens  *tensor.Tensor
		alt   *tensor.Tensor // a second u32 shaped like lens
	}
	build := func(b *tensor.Builder, prefill bool) setup {
		tensor.Scalar(b, tensor.ScalarDesc{Name: "scale", Kind: tensor.ScalarF32})
		tensor.Scalar(b, tensor.ScalarDesc{Name: "base", Kind: tensor.ScalarU32})
		tensor.Scalar(b, tensor.ScalarDesc{Name: "other", Kind: tensor.ScalarF32})
		tensor.Scalar(b, tensor.ScalarDesc{Name: "base2", Kind: tensor.ScalarU32})
		shape := tensor.Shape{4, 8}
		if prefill {
			shape = tensor.Shape{2, 4, 8}
		}
		s := setup{
			b: b,
			q: tensor.Input(b, tensor.ValueDesc{
				Name: "q", DType: accel.F32, Shape: shape,
			}),
			k: tensor.NewState(b, tensor.StateDesc{
				Name: "k", DType: accel.F32, Shape: tensor.Shape{4, 2, 8},
			}),
			v: tensor.NewState(b, tensor.StateDesc{
				Name: "v", DType: accel.F32, Shape: tensor.Shape{4, 2, 8},
			}),
			lens: tensor.Input(b, tensor.ValueDesc{
				Name: "len", DType: accel.U32, Shape: tensor.Shape{1},
			}),
			pages: tensor.Input(b, tensor.ValueDesc{
				Name: "pages", DType: accel.U32, Shape: tensor.Shape{2},
			}),
			alt: tensor.Input(b, tensor.ValueDesc{
				Name: "len2", DType: accel.U32, Shape: tensor.Shape{1},
			}),
		}
		s.opts = tensor.AttentionOptions{Lengths: s.lens, ScaleName: "scale"}
		if prefill {
			s.opts.BaseName = "base"
		}
		return s
	}

	// One row per field of AttentionOptions, exercised under **both** shapes a
	// caller can write. The shape matrix is the part that matters: issue 10 was
	// Pages reaching the decode kernel and being ignored by the prefill one, so
	// a row that tested only the shape where a field works would have passed on
	// the day the bug was introduced.
	rows := map[string]struct {
		set func(s *setup)
		// Whether the field must be *refused* rather than reach a kernel, per
		// shape. A field that does neither is accepted and ignored.
		refusedOnDecode, refusedOnPrefill bool
	}{
		// A different tensor of the same shape: an operand, so the digest moves
		// only if the operator binds the one it was given.
		"Lengths": {set: func(s *setup) { s.opts.Lengths = s.alt }},

		// No paged prefill kernel exists, so a prefill refuses it rather than
		// reading the pool in order (accel issue 10).
		"Pages": {
			set:              func(s *setup) { s.opts.Pages, s.opts.Block = s.pages, 2 },
			refusedOnPrefill: true,
		},
		"Block": {
			set:              func(s *setup) { s.opts.Pages, s.opts.Block = s.pages, 4 },
			refusedOnPrefill: true,
		},

		"ScaleName": {set: func(s *setup) { s.opts.ScaleName = "other" }},

		// A decode step has one query token and no causal mask to place, so a
		// base position is a prefill's and a decode refuses it.
		"BaseName": {
			set:             func(s *setup) { s.opts.BaseName = "base2" },
			refusedOnDecode: true,
		},
	}

	// Reflection is what makes this hold for a field nobody has added yet.
	fields := reflect.TypeOf(tensor.AttentionOptions{})
	for i := range fields.NumField() {
		name := fields.Field(i).Name
		if _, ok := rows[name]; !ok {
			t.Errorf("AttentionOptions.%s has no row here. Add one saying whether the "+
				"field reaches a kernel or is refused: a field that does neither is "+
				"accepted and ignored, which is what this test exists to stop", name)
		}
	}

	digest := func(prefill bool, mutate func(s *setup)) (tensor.Identity, error) {
		b := rt.NewBuilder("reaches")
		s := build(b, prefill)
		if mutate != nil {
			mutate(&s)
		}
		tensor.Output(b, "out", tensor.Attention(b, s.q, s.k, s.v, s.opts))
		if err := b.Err(); err != nil {
			return tensor.Identity{}, err
		}
		return b.Identity(), nil
	}

	for name, row := range rows {
		if row.set == nil {
			continue
		}
		for _, shape := range []struct {
			name    string
			prefill bool
			refused bool
		}{
			{"decode", false, row.refusedOnDecode},
			{"prefill", true, row.refusedOnPrefill},
		} {
			t.Run(name+"/"+shape.name, func(t *testing.T) {
				base, err := digest(shape.prefill, nil)
				if err != nil {
					t.Fatalf("the baseline does not build: %v", err)
				}
				got, err := digest(shape.prefill, row.set)
				switch {
				case err != nil:
					if !shape.refused {
						t.Fatalf("setting %s on a %s was refused, and this row says it "+
							"should reach a kernel: %v", name, shape.name, err)
					}
				case shape.refused:
					t.Errorf("setting %s on a %s was accepted, and this row says it "+
						"should be refused", name, shape.name)
				case got == base:
					t.Errorf("setting %s on a %s changed nothing about the plan and was "+
						"not refused, so the value reaches no kernel and the caller has "+
						"no way to tell", name, shape.name)
				}
			})
		}
	}
}
