// Copyright 2026 The golang.design Initiative Authors.
// All rights reserved. Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

package tensor_test

import (
	"testing"

	"golang.design/x/accel"
	"golang.design/x/accel/tensor"
)

// Two plans that differ only in which layer of a cache they address are
// different plans.
//
// specs/007-tensor-layer.md's reason for a digest is that "two different models
// over the same shapes are different plans, and returning one for the other
// produces a confident wrong answer". A layer view is that case exactly: same
// operator, same shapes, same kernel, different bytes.
//
// It is here because the window landed after the digest did. A layer's offset
// is the binding's rather than the view's, so none of what writeTensor covered
// -- port name, shape, strides, view offset -- moved when the layer changed.
func TestALayerViewIsItsOwnIdentity(t *testing.T) {
	rt := newRuntime(t)

	id := func(layer int) tensor.Identity {
		b := rt.NewBuilder("layers")
		tensor.Scalar(b, tensor.ScalarDesc{Name: "scale", Kind: tensor.ScalarF32})
		lens := tensor.Input(b, tensor.ValueDesc{
			Name: "len", DType: accel.U32, Shape: tensor.Shape{1},
		})
		rows := tensor.Input(b, tensor.ValueDesc{
			Name: "rows", DType: accel.F32, Shape: tensor.Shape{1, 16},
		})
		ids := tensor.Input(b, tensor.ValueDesc{
			Name: "ids", DType: accel.U32, Shape: tensor.Shape{1},
		})
		q := tensor.Input(b, tensor.ValueDesc{
			Name: "q", DType: accel.F32, Shape: tensor.Shape{4, 8},
		})
		kc := tensor.NewState(b, tensor.StateDesc{
			Name: "k", DType: accel.F32, Shape: tensor.Shape{4, 4, 2, 8},
		})
		vc := tensor.NewState(b, tensor.StateDesc{
			Name: "v", DType: accel.F32, Shape: tensor.Shape{4, 4, 2, 8},
		})
		lk := tensor.ScatterRows(b, tensor.LayerState(b, kc, layer), rows, ids)
		tensor.Output(b, "out", tensor.Attention(b, q, lk,
			tensor.LayerState(b, vc, layer),
			tensor.AttentionOptions{Lengths: lens, ScaleName: "scale"}))
		if err := b.Err(); err != nil {
			t.Fatalf("layer %d: %v", layer, err)
		}
		return b.Identity()
	}

	seen := map[tensor.Identity]int{}
	for layer := range 4 {
		got := id(layer)
		if prev, ok := seen[got]; ok {
			t.Fatalf("layers %d and %d have the same identity %v; a plan cache would "+
				"answer one with the other, and they read and write different bytes",
				prev, layer, got)
		}
		seen[got] = layer
	}
}
