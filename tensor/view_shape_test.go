// Copyright 2026 The golang.design Initiative Authors.
// All rights reserved. Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

package tensor_test

import (
	"strings"
	"testing"

	"golang.design/x/accel"
	"golang.design/x/accel/tensor"
)

func TestSamplingRejectsEmptyViews(t *testing.T) {
	for _, axis := range []int{0, 1} {
		for _, op := range []struct {
			name   string
			sample func(*tensor.Builder, *tensor.Tensor, *tensor.Tensor) *tensor.Tensor
		}{
			{"Argmax", func(b *tensor.Builder, x, d *tensor.Tensor) *tensor.Tensor { return tensor.Argmax(b, x) }},
			{"SampleCategorical", tensor.SampleCategorical},
			{"Sample", func(b *tensor.Builder, x, d *tensor.Tensor) *tensor.Tensor {
				return tensor.Sample(b, x, d, nil, nil, tensor.SamplingOptions{Temperature: 1}, "")
			}},
			{"TopKMask", func(b *tensor.Builder, x, d *tensor.Tensor) *tensor.Tensor { return tensor.TopKMask(b, x, 1) }},
			{"TopPMask", func(b *tensor.Builder, x, d *tensor.Tensor) *tensor.Tensor { return tensor.TopPMask(b, x, 0.9) }},
			{"SampleRows", func(b *tensor.Builder, x, d *tensor.Tensor) *tensor.Tensor {
				return tensor.SampleRows(b, x, d, tensor.RowSampling{Factor: d}, "")
			}},
		} {
			t.Run(op.name+string(rune('0'+axis)), func(t *testing.T) {
				rt := newRuntime(t)
				b := rt.NewBuilder("empty")
				x := tensor.Input(b, tensor.ValueDesc{Name: "x", DType: accel.F32, Shape: tensor.Shape{1, 4}})
				d := tensor.Input(b, tensor.ValueDesc{Name: "draw", DType: accel.F32, Shape: tensor.Shape{1}})
				x = tensor.Slice(b, x, axis, 0, 0)
				tensor.Output(b, "out", op.sample(b, x, d))
				if p, err := b.Compile(rt, tensor.CompileOptions{}); err == nil {
					p.Close()
					t.Fatal("Compile accepted empty sampling input")
				} else if !strings.Contains(err.Error(), op.name) || !strings.Contains(err.Error(), "empty") {
					t.Fatalf("expected sampling diagnostic for empty input, got %v", err)
				}
			})
		}
	}
}
