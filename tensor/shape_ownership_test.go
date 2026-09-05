// Copyright 2026 The golang.design Initiative Authors.
// All rights reserved. Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

package tensor_test

import (
	"testing"

	"golang.design/x/accel"
	"golang.design/x/accel/tensor"
)

func TestTensorShapesDoNotAliasCallerMemory(t *testing.T) {
	for _, kind := range []string{"input", "weight", "state", "reshape", "broadcast", "accessor", "ports"} {
		t.Run(kind, func(t *testing.T) {
			rt := newRuntime(t)
			b := rt.NewBuilder(kind)
			shape := tensor.Shape{2, 2}
			desc := tensor.ValueDesc{Name: "x", DType: accel.F32, Shape: shape}
			var x *tensor.Tensor
			switch kind {
			case "weight":
				x = tensor.Weight(b, desc)
			case "state":
				x = tensor.ReadState(b, tensor.NewState(b, tensor.StateDesc(desc)))
			default:
				x = tensor.Input(b, desc)
			}
			switch kind {
			case "reshape":
				shape = tensor.Shape{2, 2}
				x = tensor.Reshape(b, x, shape)
			case "broadcast":
				shape = tensor.Shape{2, 2}
				x = tensor.Broadcast(b, x, shape)
			case "accessor":
				shape = x.Shape()
			}
			tensor.Output(b, "out", tensor.Add(b, x, x))
			identity := b.Identity()
			if kind != "ports" {
				shape[0] = 1
			}
			if b.Identity() != identity {
				t.Fatal("mutating caller shape changed graph identity")
			}
			if !x.Shape().Equal(tensor.Shape{2, 2}) {
				t.Fatalf("tensor shape changed: %v", x.Shape())
			}
			p, err := b.Compile(rt, tensor.CompileOptions{})
			if err != nil {
				t.Fatal(err)
			}
			defer p.Close()
			if kind == "ports" {
				for _, port := range p.Ports() {
					port.Shape[0] = 1
				}
				for _, port := range p.Ports() {
					if !port.Shape.Equal(tensor.Shape{2, 2}) {
						t.Fatalf("port %s changed: %v", port.Name, port.Shape)
					}
				}
			}
			d := rt.Device()
			out := f32Buffer(t, d, "out", make([]float32, 4))
			if err := p.Submit(d.Queue(), tensor.Bindings{Buffers: map[string]accel.BufferView{
				"x": f32Buffer(t, d, "x", []float32{1, 2, 3, 4}), "out": out,
			}}).Wait(); err != nil {
				t.Fatal(err)
			}
			got := make([]float32, 4)
			if err := d.Queue().ReadBuffer(out.Buffer, 0, got); err != nil {
				t.Fatal(err)
			}
			for i, v := range got {
				if v != float32(2*(i+1)) {
					t.Fatalf("element %d = %v", i, v)
				}
			}
		})
	}
}
