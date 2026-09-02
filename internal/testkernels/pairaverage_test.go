// Copyright 2026 The golang.design Initiative Authors.
// All rights reserved. Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

package testkernels_test

import (
	"fmt"
	"math"
	"testing"

	"golang.design/x/accel"
	"golang.design/x/accel/internal/conformance/direct"
	"golang.design/x/accel/internal/conformance/numeq"
	"golang.design/x/accel/internal/testkernels"
	"golang.design/x/accel/kernelabi"
)

// TestPairAverageMatchesAuthored is spec 004's fifth level for a kernel whose
// helpers reach helpers: the generated lowering has to carry halve and putAt,
// which no kernel names, and the record has to say out is written, which only
// putAt does.
func TestPairAverageMatchesAuthored(t *testing.T) {
	if k := testkernels.PairAverageKernel; len(k.Bindings) != 2 ||
		k.Bindings[1].Name != "out" || k.Bindings[1].Access&kernelabi.Write == 0 {
		t.Fatalf("out is written two helpers deep and the record does not say so: %+v", k.Bindings)
	}
	for _, n := range []int{0, 1, 7, 32, 33, 100} {
		t.Run(fmt.Sprint(n), func(t *testing.T) {
			in := make([]float32, n*2)
			for i := range in {
				in[i] = float32(i)*0.25 - 3
			}

			authored := make([]float32, n)
			runAuthoredKernel(&testkernels.PairAverageKernel, n, func(th accel.Thread) {
				testkernels.PairAverage(th, in, authored)
			})

			generated := make([]float32, n)
			if err := direct.Run(&testkernels.PairAverageKernel,
				direct.Cover(&testkernels.PairAverageKernel, n),
				kernelabi.Args{Slices: []any{in, generated}}); err != nil {
				t.Fatalf("direct.Run: %v", err)
			}

			if r := numeq.ExactBits(generated, authored, func(f float32) uint64 {
				return uint64(math.Float32bits(f))
			}); !r.Equal {
				t.Errorf("the generated lowering and the authored function disagree: %v", r)
			}
			for i := range generated {
				want := (in[2*i] + in[2*i+1]) * 0.5
				if generated[i] != want {
					t.Errorf("out[%d] = %v, want %v", i, generated[i], want)
				}
			}
		})
	}
}
