// Copyright 2026 The golang.design Initiative Authors.
// All rights reserved. Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

package tensor_test

import (
	"math"
	"strings"
	"testing"

	"golang.design/x/accel"
	"golang.design/x/accel/tensor"
)

// A plan compiled with MaxInFlight above one runs that many submissions at
// once, each into its own graph, and refuses the one past the limit.
//
// specs/029-plan-cache.md section 5. The cache hands one plan to every request
// in a bucket and a plan binds when it is submitted, so without this a server
// could run one request per bucket at a time. Both submissions are held
// behind a blocking graph so both are in flight together, every time; each
// writes its own output, which is the property the one-in-flight rule
// protected and this keeps.
func TestMaxInFlightRunsSubmissionsSideBySide(t *testing.T) {
	const n = 1 << 8
	rt := newRuntime(t)
	d := rt.Device()
	b := rt.NewBuilder("f")
	h := tensor.Input(b, value("x", n))
	tensor.Output(b, "y", tensor.SiLU(b, h))
	plan, err := b.Compile(rt, tensor.CompileOptions{MaxInFlight: 2})
	if err != nil {
		t.Fatalf("compile: %v", err)
	}
	defer plan.Close()

	x1 := make([]float32, n)
	x2 := make([]float32, n)
	for i := range x1 {
		x1[i] = 1
		x2[i] = -2
	}
	y1 := f32Buffer(t, d, "y1", make([]float32, n))
	y2 := f32Buffer(t, d, "y2", make([]float32, n))
	y3 := f32Buffer(t, d, "y3", make([]float32, n))

	hold, release := blockingGraph(t, d)
	held := d.Queue().Submit(hold)

	bind := func(x []float32, y accel.BufferView) tensor.Bindings {
		return tensor.Bindings{Buffers: map[string]accel.BufferView{
			"x": f32Buffer(t, d, "x", x), "y": y,
		}}
	}
	f1 := plan.Submit(d.Queue(), bind(x1, y1))
	f2 := plan.Submit(d.Queue(), bind(x2, y2))
	if f1.Done() || f2.Done() {
		close(release)
		t.Fatal("a submission completed, or failed, behind a graph that has not been released")
	}
	// The third finds both instances busy and the limit reached.
	f3 := plan.Submit(d.Queue(), bind(x1, y3))
	if !f3.Done() {
		close(release)
		t.Fatal("a third submission past MaxInFlight was accepted")
	}
	if err := f3.Wait(); err == nil || !strings.Contains(err.Error(), "MaxInFlight") {
		close(release)
		t.Fatalf("the refusal should name the option: %v", err)
	}

	close(release)
	if err := held.Wait(); err != nil {
		t.Fatalf("the holding graph: %v", err)
	}
	if err := f1.Wait(); err != nil {
		t.Fatalf("first: %v", err)
	}
	if err := f2.Wait(); err != nil {
		t.Fatalf("second: %v", err)
	}
	silu := func(x float64) float32 { return float32(x / (1 + math.Exp(-x))) }
	for _, c := range []struct {
		name string
		y    accel.BufferView
		want float32
	}{{"first", y1, silu(1)}, {"second", y2, silu(-2)}} {
		got := make([]float32, n)
		if err := d.Queue().ReadBuffer(c.y.Buffer, 0, got); err != nil {
			t.Fatalf("%s readback: %v", c.name, err)
		}
		if math.Abs(float64(got[0]-c.want)) > 1e-5 || math.Abs(float64(got[n-1]-c.want)) > 1e-5 {
			t.Errorf("the %s submission's output is %v, want %v: its slots were rebound "+
				"underneath it", c.name, got[0], c.want)
		}
	}

	// Once both are done the instances are reused rather than grown.
	f4 := plan.Submit(d.Queue(), bind(x1, y3))
	if err := f4.Wait(); err != nil {
		t.Fatalf("a submission after the two completed: %v", err)
	}
}

// The plan cache tells plans compiled for different MaxInFlight apart.
func TestMaxInFlightIsPartOfThePlanKey(t *testing.T) {
	rt := newRuntime(t)
	cache := tensor.NewPlanCache(rt)
	defer cache.Close()
	record := func(b *tensor.Builder) {
		h := tensor.Input(b, value("x", 16))
		tensor.Output(b, "y", tensor.SiLU(b, h))
	}
	one, err := cache.Compile(record, tensor.CompileOptions{})
	if err != nil {
		t.Fatal(err)
	}
	two, err := cache.Compile(record, tensor.CompileOptions{MaxInFlight: 2})
	if err != nil {
		t.Fatal(err)
	}
	if one == two {
		t.Fatal("a plan compiled for one submission in flight was handed out for two")
	}
	again, err := cache.Compile(record, tensor.CompileOptions{MaxInFlight: 2})
	if err != nil {
		t.Fatal(err)
	}
	if again != two {
		t.Fatal("the same options did not hit")
	}
}
