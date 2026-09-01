// Copyright 2026 The golang.design Initiative Authors.
// All rights reserved. Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

//go:build darwin

package metal_test

import (
	"fmt"
	"sync"
	"testing"

	"golang.design/x/accel/internal/driver"
	"golang.design/x/accel/internal/kernel"
)

// A Compile on one goroutine and a Submit on another share the pipeline cache
// safely.
//
// Compile wrote the cache under the device mutex and a submission's dispatch
// read it under the executable's mutex only, so the two were a concurrent map
// write and read -- which the runtime reports as a fatal error rather than a
// wrong answer, and only when the timing lines up. Run under -race this test
// reports the race deterministically; without it, the map's own check fires
// often enough to matter.
func TestCompileAndSubmitShareThePipelineCache(t *testing.T) {
	d := open(t)
	c := d.(driver.GraphCompiler)

	b := mustAlloc(t, d, 4096)
	op, err := driver.BlockOperand(b, 0, 4096)
	if err != nil {
		t.Fatalf("operand: %v", err)
	}
	plan := func(k *kernel.Kernel, nodes int) *driver.Plan {
		p := &driver.Plan{}
		for i := range nodes {
			p.Nodes = append(p.Nodes, driver.PlanNode{
				Op: driver.OpDispatch, ID: i, Dispatch: &driver.Dispatch{
					Kernel: k, Count: kernel.ID3{X: 1, Y: 1, Z: 1},
					Bindings: []driver.Operand{op},
				},
			})
		}
		return p
	}
	// On the same device as the compiles, which is the whole point: the cache
	// is the device's, so an executable from another device shares nothing.
	//
	// Many nodes, so that each submission spends a while inside encode
	// reading the cache. Submit consults the device under its mutex just
	// before encoding, so a short encode would rarely overlap a Compile that
	// holds the same mutex for the length of a compile, and the race would
	// exist and go unreported.
	e, err := c.Compile(plan(synthetic("steady", fmt.Sprintf(oneBinding, "steady"),
		kernel.ID3{X: 1, Y: 1, Z: 1}), 512))
	if err != nil {
		t.Fatalf("compile: %v", err)
	}
	defer e.Close()

	var wg sync.WaitGroup
	wg.Add(2)
	go func() {
		defer wg.Done()
		// Distinct digests, so each Compile writes a new cache entry rather
		// than finding one.
		for i := range 30 {
			name := fmt.Sprintf("race%d", i)
			k := synthetic(name, fmt.Sprintf(oneBinding, name), kernel.ID3{X: 1, Y: 1, Z: 1})
			ex, err := c.Compile(plan(k, 1))
			if err != nil {
				t.Errorf("compile %s: %v", name, err)
				return
			}
			_ = ex.Close()
		}
	}()
	go func() {
		defer wg.Done()
		for range 200 {
			f, err := e.Submit()
			if err != nil {
				t.Errorf("submit: %v", err)
				return
			}
			if err := f.Wait(); err != nil {
				t.Errorf("wait: %v", err)
				return
			}
		}
	}()
	wg.Wait()
}
