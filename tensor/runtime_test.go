// Copyright 2026 The golang.design Initiative Authors.
// All rights reserved. Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

package tensor_test

import (
	"sync"
	"testing"

	"golang.design/x/accel/tensor"
)

// One runtime compiles plans from several goroutines at once.
//
// A runtime is shared the way a device is: a server compiles its prefill plan
// while a decode plan compiles beside it, and both reach the pipeline cache and
// the count of open plans. The cache was a bare map and the count a bare int,
// which the race detector reports as soon as two Compiles overlap -- so this
// test is meaningful under -race and a formality without it.
func TestARuntimeCompilesFromSeveralGoroutines(t *testing.T) {
	rt := newRuntime(t)

	const goroutines, rounds = 8, 4
	var wg sync.WaitGroup
	errs := make(chan error, goroutines*rounds)
	for range goroutines {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for range rounds {
				// The same kernel every time, so every goroutine reaches the
				// same pipeline-cache entry rather than its own.
				b := rt.NewBuilder("shared")
				model{}.record(b)
				plan, err := b.Compile(rt, tensor.CompileOptions{Label: "shared"})
				if err != nil {
					errs <- err
					return
				}
				if err := plan.Close(); err != nil {
					errs <- err
					return
				}
			}
		}()
	}
	wg.Wait()
	close(errs)
	for err := range errs {
		t.Fatalf("compile: %v", err)
	}
}
