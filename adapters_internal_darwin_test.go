// Copyright 2026 The golang.design Initiative Authors.
// All rights reserved. Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

//go:build darwin

package accel

import (
	"errors"
	"fmt"
	"testing"

	"golang.design/x/accel/internal/metal"
)

// A Metal probe failure is classified by what actually went wrong.
//
// Spec 006 section 6.4 requires a caller to be able to tell "this machine has no
// Metal device" from "Metal was not built", and this is the half of that
// distinction the library owns. Neither case can be produced on a machine with a
// working GPU, which is why the classification is a function: written from the
// contract, checked against it, rather than waiting for a machine that fails.
func TestMetalProbeNamesTheStage(t *testing.T) {
	for _, tc := range []struct {
		name string
		err  error
		want ProbeStage
	}{
		{"the framework would not load", errors.New("dlopen: no such file"), ProbeLoadLibrary},
		{"it loaded and found nothing", metal.ErrNoDevice, ProbeEnumerateAdapters},
		{"a wrapped no-device", fmt.Errorf("probing: %w", metal.ErrNoDevice), ProbeEnumerateAdapters},
	} {
		t.Run(tc.name, func(t *testing.T) {
			got := metalProbe(tc.err)
			if got.Stage != tc.want {
				t.Errorf("stage is %v, want %v", got.Stage, tc.want)
			}
			if got.Backend != BackendMetal {
				t.Errorf("backend is %v, want Metal", got.Backend)
			}
			if !errors.Is(got.Err, tc.err) {
				t.Errorf("the diagnostic dropped its cause: %v", got.Err)
			}
		})
	}
}
