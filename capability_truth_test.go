// Copyright 2026 The golang.design Initiative Authors.
// All rights reserved. Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

package accel_test

import (
	"errors"
	"os"
	"testing"

	"golang.design/x/accel"
	"golang.design/x/accel/internal/kernels"
)

// A reported capability agrees with what the device actually accepts.
//
// specs/006-backends.md decision 6 says absence is reported rather than
// discovered. The inverse matters just as much and had no test: Metal ran
// render passes and presented to a layer for a whole day while reporting
// Graphics false and Presentation false, because nothing compared the claim
// against the behaviour. A caller branching on Capabilities would have skipped
// the path that worked.
//
// Written over every enumerated device rather than a named one, so a backend
// added later is covered without being listed here.
func TestCapabilitiesAgreeWithWhatTheDeviceAccepts(t *testing.T) {
	for _, info := range accel.Enumerate().Devices {
		t.Run(info.Name, func(t *testing.T) {
			d, err := accel.OpenDevice(info.ID)
			if err != nil {
				// specs/006-backends.md section 7: under the Tier 2 promise a
				// device that enumerates and will not open is a failure, and a
				// skip here would let the capability claim rot unchecked.
				if os.Getenv("ACCEL_REQUIRE_METAL") != "" {
					t.Fatalf("this job promises Metal and cannot open %s: %v", info.Name, err)
				}
				t.Skipf("cannot open %s: %v", info.Name, err)
			}
			defer d.Close()
			caps := d.Capabilities()

			t.Run("graphics", func(t *testing.T) {
				p, err := d.NewRenderPipeline(accel.RenderPipelineDescriptor{
					Vertex:   &kernels.HalfTriangleVSStage,
					Fragment: &kernels.SolidFSStage,
					Targets:  []accel.ColorTargetState{{Format: accel.RGBA32Float}},
					Label:    "capability probe",
				})
				if err == nil {
					defer p.Close()
				}
				accepted := err == nil
				if accepted != caps.Graphics {
					t.Errorf("Capabilities.Graphics is %v and NewRenderPipeline %s; a "+
						"caller branching on the capability takes the wrong path either "+
						"way (err=%v)", caps.Graphics, acceptedWord(accepted), err)
				}
			})

			t.Run("presentation", func(t *testing.T) {
				// A handle this device cannot use, so the answer is about the
				// path existing rather than about the window: a device with no
				// present path reports ErrNoPresent, and one with a path gets
				// far enough to reject the handle instead.
				_, err := d.NewWindowSurface(accel.NativeHandle{
					Kind: accel.NativeMetalLayer,
				}, accel.SurfaceDescriptor{Width: 8, Height: 8})
				hasPath := !errors.Is(err, accel.ErrNoPresent)
				if hasPath != caps.Presentation {
					t.Errorf("Capabilities.Presentation is %v and the device %s an "+
						"on-screen path (err=%v)", caps.Presentation,
						acceptedWord(hasPath), err)
				}
			})
		})
	}
}

func acceptedWord(b bool) string {
	if b {
		return "has"
	}
	return "has not"
}
