// Copyright 2026 The golang.design Initiative Authors.
// All rights reserved. Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

package accel_test

import (
	"fmt"
	"strings"
	"testing"

	"golang.design/x/accel"
)

// Binding a texture to a stage, per specs/032-stage-abi.md section 5.

// A texture slot outside what a stage can hold is a diagnostic, not a panic.
//
// SetVertexBuffer's rule, and it is here for that method's reason rather than
// by analogy: a negative slot skips the grow loop and indexes the slice, which
// takes the caller's process down from inside a recording call. Every other
// method on RenderPass reports through the recorder, and a panic is the one
// diagnostic a caller cannot handle, cannot attribute to a slot, and cannot see
// beside the rest of a build's errors.
func TestATextureSlotOutsideWhatAStageHoldsIsRefused(t *testing.T) {
	for _, c := range []struct {
		name string
		slot int
		says string
	}{
		{"negative", -1, "cannot be negative"},
		{"at the ceiling", 16, "is the ceiling"},
		{"far past the ceiling", 4096, "is the ceiling"},
	} {
		t.Run(c.name, func(t *testing.T) {
			d := openDevice(t)
			target := colourTarget(t, d, "colour", 4, 4)
			src := colourTarget(t, d, "src", 4, 4)

			r := d.NewRecorder()
			p := r.RenderPass(accel.RenderPassDescriptor{
				Color: []accel.ColorAttachment{{View: view(t, target)}},
				Width: 4, Height: 4, Label: "slots",
			})
			// No recover(): a panic here fails the test by taking it down,
			// which is the behaviour this rule prevents.
			p.SetTexture(c.slot, view(t, src))

			_, err := r.Build()
			if err == nil {
				t.Fatalf("texture slot %d was accepted", c.slot)
			}
			if !strings.Contains(err.Error(), c.says) {
				t.Fatalf("the refusal should say %q, got %v", c.says, err)
			}
			// And it names the slot, so a caller with several knows which.
			if !strings.Contains(err.Error(), fmt.Sprint(c.slot)) {
				t.Errorf("the refusal should name slot %d: %v", c.slot, err)
			}
		})
	}
}

// The last slot a stage can hold is accepted.
//
// The accepting half of the ceiling, and it is not decoration: a rule written
// as ">" where it meant ">=" refuses nothing a caller writes, and a rule
// written the other way refuses the highest legal slot. Only a test at the
// boundary tells the two apart, and this project has withdrawn three rules
// that were never tested from the accepting side.
func TestTheHighestTextureSlotIsAccepted(t *testing.T) {
	d := openDevice(t)
	target := colourTarget(t, d, "colour", 4, 4)
	src := colourTarget(t, d, "src", 4, 4)

	r := d.NewRecorder()
	p := r.RenderPass(accel.RenderPassDescriptor{
		Color: []accel.ColorAttachment{{View: view(t, target)}},
		Width: 4, Height: 4, Label: "ceiling",
	})
	p.SetPipeline(solidPipeline(t, d))
	p.SetTexture(15, view(t, src))
	p.Draw(accel.Draw{VertexCount: 3})

	g, err := r.Build()
	if err != nil {
		t.Fatalf("slot 15 is the highest a stage can hold and it was refused: %v", err)
	}
	defer g.Close()
}
