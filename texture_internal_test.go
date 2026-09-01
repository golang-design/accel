// Copyright 2026 The golang.design Initiative Authors.
// All rights reserved. Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

package accel

import (
	"errors"
	"testing"
)

// Closing a texture something still holds is reported, not a silent leak.
//
// Buffer.Close returns a *LifetimeError and keeps the memory until the last
// hold goes; Texture.Close returned nil and forgot the texture, so the pool
// counted a child no caller could reach. In-package because nothing retains a
// texture through the public API yet: the hold is taken directly, which is
// what a submission's retain set will do.
func TestTextureCloseReportsAHoldRatherThanLeaking(t *testing.T) {
	d, err := OpenCPU(CPUOptions{})
	if err != nil {
		t.Fatal(err)
	}
	defer d.Close()

	desc := TextureDescriptor{
		Format: RGBA8Unorm, Size: Extent{Width: 4, Height: 4},
		Usage: TextureCopySrc, Kind: MemoryShared, Label: "held",
	}
	for _, c := range []struct {
		name string
		make func(t *testing.T) (*Texture, *Pool)
	}{{
		name: "from an explicit pool",
		make: func(t *testing.T) (*Texture, *Pool) {
			p, err := d.NewPool(PoolDescriptor{
				Kind: MemoryShared, Bytes: 1 << 20, Textures: true, Label: "p",
			})
			if err != nil {
				t.Fatal(err)
			}
			tex, err := p.AllocTexture(desc)
			if err != nil {
				t.Fatal(err)
			}
			return tex, p
		},
	}, {
		name: "owning its pool",
		make: func(t *testing.T) (*Texture, *Pool) {
			tex, err := d.NewTexture(desc)
			if err != nil {
				t.Fatal(err)
			}
			return tex, tex.pool
		},
	}} {
		t.Run(c.name, func(t *testing.T) {
			tex, p := c.make(t)
			if !tex.state.retain() {
				t.Fatal("could not hold a live texture")
			}

			err := tex.Close()
			var le *LifetimeError
			if !errors.As(err, &le) {
				t.Fatalf("Close under a hold returned %v, want *LifetimeError", err)
			}
			if le.Reason != reasonPending || le.InFlight != 1 {
				t.Errorf("Close reported %q with %d holds, want %q with 1", le.Reason, le.InFlight, reasonPending)
			}
			// The handle is retired and the memory is not: the holder still
			// has a live texture, and the pool still counts it.
			if _, err := tex.Whole(); err == nil {
				t.Error("the closed handle is still usable")
			}
			if n := p.liveChildren(); n != 1 {
				t.Errorf("the pool counts %d children under a hold, want 1", n)
			}

			// The hold going away is what frees it, and an owned pool goes
			// with it.
			tex.release()
			if n := p.liveChildren(); n != 0 {
				t.Errorf("the pool counts %d children after the last hold, want 0", n)
			}
			if tex.ownsPool {
				if !p.state.isClosed() {
					t.Error("the pool the texture owns was not closed with it")
				}
			} else if err := p.Close(); err != nil {
				t.Errorf("Close after the last hold: %v", err)
			}
		})
	}
}
