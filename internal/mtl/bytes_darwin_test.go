// Copyright 2026 The golang.design Initiative Authors.
// All rights reserved. Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

//go:build darwin

package mtl_test

import (
	"testing"

	"golang.design/x/accel/internal/mtl"
)

// An empty slice bound as stage bytes binds nothing rather than panicking.
//
// SetVertexBytes and SetFragmentBytes took &b[0] with no length check, where
// the compute encoder's SetBytes returns on an empty slice. A stage uniform of
// zero bytes therefore indexed an empty slice inside a render encoder, which is
// a Go panic with an encoder left open -- and Metal aborts on an encoder that
// is released without endEncoding, so the panic could not even be recovered.
func TestEmptyStageBytesBindNothing(t *testing.T) {
	d := open(t)
	tex, err := d.NewRenderTarget(mtl.PixelFormatRGBA32Float, 4, 4)
	if err != nil {
		t.Fatalf("target: %v", err)
	}
	defer tex.Close()
	q := d.NewQueue()
	defer q.Close()

	cb := q.Begin()
	defer cb.Close()
	enc, err := cb.Render([]mtl.RenderAttachment{{
		Texture: tex, Load: mtl.LoadActionClear, Store: mtl.StoreActionDontCare,
	}}, nil)
	if err != nil {
		t.Fatalf("encoder: %v", err)
	}
	enc.SetVertexBytes(nil, 0)
	enc.SetVertexBytes([]byte{}, 1)
	enc.SetFragmentBytes(nil, 0)
	enc.SetFragmentBytes([]byte{}, 1)
	enc.End()
	cb.Commit()
	cb.Wait()
	if err := cb.Err(); err != nil {
		t.Fatalf("an encoder that bound nothing failed: %v", err)
	}
}
