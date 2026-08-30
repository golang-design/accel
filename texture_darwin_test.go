// Copyright 2026 The golang.design Initiative Authors.
// All rights reserved. Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

package accel_test

// The texture round trip moved into specs/062-backend-parity.md's matrix, and
// it moved because one format is not the surface.
//
// TestATextureRoundTripKeepsCallerOrderOnMetal compared RGBA8Unorm between the
// two backends and nothing said which of the other twelve formats had a copy
// case -- the answer was none. It is now parity_texcopy_test.go's
// formatCopyParityCases: the same round trip over every host-copyable format,
// at an aligned width and at one that forces a repack, with the same
// row-identifiable pattern and the same flip diagnosis, held to the whole
// enumeration by section 6.8's gate.
//
// What the old entry was written for is kept rather than dropped.
// docs/conventions.md records that reading a render target back yields
// bottom-origin rows on GL and on Metal -- on Metal *despite* its top-left
// texture origin, so reasoning from the documented origin gives the wrong
// answer -- while accel guarantees caller order and the backend flips. Two
// backends that flipped identically would agree perfectly, so every case
// asserts the pattern before the matrix compares the two, and a misplaced byte
// still names the row it came from.
//
// This file holds no tests. It is a signpost, because the entry it describes
// carried the reasoning for a documented divergence and deleting it would have
// deleted that with it.
