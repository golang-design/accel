// Copyright 2026 The golang.design Initiative Authors.
// All rights reserved. Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

package front

import "testing"

// The generated file's name is spelled in two packages, and they must agree.
//
// front cannot import kernelc — kernelc imports front — so the constant is
// duplicated. A disagreement would be silent and would undo blankGenerated: the
// overlay would miss the real file, the type check would see stale output
// again, and the generator would refuse to run on the one package that needed
// it. This is the check that makes the duplication safe.
func TestGeneratedFileNameMatchesTheWriter(t *testing.T) {
	const written = "accel_kernels.go" // kernelc.GeneratedFile
	if generatedFile != written {
		t.Errorf("front overlays %q and kernelc writes %q; blankGenerated would "+
			"miss the file and the generator would refuse to regenerate stale output",
			generatedFile, written)
	}
}
