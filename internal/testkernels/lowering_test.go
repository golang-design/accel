// Copyright 2026 The golang.design Initiative Authors.
// All rights reserved. Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

package testkernels_test

import (
	"sort"
	"testing"

	"golang.design/x/accel/internal/testkernels"
)

// unlowered names every corpus kernel that carries no MSL artifact, and why.
//
// Empty, and that is the point: a kernel added here is a kernel that cannot run
// on a GPU, and writing its name down is how someone decides whether that is
// acceptable rather than discovering it later.
var unlowered = map[string]string{
	"Ballot": "MSL's simd_ballot returns a simd_vote rather than an integer " +
		"(specs/022-msl-target.md section 5), so there is nothing in the " +
		"subset to hold the mask. This is the first kernel-visible capability " +
		"the first backend does not have -- subgroup_ballot -- and " +
		"specs/058-ballot.md section 3 is why it was built anyway: it is the " +
		"case that proves the capability system does anything at all. The " +
		"refusal a caller meets names simd_vote and the capability",
	"AtomicAddF32": "the float atomic is a capability rather than a guarantee " +
		"(accel.CapAtomicFloatAddStorage) and Metal reports it false, so the " +
		"emitter declines to spell an operation the target cannot run. The " +
		"refusal a caller meets is at pipeline creation and names the " +
		"capability, which is checked in metal_darwin_test.go",
}

// Every corpus kernel lowers to MSL, or is listed with a reason.
//
// # Why this exists, and why it is not the guard beside it
//
// The differential has a completeness check in the other direction: a kernel
// that lowers to MSL and appears in no comparison case fails, so a lowering
// cannot go unverified. Nothing checked that a kernel lowers *at all*.
//
// The asymmetry had a cost. `Pack` was the only kernel of fifty-seven with no
// MSL artifact, which made `tensor.Contiguous` CPU-only -- and nothing above the
// emitter said so, because the operator is documented without qualification and
// its graph builds fine on a Metal device. The refusal arrived at plan compile,
// after a consumer had uploaded 1.4 GB of weights (accel issue 19). Every gate
// passed the whole time.
//
// # Why it is not in the darwin file
//
// The check reads a generated record, not a device, so it is portable — and it
// has to be. The differential runs only where there is a Metal device, so a
// darwin-only guard is one the Linux job cannot fail, and the Linux job is the
// one that runs on every push. A kernel silently losing its lowering is exactly
// the kind of regression that should not need a Mac to notice.
func TestEveryKernelLowersToMSLOrSaysWhyNot(t *testing.T) {
	var missing []string
	for _, k := range testkernels.Kernels {
		if k.MSL != "" {
			// Listed as not lowering and lowering anyway: the list is stale,
			// which matters because a stale entry hides the next real one.
			if why, ok := unlowered[k.Name]; ok {
				t.Errorf("%s lowers to MSL and is listed as not lowering (%q); remove "+
					"the entry, or the list stops meaning anything", k.Name, why)
			}
			continue
		}
		if _, ok := unlowered[k.Name]; !ok {
			missing = append(missing, k.Name)
		}
	}
	sort.Strings(missing)
	for _, name := range missing {
		t.Errorf("%s carries no MSL artifact, so every operator that lowers to it is "+
			"CPU-only and nothing above the emitter says so. Either lower it, or add it "+
			"to unlowered with the reason — a kernel that cannot run on a GPU is a "+
			"decision, not an accident", name)
	}
}

// The corpus is not empty, so the check above cannot pass by having nothing to
// look at. Cheap, and it is the failure mode a table-driven guard has.
func TestTheCorpusIsNotEmpty(t *testing.T) {
	if len(testkernels.Kernels) < 50 {
		t.Fatalf("the corpus has %d kernels; the lowering guard would pass on an "+
			"empty one", len(testkernels.Kernels))
	}
}
