// Copyright 2026 The golang.design Initiative Authors.
// All rights reserved. Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

package intrin

import (
	"go/token"
	"go/types"
	"testing"
)

// TestKeyString covers both spellings a key can take. The receiver-less form is
// unreachable from source today because every intrinsic is a Thread method, and
// it is exercised here rather than left dead: spec 013 adds accel/kmath's
// bounded scalar math as free functions, and a digest that formats those wrongly
// would be discovered by a mismatched digest rather than by a test.
func TestKeyString(t *testing.T) {
	for _, tc := range []struct {
		k    key
		want string
	}{
		{key{"accel/kmath", "", "Sqrt"}, "accel/kmath.Sqrt"},
		{key{kernelPkg, "Thread", "GlobalID"}, kernelPkg + ".Thread.GlobalID"},
	} {
		if got := tc.k.String(); got != tc.want {
			t.Errorf("key.String() = %q, want %q", got, tc.want)
		}
	}
}

// TestReceiverNameRejectsUnnamedReceivers covers the defensive branch that Go
// source cannot reach, since a method declaration always has a named receiver
// type. It is defensive because the front end feeds Lookup whatever go/types
// produced, and a synthesized signature is not a hypothetical: instantiating a
// generic method, which Go 1.27 now permits and the subset rejects, goes through
// the same path.
func TestReceiverNameRejectsUnnamedReceivers(t *testing.T) {
	pkg := types.NewPackage(kernelPkg, "kernel")
	recv := types.NewVar(token.NoPos, pkg, "t", types.NewStruct(nil, nil))
	sig := types.NewSignatureType(recv, nil, nil, nil, nil, false)
	fn := types.NewFunc(token.NoPos, pkg, "GlobalID", sig)

	if _, ok := Lookup(fn); ok {
		t.Error("a method whose receiver has no type name resolved to an intrinsic")
	}
	if got := receiverName(types.NewStruct(nil, nil)); got != "" {
		t.Errorf("receiverName of an unnamed struct = %q, want empty", got)
	}
	if got := receiverName(types.NewPointer(types.NewStruct(nil, nil))); got != "" {
		t.Errorf("receiverName of a pointer to an unnamed struct = %q, want empty", got)
	}
}

// TestLookupWithoutAPackage covers an object with no package, which go/types
// produces for builtins.
func TestLookupWithoutAPackage(t *testing.T) {
	fn := types.NewFunc(token.NoPos, nil, "len", types.NewSignatureType(nil, nil, nil, nil, nil, false))
	if _, ok := Lookup(fn); ok {
		t.Error("a package-less object resolved to an intrinsic")
	}
}
