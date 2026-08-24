// Copyright 2026 The golang.design Initiative Authors.
// All rights reserved. Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

package kernelabi_test

import (
	"strings"
	"testing"

	"golang.design/x/accel/kernelabi"
)

type dims struct{ N int32 }

func encodeDims(dst []byte, d dims) error {
	dst[0] = byte(d.N)
	return nil
}

// A uniform whose value does not match its codec is an error naming both
// types, not a panic.
//
// This is the branch that only a wrong caller reaches, and its absence is
// silent: a bare type assertion panics inside a driver, where the caller has
// no way to see which uniform was wrong or what it should have been.
func TestEncodeUniformNamesBothTypes(t *testing.T) {
	dst := make([]byte, 4)
	if err := kernelabi.EncodeUniform(dst, dims{N: 7}, encodeDims); err != nil {
		t.Fatalf("the matching type should encode: %v", err)
	}
	if dst[0] != 7 {
		t.Errorf("the codec wrote %d, want 7", dst[0])
	}

	err := kernelabi.EncodeUniform(dst, "not a dims", encodeDims)
	if err == nil {
		t.Fatal("a string was accepted by a codec that takes a struct")
	}
	for _, want := range []string{"string", "kernelabi_test.dims"} {
		if !strings.Contains(err.Error(), want) {
			t.Errorf("the error should name %q, got %v", want, err)
		}
	}
}
