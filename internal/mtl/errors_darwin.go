// Copyright 2026 The golang.design Initiative Authors.
// All rights reserved. Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

//go:build darwin

package mtl

import "fmt"

func errSize(n int) error {
	return fmt.Errorf("accel/mtl: a buffer of %d bytes is not a buffer", n)
}

func errAlloc(n int) error {
	return fmt.Errorf("accel/mtl: the device refused an allocation of %d bytes", n)
}
