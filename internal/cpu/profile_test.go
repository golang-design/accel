// Copyright 2026 The golang.design Initiative Authors.
// All rights reserved. Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

package cpu

import (
	"math"
	"testing"

	"golang.design/x/accel/internal/driver"
)

// Every byte count a profile reports must fit in an int on every platform accel
// builds for, including the 32-bit ones.
//
// driver.Limits measures bytes in an int, which is 32 bits on 386 and arm. A
// literal 1<<31 there is not a large limit, it is a compile error, and this
// package did not build on those platforms at all until the cross-GOOS job in
// CI found it. That job catches it again; this test catches it on the 64-bit
// machine where someone would write it, which is a shorter feedback loop than a
// push and a red matrix.
func TestProfileByteLimitsFitA32BitInt(t *testing.T) {
	for _, p := range []struct {
		name string
		lim  driver.Limits
	}{
		{"portable floor", portableFloor},
		{"developer", developerLimits},
	} {
		for _, f := range []struct {
			name string
			v    int
		}{
			{"MaxBufferBytes", p.lim.MaxBufferBytes},
			{"MaxPoolBytes", p.lim.MaxPoolBytes},
			{"MaxStorageBufferBindingBytes", p.lim.MaxStorageBufferBindingBytes},
			{"MaxSharedMemoryBytes", p.lim.MaxSharedMemoryBytes},
		} {
			if int64(f.v) > math.MaxInt32 {
				t.Errorf("the %s profile's %s is %d, which does not fit in a 32-bit "+
					"int: this package will not compile for 386 or arm", p.name, f.name, f.v)
			}
			if f.v <= 0 {
				t.Errorf("the %s profile's %s is %d, which is what an overflowed "+
					"constant looks like on a platform where it does compile",
					p.name, f.name, f.v)
			}
		}
	}
}
