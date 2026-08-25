// Copyright 2026 The golang.design Initiative Authors.
// All rights reserved. Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

//go:build darwin

package mtl

import "testing"

// BenchmarkFFICost separates the two things a Metal call through this package
// costs: the foreign call itself, and the autorelease pool wrapped around it.
//
// It exists because the first attribution of a slow encode path named the pool
// as the cost, on the evidence that an empty [withPool] takes most of a
// microsecond. That evidence does not support the claim: withPool makes two
// foreign calls of its own, so an empty one measures the foreign call twice and
// says nothing about what a pool costs on top of it. Deciding what to make
// faster needs the two apart, because the fixes are different and only one of
// them touches an object-lifetime rule.
//
//	send-in-pool     one foreign call, pool already held
//	withPool-empty   two foreign calls, no work inside
//	send-with-pool   the whole thing: pool, call, drain
//
// Read it as: if send-in-pool is close to half of withPool-empty, the foreign
// call is the unit of cost and the pool is one more of them. If send-in-pool is
// far smaller, the pool is genuinely the expensive part.
func BenchmarkFFICost(b *testing.B) {
	ds, err := Devices()
	if err != nil || len(ds) == 0 {
		b.Skipf("no Metal device (err=%v)", err)
	}
	d := ds[0]

	b.Run("send-in-pool", func(b *testing.B) {
		withPool(func() {
			b.ResetTimer()
			for range b.N {
				_ = d.id.Send(selRegistryID)
			}
			b.StopTimer()
		})
	})
	b.Run("withPool-empty", func(b *testing.B) {
		for range b.N {
			withPool(func() {})
		}
	})
	b.Run("send-with-pool", func(b *testing.B) {
		for range b.N {
			withPool(func() { _ = d.id.Send(selRegistryID) })
		}
	})
}
