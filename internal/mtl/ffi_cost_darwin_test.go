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
//	send-reflected   one message send through purego's reflect path
//	send-direct      the same send through [send], pool already held
//	withPool-empty   a pool push and pop, no work inside
//	send-with-pool   the whole thing: pool, reflected call, drain
//
// The first two are also the before and after of the change that made this
// package call objc_msgSend without reflection, which is why both spellings
// are kept reachable rather than one deleted.
//
// Read it as: the foreign call is the unit of cost, and a pool is two more of
// them. Measured on an M2 after the change, 180ns direct against 667ns
// reflected, and an empty pool 389ns for its two.
func BenchmarkFFICost(b *testing.B) {
	ds, err := Devices()
	if err != nil || len(ds) == 0 {
		b.Skipf("no Metal device (err=%v)", err)
	}
	d := ds[0]

	b.Run("send-reflected", func(b *testing.B) {
		withPool(func() {
			b.ResetTimer()
			for range b.N {
				_ = d.id.Send(selRegistryID)
			}
			b.StopTimer()
		})
	})
	b.Run("send-direct", func(b *testing.B) {
		withPool(func() {
			b.ResetTimer()
			for range b.N {
				_ = send(d.id, selRegistryID, 0, 0, 0)
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
