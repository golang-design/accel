// Copyright 2026 The golang.design Initiative Authors.
// All rights reserved. Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

package pagetable_test

import (
	"strings"
	"testing"

	"golang.design/x/accel/tensor/internal/pagetable"
)

func TestBlockPool(t *testing.T) {
	p, err := pagetable.NewBlockPool(4, 8) // 4 blocks of 8 positions
	if err != nil {
		t.Fatalf("pool: %v", err)
	}
	if p.BlockSize() != 8 || p.Available() != 4 {
		t.Fatalf("a fresh pool reports block %d and %d available", p.BlockSize(), p.Available())
	}

	// Growing takes only what it needs, and rounds up to whole blocks.
	a, err := p.Grow(nil, 9) // two blocks
	if err != nil {
		t.Fatalf("grow: %v", err)
	}
	if len(a) != 2 || p.Available() != 2 {
		t.Fatalf("9 positions took %d blocks and left %d", len(a), p.Available())
	}
	if p.Positions(a) != 16 {
		t.Errorf("two blocks of eight address %d positions", p.Positions(a))
	}

	// Growing again keeps the earlier pages where they are: a decode step's
	// cache must not move under it.
	before := append([]uint32(nil), a...)
	a, err = p.Grow(a, 17) // a third block
	if err != nil {
		t.Fatalf("grow: %v", err)
	}
	if len(a) != 3 {
		t.Fatalf("17 positions took %d blocks", len(a))
	}
	for i := range before {
		if a[i] != before[i] {
			t.Fatalf("page %d moved from %d to %d when the sequence grew",
				i, before[i], a[i])
		}
	}

	// Growing to something the existing pages already hold takes nothing.
	same, err := p.Grow(a, 20)
	if err != nil {
		t.Fatalf("grow: %v", err)
	}
	if len(same) != 3 || p.Available() != 1 {
		t.Fatalf("a grow inside the existing pages took a block")
	}

	// And the pool refuses rather than evicting.
	if _, err := p.Grow(a, 100); err == nil {
		t.Fatal("the pool handed out blocks it does not have")
	} else if !strings.Contains(err.Error(), "does not evict") {
		t.Errorf("the refusal should say why: %v", err)
	}

	if err := p.Free(a); err != nil {
		t.Fatalf("free: %v", err)
	}
	if p.Available() != 4 {
		t.Errorf("after freeing three blocks the pool has %d of 4", p.Available())
	}

	// Freeing twice is refused, because it would hand one block to two
	// sequences -- and the symptom is one conversation reading another's.
	if err := p.Free(a); err == nil {
		t.Error("a block was freed twice")
	}

	if _, err := pagetable.NewBlockPool(0, 8); err == nil {
		t.Error("a pool of no blocks was accepted")
	}
	if _, err := pagetable.NewBlockPool(4, 0); err == nil {
		t.Error("a block of no positions was accepted")
	}
	if _, err := p.Grow(nil, -1); err == nil {
		t.Error("a negative length was accepted")
	}
}
