// Copyright 2026 The golang.design Initiative Authors.
// All rights reserved. Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

package tensor

import (
	"errors"
	"fmt"
	"sync"
)

// BlockPool hands out fixed-size pieces of one KV cache so sequences of
// different lengths can share it.
//
//	pool := tensor.NewBlockPool(blocks, positionsPerBlock)
//	pages, err := pool.Grow(nil, 40) // enough blocks for 40 positions
//	...
//	pool.Free(pages)
//
// A sequence's page table is the list of physical blocks it holds, and the
// blocks need not be adjacent — which is the whole point. A cache sized for the
// longest sequence a server will ever see holds a fraction of what it reserved
// when the sequence is short, and that reservation is device memory nothing
// else can use.
//
// # It does not evict
//
// [BlockPool.Grow] fails when the pool is empty rather than choosing a victim.
// Choosing one is a policy question about which sequence matters, and a wrong
// answer silently truncates somebody's context — which is the same reason
// specs/029-plan-cache.md refuses to truncate a prompt. A caller who wants
// eviction implements the policy they can defend and frees the pages
// themselves.
//
// specs/030-paged-kv.md has the addressing.
type BlockPool struct {
	block int

	mu   sync.Mutex
	free []uint32
	held int
}

// NewBlockPool divides a cache of blocks*positions into blocks.
func NewBlockPool(blocks, positionsPerBlock int) (*BlockPool, error) {
	if blocks <= 0 || positionsPerBlock <= 0 {
		return nil, fmt.Errorf("accel/tensor: a pool of %d blocks of %d positions",
			blocks, positionsPerBlock)
	}
	p := &BlockPool{block: positionsPerBlock, free: make([]uint32, blocks)}
	for i := range p.free {
		// Handed out from the end, so the first sequence gets low indices and
		// the order a test sees is stable rather than incidental.
		p.free[i] = uint32(blocks - 1 - i)
	}
	return p, nil
}

// BlockSize is how many positions one block holds.
func (p *BlockPool) BlockSize() int { return p.block }

// Available reports how many blocks are unheld.
func (p *BlockPool) Available() int {
	p.mu.Lock()
	defer p.mu.Unlock()
	return len(p.free)
}

// Grow extends a page table to hold at least n positions.
//
// The existing pages are kept and appended to, so a sequence's earlier
// positions stay where they are — a decode step's cache must not move under it.
// It returns the same slice when nothing more is needed.
func (p *BlockPool) Grow(pages []uint32, n int) ([]uint32, error) {
	if n < 0 {
		return nil, fmt.Errorf("accel/tensor: %d positions", n)
	}
	want := (n + p.block - 1) / p.block
	if want <= len(pages) {
		return pages, nil
	}

	p.mu.Lock()
	defer p.mu.Unlock()
	need := want - len(pages)
	if need > len(p.free) {
		return nil, fmt.Errorf("accel/tensor: %d positions need %d more block(s) and %d "+
			"are free; the pool does not evict, because choosing which sequence to truncate "+
			"is a policy this cannot make for you", n, need, len(p.free))
	}
	out := append(append([]uint32(nil), pages...), p.free[len(p.free)-need:]...)
	p.free = p.free[:len(p.free)-need]
	p.held += need
	return out, nil
}

// Free returns a sequence's blocks to the pool.
//
// Freeing a page twice is refused rather than ignored: it would hand one block
// to two sequences, and the symptom is one sequence reading another's tokens —
// which reads as a model producing text from the wrong conversation.
func (p *BlockPool) Free(pages []uint32) error {
	p.mu.Lock()
	defer p.mu.Unlock()
	held := map[uint32]bool{}
	for _, f := range p.free {
		held[f] = true
	}
	var errs []error
	for _, pg := range pages {
		if held[pg] {
			errs = append(errs, fmt.Errorf("accel/tensor: block %d is already free", pg))
			continue
		}
		held[pg] = true
		p.free = append(p.free, pg)
		p.held--
	}
	return errors.Join(errs...)
}

// Positions reports how many positions a page table addresses.
func (p *BlockPool) Positions(pages []uint32) int { return len(pages) * p.block }
