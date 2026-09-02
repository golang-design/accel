// Copyright 2026 The golang.design Initiative Authors.
// All rights reserved. Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

package accel

import (
	"fmt"
	"sync"

	"golang.design/x/accel/internal/driver"
)

// TransientPool is device memory several graphs plan their intermediates into.
//
//	pool, _ := dev.NewTransientPool("buckets")
//	defer pool.Close()
//
//	r := dev.NewRecorder()
//	r.UseTransientPool(pool)
//	...
//
// # Why you would want one
//
// A set of graphs over one model — a plan per prefill bucket, say — has one set
// of intermediates each, and only one of them ever runs at a time. Five buckets
// at 200 MiB is a gigabyte of device memory of which 800 MiB is idle. Sharing a
// pool makes that 200 MiB.
//
// # What it costs
//
// Graphs sharing a pool cannot execute together, and the second one is
// **refused** rather than queued: its fence carries the error, which is how
// every other submission failure arrives. For a bucket set that costs nothing,
// because a request runs in one bucket. For two graphs you wanted to overlap it
// is the wrong tool, and the error says so — a pool that queued silently would
// turn a design mistake into a latency mystery.
//
// Submitting them one after another is always fine. A claim covers execution
// and not the wait in front of it, so a queue that runs them in turn never
// trips the rule.
//
// It grows to fit the largest graph built into it and never shrinks. Building
// is the only moment it may resize, because a submission holds device addresses
// into it.
//
// See specs/031-shared-transients.md.
type TransientPool struct {
	_ noCopy

	dev   *Device
	label string

	mu sync.Mutex

	// block is the handle every graph's operands hold. It is a wrapper rather
	// than the device allocation itself, so growing the pool swaps what is
	// underneath without invalidating anything above.
	//
	// Without that indirection, growing frees the allocation that graphs built
	// earlier still reference -- their transients, their plan operands, and
	// their compiled executables all captured it at build. The first version
	// did exactly that and the first test caught it: "the block has been
	// freed", from a graph that had been correct until a larger one was built
	// beside it.
	block *poolBlock
	bytes int

	// graphs is how many built graphs reference this pool, so closing it while
	// one could still submit is refused rather than discovered.
	graphs int

	// inFlight is true while a graph sharing this pool is executing. The rule a
	// graph has for itself, widened to the set sharing a pool. It is set and
	// cleared inside Graph.run, so it is true for exactly as long as the bytes
	// are live.
	inFlight bool

	closed bool
}

// NewTransientPool creates a pool this device's graphs can share.
func (d *Device) NewTransientPool(label string) (*TransientPool, error) {
	// A child of the device from here, counted by Device.Close as an explicit
	// pool is, and registered under the same lock Close counts under.
	d.lifecycle.RLock()
	defer d.lifecycle.RUnlock()
	if err := d.state.checkOpen("NewTransientPool"); err != nil {
		return nil, err
	}
	d.mu.Lock()
	d.transientPools++
	d.mu.Unlock()
	return &TransientPool{dev: d, label: label}, nil
}

// Bytes reports what the pool has allocated, which is the largest requirement
// among the graphs built into it.
//
// A different question from [Graph.Memory], which reports what one graph's
// transients need. One is "what did we reserve" and the other is "what does
// this plan cost".
func (p *TransientPool) Bytes() int {
	p.mu.Lock()
	defer p.mu.Unlock()
	return p.bytes
}

// Graphs reports how many built graphs share this pool.
func (p *TransientPool) Graphs() int {
	p.mu.Lock()
	defer p.mu.Unlock()
	return p.graphs
}

// reserve makes the pool at least n bytes, and is called during Build.
//
// Growing is safe only because building is not submitting: a live submission
// holds device addresses into the block, and reallocating under one would be a
// use-after-free whose symptom is a wrong answer rather than a crash. So a
// build while anything is in flight is refused.
func (p *TransientPool) reserve(n int, label string) (driver.Block, error) {
	p.mu.Lock()
	defer p.mu.Unlock()
	if p.closed {
		return nil, fmt.Errorf("accel: Build: the transient pool %q is closed", p.label)
	}
	if p.inFlight {
		return nil, &LifetimeError{
			Op: "Build", Resource: "transient pool", Reason: reasonInFlight,
		}
	}
	if n > p.bytes {
		blk, err := p.dev.dev.Alloc(driver.MemoryDevice, n, "shared transients: "+p.label)
		if err != nil {
			return nil, fmt.Errorf("accel: Build: growing the transient pool %q to %s: %w",
				p.label, humanBytes(n), err)
		}
		p.dev.countImplicit(1)
		if p.block == nil {
			p.block = &poolBlock{}
		}
		// The previous allocation is freed only after the wrapper points at the
		// new one, so nothing can observe a freed block through it. Transients
		// hold no data between submissions -- Build checks that a graph writes
		// one before reading it -- so there is nothing to copy across.
		if old := p.block.swap(blk); old != nil {
			old.Free()
			p.dev.countImplicit(-1)
		}
		p.bytes = n
	}
	p.graphs++
	return p.block, nil
}

// release drops a graph's reference, called when it closes.
func (p *TransientPool) release() {
	p.mu.Lock()
	defer p.mu.Unlock()
	p.graphs--
}

// begin claims the pool for one execution, and is called from [Graph.run].
//
// A claim is refused rather than queued. Queuing would make two graphs the
// caller expected to overlap run one after the other instead, which reads as
// the machine being slow rather than as the design mistake it is.
func (p *TransientPool) begin() error {
	p.mu.Lock()
	defer p.mu.Unlock()
	if p.closed {
		return fmt.Errorf("accel: Submit: the transient pool %q is closed", p.label)
	}
	if p.inFlight {
		return fmt.Errorf("accel: Submit: another graph sharing the transient pool %q is in "+
			"flight; graphs sharing a pool share its memory and so cannot overlap. Wait on "+
			"that submission, or give this graph its own transients", p.label)
	}
	p.inFlight = true
	return nil
}

// end releases the pool after an execution. Deferred immediately after the
// claim succeeds, because a claim leaked on an error path is not recoverable:
// the pool refuses every later submission and nothing can clear it.
func (p *TransientPool) end() {
	p.mu.Lock()
	defer p.mu.Unlock()
	p.inFlight = false
}

// Close frees the pool's memory.
//
// Refused while any graph built into it is open, because those graphs hold
// offsets into memory this would free. Closing a graph built into a pool does
// not free the pool: the pool is the caller's, which is what makes it shareable.
func (p *TransientPool) Close() error {
	p.mu.Lock()
	defer p.mu.Unlock()
	if p.closed {
		return nil
	}
	if p.graphs > 0 {
		return fmt.Errorf("accel: Close: %d graph(s) still share the transient pool %q; "+
			"close them first, because they hold offsets into memory this frees",
			p.graphs, p.label)
	}
	p.closed = true
	if p.block != nil {
		p.block.Free()
		p.dev.countImplicit(-1)
		p.block = nil
	}
	p.bytes = 0
	p.dev.mu.Lock()
	p.dev.transientPools--
	p.dev.mu.Unlock()
	return nil
}

// poolBlock is a driver.Block whose allocation can be replaced.
//
// It exists so a pool can grow after graphs have been built into it. Every
// operand, transient and executable captured at build holds this wrapper, so
// swapping what is inside reaches all of them at once -- and the alternative is
// recomputing every plan's operands, which is most of Build.
type poolBlock struct {
	mu sync.Mutex
	b  driver.Block
}

func (p *poolBlock) swap(next driver.Block) driver.Block {
	p.mu.Lock()
	defer p.mu.Unlock()
	old := p.b
	p.b = next
	return old
}

// Unwrap lets a backend reach the real allocation. See [driver.Unwrap].
func (p *poolBlock) Unwrap() driver.Block { return p.inner() }

func (p *poolBlock) inner() driver.Block {
	p.mu.Lock()
	defer p.mu.Unlock()
	return p.b
}

func (p *poolBlock) Bytes() []byte {
	if b := p.inner(); b != nil {
		return b.Bytes()
	}
	return nil
}

func (p *poolBlock) Size() int {
	if b := p.inner(); b != nil {
		return b.Size()
	}
	return 0
}

func (p *poolBlock) Write(off int, src []byte) error {
	b := p.inner()
	if b == nil {
		return fmt.Errorf("accel: the transient pool has been freed")
	}
	return b.Write(off, src)
}

func (p *poolBlock) Read(off int, dst []byte) error {
	b := p.inner()
	if b == nil {
		return fmt.Errorf("accel: the transient pool has been freed")
	}
	return b.Read(off, dst)
}

// Free releases the current allocation. Called by the pool, not by a graph:
// releaseTransients leaves a shared pool alone.
func (p *poolBlock) Free() {
	if b := p.swap(nil); b != nil {
		b.Free()
	}
}
