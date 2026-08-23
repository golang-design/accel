// Copyright 2026 The golang.design Initiative Authors.
// All rights reserved. Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

package kernel

import "sync"

// RunAuthored runs an authored cooperative kernel with a real rendezvous.
//
// # Why this exists
//
// specs/004-kernel-authoring.md's fifth testing level compares the generated
// lowering against the authored function, and a cooperative kernel's authored
// form cannot be run one invocation at a time: its invocations rendezvous, so
// running them in sequence is a different program.
//
// The obvious workaround — run every invocation through the whole function once
// per barrier — is **unsound**, and finding out how was worth writing down. It
// works only while every pre-barrier statement is idempotent, and a tree
// reduction is not: it overwrites the shared array it reduces, so a second pass
// reduces its own output. Tests built that way pass by luck on some shapes and
// produce NaN on others, which is the worst of both.
//
// So this runs each invocation in its own goroutine with a cyclic barrier,
// which is what a workgroup actually is. It is not how the *backend* executes a
// kernel — the scheduler's one-at-a-time advance is deterministic and this is
// not — and that is exactly why it belongs in a reference rather than in the
// runtime: the two arrive at the same answer by different means, which is what
// makes the comparison worth making.
//
// It is for testing. A kernel run this way has no diagnostics, no definition
// tracking, and no arrival checking.
func RunAuthored(size, group, count ID3, subgroup uint32, body func(t Thread)) {
	n := int(linear(size))
	if n == 0 {
		return
	}
	b := newCyclicBarrier(n)

	var wg sync.WaitGroup
	wg.Add(n)
	i := 0
	for lz := range max(size.Z, 1) {
		for ly := range max(size.Y, 1) {
			for lx := range max(size.X, 1) {
				t := NewThreadWithSubgroup(
					ID3{
						X: group.X*size.X + lx,
						Y: group.Y*size.Y + ly,
						Z: group.Z*size.Z + lz,
					},
					ID3{X: lx, Y: ly, Z: lz}, group, size, count, subgroup,
				)
				t.rendezvous = b.wait
				go func(t Thread) {
					defer wg.Done()
					// An invocation that returns without reaching a barrier its
					// peers are waiting at would deadlock them, so it retires
					// from the barrier on the way out.
					defer b.leave()
					body(t)
				}(t)
				i++
			}
		}
	}
	wg.Wait()
}

// cyclicBarrier releases every participant once all of them have arrived, and
// then rearms.
//
// Rearms, because a kernel has many barriers and a one-shot would deadlock at
// the second. The generation counter is what makes a fast participant that
// loops round and arrives again wait for the current round rather than
// releasing the next one early.
type cyclicBarrier struct {
	mu      sync.Mutex
	cond    *sync.Cond
	total   int
	waiting int
	round   int
}

func newCyclicBarrier(n int) *cyclicBarrier {
	b := &cyclicBarrier{total: n}
	b.cond = sync.NewCond(&b.mu)
	return b
}

func (b *cyclicBarrier) wait() {
	b.mu.Lock()
	defer b.mu.Unlock()
	if b.total == 0 {
		return
	}
	round := b.round
	b.waiting++
	if b.waiting >= b.total {
		b.waiting = 0
		b.round++
		b.cond.Broadcast()
		return
	}
	for round == b.round {
		b.cond.Wait()
	}
}

// leave retires a participant. Without it, an invocation that finished while
// its peers wait at a barrier would leave them waiting for an arrival that can
// never come — which is the deadlock a real backend reports as a non-uniform
// arrival and a reference has no diagnostics to report at all.
func (b *cyclicBarrier) leave() {
	b.mu.Lock()
	defer b.mu.Unlock()
	b.total--
	if b.total > 0 && b.waiting >= b.total {
		b.waiting = 0
		b.round++
	}
	b.cond.Broadcast()
}
