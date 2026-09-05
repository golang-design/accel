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
// # Why it takes the kernel rather than a workgroup extent
//
// The extent used to be an argument, and the caller wrote a literal beside a
// comparison against `k`'s generated form. Two numbers that must agree
// eventually disagree, and here the disagreement is silent: a differential that
// runs the authored function at one width and its lowering at another compares
// two different programs and reports them equal whenever the shape happens not
// to matter.
//
// It matters more since specs/052-dispatch-shape.md. `Thread.WorkgroupSize()`
// lowers to a **compile-time literal** on Metal, taken from the kernel's
// `accel:kernel` directive, and to `t.groupSize` in the authored form. Those
// two are the same number only because this function reads the declaration
// rather than being told one.
//
// It is for testing. A kernel run this way has no diagnostics, no definition
// tracking, and no arrival checking.
func RunAuthored(k *Kernel, group, count ID3, subgroup uint32, body func(t Thread)) {
	size := k.WorkgroupSize
	n := int(linear(size))
	if n == 0 {
		return
	}
	b := newCyclicBarrier(n)

	// One rendezvous per subgroup, beside the workgroup's, because that is what
	// a subgroup barrier is: it releases its own lanes and no others
	// (specs/002-compute-model.md §5.3).
	//
	// **No test distinguishes this from reusing the workgroup's barrier**, and
	// that was measured rather than assumed: routing SubgroupBarrier back to
	// the workgroup rendezvous leaves every test green. The reason is `leave`
	// below — a lane that finishes retires from the barrier, so subgroups
	// running different numbers of barriers cascade instead of deadlocking, and
	// no corpus kernel's *value* depends on which lanes were released together.
	//
	// It is still the right model, and the wrong one is a reference that
	// happens to agree. A witness would need a kernel whose result depends on
	// the release grouping, which for same-subgroup data is exactly what the
	// barrier orders and for cross-subgroup data is a race — so there may be no
	// legal witness, and that is worth knowing rather than papering over.
	//
	// Zero means one subgroup spanning the workgroup, which is what
	// [Thread.SubgroupSize] reports for it, so the arithmetic below needs no
	// special case.
	lanes := subgroup
	if lanes == 0 {
		lanes = uint32(n)
	}
	subs := make([]*cyclicBarrier, (uint32(n)+lanes-1)/lanes)
	reduces := make([]*authoredSubgroup, len(subs))
	for i := range subs {
		size := int(lanes)
		if rest := n - i*int(lanes); rest < size {
			size = rest
		}
		subs[i] = newCyclicBarrier(size)
		reduces[i] = &authoredSubgroup{
			vals: make([]float32, size), active: make([]bool, size), wait: subs[i].wait,
		}
	}

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
				sub := subs[uint32(i)/lanes]
				t.subRendezvous = sub.wait
				t.subReduce = reduces[uint32(i)/lanes]
				go func(t Thread) {
					defer wg.Done()
					// An invocation that returns without reaching a barrier its
					// peers are waiting at would deadlock them, so it retires
					// from both barriers on the way out.
					defer b.leave()
					defer sub.leave()
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

// authoredSubgroup combines one subgroup's lanes for the f32 reductions when
// an authored kernel runs under [RunAuthored].
//
// An authored subgroup operation called outside the runner combines nothing
// (see subgroup.go). Under the runner it has a real rendezvous: each lane
// writes its value, the subgroup's lanes wait, every lane folds the values in
// lane order -- the scheduler's order, seeded from the first active lane, so
// the authored form and the lowering agree bit for bit -- and the lanes wait
// again so the slots can be reused. A lane that returned early never marks
// itself active and so is not part of the fold, which is the scheduler's
// active set.
type authoredSubgroup struct {
	vals   []float32
	active []bool
	wait   func()
}

func (a *authoredSubgroup) reduce(lane uint32, v float32, fold func(acc, x float32) float32) float32 {
	a.vals[lane] = v
	a.active[lane] = true
	a.wait()
	seeded := false
	var acc float32
	for i, on := range a.active {
		if !on {
			continue
		}
		if !seeded {
			acc, seeded = a.vals[i], true
			continue
		}
		acc = fold(acc, a.vals[i])
	}
	a.wait()
	a.active[lane] = false
	return acc
}
