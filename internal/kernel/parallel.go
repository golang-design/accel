// Copyright 2026 The golang.design Initiative Authors.
// All rights reserved. Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

package kernel

import (
	"runtime"
	"sync"
	"sync/atomic"
)

// A dispatch's workgroups do not depend on each other, so the grid walk is
// parallel by construction. This file is that walk, shared by both execution
// strategies: the flat path in dispatch.go and the cooperative one in
// schedule.go run the same grid and differ only in what one workgroup costs.
//
// # What "the same result" has to mean here
//
// specs/006-backends.md makes this backend the oracle, so a dispatch that ran
// on eight goroutines must produce the bytes a dispatch on one produced. Three
// things carry that:
//
//   - A kernel whose result can depend on workgroup order runs on one worker.
//     [Kernel.OrderIndependent] is what says it cannot, and it is inferred by
//     the compiler rather than declared. See its doc comment.
//   - Workgroups are numbered x-fastest, the order the serial loop visited
//     them, so "workgroup n" names one workgroup whatever the worker count is.
//   - When several workgroups fail, the reported failure is the lowest
//     numbered one, which is the one the serial loop would have reported.

// parallelThreshold is the invocation count a dispatch needs before a worker
// pool is worth starting.
//
// Measured, not guessed, and measured on the *cheapest* kernel available: an
// elementwise f32 scale, one multiply per invocation, which is
// BenchmarkDispatchScale. The cheapest kernel is the right one to measure on,
// because a pool's cost is fixed and a workgroup's work is not: the size at
// which the pool starts paying for itself is highest when the work per
// invocation is lowest, so a threshold that holds here holds for every heavier
// kernel.
//
// On an eight-core M2 the crossover is between 128 and 256 invocations: at 128
// the pool costs about 2us more than it saves, and at 256 the two are level.
// This is four times the crossover, which is the margin rather than the number:
// below it a dispatch pays nothing at all, and at it the same scale dispatch is
// already about 1.7x faster, so nothing lands in the region where the answer is
// arguable.
//
// In invocations rather than workgroups because a workgroup is not a fixed
// amount of work: 4 workgroups of 1024 invocations is worth a pool and 64
// workgroups of 1 invocation is not.
const parallelThreshold = 1024

// workerCount is how many goroutines one dispatch asks for.
//
// Asks for, because [runGrid] takes no more workers than there are workgroups.
// Clamping in one place rather than two keeps the two from disagreeing about
// what a pool of eight over three workgroups means.
func workerCount(orderIndependent bool, invocations, want int) int {
	// Order dependence is the gate, and it is checked first so that nothing
	// else can get past it. See [Kernel.OrderIndependent].
	if !orderIndependent {
		return 1
	}
	// A caller who names a count has asked about the strategy rather than about
	// the size, which is what a determinism test needs.
	if want > 0 {
		return want
	}
	if invocations < parallelThreshold {
		return 1
	}
	return runtime.GOMAXPROCS(0)
}

// normalizeCount reads an omitted axis as one.
//
// A caller writing ID3{X: n} means one workgroup deep and one tall, which is
// what the serial loop's max(count.Z, 1) already said. It is done once here so
// that the linearization below and the loop agree about how many workgroups
// there are.
func normalizeCount(count ID3) ID3 {
	return ID3{X: max(count.X, 1), Y: max(count.Y, 1), Z: max(count.Z, 1)}
}

// workgroupAt is the workgroup numbered n, x fastest.
//
// x fastest is the order the nested loops visited, and matching it is not a
// detail: a diagnostic names a workgroup, and a report that named a different
// one when the pool grew would be a report a reader could not reproduce.
func workgroupAt(count ID3, n int) ID3 {
	x := uint32(n) % count.X
	n /= int(count.X)
	y := uint32(n) % count.Y
	n /= int(count.Y)
	return ID3{X: x, Y: y, Z: uint32(n)}
}

// chunkSize is how many consecutive workgroups a worker claims at a time.
//
// A claim is an atomic add, and one per workgroup is a measurable share of a
// cheap workgroup's cost. A large claim amortizes it and unbalances the pool
// instead: a causal-masked attention kernel's workgroups get steadily more
// expensive with their index, so a worker holding the last eighth of the grid
// finishes long after the rest.
//
// So: enough chunks that every worker takes many of them, which lets a late
// worker pick up the tail, and enough workgroups per chunk that the add
// disappears into them.
func chunkSize(total, workers int) int {
	return max(1, min(64, total/(workers*64)))
}

// runGrid runs body once per workgroup, on at most workers goroutines.
//
// # What it promises
//
// The error it returns belongs to the lowest-numbered workgroup that produced
// one, which is the workgroup the serial loop would have stopped at. That holds
// even though workgroups finish out of order, and it is why a claim is not
// abandoned half way: a worker checks for a stop between chunks and not inside
// one, so every workgroup numbered below a failure has been claimed and run.
//
// A panic crosses back to the caller's goroutine and is re-raised there as a
// [*Panic] carrying the site and stack it was raised at, which the worker's
// recovery captured while the panicking frames were still on the stack.
// [internal/cpu] turns a panicking kernel into an error through the fence, and
// it recovers on the goroutine that calls this; a panic left in a worker would
// take the process down instead, which is the behaviour that recover exists to
// prevent.
func runGrid(count ID3, workers int, body func(worker int, group ID3) error) error {
	total := int(count.X) * int(count.Y) * int(count.Z)
	// A worker with no workgroup to claim is a goroutine started and joined for
	// nothing, and at one workgroup that is the whole pool.
	if workers > total {
		workers = total
	}
	if workers < 2 {
		for i := range total {
			if err := body(0, workgroupAt(count, i)); err != nil {
				return err
			}
		}
		return nil
	}

	var (
		cursor  atomic.Int64
		stopped atomic.Bool
		mu      sync.Mutex
		wg      sync.WaitGroup

		failedAt = total
		failErr  error
		panicAt  = total
		panicVal *Panic
	)
	chunk := chunkSize(total, workers)

	for w := range workers {
		wg.Add(1)
		go func(w int) {
			defer wg.Done()
			// The workgroup being run, so a panic can be attributed to one and
			// compared against a returned error from elsewhere in the grid.
			cur := total
			defer func() {
				r := recover()
				if r == nil {
					return
				}
				// Captured here, on the worker, because this is the last
				// place the panicking frames exist.
				p := Recovered(r)
				mu.Lock()
				if cur < panicAt {
					panicAt, panicVal = cur, p
				}
				mu.Unlock()
				stopped.Store(true)
			}()
			for {
				if stopped.Load() {
					return
				}
				start := int(cursor.Add(int64(chunk))) - chunk
				if start >= total {
					return
				}
				for i := start; i < min(start+chunk, total); i++ {
					cur = i
					if err := body(w, workgroupAt(count, i)); err != nil {
						mu.Lock()
						if i < failedAt {
							failedAt, failErr = i, err
						}
						mu.Unlock()
						stopped.Store(true)
						break
					}
				}
			}
		}(w)
	}
	wg.Wait()

	// Whichever came first in workgroup order is what the serial loop would
	// have reported, and a panic and an error cannot come from the same one.
	if panicVal != nil && panicAt < failedAt {
		panic(panicVal)
	}
	return failErr
}
