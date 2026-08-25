// Copyright 2026 The golang.design Initiative Authors.
// All rights reserved. Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

package kernel

import (
	"fmt"
	"math"
)

// DispatchCooperative runs a cooperative kernel over a grid of workgroups.
//
// # What a workgroup's execution looks like
//
// Every invocation of one workgroup is advanced to its next suspension point,
// and only then is the epoch released and the next round begun:
//
//	epoch 0:  inv0 ──▶│  inv1 ──▶│  inv2 ──▶│   all suspended at barrier 0
//	epoch 1:  inv0 ──▶│  inv1 ──▶│  inv2 ──▶│   all suspended at barrier 1
//	epoch 2:  inv0 ──●  inv1 ──●  inv2 ──●      all finished
//
// That is what a barrier means: no invocation passes it until every invocation
// reaches it. Advancing them in sequence rather than concurrently is not a
// simplification, it is what makes the schedule deterministic and therefore
// makes a diagnostic reproducible.
//
// # Workgroups do not coordinate
//
// Each workgroup runs to completion independently, because
// specs/002-compute-model.md section 2.7 does not guarantee forward progress
// between workgroups: a kernel that waited on another workgroup would be
// relying on something no target promises.
func DispatchCooperative(k *Kernel, count ID3, args Args) error {
	return DispatchCooperativeWith(k, count, args, Options{Diagnostics: true})
}

// Options is how a dispatch is run.
type Options struct {
	// SubgroupSize is the emulated lane count. Zero means the workgroup's own
	// invocation count, which is the degenerate case of one subgroup.
	//
	// It is an option rather than a constant so a kernel can be swept across
	// the sizes real hardware has, which is spec 009's criterion: a subgroup
	// reduction that agrees at 32 and disagrees at 4 has a boundary bug, and a
	// boundary bug at v0 becomes a wrong answer on hardware nobody here owns.
	SubgroupSize uint32

	// Diagnostics turns the instrumentation on. It is the CPU backend's
	// developer mode: the checks are what make this backend an oracle rather
	// than an executor, so they are on by default and off only when a caller
	// asks for the speed. See specs/006-backends.md section 5.
	Diagnostics bool

	// ShuffleSeed varies the order invocations are advanced in within an epoch.
	// Zero keeps signature order, which is the default and what every
	// reproducible run wants.
	//
	// A non-zero seed exists to break the habit a deterministic order teaches. A
	// kernel whose result depends on which invocation runs first inside an epoch
	// is wrong on hardware, and with one fixed order it is wrong *consistently*,
	// which reads as correct. Sweeping the seed turns that into a disagreement
	// between two runs on one machine.
	//
	// It permutes only the order within an epoch. Epoch boundaries are what a
	// barrier means, so shuffling across them would not model any real device.
	ShuffleSeed uint64

	// Workers is how many workgroups run at once. Zero picks a size from
	// GOMAXPROCS and the dispatch's own extent; one runs the whole grid in the
	// calling goroutine.
	//
	// It is an option rather than a package-level knob because a test that pins
	// it is testing both strategies rather than the machine it runs on: a
	// parallel-agrees-with-serial assertion that depended on GOMAXPROCS would
	// pass on a laptop and prove nothing in CI.
	//
	// It sizes the pool, it does not license one. A kernel the compiler did not
	// prove order-independent runs on one worker whatever this says, because a
	// knob that turned the oracle off is a knob whose default nobody could
	// trust. See [Kernel.OrderIndependent].
	Workers int
}

// DispatchCooperativeWith runs a cooperative kernel with explicit options.
func DispatchCooperativeWith(k *Kernel, count ID3, args Args, opts Options) error {
	if k == nil {
		return fmt.Errorf("accel: dispatch: no kernel")
	}
	if k.Cooperative == nil {
		return fmt.Errorf("accel: dispatch: kernel %q has no cooperative entry point", k.Name)
	}
	size := k.WorkgroupSize
	if size.X == 0 || size.Y == 0 || size.Z == 0 {
		return fmt.Errorf("accel: dispatch: kernel %q has workgroup extent %d,%d,%d, and an "+
			"axis of zero dispatches nothing", k.Name, size.X, size.Y, size.Z)
	}
	if err := k.Bind(args); err != nil {
		return err
	}

	grid := normalizeCount(count)
	groups := int(grid.X) * int(grid.Y) * int(grid.Z)
	invocations := int(size.X) * int(size.Y) * int(size.Z)
	workers := workerCount(k.OrderIndependent, groups*invocations, opts.Workers)

	// One scheduler per worker, not one per dispatch. Every field of it is
	// state for the workgroup currently being advanced -- frames, ids, the
	// tracker, the shared storage -- and specs/002-compute-model.md section 2.7
	// gives no ordering between workgroups at all, so two workgroups sharing
	// any of it would be two programs writing one scratchpad.
	//
	// Per worker rather than per workgroup because the allocation is what the
	// serial loop already avoided: a dispatch is many workgroups, and one
	// allocation per invocation per workgroup is a cost the flat path does not
	// pay.
	schedulers := make([]*scheduler, workers)
	for i := range schedulers {
		schedulers[i] = newScheduler(k, args, invocations, opts)
	}

	return runGrid(grid, workers, func(w int, group ID3) error {
		return schedulers[w].workgroup(k, group, size, grid, opts)
	})
}

// scheduler is one worker's state for the workgroup it is advancing.
type scheduler struct {
	frames  []Frame
	threads []Thread

	// tracker is one tracker for the whole workgroup, shared by every
	// invocation, because what it checks is what the invocations did to each
	// other. Nil in strict mode, where every call the generated code makes on
	// it is a no-op the compiler removes.
	tracker *SharedTracker

	// args is this worker's own copy. The slices in it are the dispatch's
	// bindings and are shared; Shared is the workgroup's own storage and is
	// replaced per workgroup, which is why the struct is copied rather than
	// pointed at.
	args Args

	// order is the advance order within an epoch, reused across workgroups so
	// that a shuffled dispatch does not allocate one per workgroup.
	order []int
}

func newScheduler(k *Kernel, args Args, invocations int, opts Options) *scheduler {
	s := &scheduler{
		frames:  make([]Frame, invocations),
		threads: make([]Thread, invocations),
		args:    args,
		order:   make([]int, invocations),
	}
	if opts.Diagnostics && len(k.SharedSizes) > 0 {
		s.tracker = NewSharedTracker(k.Name, ID3{}, k.SharedSizes)
	}
	return s
}

// workgroup runs one workgroup to completion.
func (s *scheduler) workgroup(k *Kernel, group, size, count ID3, opts Options) error {
	// Frames are reset rather than reallocated. Their *contents* are dropped,
	// though, and that is not an optimization to reclaim later: a frame carried
	// into the next workgroup would resume an invocation mid-kernel with
	// another workgroup's locals.
	for i := range s.frames {
		s.frames[i] = Frame{}
	}
	// Shared storage is fresh per workgroup, and the generated code allocates
	// it because only that knows each array's element type and extent. It
	// arrives poisoned rather than zeroed: zero is a value a kernel
	// legitimately expects, so a read-before-write would return something
	// plausible and survive every test.
	if k.NewShared != nil {
		s.args.Shared = k.NewShared()
	}
	fill(s.threads, group, size, count, opts.SubgroupSize)
	s.tracker.Reset(group)
	for i := range s.frames {
		s.frames[i].Shared = s.tracker
	}
	if err := runWorkgroup(k, s.args, s.threads, s.frames, s.tracker, s.order,
		opts.ShuffleSeed, opts.Diagnostics); err != nil {
		return err
	}
	// Reported per workgroup rather than accumulated, because the first
	// offending workgroup is the one a reader wants and a dispatch of a
	// thousand would otherwise report a thousand copies of one mistake. Which
	// workgroup is "first" is the lowest numbered one whatever the worker count
	// is: see [runGrid].
	if ds := s.tracker.Diagnostics(); len(ds) > 0 {
		return ds
	}
	return nil
}

// combineSubgroups performs one epoch's subgroup rendezvous.
//
// # Why per subgroup and not per workgroup
//
// A subgroup is the unit these operations are defined over, and a workgroup
// holds several. Combining across the whole workgroup would give the right
// answer only when the subgroup size equals the workgroup size, which is
// exactly the configuration a sweep across sizes 1, 4, 32 and 64 is written to
// distinguish.
//
// # Why the reduction and scan order is fixed
//
// Lane order, ascending, over the *active* lanes: a scan skips an inactive lane
// rather than adding an identity element in its place, which is what makes an
// exclusive scan over active lanes {0, 2, 3} give lane 3 the sum of lanes 0 and
// 2. f32 addition is not associative, so a reduction's
// result depends on the order, and an oracle whose answer moved between runs
// would be an oracle no test could be written against. Real hardware may use a
// different order and produce a different last bit; that is what
// specs/008-numerics.md section 7's budget is for, and it is why a *same
// backend* determinism test is meaningful while a cross-backend exact one is
// not.
//
// # The active set, and the lanes that are not in it
//
// A subgroup's active lanes at one rendezvous are the lanes suspended there.
// Everything else -- a lane of the workgroup's last, partly filled subgroup, a
// lane that finished, a lane somewhere else in the program -- is inactive, and
// specs/002-compute-model.md section 5.2's five rules are about what those
// contribute, which is nothing: not a zero, not an identity element, and not a
// bit in a ballot.
func combineSubgroups(kernel string, threads []Thread, frames []Frame, diag bool) Diagnostics {
	// Group the suspended lanes by subgroup. Ascending lane order within each,
	// which the invocation order already gives: fill visits x fastest and
	// SubgroupLane is LocalIndex modulo the size.
	type group struct {
		op    SubgroupOp
		lanes []int
	}
	groups := map[uint32]*group{}
	var order []uint32

	for i := range frames {
		if frames[i].Done || frames[i].Sub == SubNone {
			continue
		}
		id := threads[i].SubgroupIndex()
		g := groups[id]
		if g == nil {
			g = &group{op: frames[i].Sub}
			groups[id] = g
			order = append(order, id)
		}
		g.lanes = append(g.lanes, i)
	}

	// In discovery order, which is invocation order, so a report or a
	// floating-point sum is the same every run.
	var ds Diagnostics
	for _, id := range order {
		ds = append(ds, combineOne(kernel, threads, frames, groups[id].op, groups[id].lanes, diag)...)
	}
	if len(ds) > 0 {
		ds.sortStable()
	}
	return ds
}

// combineOne applies one operation across one subgroup's suspended lanes.
func combineOne(kernel string, threads []Thread, frames []Frame, op SubgroupOp, lanes []int, diag bool) Diagnostics {
	if op.isLaneRead() {
		return laneRead(kernel, threads, frames, op, lanes, diag)
	}
	switch op {
	case SubAddF32:
		var acc float32
		for n, i := range lanes {
			if n == 0 {
				// The first lane's value rather than zero plus it: a reduction
				// over an active set of one returns that lane's value. The two
				// differ on exactly one input, which is why the rule needs a
				// witness rather than an assertion -- 0 + v is exactly v for
				// every finite v, and 0 + (-0) is +0. See
				// specs/002-compute-model.md section 5.2, rule 5.
				acc = frames[i].SubF32
				continue
			}
			acc += frames[i].SubF32
		}
		broadcastF32(frames, lanes, acc)

	case SubMinF32:
		acc := frames[lanes[0]].SubF32
		for _, i := range lanes[1:] {
			if v := frames[i].SubF32; v < acc {
				acc = v
			}
		}
		broadcastF32(frames, lanes, acc)

	case SubMaxF32:
		acc := frames[lanes[0]].SubF32
		for _, i := range lanes[1:] {
			if v := frames[i].SubF32; v > acc {
				acc = v
			}
		}
		broadcastF32(frames, lanes, acc)

	case SubBroadcastFirstF32:
		broadcastF32(frames, lanes, frames[lanes[0]].SubF32)

	case SubInclusiveAddF32:
		// The first active lane reads back its own value rather than zero plus
		// it, for the reason SubAddF32 does: a scan whose first step is 0 + v
		// turns a negative zero into a positive one.
		acc := frames[lanes[0]].SubF32
		for _, i := range lanes[1:] {
			acc += frames[i].SubF32
			frames[i].SubF32 = acc
		}

	case SubExclusiveAddF32:
		// The lowest active lane sums nothing, and nothing is +0. That is the
		// one row in specs/002-compute-model.md section 5.2 where an identity
		// is the answer, and it does not contradict the rule above: this lane's
		// prefix is empty rather than one element long.
		acc := frames[lanes[0]].SubF32
		frames[lanes[0]].SubF32 = 0
		for _, i := range lanes[1:] {
			v := frames[i].SubF32
			frames[i].SubF32 = acc
			acc += v
		}

	case SubElect:
		// True for exactly one lane, and accel pins *which*: the lowest
		// numbered. Hardware guarantees only "exactly one", so leaving it
		// unpinned would make a correct kernel's output depend on the device.
		for n, i := range lanes {
			frames[i].SubBool = n == 0
		}

	case SubAny:
		any := false
		for _, i := range lanes {
			any = any || frames[i].SubBool
		}
		for _, i := range lanes {
			frames[i].SubBool = any
		}

	case SubAll:
		all := true
		for _, i := range lanes {
			all = all && frames[i].SubBool
		}
		for _, i := range lanes {
			frames[i].SubBool = all
		}

	case SubBallot:
		var m Mask
		for _, i := range lanes {
			if frames[i].SubBool {
				m.set(threads[i].SubgroupLane())
			}
		}
		for _, i := range lanes {
			frames[i].SubMask = m
		}
	}
	return nil
}

// laneRead applies one broadcast or shuffle across one subgroup.
//
// # Why every contribution is read before any result is written
//
// Both travel in the same field. Writing lane 0's result before lane 31 has
// read lane 0's value would make a reversing shuffle return a copy of itself
// for half the subgroup, which is a wrong answer that looks like a plausible
// permutation. So the values are taken first, in one pass.
func laneRead(kernel string, threads []Thread, frames []Frame, op SubgroupOp, lanes []int, diag bool) Diagnostics {
	width := uint32(0)
	if len(lanes) > 0 {
		width = threads[lanes[0]].SubgroupSize()
	}
	values := make(map[uint32]float32, len(lanes))
	for _, i := range lanes {
		values[threads[i].SubgroupLane()] = frames[i].SubF32
	}

	// Both checks below belong to the developer mode: they are what make this
	// backend an oracle rather than an executor, and specs/006-backends.md
	// section 5 says the instrumentation is what a caller turns off when it
	// wants the speed. Two sibling checks in one function that disagreed about
	// which mode they live in would make the same kernel pass or fail on a
	// switch neither of them names.
	var ds Diagnostics
	if diag && op == SubBroadcastF32 {
		ds = append(ds, checkUniformLane(kernel, threads, frames, lanes)...)
	}

	for _, i := range lanes {
		lane, operand := threads[i].SubgroupLane(), frames[i].SubLane
		src, inRange := sourceLane(op, lane, operand, width)
		if inRange {
			if v, active := values[src]; active {
				frames[i].SubF32 = v
				continue
			}
			// In this subgroup and not taking part. The rule this reports is
			// the one section 5.2 says everyone gets wrong: the result is
			// undefined, and a plausible number propagating out of it is a
			// wrong answer nothing else would catch.
			if diag {
				ds = append(ds, Diagnostic{
					Kind: DiagUndefinedLane, Kernel: kernel, Workgroup: threads[i].GroupID(),
					Invocation: threads[i].LocalID(), Element: -1,
					Detail: fmt.Sprintf("lane %d read lane %d of its subgroup through %v, and "+
						"lane %d is not active at that operation: an inactive lane holds no "+
						"value, so the result is undefined rather than zero "+
						"(specs/002-compute-model.md section 5.2 rule 3)",
						lane, src, op, src),
				})
			}
		}
		// Undefined, and loud: a quiet NaN rather than a number, so a kernel
		// that does depend on it produces something nobody mistakes for an
		// answer. A lane index outside the subgroup entirely arrives here
		// without a diagnostic -- see [Thread.SubgroupShuffleF32].
		frames[i].SubF32 = math.Float32frombits(poisonBits)
	}
	return ds
}

// sourceLane is which lane a read addresses, and whether that lane is inside
// the subgroup at all.
//
// Outside covers both ends: a shuffle up from a low lane underflows, and a
// shuffle down from a high one runs past the width. Both are undefined by
// specs/002-compute-model.md section 5.2, and neither is reported, because the
// idiomatic scan has every lane at one end read out of range and discard the
// answer.
func sourceLane(op SubgroupOp, lane, operand, width uint32) (src uint32, inRange bool) {
	switch op {
	case SubBroadcastF32, SubShuffleF32:
		src = operand
	case SubShuffleXorF32:
		src = lane ^ operand
	case SubShuffleUpF32:
		if operand > lane {
			return 0, false
		}
		src = lane - operand
	case SubShuffleDownF32:
		src = lane + operand
		if src < lane {
			return 0, false
		}
	}
	return src, src < width
}

// checkUniformLane reports a broadcast whose lane operand is not the same for
// every active lane.
//
// specs/002-compute-model.md section 5.2 requires it to be dynamically uniform.
// Picking a winner would be the wrong answer to give: on hardware the winner is
// the device's, so a kernel whose output depends on it is already wrong, and
// this backend exists to say so rather than to make one device's answer look
// right.
func checkUniformLane(kernel string, threads []Thread, frames []Frame, lanes []int) Diagnostics {
	var ds Diagnostics
	for _, i := range lanes[1:] {
		if frames[i].SubLane == frames[lanes[0]].SubLane {
			continue
		}
		ds = append(ds, Diagnostic{
			Kind: DiagUndefinedLane, Kernel: kernel, Workgroup: threads[i].GroupID(),
			Invocation: threads[i].LocalID(), Other: threads[lanes[0]].LocalID(),
			HasOther: true, Element: -1,
			Detail: fmt.Sprintf("lane %d asked BroadcastF32 for lane %d while lane %d asked "+
				"for lane %d: the lane a broadcast reads must be dynamically uniform, and "+
				"which of the two a device honours is not defined "+
				"(specs/002-compute-model.md section 5.2)",
				threads[i].SubgroupLane(), frames[i].SubLane,
				threads[lanes[0]].SubgroupLane(), frames[lanes[0]].SubLane),
		})
	}
	return ds
}

func broadcastF32(frames []Frame, lanes []int, v float32) {
	for _, i := range lanes {
		frames[i].SubF32 = v
	}
}

// checkArrival reports invocations that did not all reach the same barrier.
//
// # Two failures, one rule
//
// An invocation that returned while its peers wait, and one that reached a
// different barrier, are the same mistake seen from two sides: the epoch is
// keyed by barrier identity, so arriving at A while a peer waits at B is a
// reported mismatch rather than a silent pairing.
//
// It is a detection rather than a timeout, so it is not flaky and it fires on
// the first offending run. specs/002-compute-model.md section 3.4 says the CPU
// backend catching this is a large part of why it is worth having.
func checkArrival(k *Kernel, threads []Thread, frames []Frame, tracker *SharedTracker) error {
	if tracker == nil {
		return nil
	}
	// An invocation that suspended without saying where is a defect in the
	// generated lowering, and it is reported as one rather than skipped.
	//
	// Skipping is the tempting reading -- there is no id to compare against --
	// and it is absence of evidence read as evidence of absence: with no
	// suspended invocation carrying an id, the whole epoch would pass. A
	// transform that forgets to emit the id would then disable this check
	// silently, which is worse than the mismatch it exists to find.
	var ds Diagnostics
	for i := range frames {
		if !frames[i].Done && frames[i].Barrier.Index < 0 {
			ds = append(ds, Diagnostic{
				Kind: DiagArrival, Kernel: k.Name, Workgroup: threads[i].GroupID(),
				Invocation: threads[i].LocalID(), Element: -1,
				Detail: "it suspended but did not identify which barrier it stopped at, " +
					"so its arrival cannot be checked against its peers': the generated " +
					"lowering is not recording a barrier id",
			})
		}
	}
	if len(ds) > 0 {
		ds.sortStable()
		return ds
	}

	// The expected barrier is the first active invocation's, in invocation
	// order, so the report names the same pair every run.
	expect := BarrierID{Index: -1}
	var expectBy ID3
	found := false
	for i := range frames {
		if frames[i].Done {
			continue
		}
		expect, expectBy, found = frames[i].Barrier, threads[i].LocalID(), true
		break
	}
	if !found {
		return nil
	}

	for i := range frames {
		switch {
		case frames[i].Done:
			// It returned while its peers are waiting. Not a suspicion drawn
			// from a count: this invocation has finished and can never arrive.
			ds = append(ds, Diagnostic{
				Kind: DiagArrival, Kernel: k.Name, Workgroup: threads[i].GroupID(),
				Invocation: threads[i].LocalID(), Other: expectBy, HasOther: true,
				Element: -1,
				Detail: "it returned while its peers wait at " + expect.Describe() +
					", so that barrier can never be reached by every invocation",
			})
		case frames[i].Barrier != expect:
			ds = append(ds, Diagnostic{
				Kind: DiagArrival, Kernel: k.Name, Workgroup: threads[i].GroupID(),
				Invocation: threads[i].LocalID(), Other: expectBy, HasOther: true,
				Element: -1,
				Detail: "it suspended at " + frames[i].Barrier.Describe() +
					" while its peer waits at " + expect.Describe(),
			})
		}
	}
	if len(ds) == 0 {
		return nil
	}
	ds.sortStable()
	return ds
}

// fill computes every invocation's ids for one workgroup, in x-fastest order.
//
// x-fastest is guaranteed by specs/002-compute-model.md section 1.4 rather than
// incidental, because a kernel indexing shared memory by LocalIndex depends on
// which invocation gets which slot.
func fill(threads []Thread, group, size, count ID3, subgroup uint32) {
	if subgroup == 0 {
		// One subgroup spanning the workgroup, which is the degenerate case
		// rather than a special one: every operation still combines, over every
		// lane.
		subgroup = linear(size)
	}
	i := 0
	for lz := range size.Z {
		for ly := range size.Y {
			for lx := range size.X {
				threads[i] = NewThreadWithSubgroup(
					ID3{
						X: group.X*size.X + lx,
						Y: group.Y*size.Y + ly,
						Z: group.Z*size.Z + lz,
					},
					ID3{X: lx, Y: ly, Z: lz}, group, size, count, subgroup,
				)
				i++
			}
		}
	}
}

// runWorkgroup advances every invocation epoch by epoch until all have
// finished.
func runWorkgroup(k *Kernel, args Args, threads []Thread, frames []Frame, tracker *SharedTracker, order []int, shuffleSeed uint64, diag bool) error {
	// The bound is a backstop against a generated program counter that does not
	// advance, and it is deliberately loose.
	//
	// A tight bound is not available: a barrier inside a loop suspends once per
	// iteration, and the trip count is data. Suspensions counts the barriers in
	// the *source*, so a kernel with one barrier in a thousand-round loop needs
	// a thousand epochs and is perfectly correct. Anything derived from the
	// static count would refuse it.
	//
	// So this catches a machine that is stuck rather than one that is slow, and
	// the number is large enough that no data-bounded loop reaches it. A stuck
	// machine spins to the cap and reports, which is fast and terminates; the
	// alternative is a hang, and a hang is what this whole backend exists to
	// turn into a report.
	const maxEpochs = 1 << 20
	bound := maxEpochs
	// The advance order within an epoch. Signature order unless a seed asks for
	// a permutation; see Options.ShuffleSeed. The slice belongs to the worker
	// so that a shuffled dispatch does not allocate one per workgroup.
	for i := range order {
		order[i] = i
	}
	for epoch := 0; epoch < bound; epoch++ {
		if shuffleSeed != 0 {
			shuffleOrder(order, shuffleSeed, uint64(epoch))
		}
		active := 0
		for _, i := range order {
			if frames[i].Done {
				continue
			}
			tracker.Begin(threads[i].LocalID())
			frames[i].Barrier = BarrierID{Index: -1}
			frames[i].Sub = SubNone
			if k.Cooperative(threads[i], args, &frames[i]) {
				active++
				continue
			}
			frames[i].Done = true
		}
		if active == 0 {
			return nil
		}
		// A subgroup rendezvous is combined before the arrival check, because
		// lanes suspended at one are at the same suspension point by
		// construction -- the check below is about barriers.
		if ds := combineSubgroups(k.Name, threads, frames, diag); len(ds) > 0 {
			return ds
		}
		if err := checkArrival(k, threads, frames, tracker); err != nil {
			return err
		}
		// The epoch ends here, which is what a barrier means and what bounds
		// the window conflicting accesses are compared in.
		tracker.Epoch()
	}
	return fmt.Errorf("accel: kernel %q did not finish within %d rendezvous epochs, so "+
		"either its generated program counter is not advancing or a loop in it does not "+
		"terminate", k.Name, bound)
}

// shuffleOrder permutes order in place, deterministically in the seed and the
// epoch.
//
// Deterministic in both, so a failing run reproduces from the seed alone: a
// scheduler that reached for a global random source would report a bug nobody
// could re-run, which is the opposite of what this backend is for.
//
// The generator is a splitmix64 step rather than math/rand, because this package
// is reached from a kernel's execution path and pulling in a source with its own
// locking would put a mutex in the epoch loop.
func shuffleOrder(order []int, seed, epoch uint64) {
	state := seed ^ (epoch * 0x9E3779B97F4A7C15)
	next := func() uint64 {
		state += 0x9E3779B97F4A7C15
		z := state
		z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9
		z = (z ^ (z >> 27)) * 0x94D049BB133111EB
		return z ^ (z >> 31)
	}
	for i := len(order) - 1; i > 0; i-- {
		j := int(next() % uint64(i+1))
		order[i], order[j] = order[j], order[i]
	}
}
