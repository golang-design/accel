// Copyright 2026 The golang.design Initiative Authors.
// All rights reserved. Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

package kernel

import "fmt"

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

	invocations := int(size.X) * int(size.Y) * int(size.Z)
	frames := make([]Frame, invocations)
	threads := make([]Thread, invocations)

	// One tracker for the whole workgroup, shared by every invocation, because
	// what it checks is what the invocations did to each other. Nil in strict
	// mode, where every call the generated code makes on it is a no-op the
	// compiler removes.
	var tracker *SharedTracker
	if opts.Diagnostics && len(k.SharedSizes) > 0 {
		tracker = NewSharedTracker(k.Name, ID3{}, k.SharedSizes)
	}

	for gz := range max(count.Z, 1) {
		for gy := range max(count.Y, 1) {
			for gx := range max(count.X, 1) {
				group := ID3{X: gx, Y: gy, Z: gz}
				// Frames are reset rather than reallocated: a dispatch is many
				// workgroups, and one allocation per invocation per workgroup
				// is a cost the flat path does not pay.
				//
				// Their *contents* are dropped, though, and that is not an
				// optimization to reclaim later: a frame carried into the next
				// workgroup would resume an invocation mid-kernel with another
				// workgroup's locals.
				for i := range frames {
					frames[i] = Frame{}
				}
				// Shared storage is fresh per workgroup for the same reason,
				// and the generated code allocates it because only that knows
				// each array's element type and extent. It arrives poisoned
				// rather than zeroed: zero is a value a kernel legitimately
				// expects, so a read-before-write would return something
				// plausible and survive every test.
				if k.NewShared != nil {
					args.Shared = k.NewShared()
				}
				fill(threads, group, size, count, opts.SubgroupSize)
				tracker.Reset(group)
				for i := range frames {
					frames[i].Shared = tracker
				}
				if err := runWorkgroup(k, args, threads, frames, tracker); err != nil {
					return err
				}
				// Reported per workgroup rather than accumulated, because the
				// first offending workgroup is the one a reader wants and a
				// dispatch of a thousand would otherwise report a thousand
				// copies of one mistake.
				if ds := tracker.Diagnostics(); len(ds) > 0 {
					return ds
				}
			}
		}
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
// # Why the reduction order is fixed
//
// Lane order, ascending. f32 addition is not associative, so a reduction's
// result depends on the order, and an oracle whose answer moved between runs
// would be an oracle no test could be written against. Real hardware may use a
// different order and produce a different last bit; that is what
// specs/008-numerics.md section 7's budget is for, and it is why a *same
// backend* determinism test is meaningful while a cross-backend exact one is
// not.
func combineSubgroups(threads []Thread, frames []Frame) {
	// Group the suspended lanes by subgroup. Ascending lane order within each,
	// which the invocation order already gives: fill visits x fastest and
	// SubgroupInvocationID is LocalIndex modulo the size.
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
		id := threads[i].SubgroupID()
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
	for _, id := range order {
		combineOne(threads, frames, groups[id].op, groups[id].lanes)
	}
}

// combineOne applies one operation across one subgroup's suspended lanes.
func combineOne(threads []Thread, frames []Frame, op SubgroupOp, lanes []int) {
	switch op {
	case SubAddF32:
		var acc float32
		for n, i := range lanes {
			if n == 0 {
				// The first lane's value rather than zero plus it: a reduction
				// over an active set of one returns that lane's value, and for
				// non-associative f32 those differ in the last bit. See
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
				m.set(threads[i].SubgroupInvocationID())
			}
		}
		for _, i := range lanes {
			frames[i].SubMask = m
		}
	}
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
func runWorkgroup(k *Kernel, args Args, threads []Thread, frames []Frame, tracker *SharedTracker) error {
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
	for epoch := 0; epoch < bound; epoch++ {
		active := 0
		for i := range threads {
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
		combineSubgroups(threads, frames)
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
