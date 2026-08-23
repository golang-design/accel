// Copyright 2026 The golang.design Initiative Authors.
// All rights reserved. Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

package kernel

import "fmt"

// Subgroup operations, emulated.
//
// # Why a subgroup operation is a rendezvous here
//
// It needs every lane's contribution at the point of the call, and the
// workgroup scheduler advances one invocation at a time: when lane 0 reaches
// `t.SubgroupAddF32(v)`, lane 1 has not computed its v yet. So the generated
// lowering suspends at a subgroup operation exactly as it does at a barrier,
// the scheduler combines the contributions between epochs, and each lane reads
// its result on resume.
//
// That costs an epoch per operation, which is the right trade for an oracle:
// the CPU backend exists to be believed rather than to be fast, and a design
// that ran lanes concurrently to avoid the epoch would reintroduce the
// scheduling non-determinism the whole backend is built to remove.
//
// # Why only in uniform control flow at v0
//
// specs/002-compute-model.md section 5.3 says subgroup operations do not
// *require* uniform control flow, and section 5.1 says their result is portable
// only in it — because whether lanes reconverge after a divergent region is
// implementation-defined, and inactive lanes contribute nothing rather than an
// identity. Emulating the divergent case faithfully means modelling an active
// set that no two backends agree on.
//
// So v0 admits them where the answer is portable and rejects them elsewhere,
// with the position and the reason. The rejection is a v0 boundary rather than
// a permanent rule, and specs/020-cooperative-atomics.md says so.

// SubgroupOp identifies which rendezvous a lane is suspended at.
type SubgroupOp uint8

const (
	// SubNone means the suspension is an ordinary barrier.
	SubNone SubgroupOp = iota

	SubAddF32
	SubMinF32
	SubMaxF32
	SubBroadcastFirstF32
	SubElect
	SubAny
	SubAll
	SubBallot
)

func (o SubgroupOp) String() string {
	switch o {
	case SubAddF32:
		return "SubgroupAddF32"
	case SubMinF32:
		return "SubgroupMinF32"
	case SubMaxF32:
		return "SubgroupMaxF32"
	case SubBroadcastFirstF32:
		return "BroadcastFirstF32"
	case SubElect:
		return "Elect"
	case SubAny:
		return "Any"
	case SubAll:
		return "All"
	case SubBallot:
		return "Ballot"
	}
	return "barrier"
}

// Mask is a subgroup ballot: one bit per lane.
//
// An opaque value with methods rather than a uint64, because the dtype set has
// no 64-bit integer and Vulkan's ballot is 128 bits wide -- a uint64 would
// foreclose a real device. See specs/002-compute-model.md section 5.2.
type Mask struct {
	// bits is two words, which covers the 128-lane ballot Vulkan reports. A
	// device with wider subgroups than that does not exist.
	bits [2]uint64
}

// Count is how many lanes set their predicate.
func (m Mask) Count() int {
	n := 0
	for _, w := range m.bits {
		n += popcount(w)
	}
	return n
}

// Bit reports one lane's predicate. An inactive lane's bit is zero, so
// Ballot(true).Count() is the *active* count -- usually what was wanted, and
// occasionally a bug.
func (m Mask) Bit(lane uint32) bool {
	if lane >= 128 {
		return false
	}
	return m.bits[lane/64]&(1<<(lane%64)) != 0
}

// LowestSet is the lowest lane whose bit is set, or the mask's width when none
// is.
func (m Mask) LowestSet() uint32 {
	for w, word := range m.bits {
		if word == 0 {
			continue
		}
		for b := range uint32(64) {
			if word&(1<<b) != 0 {
				return uint32(w)*64 + b
			}
		}
	}
	return 128
}

// CountLower is how many set bits are below lane, which is what an exclusive
// scan over a ballot needs.
func (m Mask) CountLower(lane uint32) int {
	n := 0
	for l := range lane {
		if m.Bit(l) {
			n++
		}
	}
	return n
}

// Any reports whether any lane set its predicate.
func (m Mask) Any() bool { return m.bits[0] != 0 || m.bits[1] != 0 }

func (m Mask) String() string { return fmt.Sprintf("mask{%016x %016x}", m.bits[1], m.bits[0]) }

func (m *Mask) set(lane uint32) {
	if lane < 128 {
		m.bits[lane/64] |= 1 << (lane % 64)
	}
}

func popcount(w uint64) int {
	n := 0
	for w != 0 {
		w &= w - 1
		n++
	}
	return n
}

// SubgroupSize is how many lanes a subgroup has on this device.
func (t Thread) SubgroupSize() uint32 { return t.subgroupSize }

// SubgroupID is which subgroup of the workgroup this invocation is in.
//
// The mapping is LocalIndex/size, which is what most hardware does and what
// this oracle uses. It is **not promised by every backend**: which invocation
// lands in which subgroup is implementation-defined, and a kernel needing a
// specific lane-to-data mapping must build it from LocalIndex itself. See
// specs/002-compute-model.md section 5.1.
func (t Thread) SubgroupID() uint32 {
	if t.subgroupSize == 0 {
		return 0
	}
	return t.LocalIndex() / t.subgroupSize
}

// SubgroupInvocationID is this invocation's lane within its subgroup, under the
// same implementation-defined mapping as [Thread.SubgroupID].
func (t Thread) SubgroupInvocationID() uint32 {
	if t.subgroupSize == 0 {
		return 0
	}
	return t.LocalIndex() % t.subgroupSize
}

// The authored subgroup operations.
//
// Their bodies do nothing, exactly as [Thread.Barrier]'s does and for the same
// reason: what runs is the generated lowering, where each is a suspension and a
// resume. The authored function is type-checking input, and these exist so that
// input names something and can be run as a reference under an emulated
// rendezvous.
//
// Calling one directly does not combine anything across lanes. That is why
// every place this repository runs an authored cooperative kernel emulates the
// rendezvous explicitly.

// SubgroupAddF32 sums v across the subgroup's active lanes and gives every lane
// the total.
func (t Thread) SubgroupAddF32(v float32) float32 { return v }

// SubgroupMinF32 and SubgroupMaxF32 are the same over the minimum and maximum.
func (t Thread) SubgroupMinF32(v float32) float32 { return v }
func (t Thread) SubgroupMaxF32(v float32) float32 { return v }

// BroadcastFirstF32 gives every active lane the lowest-numbered active lane's
// v. It is the safe broadcast: unlike a broadcast from a chosen lane, it is
// always defined, because the lane it reads is active by construction.
func (t Thread) BroadcastFirstF32(v float32) float32 { return v }

// Elect is true for exactly one lane, and accel pins which: the lowest
// numbered. Hardware guarantees only "exactly one", so leaving it unpinned
// would make a correct kernel's output depend on the device.
func (t Thread) Elect() bool { return true }

// Any and All reduce a predicate over the active lanes.
func (t Thread) Any(pred bool) bool { return pred }
func (t Thread) All(pred bool) bool { return pred }

// Ballot reports each lane's predicate as a bit. An inactive lane's bit is
// zero, so Ballot(true).Count() is the *active* count.
func (t Thread) Ballot(pred bool) Mask {
	var m Mask
	if pred {
		m.set(t.SubgroupInvocationID())
	}
	return m
}
