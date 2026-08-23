// Copyright 2026 The golang.design Initiative Authors.
// All rights reserved. Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

//go:build darwin

package metal

import (
	"fmt"

	"golang.design/x/accel/internal/kernel"
	"golang.design/x/accel/internal/mslabi"
	"golang.design/x/accel/internal/mtl"
)

// # Clamping a device-written workgroup count
//
// specs/003-command-graph.md is normative and unusually blunt about this:
//
//	**Every build mode clamps**, not only a debug one. Correctness cannot depend
//	on a flag, and no backend may submit an out-of-range indirect count.
//
// The obvious implementation reads the count back, clamps on the host, and
// encodes an ordinary dispatch. That is correct and it destroys the point: a
// readback is a synchronisation point in the middle of a graph, so an indirect
// dispatch would cost more than the direct one it replaces, and a caller would
// reasonably stop using it.
//
// So the clamp runs on the device. A one-thread kernel reads the count, writes
// the clamped triple into a private buffer, and records what it saw; the real
// dispatch then reads *that* buffer indirectly. The encoder boundary between
// them is the ordering, which is the same barrier every other node in this
// backend uses.
//
// What a caller gives up by not asking for stats is being *told* a clamp
// happened, which costs a readback. The clamp itself is unconditional, and the
// statistics buffer is written either way because a branch to avoid four stores
// would cost more than the stores.
const clampSource = `#include <metal_stdlib>
using namespace metal;
` + mslabi.ContractOff + `

kernel void _accel_clamp(const device uint *count [[buffer(0)]],
                         device uint *clamped [[buffer(1)]],
                         device uint *stats [[buffer(2)]],
                         constant uint *limit [[buffer(3)]]) {
  uint3 c = uint3(count[0], count[1], count[2]);
  uint3 m = uint3(limit[0], limit[1], limit[2]);
  uint3 r = min(c, m);
  clamped[0] = r.x;
  clamped[1] = r.y;
  clamped[2] = r.z;
  stats[0] = c.x;
  stats[1] = c.y;
  stats[2] = c.z;
  stats[3] = (c.x > m.x || c.y > m.y || c.z > m.z) ? 1u : 0u;
}
`

// indirectSlot is the per-node state an indirect dispatch needs.
type indirectSlot struct {
	node int

	// clamped holds the three counts the real dispatch reads. Private, because
	// nothing on the host reads it: the host reads stats.
	clamped *mtl.Buffer

	// stats holds the three counts the device *supplied* and whether they were
	// clamped, so IndirectStats can report what happened. Shared, because that
	// is the readback a caller opted into.
	stats *mtl.Buffer

	max kernel.ID3
}

// clampPipeline compiles the clamp kernel, once per device.
//
// The caller holds d.mu.
func (d *device) clampPipeline() (*mtl.Pipeline, error) {
	const key = "accel:clamp"
	if p, ok := d.pipelines[key]; ok {
		return p, nil
	}
	p, err := d.dev.Compile(clampSource, "_accel_clamp")
	if err != nil {
		return nil, fmt.Errorf("accel: compiling the indirect clamp: %w", err)
	}
	if d.pipelines == nil {
		d.pipelines = map[string]*mtl.Pipeline{}
	}
	d.pipelines[key] = p
	return p, nil
}

// newIndirectSlot allocates one node's clamp buffers.
//
// The caller holds d.mu.
func (d *device) newIndirectSlot(node int, max kernel.ID3) (*indirectSlot, error) {
	if _, err := d.clampPipeline(); err != nil {
		return nil, err
	}
	s := &indirectSlot{node: node, max: max}
	var err error
	// Three uint32 for the counts. Private, so Bytes() is nil and nothing can
	// read it without going through a transfer, which is the same rule every
	// device allocation follows.
	if s.clamped, err = d.dev.NewBuffer(12, mtl.StoragePrivate); err != nil {
		return nil, err
	}
	if s.stats, err = d.dev.NewBuffer(16, mtl.StorageShared); err != nil {
		s.clamped.Close()
		return nil, err
	}
	return s, nil
}

func (s *indirectSlot) close() {
	if s.clamped != nil {
		s.clamped.Close()
	}
	if s.stats != nil {
		s.stats.Close()
	}
	s.clamped, s.stats = nil, nil
}

// encodeClamp runs the clamp for one node, in its own compute pass.
//
// Its own pass on purpose: the real dispatch reads what this wrote, and an
// encoder boundary is what orders one against the other on Metal. Sharing an
// encoder would be a read of memory written by a dispatch nothing ordered
// against it, which is undefined rather than merely fast.
func (e *executable) encodeClamp(p *pass, s *indirectSlot, count resolved) error {
	pipe, err := e.dev.clampPipeline()
	if err != nil {
		return err
	}
	enc := p.compute()
	enc.SetPipeline(pipe)
	enc.SetBuffer(count.buf, count.off, 0)
	enc.SetBuffer(s.clamped, 0, 1)
	enc.SetBuffer(s.stats, 0, 2)
	enc.SetBytes(u32Bytes([]uint32{s.max.X, s.max.Y, s.max.Z}), 3)
	one := mtl.Size{Width: 1, Height: 1, Depth: 1}
	enc.Dispatch(one, one)
	// End it here rather than letting the next node's encoder switch do it: the
	// dependency is on the *next* node specifically, and making that explicit
	// is cheaper to read than inferring it from what happens to follow.
	p.end()
	return nil
}
