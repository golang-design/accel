// Copyright 2026 The golang.design Initiative Authors.
// All rights reserved. Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

package accel

import (
	"fmt"
	"sync"

	"golang.design/x/accel/internal/alloc"
	"golang.design/x/accel/internal/driver"
)

// defaultGranularity is the suballocation granularity a pool uses unless told
// otherwise. It is 256 rather than the device's queried alignment on purpose:
// spec 001 section 3.1 makes 256 always sufficient, and taking the queried
// number as the default would make identical code produce different pool
// layouts and different PoolStats on different devices, so a memory
// requirement computed on one machine would be wrong on another.
const defaultGranularity = 256

// implicitBlockBytes is how much the implicit pool behind Device.NewBuffer adds
// when no existing block can serve a request. Blocks are kept large because the
// driver caps how many live allocations a process may hold.
const implicitBlockBytes = 64 << 20

// NewPool allocates a memory pool of the given kind and size, from which buffers
// are suballocated. See specs/001-device-resources.md for why allocation is
// pooled rather than per resource.
//
// It is [Device.NewPoolWith] with the general-purpose policy. A pool never grows:
// a pool is one device allocation, no backend can resize one in place, and moving
// it would invalidate every address already handed out.
func (d *Device) NewPool(desc PoolDescriptor) (*Pool, error) {
	if err := d.state.checkOpen("NewPoolWith"); err != nil {
		return nil, err
	}
	if desc.Bytes <= 0 {
		return nil, fmt.Errorf("accel: NewPool %q: %d bytes is not a positive size", desc.Label, desc.Bytes)
	}
	if lim := d.info.Limits.MaxPoolBytes; desc.Bytes > lim {
		return nil, fmt.Errorf("accel: NewPool %q: %s exceeds this device's MaxPoolBytes of %s",
			desc.Label, humanBytes(desc.Bytes), humanBytes(lim))
	}

	// A kind the device reports absent is an error naming the kind and the
	// device. It is never an Upload pool handed back under a different name: a
	// caller who sized a KV cache against "no staging copy" and silently got
	// staging has a performance mystery of exactly the kind device selection
	// refuses to create.
	if err := d.supportsKind(desc.Kind); err != nil {
		return nil, err
	}

	p, err := d.newPool(desc)
	if err != nil {
		return nil, err
	}
	d.addPool(p)
	return p, nil
}

// supportsKind reports whether this device backs a memory kind, with the error
// spec 001 section 2 specifies for one it does not.
func (d *Device) supportsKind(kind MemoryKind) error {
	if kind == MemoryShared && !d.info.Capabilities.SharedMemoryKind {
		return fmt.Errorf("%w: NewPool(MemoryShared): this device reports no unified memory "+
			"(%v, %q). Use MemoryUpload plus a copy, or check Capabilities.SharedMemoryKind first",
			ErrUnsupported, d.info.Backend, d.info.Name)
	}
	if !d.dev.Supports(driver.MemoryKind(kind)) {
		return fmt.Errorf("%w: NewPool(%v): this device does not provide that memory kind (%v, %q)",
			ErrUnsupported, kind, d.info.Backend, d.info.Name)
	}
	return nil
}

// newPool builds one pool without registering it, so that the implicit pool can
// reuse the construction without appearing in the device's live-child count
// twice.
func (d *Device) newPool(desc PoolDescriptor) (*Pool, error) {
	if n := d.livePools(); n >= d.info.Limits.MaxPools {
		return nil, fmt.Errorf("%w: NewPool %q: this device holds its cap of %d live pools. "+
			"Pooling is mandatory rather than merely efficient because of this number",
			ErrOutOfDeviceMemory, desc.Label, d.info.Limits.MaxPools)
	}

	blk, err := d.dev.Alloc(driver.MemoryKind(desc.Kind), desc.Bytes, desc.Label)
	if err != nil {
		return nil, err
	}

	var a alloc.Allocator
	switch desc.Policy {
	case PoolGeneral:
		a, err = alloc.NewTLSF(desc.Bytes, defaultGranularity)
	case PoolLinear:
		a, err = alloc.NewBump(desc.Bytes, defaultGranularity)
	default:
		blk.Free()
		return nil, fmt.Errorf("accel: NewPool %q: unknown pool policy %d", desc.Label, desc.Policy)
	}
	if err != nil {
		blk.Free()
		return nil, err
	}

	p := &Pool{dev: d, desc: desc, block: blk, alloc: a}
	p.state.init(desc.Label)
	return p, nil
}

func (d *Device) addPool(p *Pool) {
	d.mu.Lock()
	d.pools = append(d.pools, p)
	d.mu.Unlock()
}

func (d *Device) removePool(p *Pool) {
	d.mu.Lock()
	for i, q := range d.pools {
		if q == p {
			d.pools = append(d.pools[:i], d.pools[i+1:]...)
			break
		}
	}
	d.mu.Unlock()
}

// livePools counts every device allocation this device holds, explicit and
// implicit alike, because the driver's cap does not distinguish them.
//
// The implicit count is a counter on the device rather than a walk over the
// block sets. A block set is guarded by its own lock and takes the device lock
// underneath it, so reaching the other way to read its slice would both race
// and invert the lock order.
func (d *Device) livePools() int {
	d.mu.Lock()
	defer d.mu.Unlock()
	return len(d.pools) + d.implicitBlocks
}

// countImplicit adjusts the device's implicit block count.
func (d *Device) countImplicit(delta int) {
	d.mu.Lock()
	d.implicitBlocks += delta
	d.mu.Unlock()
}

// NewBuffer allocates a single buffer from an implicit pool. It is a convenience
// for callers with a handful of buffers; anything allocating at scale should use
// [Device.NewPool].
//
// The implicit pool is the one thing here that grows, and it grows the only way
// a device allocation can: by adding another one. It is a *set* of fixed-size
// blocks, and when none can serve a request the set adds one. Nothing moves and
// no address is invalidated, which is why this is expressible where growing an
// explicit pool is not. See specs/001-device-resources.md section 5.5.
func (d *Device) NewBuffer(desc BufferDescriptor) (*Buffer, error) {
	if err := d.state.checkOpen("NewBuffer"); err != nil {
		return nil, err
	}
	size, err := d.bufferBytes(desc)
	if err != nil {
		return nil, err
	}

	// One block set per memory kind: a buffer wanting Upload memory cannot be
	// served out of a Device block.
	kind := MemoryDevice
	d.mu.Lock()
	if d.implicit == nil {
		d.implicit = make(map[MemoryKind]*blockSet)
	}
	set := d.implicit[kind]
	if set == nil {
		set = &blockSet{kind: kind}
		d.implicit[kind] = set
	}
	d.mu.Unlock()

	return set.alloc(d, desc, size)
}

// blockSet is the implicit pool: a growable set of fixed-size device
// allocations, each one an ordinary Pool underneath.
type blockSet struct {
	mu     sync.Mutex
	kind   MemoryKind
	blocks []*Pool
}

func (s *blockSet) alloc(d *Device, desc BufferDescriptor, size int) (*Buffer, error) {
	s.mu.Lock()
	defer s.mu.Unlock()

	for _, p := range s.blocks {
		if b, err := p.AllocBuffer(desc); err == nil {
			return b, nil
		}
	}

	// No block can serve it, so add one. A request larger than the standard
	// block gets a block of its own, rounded up.
	bytes := implicitBlockBytes
	if size > bytes {
		bytes = size
	}
	p, err := d.newPool(PoolDescriptor{
		Kind:   s.kind,
		Bytes:  bytes,
		Policy: PoolGeneral,
		Label:  fmt.Sprintf("implicit %v block %d", s.kind, len(s.blocks)),
	})
	if err != nil {
		return nil, err
	}
	s.blocks = append(s.blocks, p)
	d.countImplicit(1)
	return p.AllocBuffer(desc)
}

func (s *blockSet) close(d *Device) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	var first error
	for _, p := range s.blocks {
		if err := p.Close(); err != nil && first == nil {
			first = err
		}
	}
	d.countImplicit(-len(s.blocks))
	s.blocks = nil
	return first
}

// Alloc suballocates a buffer from the pool.
func (p *Pool) AllocBuffer(desc BufferDescriptor) (*Buffer, error) {
	if err := p.state.checkOpen("Alloc"); err != nil {
		return nil, err
	}
	if p.desc.Textures {
		return nil, fmt.Errorf("%w: Alloc %q: pool %q holds textures. Texture placement "+
			"alignment is far coarser than any buffer alignment, so a pool is one or the "+
			"other and never both (spec 001 section 4.4)",
			ErrUsage, desc.Label, p.desc.Label)
	}

	size, err := p.dev.bufferBytes(desc)
	if err != nil {
		return nil, err
	}
	align := p.dev.allocAlignment(desc.Usage)

	p.mu.Lock()
	a, err := p.alloc.Alloc(size, align)
	if err != nil {
		s := p.alloc.Stats()
		p.mu.Unlock()
		return nil, &AllocError{
			Label:       desc.Label,
			Pool:        p.desc.Label,
			Kind:        p.desc.Kind,
			Requested:   size,
			Alignment:   align,
			Free:        s.Free,
			LargestFree: s.LargestFree,
			PoolSize:    s.Size,
		}
	}
	b := &Buffer{pool: p, desc: desc, alloc: a, bytes: size}
	b.state.init(desc.Label)
	p.live = append(p.live, b)
	p.mu.Unlock()

	return b, nil
}

// allocAlignment is the strictest alignment any declared usage implies.
//
// This is the second reason usage is declared at creation rather than inferred:
// the allocator needs the number before it places the allocation, and a buffer
// that later turns out to be bound as a uniform cannot be moved, because its
// address is already in descriptors and, on two backends, in a recorded command
// buffer. See specs/001-device-resources.md section 3.1.
func (d *Device) allocAlignment(u BufferUsage) int {
	lim := d.info.Limits
	align := 4 // the dtype floor, always
	if u&BufferStorage != 0 {
		align = max(align, lim.MinStorageBufferOffsetAlignment)
	}
	if u&BufferUniform != 0 {
		align = max(align, lim.MinUniformBufferOffsetAlignment)
	}
	if u&(BufferCopySrc|BufferCopyDst) != 0 {
		align = max(align, lim.MinBufferCopyOffsetAlignment)
	}
	if u&BufferIndirect != 0 {
		align = max(align, 4) // indirect args are u32 triples
	}
	return align
}

// bufferBytes validates a descriptor and returns the buffer's size in bytes.
func (d *Device) bufferBytes(desc BufferDescriptor) (int, error) {
	if desc.Count <= 0 {
		return 0, fmt.Errorf("accel: buffer %q: count %d is not positive", desc.Label, desc.Count)
	}
	elem := desc.DType.Size()
	if elem == 0 {
		return 0, fmt.Errorf("accel: buffer %q: %v is not a dtype", desc.Label, desc.DType)
	}
	size := elem * desc.Count
	if size/elem != desc.Count {
		return 0, fmt.Errorf("accel: buffer %q: %d elements of %v overflow", desc.Label, desc.Count, desc.DType)
	}
	if lim := d.info.Limits.MaxBufferBytes; size > lim {
		return 0, fmt.Errorf("%w: buffer %q: %s exceeds this device's MaxBufferBytes of %s",
			ErrOutOfDeviceMemory, desc.Label, humanBytes(size), humanBytes(lim))
	}
	return size, nil
}

// Reset releases every allocation in a linear pool at once. It rejects general
// pools and a linear pool with resources retained by an in-flight submission.
func (p *Pool) Reset() error {
	if err := p.state.checkOpen("Reset"); err != nil {
		return err
	}
	if p.desc.Policy != PoolLinear {
		return fmt.Errorf("%w: Reset %q: only a linear pool resets as a whole; a general "+
			"pool frees one buffer at a time", ErrUsage, p.desc.Label)
	}

	p.mu.Lock()
	defer p.mu.Unlock()
	// Textures are children exactly as buffers are, and the rule is the same
	// for both: a hold beyond the caller's handle refuses the reset, and a
	// successful reset closes every handle. Handling only the buffers was the
	// version that left a texture open over bytes the next allocation would
	// reuse, and counted it as a live child no Close could get past.
	for _, b := range p.live {
		// At M1 the only hold on a buffer is a queue write waiting for a flush;
		// a submission's retain set joins it at M3 and will need telling apart.
		if n := b.state.holds() - 1; n > 0 {
			return &LifetimeError{Op: "Reset", Resource: p.desc.Label, Reason: reasonPending, InFlight: n}
		}
	}
	for _, t := range p.liveTextures {
		if n := t.state.holds() - 1; n > 0 {
			return &LifetimeError{Op: "Reset", Resource: p.desc.Label, Reason: reasonPending, InFlight: n}
		}
	}
	// After a successful reset every child handle allocated from the pool is
	// closed, so a stale one is reported rather than addressing whatever now
	// occupies that offset.
	for _, b := range p.live {
		b.state.beginClose()
	}
	for _, t := range p.liveTextures {
		t.state.beginClose()
	}
	p.live = nil
	p.liveTextures = nil
	return p.alloc.Reset()
}

// Stats reports the pool's size, how much is in use, and how much is free.
func (p *Pool) Stats() PoolStats {
	p.mu.Lock()
	defer p.mu.Unlock()
	s := p.alloc.Stats()
	return PoolStats{
		Size:        s.Size,
		Used:        s.Used,
		Free:        s.Free,
		LargestFree: s.LargestFree,
		Allocations: s.Allocations,
		Blocks:      s.Blocks,
	}
}

// Kind reports the pool's memory kind.
func (p *Pool) Kind() MemoryKind { return p.desc.Kind }

// Close releases the pool. Buffers suballocated from it must be closed first:
// closing a pool with live buffers reports a *LifetimeError and frees nothing,
// because closing children out from under a caller who still holds them turns a
// bug into a silent success.
func (p *Pool) Close() error {
	p.mu.Lock()
	// Textures are children exactly as buffers are: each has a handle a caller
	// holds, and closing one out from under them turns a leak into a
	// use-after-free. Counting only buffers was the version that let a pool
	// close while a texture was live.
	if n := len(p.live) + len(p.liveTextures); n > 0 {
		p.mu.Unlock()
		return &LifetimeError{Op: "Close", Resource: p.desc.Label, Reason: reasonChildren, Children: n}
	}
	p.mu.Unlock()

	if !p.state.beginClose() {
		return nil
	}
	if p.state.release() {
		p.block.Free()
	}
	p.dev.removePool(p)
	return nil
}

// liveChildren counts the buffers this pool has handed out and not taken back.
func (p *Pool) liveChildren() int {
	p.mu.Lock()
	defer p.mu.Unlock()
	return len(p.live) + len(p.liveTextures)
}

// liveChildren counts the buffers the implicit pool has handed out.
//
// They have handles, so they are live children of the device exactly as a buffer
// from an explicit pool is. Leaving them out would let Device.Close decide it
// can close, mark the handle dead, and only then discover a pool that refuses.
func (s *blockSet) liveChildren() int {
	s.mu.Lock()
	defer s.mu.Unlock()
	n := 0
	for _, p := range s.blocks {
		n += p.liveChildren()
	}
	return n
}

// forget drops a buffer's accounting from its pool once its memory is back.
func (p *Pool) forget(b *Buffer) {
	p.mu.Lock()
	defer p.mu.Unlock()
	for i, q := range p.live {
		if q == b {
			p.live = append(p.live[:i], p.live[i+1:]...)
			break
		}
	}
}
