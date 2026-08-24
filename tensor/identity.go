// Copyright 2026 The golang.design Initiative Authors.
// All rights reserved. Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

package tensor

import (
	"crypto/sha256"
	bin "encoding/binary"
	"fmt"
	"io"
)

// Identity is a digest over everything about a recorded graph that changes what
// compiling it produces.
//
// It exists so a [PlanCache] can tell one model from another, and
// specs/007-tensor-layer.md is explicit that a shape is not enough: two
// different models over the same shapes are different plans, and returning one
// for the other produces a confident wrong answer.
//
// What it covers: every operator in record order, the kernel each selected and
// that kernel's own digest, every operand's shape, dtype and layout, and every
// port and scalar's name and kind. What it deliberately does not cover: the
// *values* a caller binds, which are what a plan is reused across.
func (b *Builder) Identity() Identity {
	h := sha256.New()
	// Versioned, so a change to what this digest covers cannot collide with a
	// key computed by an older build. Without it, adding a component to the
	// digest would leave old keys valid for plans that no longer match them.
	//
	// v2 added the window a value occupies within its port. v1 covered the port
	// name and the view's offset, and a layer view carries neither -- its
	// offset is the *binding's* -- so two plans differing only in which layer
	// of a cache they address hashed alike, and a cache would have answered one
	// with the other.
	//
	// v3 added the attributes an operator records rather than reads from a
	// scalar. Those reach the kernel through a closure the digest cannot see,
	// so a top-k of 40 and a top-k of 5, an RMSNorm at two epsilons, a RoPE at
	// two rotary widths and a paged attention at two block sizes were each one
	// key -- and [PlanCache.Compile] discards the graph it just recorded on a
	// hit, so the second caller ran the first caller's configuration.
	writeString(h, "accel/tensor identity v3")

	for _, d := range b.ports {
		writeString(h, "port")
		writeString(h, d.Name)
		writeUint(h, uint64(d.Kind))
		writeUint(h, uint64(d.DType))
		writeShape(h, d.Shape)
	}
	for _, s := range b.scalars {
		writeString(h, "scalar")
		writeString(h, s.Name)
		writeUint(h, uint64(s.Kind))
	}
	for i := range b.nodes {
		n := &b.nodes[i]
		writeString(h, "node")
		writeString(h, n.op)
		if n.kernel != nil {
			writeString(h, n.kernel.Name)
			// The kernel's own digest, which is the component a naive cache
			// omits and the one whose absence survives longest: a plan
			// compiled before go generate ran and reused after would run a
			// lowering whose source no longer exists.
			writeString(h, n.kernel.Digest)
		}
		writeUint(h, uint64(len(n.inputs)))
		for _, in := range n.inputs {
			writeTensor(h, in)
		}
		writeTensor(h, n.out)
		writeUint(h, boolBit(n.inPlace)|boolBit(n.bcast)<<1)
		if n.outPort != "" {
			writeString(h, n.outPort)
			// Which range of that port, because a write to layer 3 and a write
			// to layer 0 are the same operator over the same shapes.
			writeUint(h, uint64(n.outOff))
		}
		for _, r := range n.reads {
			writeString(h, r)
		}
		// The recorded attributes. Length-prefixed like everything else, so an
		// operator with two of them cannot hash like the next node's first.
		writeUint(h, uint64(len(n.attrs)))
		for _, a := range n.attrs {
			writeUint(h, a)
		}
	}
	for _, o := range b.outputs {
		writeString(h, "output")
		writeString(h, o.name)
		writeTensor(h, o.t)
	}

	var id Identity
	copy(id[:], h.Sum(nil))
	return id
}

// Identity is the digest itself.
type Identity [32]byte

func (id Identity) String() string { return fmt.Sprintf("%x", id[:8]) }

// The writers below are length-prefixed rather than concatenated, so that two
// different structures cannot hash to the same bytes. Writing "ab" then "c" and
// writing "a" then "bc" would otherwise be indistinguishable, which is the
// classic way a digest over a structure stops distinguishing structures.
func writeString(h io.Writer, s string) {
	writeUint(h, uint64(len(s)))
	_, _ = io.WriteString(h, s)
}

func writeUint(h io.Writer, v uint64) {
	var b [8]byte
	bin.LittleEndian.PutUint64(b[:], v)
	_, _ = h.Write(b[:])
}

func writeShape(h io.Writer, s Shape) {
	writeUint(h, uint64(len(s)))
	for _, d := range s {
		writeUint(h, uint64(d))
	}
}

// writeTensor covers a value's layout as well as its extent, because a view and
// a copy of the same shape lower differently.
func writeTensor(h io.Writer, t *Tensor) {
	if t == nil {
		writeString(h, "nil")
		return
	}
	writeUint(h, uint64(t.dtype))
	writeShape(h, t.shape)
	writeUint(h, uint64(len(t.strides)))
	for _, s := range t.strides {
		writeUint(h, uint64(s))
	}
	writeUint(h, uint64(t.offset))
	// The producing node, so two operands of the same shape from different
	// producers do not look alike.
	writeUint(h, uint64(int64(t.node)+1))
	writeString(h, t.port)
	// The window within that port. A layer view is contiguous from its own
	// element zero and carries no view offset, so without this a read of layer
	// 3 is indistinguishable from a read of layer 0.
	if t.win == nil {
		writeString(h, "whole")
	} else {
		writeString(h, "window")
		writeUint(h, uint64(t.win.off))
		writeUint(h, uint64(t.win.count))
	}
}

func boolBit(b bool) uint64 {
	if b {
		return 1
	}
	return 0
}
