// Copyright 2026 The golang.design Initiative Authors.
// All rights reserved. Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

package tensor

import (
	"crypto/sha256"
	"errors"
	"fmt"
	"sort"
	"sync"
)

// PlanCache keeps compiled plans so a caller can vary shape without paying for
// compilation on every request.
//
//	cache := tensor.NewPlanCache(rt)
//	defer cache.Close()
//
//	plan, err := cache.Compile(func(b *tensor.Builder) { buildModel(b, n) },
//	    tensor.CompileOptions{Label: "prefill"})
//
// The second call with the same model and the same shapes returns the first
// call's plan.
//
// # What it does not do
//
// It never evicts. A plan owns transient device memory, and a cache that freed
// one on its own would free memory a caller might be about to submit against —
// so it grows until [PlanCache.Close]. A caller who needs a bound gives it a
// bounded set of shapes, which is what [Buckets] is for.
//
// It is also not a substitute for holding a plan in a variable. A decode loop
// runs one shape and should keep its plan; this is for the case where the shape
// varies, and a lookup still costs a digest and a map probe.
//
// specs/029-plan-cache.md has the reasoning, and specs/007-tensor-layer.md has
// the requirement its key satisfies.
type PlanCache struct {
	rt *Runtime

	mu     sync.Mutex
	plans  map[Key]*Plan
	closed bool
}

// Key identifies a compiled plan.
//
// A digest over the six things specs/007-tensor-layer.md requires: the DAG's
// structure, every port and scalar, the selected kernels' own digests, the
// device's identity, and the compile options. That spec's last word on the
// subject is the one that matters -- "shape alone is never a sufficient key" --
// because two different models over the same shapes are different plans and
// returning one for the other is a confident wrong answer.
type Key [32]byte

func (k Key) String() string { return fmt.Sprintf("%x", k[:8]) }

// NewPlanCache returns a cache over one runtime.
func NewPlanCache(rt *Runtime) *PlanCache {
	return &PlanCache{rt: rt, plans: map[Key]*Plan{}}
}

// key combines a recorded graph's identity with everything outside it that
// changes what compiling produces.
func (c *PlanCache) key(id Identity, opts CompileOptions) Key {
	h := sha256.New()
	writeString(h, "accel/tensor plan key v1")
	_, _ = h.Write(id[:])

	// The device. A Metal plan handed to a CPU runtime would fail at bind, but
	// one device's plan handed to another of the same backend would not -- it
	// would submit against pipelines that belong to a different device.
	token := c.rt.dev.Info().ID.String()
	writeString(h, token)

	// Every compile option that affects lowering. Label does not, and is
	// excluded deliberately: two plans differing only in what they are called
	// are the same plan, and including it would double the cache for nothing.
	writeString(h, "opts v1")

	var k Key
	copy(k[:], h.Sum(nil))
	return k
}

// Compile returns the plan for a recorded graph, compiling it once.
//
// The graph is recorded by the callback rather than passed in, because the
// point of a hit is not to record it: a caller who built the graph to ask for
// its key would have paid most of what the cache saves.
func (c *PlanCache) Compile(record func(*Builder), opts CompileOptions) (*Plan, error) {
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.closed {
		return nil, errors.New("accel/tensor: the plan cache is closed")
	}

	// Recording is cheap -- it allocates a few slices and touches no device --
	// and it is the only way to learn the identity. What the cache saves is
	// everything after: kernel selection, pipeline creation, graph planning,
	// transient aliasing, and the device allocation behind them.
	b := c.rt.NewBuilder(opts.Label)
	record(b)
	if err := b.Err(); err != nil {
		return nil, err
	}
	k := c.key(b.Identity(), opts)
	if p, ok := c.plans[k]; ok {
		return p, nil
	}

	p, err := b.Compile(c.rt, opts)
	if err != nil {
		return nil, err
	}
	c.plans[k] = p
	return p, nil
}

// Len reports how many plans are held.
func (c *PlanCache) Len() int {
	c.mu.Lock()
	defer c.mu.Unlock()
	return len(c.plans)
}

// Close releases every plan the cache holds.
func (c *PlanCache) Close() error {
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.closed {
		return nil
	}
	c.closed = true
	var errs []error
	for _, p := range c.plans {
		errs = append(errs, p.Close())
	}
	c.plans = nil
	return errors.Join(errs...)
}

// Buckets is a sorted set of sequence lengths a caller compiles plans for.
//
// A prompt runs in the smallest bucket that fits, and the extra positions are
// padding. Padding needs no mask: a causal prefill's query position attends to
// at most its own position, and padding sits after the real tokens, so a real
// position's window never reaches it. What the padded rows compute is discarded.
//
// The trade is arithmetic for compilation. Fewer buckets waste more work per
// request; more buckets hold more plans and more memory. That is a policy, which
// is why it is a caller's list rather than a rule here.
type Buckets []int

// NewBuckets sorts and de-duplicates a bucket set.
func NewBuckets(sizes ...int) (Buckets, error) {
	if len(sizes) == 0 {
		return nil, errors.New("accel/tensor: a bucket set needs at least one size")
	}
	out := append(Buckets(nil), sizes...)
	sort.Ints(out)
	for i, s := range out {
		if s <= 0 {
			return nil, fmt.Errorf("accel/tensor: bucket %d is %d, and a length is positive",
				i, s)
		}
		if i > 0 && out[i] == out[i-1] {
			return nil, fmt.Errorf("accel/tensor: bucket size %d appears twice", s)
		}
	}
	return out, nil
}

// For returns the smallest bucket that holds n tokens.
//
// A prompt longer than the largest bucket is an error rather than a truncation.
// Truncating changes what the model was asked and produces a plausible answer to
// a different question, which is the worst failure a serving path can have.
func (b Buckets) For(n int) (int, error) {
	if n <= 0 {
		return 0, fmt.Errorf("accel/tensor: a prompt of %d tokens", n)
	}
	i := sort.SearchInts(b, n)
	if i == len(b) {
		return 0, fmt.Errorf("accel/tensor: a prompt of %d tokens exceeds the largest "+
			"bucket, %d; add a bucket or split the prompt, because truncating it would "+
			"answer a different question", n, b[len(b)-1])
	}
	return b[i], nil
}
