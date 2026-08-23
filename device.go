// Copyright 2026 The golang.design Initiative Authors.
// All rights reserved. Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

package accel

import (
	"fmt"

	"golang.design/x/accel/internal/cpu"
	"golang.design/x/accel/internal/driver"
)

// The conversions between the public records and the backend-facing ones are
// whole-struct conversions on purpose. Go permits them only when the field
// lists are identical, so a field added to one side and not the other is a
// build failure rather than a field silently left at its zero value, which is
// the exact failure spec 001 section 1.1 warns about for limits.
func publicLimits(l driver.Limits) Limits           { return Limits(l) }
func publicCaps(c driver.Capabilities) Capabilities { return Capabilities(c) }
func driverLimits(l Limits) driver.Limits           { return driver.Limits(l) }
func driverCaps(c Capabilities) driver.Capabilities { return driver.Capabilities(c) }

// String returns the backend's name.
func (b Backend) String() string {
	switch b {
	case BackendCPU:
		return "CPU"
	case BackendMetal:
		return "Metal"
	case BackendVulkan:
		return "Vulkan"
	case BackendD3D12:
		return "D3D12"
	case BackendOpenGL:
		return "OpenGL"
	}
	return fmt.Sprintf("Backend(%d)", int(b))
}

// adapters returns every backend adapter compiled into this build, in a stable
// order. The CPU backend is always present on every platform.
//
// A backend that is compiled in but cannot probe contributes a diagnostic
// rather than disappearing, so a caller can tell "no Metal device" from "Metal
// was not built".
func adapters() ([]driver.Adapter, []ProbeDiagnostic) {
	all := []driver.Adapter{cpu.Adapter{}}
	native, diags := platformAdapters()
	return append(all, native...), diags
}

func publicInfo(a driver.Adapter) DeviceInfo {
	return infoFrom(a.Token(), a.Info())
}

func infoFrom(token [16]byte, i driver.Info) DeviceInfo {
	return DeviceInfo{
		ID:           AdapterID{token: token},
		Backend:      Backend(i.Backend),
		Name:         i.Name,
		Vendor:       i.Vendor,
		Software:     i.Software,
		Capabilities: publicCaps(i.Capabilities),
		Limits:       publicLimits(i.Limits),
	}
}

// Enumerate reports every openable synchronous adapter and all probe failures.
func Enumerate() Enumeration {
	all, diags := adapters()
	e := Enumeration{Diagnostics: diags}
	for _, a := range all {
		e.Devices = append(e.Devices, publicInfo(a))
	}
	return e
}

// OpenDevice opens exactly the enumerated adapter id names.
//
// It never falls back to another adapter or backend. Use [OpenBest] to ask for
// automatic selection explicitly.
func OpenDevice(id AdapterID) (*Device, error) {
	all, _ := adapters()
	for _, a := range all {
		if a.Token() != id.token {
			continue
		}
		// Opening the enumerated CPU adapter is exactly OpenCPU(CPUOptions{}),
		// per spec 006 section 5: automatic and explicit paths never fabricate
		// non-default CPU options.
		dev, err := a.Open(nil)
		if err != nil {
			return nil, err
		}
		return newDevice(a.Token(), dev), nil
	}
	return nil, fmt.Errorf("accel: no enumerated adapter has that ID; call Enumerate first")
}

// OpenCPU opens the CPU backend with an explicit oracle profile.
func OpenCPU(opts CPUOptions) (*Device, error) {
	o := cpu.Options{
		Mode:             cpu.Mode(opts.Mode),
		SubgroupSize:     opts.SubgroupSize,
		ShuffleSeed:      opts.ShuffleSeed,
		LoseAtSubmission: opts.LoseAtSubmission,
	}
	for _, t := range opts.StrictTargets {
		o.StrictTargets = append(o.StrictTargets, driver.Backend(t))
	}
	if opts.Mimic != nil {
		o.Mimic = &driver.Info{
			Backend:      driver.Backend(opts.Mimic.Info.Backend),
			Name:         opts.Mimic.Info.Name,
			Vendor:       opts.Mimic.Info.Vendor,
			Software:     opts.Mimic.Info.Software,
			Capabilities: driverCaps(opts.Mimic.Info.Capabilities),
			Limits:       driverLimits(opts.Mimic.Info.Limits),
		}
	}

	a := cpu.Adapter{}
	dev, err := a.Open(&o)
	if err != nil {
		return nil, err
	}
	return newDevice(a.Token(), dev), nil
}

// OpenBest opens the best available device under an explicit policy. Unlike
// [OpenDevice] this is a request to choose, and the policy is what it chooses by: it
// fails rather than descending into something the caller did not sanction.
func OpenBest(p Policy) (*Device, error) {
	all, _ := adapters()
	report := SelectionReport{}

	order := p.Prefer
	if len(order) == 0 {
		order = []Backend{BackendMetal, BackendVulkan, BackendD3D12, BackendOpenGL, BackendCPU}
	}

	var chosen driver.Adapter
	for _, want := range order {
		for _, a := range all {
			info := publicInfo(a)
			if info.Backend != want {
				continue
			}
			if reason := rejects(p, info); reason != "" {
				report.Rejected = append(report.Rejected, AdapterRejection{ID: info.ID, Reason: reason})
				continue
			}
			chosen = a
			break
		}
		if chosen != nil {
			break
		}
	}

	if chosen == nil {
		return nil, fmt.Errorf("accel: OpenBest: no adapter satisfied the policy (%d rejected). "+
			"The CPU backend is never selected unless Policy.AllowCPU says so, because it is not a fast path",
			len(report.Rejected))
	}

	dev, err := chosen.Open(nil)
	if err != nil {
		return nil, err
	}
	d := newDevice(chosen.Token(), dev)
	report.Selected = d.info.ID
	d.selection = &report
	return d, nil
}

// rejects reports why a policy excludes an adapter, or "" if it does not.
func rejects(p Policy, info DeviceInfo) string {
	if info.Backend == BackendCPU && !p.AllowCPU {
		return "the CPU backend was not sanctioned: set Policy.AllowCPU"
	}
	if info.Software && !p.AllowSoftware {
		return "a software device was not sanctioned: set Policy.AllowSoftware"
	}
	if missing := p.Require &^ available(info.Capabilities); missing != 0 {
		return fmt.Sprintf("required capabilities %#x are absent", uint32(missing))
	}
	if why := violatesLimits(p.Limits, info.Limits); why != "" {
		return why
	}
	return ""
}

// available maps a device's reported capabilities onto the requirement set a
// kernel is expressed in.
func available(c Capabilities) Capability {
	var got Capability
	if c.Subgroups {
		for _, m := range []struct {
			op  SubgroupOpSet
			cap Capability
		}{
			{SubgroupBasic, CapSubgroupBasic},
			{SubgroupVote, CapSubgroupVote},
			{SubgroupBallot, CapSubgroupBallot},
			{SubgroupShuffle, CapSubgroupShuffle},
			{SubgroupArithmetic, CapSubgroupArithmetic},
		} {
			if c.SubgroupOps&m.op != 0 {
				got |= m.cap
			}
		}
	}
	for _, m := range []struct {
		have bool
		cap  Capability
	}{
		{c.F16Arithmetic, CapF16Arithmetic},
		{c.BF16Arithmetic, CapBF16Arithmetic},
		{c.AtomicFloatAddStorage, CapAtomicFloatAddStorage},
		{c.AtomicFloatAddShared, CapAtomicFloatAddShared},
		{c.I8DotProduct, CapI8DotProduct},
	} {
		if m.have {
			got |= m.cap
		}
	}
	return got
}

// violatesLimits compares an adapter against the policy's numeric constraints.
// Zero fields are unconstrained and array components compare independently.
func violatesLimits(c LimitConstraints, have Limits) string {
	got := LimitValues(have)
	for i, f := range LimitValues(c.AtLeast) {
		if f.Value != 0 && got[i].Value < f.Value {
			return fmt.Sprintf("%s is %d, below the required %d", f.Name, got[i].Value, f.Value)
		}
	}
	for i, f := range LimitValues(c.AtMost) {
		if f.Value != 0 && got[i].Value > f.Value {
			return fmt.Sprintf("%s is %d, above the permitted %d", f.Name, got[i].Value, f.Value)
		}
	}
	return ""
}

// Device is an opened accelerator.
func newDevice(token [16]byte, dev driver.Device) *Device {
	d := &Device{dev: dev}
	d.info = infoFrom(token, dev.Info())
	d.state.init(d.info.Name)
	// At v0 both backends report exactly one queue: Metal has one general queue
	// and the CPU backend has one by construction. Every multi-queue path in this
	// design is specified and unexercised until Vulkan or D3D12 lands, so the
	// topology comes from the backend rather than being assumed here.
	d.queues = []QueueInfo{{Kind: QueueUniversal, Index: 0, Label: "universal"}}
	d.handles = make([]*Queue, len(d.queues))
	for i := range d.queues {
		d.handles[i] = &Queue{dev: d, info: d.queues[i]}
	}
	d.queue = d.handles[0]
	return d
}

// Info reports what this device is and what it can do.
func (d *Device) Info() DeviceInfo { return d.info }

// Limits reports the device's numeric bounds.
func (d *Device) Limits() Limits { return d.info.Limits }

// SelectionReport reports how OpenBest selected this device. The bool is false
// for a device opened explicitly with OpenDevice or OpenCPU.
func (d *Device) SelectionReport() (SelectionReport, bool) {
	if d.selection == nil {
		return SelectionReport{}, false
	}
	return *d.selection, true
}

// Queue returns the device's default queue, which is always [QueueUniversal].
func (d *Device) Queue() *Queue { return d.queue }

// Queues reports every queue this device exposes, in a stable order whose first
// entry is what [Device.Queue] returns.
//
// Queue topology is reported rather than inferred from the platform, because the
// backends disagree completely: Vulkan exposes queue families with capability
// bits, D3D12 has typed command queues, and Metal, GL and the CPU backend have
// exactly one. Ordering between submissions depends on which queue they went to,
// so a caller who cannot enumerate them cannot use that rule.
func (d *Device) Queues() []QueueInfo {
	out := make([]QueueInfo, len(d.queues))
	copy(out, d.queues)
	return out
}

// QueueFor returns a queue able to run kind.
//
// It never fails and never invents a queue: on a device with one universal queue
// it returns that queue, and the caller sees which one they got through
// [Device.Queues]. That is not the silent substitution [OpenDevice] refuses, because
// nothing about the result is weaker than what was asked for, only less parallel.
func (d *Device) QueueFor(kind QueueKind) *Queue {
	// At v0 every backend reports exactly one universal queue, which runs
	// everything. Returning it is not the silent substitution OpenDevice refuses:
	// nothing about the result is weaker than what was asked for, only less
	// parallel, and Queues reports which one the caller got.
	for i, q := range d.queues {
		if q.Kind == kind {
			return d.handles[i]
		}
	}
	return d.queue
}

// Close releases the device. Resources created from it must be closed first.
//
// Closing is ordered rather than recursive: a device with live pools reports a
// *LifetimeError counting them and frees nothing. The API could close children
// on the caller's behalf and deliberately does not, because a caller who closed
// a device out from under a pool they still hold has a bug, and turning that
// bug into a silent success makes the next use of the pool undefined instead of
// reported.
//
// The implicit pool behind [Device.NewBuffer] is the exception, because the
// caller never named it: it has no handle to close, so the device owns it.
func (d *Device) Close() error {
	d.mu.Lock()
	live := len(d.pools) + d.graphs + d.pipelines
	implicit := make([]*blockSet, 0, len(d.implicit))
	for _, set := range d.implicit {
		implicit = append(implicit, set)
	}
	d.mu.Unlock()

	// A buffer from the implicit pool has a handle, so it is a live child exactly
	// as one from an explicit pool is. Counting only d.pools would let Close
	// decide it can proceed, mark the handle dead, and only then discover a pool
	// that refuses, leaving a device that reports closed and never closed.
	for _, set := range implicit {
		live += set.liveChildren()
	}

	// Close does not hide asynchronous work: an unflushed batch is reported as a
	// live child, so orderly teardown is Flush().Wait(), then resource closes,
	// then this. A device that silently dropped queued writes would turn a
	// missing flush into missing data rather than into an error.
	for _, q := range d.handles {
		live += q.pendingCount()
	}

	// The children are counted before the handle is marked dead, so a device that
	// refuses to close stays fully usable: marking first and rolling back would
	// give a concurrent NewPool a spurious closed error, and would let a
	// concurrent second Close report success for a device that never closed.
	if live > 0 {
		return &LifetimeError{Op: "Close", Resource: d.info.Name, Reason: reasonChildren, Children: live}
	}
	if !d.state.beginClose() {
		return nil
	}

	for _, set := range implicit {
		if err := set.close(d); err != nil {
			return err
		}
	}
	d.mu.Lock()
	d.implicit = nil
	d.mu.Unlock()

	return d.dev.Close()
}
