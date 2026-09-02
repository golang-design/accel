// Copyright 2026 The golang.design Initiative Authors.
// All rights reserved. Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

package accel_test

import (
	"go/ast"
	"go/parser"
	"go/token"
	"math/bits"
	"path/filepath"
	"strings"
	"testing"

	"golang.design/x/accel"
)

func openCPU(t *testing.T, opts accel.CPUOptions) *accel.Device {
	t.Helper()
	d, err := accel.OpenCPU(opts)
	if err != nil {
		t.Fatalf("OpenCPU(%+v): %v", opts, err)
	}
	t.Cleanup(func() {
		if err := d.Close(); err != nil {
			t.Errorf("Close: %v", err)
		}
	})
	return d
}

// TestEnumerateReportsCPU is spec 006's rule that the CPU backend is always
// available on every platform, seen from the enumeration side.
func TestEnumerateReportsCPU(t *testing.T) {
	e := accel.Enumerate()
	if len(e.Devices) == 0 {
		t.Fatal("Enumerate reported no devices; the CPU backend is mandatory on every platform")
	}
	var cpu *accel.DeviceInfo
	for i := range e.Devices {
		if e.Devices[i].Backend == accel.BackendCPU {
			cpu = &e.Devices[i]
		}
	}
	if cpu == nil {
		t.Fatal("Enumerate reported no CPU adapter")
	}
	if cpu.Name == "" || cpu.Vendor == "" {
		t.Errorf("CPU adapter has an empty identity: %+v", *cpu)
	}
	if cpu.Software {
		t.Error("the CPU backend is not a software GPU and must not report Software")
	}
}

// TestAdapterIDIsStable checks the AdapterID contract: comparable and stable
// across repeated enumerations while the adapter is present.
func TestAdapterIDIsStable(t *testing.T) {
	first, second := accel.Enumerate(), accel.Enumerate()
	if len(first.Devices) != len(second.Devices) {
		t.Fatalf("enumeration is unstable: %d then %d devices", len(first.Devices), len(second.Devices))
	}
	for i := range first.Devices {
		if first.Devices[i].ID != second.Devices[i].ID {
			t.Errorf("device %d: AdapterID changed between enumerations", i)
		}
	}
}

// TestOpenDeviceIsCPUDefault is spec 006 section 5: OpenDevice on the enumerated
// CPU adapter is equivalent to OpenCPU(CPUOptions{}), and automatic selection
// never fabricates non-default CPU options.
func TestOpenDeviceIsCPUDefault(t *testing.T) {
	var id accel.AdapterID
	for _, info := range accel.Enumerate().Devices {
		if info.Backend == accel.BackendCPU {
			id = info.ID
		}
	}
	byID, err := accel.OpenDevice(id)
	if err != nil {
		t.Fatalf("OpenDevice: %v", err)
	}
	defer byID.Close()

	byCPU := openCPU(t, accel.CPUOptions{})
	if byID.Info().Name != byCPU.Info().Name {
		t.Errorf("OpenDevice(cpu) = %q, OpenCPU{} = %q; they must be equivalent",
			byID.Info().Name, byCPU.Info().Name)
	}
	if byID.Limits() != byCPU.Limits() {
		t.Error("OpenDevice(cpu) and OpenCPU{} report different limits")
	}
	if byID.Info().Capabilities != byCPU.Info().Capabilities {
		t.Error("OpenDevice(cpu) and OpenCPU{} report different capabilities")
	}
	if _, ok := byID.SelectionReport(); ok {
		t.Error("a device opened explicitly must not carry a selection report")
	}
}

// TestOpenDeviceRejectsUnknownID checks that opening never falls back.
func TestOpenDeviceRejectsUnknownID(t *testing.T) {
	if _, err := accel.OpenDevice(accel.AdapterID{}); err == nil {
		t.Fatal("OpenDevice with an unknown ID returned a device; it must never fall back")
	}
}

// TestOpenBestNeverReturnsAnUnsanctionedBackend is spec 001 section 11.7 and
// spec 006's selection rule: the CPU backend is not selected unless the policy
// names it, and no other backend is substituted for the one asked for.
//
// What the default policy does depends on what this machine has, so the test
// asks rather than assuming. It used to assert that OpenBest(Policy{}) fails,
// on the premise that only the CPU backend was compiled in; that premise stopped
// being true the moment Metal landed, and the rule it was checking did not
// change. The rule is the two clauses below, which hold either way.
func TestOpenBestNeverReturnsAnUnsanctionedBackend(t *testing.T) {
	gpu := false
	for _, d := range accel.Enumerate().Devices {
		if d.Backend != accel.BackendCPU {
			gpu = true
		}
	}

	d, err := accel.OpenBest(accel.Policy{})
	switch {
	case gpu && err != nil:
		t.Fatalf("this machine enumerates a GPU adapter, so the default policy should "+
			"select it: %v", err)
	case gpu:
		got := d.Info().Backend
		d.Close()
		if got == accel.BackendCPU {
			t.Fatal("OpenBest(Policy{}) selected the CPU backend without AllowCPU")
		}
	case err == nil:
		d.Close()
		t.Fatal("OpenBest(Policy{}) returned a device on a machine with only the CPU " +
			"backend; the CPU backend requires AllowCPU")
	case !strings.Contains(err.Error(), "AllowCPU"):
		t.Errorf("OpenBest error does not say how to sanction the CPU backend: %v", err)
	}

	// Asking for one backend never yields another, whichever way round this
	// machine is. This is the clause that has nothing to do with what is
	// installed.
	if d, err := accel.OpenBest(accel.Policy{Prefer: []accel.Backend{accel.BackendVulkan}}); err == nil {
		got := d.Info().Backend
		d.Close()
		t.Fatalf("OpenBest asked for Vulkan and returned %v", got)
	}

	d, err = accel.OpenBest(accel.Policy{AllowCPU: true, Prefer: []accel.Backend{accel.BackendCPU}})
	if err != nil {
		t.Fatalf("OpenBest(AllowCPU, Prefer CPU): %v", err)
	}
	defer d.Close()
	if got := d.Info().Backend; got != accel.BackendCPU {
		t.Errorf("OpenBest asked for the CPU backend and selected %v", got)
	}
	report, ok := d.SelectionReport()
	if !ok {
		t.Fatal("a device opened by OpenBest must carry a selection report")
	}
	if report.Selected != d.Info().ID {
		t.Error("the selection report names a different adapter than the one opened")
	}
}

// TestOpenBestRejectsOnRequirements checks that Policy.Require and
// Policy.Limits filter, and that the rejection says why.
func TestOpenBestRejectsOnRequirements(t *testing.T) {
	// The CPU backend in developer mode has every emulatable capability, so ask
	// for a limit it cannot meet instead.
	huge := accel.Limits{MaxSharedMemoryBytes: 1 << 40}
	_, err := accel.OpenBest(accel.Policy{
		AllowCPU: true,
		Limits:   accel.LimitConstraints{AtLeast: huge},
	})
	if err == nil {
		t.Fatal("OpenBest accepted a device below the required limit")
	}

	_, err = accel.OpenBest(accel.Policy{
		AllowCPU: true,
		Limits:   accel.LimitConstraints{AtMost: accel.Limits{MaxSharedMemoryBytes: 1}},
	})
	if err == nil {
		t.Fatal("OpenBest accepted a device above the permitted limit")
	}

	// A capability no device reports rejects every candidate, and the rejection
	// says which requirement was unmet rather than only that none qualified.
	_, err = accel.OpenBest(accel.Policy{AllowCPU: true, Require: accel.Capability(1 << 20)})
	if err == nil {
		t.Fatal("OpenBest accepted a device lacking a required capability")
	}

	// The CPU backend in developer mode reports every emulatable capability, so
	// requiring them all must still select it. This is the positive half: a
	// filter that rejects everything is not evidence the filter works.
	d, err := accel.OpenBest(accel.Policy{
		AllowCPU: true,
		Require: accel.CapSubgroupBasic | accel.CapSubgroupVote | accel.CapSubgroupBallot |
			accel.CapSubgroupShuffle | accel.CapSubgroupArithmetic | accel.CapF16Arithmetic |
			accel.CapBF16Arithmetic | accel.CapAtomicFloatAddStorage |
			accel.CapAtomicFloatAddShared | accel.CapI8DotProduct,
	})
	if err != nil {
		t.Fatalf("OpenBest rejected the CPU backend for capabilities it emulates: %v", err)
	}
	d.Close()

	// A strict profile reports no subgroups, so the same requirement set must be
	// unmet there. Requirements are compared against Capabilities, not against
	// what the backend could do.
	strict := openCPU(t, accel.CPUOptions{
		Mode:          accel.CPUStrict,
		StrictTargets: []accel.Backend{accel.BackendVulkan},
	})
	if strict.Info().Capabilities.SubgroupOps != 0 {
		t.Error("a strict profile without subgroups still reports subgroup operations")
	}
}

// TestLimitsArePopulated is spec 001 section 11.2's cheapest catch for a backend
// that forgot a field: every alignment is a positive power of two, and no limit
// on an opened device is zero.
func TestLimitsArePopulated(t *testing.T) {
	for _, info := range accel.Enumerate().Devices {
		t.Run(info.Backend.String(), func(t *testing.T) {
			d, err := accel.OpenDevice(info.ID)
			if err != nil {
				t.Fatalf("OpenDevice: %v", err)
			}
			defer d.Close()

			for _, v := range accel.LimitValues(d.Limits()) {
				if v.Value <= 0 {
					t.Errorf("%s is %d; an opened device has no zero-valued limit "+
						"(spec 001 section 1.1)", v.Name, v.Value)
				}
				if strings.Contains(v.Name, "Alignment") && bits.OnesCount(uint(v.Value)) != 1 {
					t.Errorf("%s is %d, which is not a positive power of two", v.Name, v.Value)
				}
			}
		})
	}
}

// TestLimitsMatchTheAdapter checks that what enumeration promised is what the
// opened device reports, so a caller really can choose before opening.
func TestLimitsMatchTheAdapter(t *testing.T) {
	for _, info := range accel.Enumerate().Devices {
		d, err := accel.OpenDevice(info.ID)
		if err != nil {
			t.Fatalf("OpenDevice: %v", err)
		}
		if d.Limits() != info.Limits {
			t.Errorf("%v: opened limits differ from the enumerated ones", info.Backend)
		}
		if d.Info().Capabilities != info.Capabilities {
			t.Errorf("%v: opened capabilities differ from the enumerated ones", info.Backend)
		}
		d.Close()
	}
}

// TestQueueTopologyIsReported is spec 001 section 1: at v0 every backend reports
// exactly one universal queue, and QueueFor never invents one.
func TestQueueTopologyIsReported(t *testing.T) {
	d := openCPU(t, accel.CPUOptions{})

	qs := d.Queues()
	if len(qs) != 1 {
		t.Fatalf("the CPU backend reported %d queues, want exactly 1", len(qs))
	}
	if qs[0].Kind != accel.QueueUniversal {
		t.Errorf("the first queue is %v, and Device.Queue always returns a universal queue", qs[0].Kind)
	}
	if qs[0].Label == "" {
		t.Error("a queue carries the backend's own name for it, for logs")
	}

	if d.Queue() != d.QueueFor(accel.QueueUniversal) {
		t.Error("QueueFor(QueueUniversal) must return the default queue")
	}
	for _, kind := range []accel.QueueKind{accel.QueueCompute, accel.QueueTransfer} {
		if d.QueueFor(kind) != d.Queue() {
			t.Errorf("QueueFor(%v) invented a queue; on a one-queue device it returns that queue", kind)
		}
	}

	// Queues returns a copy: mutating it must not reach the device.
	qs[0].Label = "clobbered"
	if d.Queues()[0].Label == "clobbered" {
		t.Error("Queues handed out its own slice")
	}
}

// TestStrictTargetSetValidation is spec 006 section 5's rules for the strict
// target list: non-empty, no duplicates, no CPU, no unpublished baseline, and
// order-independent.
func TestStrictTargetSetValidation(t *testing.T) {
	tests := []struct {
		name    string
		targets []accel.Backend
		want    string
	}{
		{"empty", nil, "non-empty"},
		{"cpu", []accel.Backend{accel.BackendCPU}, "own portability target"},
		{"duplicate", []accel.Backend{accel.BackendVulkan, accel.BackendVulkan}, "twice"},
		{"unpublished", []accel.Backend{accel.Backend(99)}, "no published baseline"},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			_, err := accel.OpenCPU(accel.CPUOptions{Mode: accel.CPUStrict, StrictTargets: tc.targets})
			if err == nil {
				t.Fatalf("strict mode accepted %v", tc.targets)
			}
			if !strings.Contains(err.Error(), tc.want) {
				t.Errorf("error %q does not explain %q", err, tc.want)
			}
		})
	}

	forward := openCPU(t, accel.CPUOptions{
		Mode:          accel.CPUStrict,
		StrictTargets: []accel.Backend{accel.BackendMetal, accel.BackendVulkan},
	})
	reverse := openCPU(t, accel.CPUOptions{
		Mode:          accel.CPUStrict,
		StrictTargets: []accel.Backend{accel.BackendVulkan, accel.BackendMetal},
	})
	if forward.Info() != reverse.Info() {
		t.Error("the strict target order changed the reported profile; it is normalized")
	}
}

// TestStrictModeIntersects checks that strict mode reports less than developer
// mode, and that adding a target never adds a capability.
func TestStrictModeIntersects(t *testing.T) {
	dev := openCPU(t, accel.CPUOptions{})
	if !dev.Info().Capabilities.Subgroups {
		t.Fatal("developer mode reports no subgroups; the CPU backend emulates them")
	}

	metal := openCPU(t, accel.CPUOptions{
		Mode:          accel.CPUStrict,
		StrictTargets: []accel.Backend{accel.BackendMetal},
	})
	if metal.Info().Capabilities.Subgroups {
		t.Error("Metal's subgroup support is `cap` in spec 006's matrix, so a conservative " +
			"baseline must report it absent")
	}
	if !metal.Info().Capabilities.F16Arithmetic {
		t.Error("Metal's f16 arithmetic is `yes` in spec 006's matrix and must survive the baseline")
	}

	both := openCPU(t, accel.CPUOptions{
		Mode:          accel.CPUStrict,
		StrictTargets: []accel.Backend{accel.BackendMetal, accel.BackendOpenGL},
	})
	if both.Info().Capabilities.F16Arithmetic {
		t.Error("GLES 3.1 has no f16 arithmetic, so the intersection with Metal must not have it")
	}

	// Strict limits are the portable floor, which is below developer mode's.
	if both.Limits().MaxWorkgroupInvocations >= dev.Limits().MaxWorkgroupInvocations {
		t.Error("strict mode must not report a more generous workgroup bound than developer mode")
	}
	if got := both.Limits().MaxWorkgroupInvocations; got != 128 {
		t.Errorf("MaxWorkgroupInvocations = %d, want spec 002 section 1.5's portable floor of 128", got)
	}
}

// TestStrictModeReportsNoSubgroupSentinel is spec 001 section 1.1: a device
// without subgroups reports 1/1 rather than zero.
func TestStrictModeReportsNoSubgroupSentinel(t *testing.T) {
	d := openCPU(t, accel.CPUOptions{
		Mode:          accel.CPUStrict,
		StrictTargets: []accel.Backend{accel.BackendVulkan},
	})
	if d.Info().Capabilities.Subgroups {
		t.Fatal("Vulkan's subgroup support is `cap`, so the baseline reports it absent")
	}
	if got, want := d.Limits().MinSubgroupSize, 1; got != want {
		t.Errorf("MinSubgroupSize = %d, want the %d sentinel", got, want)
	}
	if got, want := d.Limits().MaxSubgroupSize, 1; got != want {
		t.Errorf("MaxSubgroupSize = %d, want the %d sentinel", got, want)
	}
	if _, err := accel.OpenCPU(accel.CPUOptions{
		Mode:          accel.CPUStrict,
		StrictTargets: []accel.Backend{accel.BackendVulkan},
		SubgroupSize:  4,
	}); err == nil {
		t.Error("a profile reporting no subgroups accepted a subgroup size of 4")
	}
}

// TestSubgroupSizeValidation is spec 006 section 5: the size is 1 or a power of
// two within the profile's bounds, and it defaults to 4 rather than 1.
func TestSubgroupSizeValidation(t *testing.T) {
	if got := openCPU(t, accel.CPUOptions{}).Limits().MaxSubgroupSize; got != 4 {
		t.Errorf("the default emulated subgroup size is %d, want 4: a size of 1 makes "+
			"shuffle and ballot degenerate and hides the bugs the emulation exists to find", got)
	}
	for _, size := range []int{1, 2, 4, 32, 64, 128} {
		d := openCPU(t, accel.CPUOptions{SubgroupSize: size})
		if got := d.Limits().MinSubgroupSize; got != size {
			t.Errorf("SubgroupSize %d reported as %d", size, got)
		}
	}
	for _, size := range []int{-1, 3, 6, 256} {
		if _, err := accel.OpenCPU(accel.CPUOptions{SubgroupSize: size}); err == nil {
			t.Errorf("accepted subgroup size %d", size)
		}
	}
}

// TestMimicModeRequiresAProfile checks that mimic mode reproduces a captured
// contract and refuses to invent one.
func TestMimicModeRequiresAProfile(t *testing.T) {
	if _, err := accel.OpenCPU(accel.CPUOptions{Mode: accel.CPUMimic}); err == nil {
		t.Fatal("mimic mode opened without a profile")
	}

	captured := accel.DeviceProfile{Info: accel.DeviceInfo{
		Backend:      accel.BackendMetal,
		Name:         "Apple M3 Max",
		Capabilities: accel.Capabilities{F16Arithmetic: true, SharedMemoryKind: true},
		Limits:       accel.Limits{MaxSharedMemoryBytes: 32768, MinSubgroupSize: 1, MaxSubgroupSize: 1},
	}}
	d := openCPU(t, accel.CPUOptions{Mode: accel.CPUMimic, Mimic: &captured})

	if !strings.Contains(d.Info().Name, "Apple M3 Max") {
		t.Errorf("a mimicking device does not name what it mimics: %q", d.Info().Name)
	}
	if d.Info().Backend != accel.BackendCPU {
		t.Error("mimic mode reproduces another device's contract; it does not claim to be that device")
	}
	if got := d.Limits().MaxSharedMemoryBytes; got != 32768 {
		t.Errorf("MaxSharedMemoryBytes = %d, want the captured 32768", got)
	}
	if d.Info().Capabilities.Subgroups {
		t.Error("the captured profile has no subgroups")
	}
}

// TestUnknownModeIsRejected keeps the mode switch honest.
func TestUnknownModeIsRejected(t *testing.T) {
	if _, err := accel.OpenCPU(accel.CPUOptions{Mode: accel.CPUMode(42)}); err == nil {
		t.Fatal("an unknown CPU mode opened a device")
	}
}

// TestCloseIsIdempotent checks that a second Close is not an error, so a
// deferred Close after an explicit one is safe.
func TestCloseIsIdempotent(t *testing.T) {
	d, err := accel.OpenCPU(accel.CPUOptions{})
	if err != nil {
		t.Fatal(err)
	}
	if err := d.Close(); err != nil {
		t.Fatalf("first Close: %v", err)
	}
	if err := d.Close(); err != nil {
		t.Fatalf("second Close: %v", err)
	}
}

// TestBackendString covers the name every error message is written against.
func TestBackendString(t *testing.T) {
	for b, want := range map[accel.Backend]string{
		accel.BackendCPU:    "CPU",
		accel.BackendMetal:  "Metal",
		accel.BackendVulkan: "Vulkan",
		accel.BackendD3D12:  "D3D12",
		accel.BackendOpenGL: "OpenGL",
		accel.Backend(42):   "Backend(42)",
	} {
		if got := b.String(); got != want {
			t.Errorf("Backend(%d).String() = %q, want %q", int(b), got, want)
		}
	}
}

// TestDTypeSizeAndName pins the widths a storage buffer's element stride is,
// per spec 001 section 3.2: tightly packed, no padding ever.
func TestDTypeSizeAndName(t *testing.T) {
	tests := []struct {
		d    accel.DType
		size int
		name string
	}{
		{accel.F32, 4, "f32"},
		{accel.F16, 2, "f16"},
		{accel.BF16, 2, "bf16"},
		{accel.I32, 4, "i32"},
		{accel.U32, 4, "u32"},
		{accel.I8, 1, "i8"},
		{accel.U8, 1, "u8"},
	}
	for _, tc := range tests {
		if got := tc.d.Size(); got != tc.size {
			t.Errorf("%v.Size() = %d, want %d", tc.name, got, tc.size)
		}
		if got := tc.d.String(); got != tc.name {
			t.Errorf("DType(%d).String() = %q, want %q", int(tc.d), got, tc.name)
		}
	}
	if got := accel.DType(99).Size(); got != 0 {
		t.Errorf("an unknown dtype has size %d, want 0", got)
	}
	if got := accel.DType(99).String(); got != "DType(99)" {
		t.Errorf("DType(99).String() = %q", got)
	}
}

// ErrNotImplemented has no reachable user, which is the state to notice.
//
// It existed for Sampler, and specs/036-documentation.md's freeze record
// withdrew that whole family: exported with no producer, no consumer, and a
// Close that panicked, so a caller who found it could build a descriptor, never
// obtain one, and crash if they somehow did. Naming a type in a public API is a
// promise to keep it.
//
// The constant stays, because specs/032-stage-abi.md and
// specs/033-render-api.md will give it users again. This test asserts the
// boundary is empty rather than that it is gone, so re-exporting something
// unbuilt is a deliberate act rather than a drift.
func TestErrNotImplementedHasNoReachableUser(t *testing.T) {
	if accel.ErrNotImplemented == nil {
		t.Fatal("ErrNotImplemented was removed; if that is deliberate, this test goes with it")
	}
	// Read from the source rather than probed through the API: a stub is a
	// function nobody can reach *because* it panics, so the only way to count
	// them is to look. Every mention of the identifier in the package's
	// non-test files, other than its own declaration, is a use.
	files, err := filepath.Glob("*.go")
	if err != nil {
		t.Fatal(err)
	}
	fset := token.NewFileSet()
	var uses []string
	for _, name := range files {
		if strings.HasSuffix(name, "_test.go") {
			continue
		}
		f, err := parser.ParseFile(fset, name, nil, 0)
		if err != nil {
			t.Fatalf("parse %s: %v", name, err)
		}
		ast.Inspect(f, func(n ast.Node) bool {
			switch x := n.(type) {
			case *ast.ValueSpec:
				// The declaration itself: var ErrNotImplemented = ...
				for _, id := range x.Names {
					if id.Name == "ErrNotImplemented" {
						return false
					}
				}
			case *ast.Ident:
				if x.Name == "ErrNotImplemented" {
					uses = append(uses, fset.Position(x.Pos()).String())
				}
			}
			return true
		})
	}
	if len(uses) > 0 {
		t.Errorf("ErrNotImplemented is used at %s; the design-stage boundary is meant "+
			"to be empty, so a new stub is a deliberate act that starts by editing this "+
			"test", strings.Join(uses, ", "))
	}
}

// And what *was* on that list is now built, which is the other half: a boundary
// nobody moves is a boundary nobody is working on.
func TestTexturesAreNoLongerAStub(t *testing.T) {
	d := openCPU(t, accel.CPUOptions{})
	tex, err := d.NewTexture(accel.TextureDescriptor{
		Format: accel.RGBA8Unorm, Size: accel.Extent{Width: 4, Height: 4},
		Usage: accel.TextureCopyDst, Label: "tex",
	})
	if err != nil {
		t.Fatalf("NewTexture: %v", err)
	}
	if err := tex.Close(); err != nil {
		t.Fatalf("close: %v", err)
	}
}

// A caller can compare what a device offers against what a kernel requires.
//
// The two halves of that question were expressed in different types with no
// public bridge: a device reports Capabilities, a struct of flags plus a
// SubgroupOpSet; a kernel and Policy.Require use a Capability bitmask. Mapping
// one to the other meant reimplementing an unexported function, including the
// part that is easy to miss — every subgroup bit is gated on Subgroups as well
// as on its own op bit.
//
// Found by the public-surface review of specs/036-documentation.md: it is the
// first question a portability tutorial asks.
func TestCapabilitiesBridgeToTheRequirementSet(t *testing.T) {
	d, err := accel.OpenCPU(accel.CPUOptions{})
	if err != nil {
		t.Fatal(err)
	}
	defer d.Close()

	// Capabilities() is the sibling of Limits(), and its absence was the trap.
	caps := d.Capabilities()
	if caps != d.Info().Capabilities {
		t.Error("Device.Capabilities disagrees with Info().Capabilities")
	}

	set := caps.Set()
	if caps.Subgroups && set&accel.CapSubgroupBasic == 0 {
		t.Error("a device reporting subgroups offers no subgroup capability")
	}
	if !caps.Subgroups && set&accel.CapSubgroupBasic != 0 {
		t.Error("a device without subgroups offers a subgroup capability; every subgroup " +
			"bit is gated on Subgroups as well as on its own op bit")
	}

	// Has is the whole point: one call, no reimplementation.
	if !caps.Has(0) {
		t.Error("the empty requirement is unsatisfied")
	}
	if !caps.Has(set) {
		t.Error("a device does not satisfy its own capability set")
	}

	// And a capability it lacks is reported as lacking rather than as present.
	absent := ^set
	if absent != 0 && caps.Has(absent) {
		t.Error("a device satisfies capabilities it does not have")
	}
}

// Two devices that disagree about what they can do must not share an identity.
//
// The CPU adapter's token is a constant — there is one CPU adapter — but an
// opened device's profile is not: CPUStrict and CPUMimic resolve different
// names, capabilities and limits. The identity used to come from the adapter
// alone, so two open devices reported equal Info().ID and disagreed about what
// they could run.
//
// That matters beyond tidiness. DeviceInfo promises a caller can choose "on
// reported capabilities and limits rather than by trying and catching
// failures", and specs/007-tensor-layer.md keys its plan cache on the device
// identity — so a plan compiled against the developer profile could be reused
// on a strict device that cannot run its kernels.
func TestDeviceIdentityCoversTheResolvedProfile(t *testing.T) {
	dev := openCPU(t, accel.CPUOptions{})
	strict := openCPU(t, accel.CPUOptions{
		Mode:          accel.CPUStrict,
		StrictTargets: []accel.Backend{accel.BackendMetal},
	})

	if dev.Info().Capabilities == strict.Info().Capabilities &&
		dev.Limits() == strict.Limits() {
		t.Skip("this build's strict profile equals the developer one, so there is " +
			"nothing for the identity to distinguish")
	}
	if dev.Info().ID == strict.Info().ID {
		t.Error("two devices with different capabilities share an AdapterID; a plan " +
			"cache keyed on it would reuse a plan the strict device cannot run")
	}

	// And it is stable: the same profile opened twice is the same identity, or
	// every cache lookup misses.
	again := openCPU(t, accel.CPUOptions{})
	if dev.Info().ID != again.Info().ID {
		t.Error("the same profile opened twice reports different identities, so no " +
			"cache keyed on it can ever hit")
	}
}
