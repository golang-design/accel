// Copyright 2026 The golang.design Initiative Authors.
// All rights reserved. Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

package accel

import (
	"errors"
	"runtime"
	"strings"
	"testing"
)

// A big-endian host is refused at open, naming the architecture.
//
// specs/001-device-resources.md section 3.5 puts the refusal at OpenDevice;
// it lived only in the transfer path, so such a host opened a device, built
// and submitted graphs, and failed at its first host transfer with a message
// that named nothing. No supported platform is big-endian, so the flag is
// flipped here to reach the branch.
func TestABigEndianHostIsRefusedAtOpen(t *testing.T) {
	was := hostIsLittleEndian
	hostIsLittleEndian = false
	t.Cleanup(func() { hostIsLittleEndian = was })

	for _, open := range []struct {
		name string
		open func() (*Device, error)
	}{
		{"OpenCPU", func() (*Device, error) { return OpenCPU(CPUOptions{}) }},
		{"OpenDevice", func() (*Device, error) { return OpenDevice(AdapterID{}) }},
	} {
		d, err := open.open()
		if err == nil {
			_ = d.Close()
			t.Fatalf("%s opened a device on a big-endian host", open.name)
		}
		if !errors.Is(err, ErrUnsupported) {
			t.Errorf("%s: %v is not ErrUnsupported", open.name, err)
		}
		for _, want := range []string{open.name, runtime.GOARCH, "big-endian"} {
			if !strings.Contains(err.Error(), want) {
				t.Errorf("%s: the refusal does not name %q: %v", open.name, want, err)
			}
		}
	}
	hostIsLittleEndian = was
	d, err := OpenCPU(CPUOptions{})
	if err != nil {
		t.Fatalf("with the flag restored OpenCPU failed: %v", err)
	}
	_ = d.Close()
}
