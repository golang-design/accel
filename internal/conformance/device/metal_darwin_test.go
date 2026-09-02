// Copyright 2026 The golang.design Initiative Authors.
// All rights reserved. Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

//go:build darwin

package device

import (
	"strings"
	"testing"

	"golang.design/x/accel"
)

// The Metal row is absent without an adapter, and a failing row when the job
// promised one.
//
// specs/006-backends.md section 7: a job that promises only the CPU must not go
// red on a Mac without a usable GPU, and a job that promises Metal must not go
// green by skipping. Both branches are reached here through an empty
// enumeration, because the machine that runs this test has an adapter.
func TestTheMetalRowWithoutAnAdapter(t *testing.T) {
	none := accel.Enumeration{}
	if rows := metalProfilesFrom(none, false); len(rows) != 0 {
		t.Fatalf("with no adapter and no promise the Metal row should be absent, got %d rows", len(rows))
	}
	rows := metalProfilesFrom(none, true)
	if len(rows) != 1 {
		t.Fatalf("with no adapter and the promise there should be one failing row, got %d", len(rows))
	}
	d, err := rows[0].open()
	if err == nil || d != nil {
		t.Fatalf("the promised row should fail to open, got device %v, err %v", d, err)
	}
	if !strings.Contains(err.Error(), "promises Metal") {
		t.Errorf("the failure should say the job promised Metal: %v", err)
	}
}

// With an adapter the row is the adapter, whatever the promise.
func TestTheMetalRowWithAnAdapter(t *testing.T) {
	e := accel.Enumerate()
	var metal []accel.DeviceInfo
	for _, info := range e.Devices {
		if info.Backend == accel.BackendMetal {
			metal = append(metal, info)
		}
	}
	if len(metal) == 0 {
		t.Skip("no Metal adapter on this machine")
	}
	for _, required := range []bool{false, true} {
		rows := metalProfilesFrom(e, required)
		if len(rows) != 1 || rows[0].DeviceName != metal[0].Name {
			t.Fatalf("required=%v: want one row naming %q, got %+v", required, metal[0].Name, rows)
		}
	}
}
