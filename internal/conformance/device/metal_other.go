// Copyright 2026 The golang.design Initiative Authors.
// All rights reserved. Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

//go:build !darwin

package device

// metalProfiles is empty where Metal cannot exist. The row is a darwin fact,
// not a skip: on Linux and Windows there is nothing to promise and nothing to
// report absent.
func metalProfiles() []Profile { return nil }
