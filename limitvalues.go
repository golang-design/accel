// Copyright 2026 The golang.design Initiative Authors.
// All rights reserved. Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

package accel

import (
	"fmt"
	"reflect"
)

// LimitValue is one numeric bound from [Limits], named.
type LimitValue struct {
	Name  string
	Value int
}

// LimitValues flattens [Limits] into a stable, named sequence, expanding a
// per-axis limit into one entry per axis.
//
// It exists because a limit is one kind of thing repeated twenty-odd times, and
// every consumer wants to treat them uniformly: [Policy] filters on them,
// diagnostics print them, and the conformance requirement that an opened device
// has no zero-valued limit is one loop rather than one assertion per field. That
// last check is the cheapest possible catch for a backend that forgot to fill a
// limit in, which is the failure spec 001 section 1.1 is written against.
//
// The order is the declaration order of [Limits] and is part of the contract:
// two calls are index-comparable.
func LimitValues(l Limits) []LimitValue {
	v := reflect.ValueOf(l)
	t := v.Type()
	out := make([]LimitValue, 0, t.NumField()+4)
	for i := range t.NumField() {
		f, fv := t.Field(i), v.Field(i)
		switch f.Type.Kind() {
		case reflect.Int:
			out = append(out, LimitValue{Name: f.Name, Value: int(fv.Int())})
		case reflect.Array:
			for j := range fv.Len() {
				out = append(out, LimitValue{
					Name:  fmt.Sprintf("%s[%d]", f.Name, j),
					Value: int(fv.Index(j).Int()),
				})
			}
		default:
			// Limits holds numeric bounds and nothing else. A field of another
			// kind means the struct grew something that does not belong here.
			panic(fmt.Sprintf("accel: Limits.%s has non-numeric kind %s", f.Name, f.Type.Kind()))
		}
	}
	return out
}
