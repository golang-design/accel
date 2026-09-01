// Copyright 2026 The golang.design Initiative Authors.
// All rights reserved. Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

package tensor

import (
	"reflect"
	"testing"

	"golang.design/x/accel"
)

// The plan key covers every compile option except Label.
//
// writeOptions hashes the fields by name, and a field added to CompileOptions
// without a line there would be left out of the key: two plans compiled with
// different options would share one, and the cache would answer one with the
// other. That is the failure specs/029-plan-cache.md exists to rule out, so the
// struct is walked by reflection here and each field is varied on its own.
//
// Label is the one field that must *not* move the key, for the reason
// writeOptions gives, and that is asserted too.
func TestThePlanKeyCoversEveryCompileOption(t *testing.T) {
	dev, err := accel.OpenCPU(accel.CPUOptions{})
	if err != nil {
		t.Fatalf("open: %v", err)
	}
	defer dev.Close()
	rt, err := NewRuntime(dev)
	if err != nil {
		t.Fatalf("runtime: %v", err)
	}
	defer rt.Close()
	c := NewPlanCache(rt)

	var id Identity
	base := c.key(id, CompileOptions{})

	typ := reflect.TypeFor[CompileOptions]()
	for i := range typ.NumField() {
		f := typ.Field(i)
		t.Run(f.Name, func(t *testing.T) {
			var opts CompileOptions
			set(t, reflect.ValueOf(&opts).Elem().Field(i))
			moved := c.key(id, opts) != base
			if f.Name == "Label" {
				if moved {
					t.Fatal("Label moved the key; two plans differing only in what " +
						"they are called are the same plan")
				}
				return
			}
			if !moved {
				t.Fatalf("CompileOptions.%s did not move the key: add it to writeOptions, "+
					"or two plans compiled with different %s would share one", f.Name, f.Name)
			}
		})
	}
}

// set gives a field a value distinct from its zero, so varying it is
// observable. A kind this cannot vary fails the test rather than passing it
// quietly: extend this, do not skip the field.
func set(t *testing.T, v reflect.Value) {
	t.Helper()
	switch v.Kind() {
	case reflect.Bool:
		v.SetBool(true)
	case reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64:
		v.SetInt(1)
	case reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64:
		v.SetUint(1)
	case reflect.Float32, reflect.Float64:
		v.SetFloat(1)
	case reflect.String:
		v.SetString("x")
	case reflect.Slice:
		v.Set(reflect.MakeSlice(v.Type(), 1, 1))
		set(t, v.Index(0))
	case reflect.Pointer:
		v.Set(reflect.New(v.Type().Elem()))
		set(t, v.Elem())
	case reflect.Struct:
		for i := range v.NumField() {
			set(t, v.Field(i))
		}
	default:
		t.Fatalf("a %v field is not one this test knows how to vary", v.Kind())
	}
}
