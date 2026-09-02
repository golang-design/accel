// Copyright 2026 The golang.design Initiative Authors.
// All rights reserved. Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

package emit_test

import (
	"go/constant"
	"go/token"
	"strings"
	"testing"

	"golang.design/x/accel/internal/kernelc/emit"
	"golang.design/x/accel/internal/kernelc/ir"
)

// A float minimum or maximum goes through a NaN-propagating helper, and an
// integer one keeps MSL's built-in.
//
// kmath.Min and kmath.Max call NaN propagation a contract. MSL's min and max on
// float are fmin and fmax, which return the other operand when one is NaN, so
// the bare built-in agreed with the oracle on every input except the one the
// contract is about. The differential that runs this on a device is in the
// root package; this is the assertion about the text.
func TestFloatMinMaxPropagateNaNInMSL(t *testing.T) {
	f32 := &ir.Type{Kind: ir.F32}
	one := ir.NewConst(0, f32, constant.MakeFloat64(1))
	two := ir.NewConst(0, f32, constant.MakeFloat64(2))
	for _, tc := range []struct {
		op     ir.Opcode
		helper string
	}{{ir.OpMin, "_accel_fmin"}, {ir.OpMax, "_accel_fmax"}} {
		x := ir.NewLocal(0, f32, 0, "x", nil)
		src, err := emit.MSL(kernelWith(ir.NewDeclare(0, x,
			ir.NewIntrinsic(0, f32, tc.op, nil, []ir.Value{one, two}))))
		if err != nil {
			t.Fatalf("MSL: %v", err)
		}
		if !strings.Contains(src, tc.helper+"(float(1), float(2))") {
			t.Errorf("%v should call %s, got:\n%s", tc.op, tc.helper, src)
		}
		if !strings.Contains(src, "static float "+tc.helper+"(float a, float b)") {
			t.Errorf("%v should emit the helper it calls, got:\n%s", tc.op, src)
		}
		// The quiet NaN kmath returns, by its bits, so the two backends agree
		// exactly rather than both being some NaN.
		if !strings.Contains(src, "as_type<float>(0x7FC00000u)") {
			t.Errorf("the helper should return kmath's quiet NaN, got:\n%s", src)
		}
	}

	// The integer forms have no NaN to propagate and keep the built-in.
	i32 := &ir.Type{Kind: ir.I32}
	x := ir.NewLocal(0, i32, 0, "x", nil)
	src, err := emit.MSL(kernelWith(ir.NewDeclare(0, x,
		ir.NewIntrinsic(0, i32, ir.OpMin, nil, []ir.Value{
			ir.NewConst(0, i32, constant.MakeInt64(1)),
			ir.NewConst(0, i32, constant.MakeInt64(2)),
		}))))
	if err != nil {
		t.Fatalf("MSL: %v", err)
	}
	if !strings.Contains(src, "min(int(1), int(2))") {
		t.Errorf("an integer minimum should keep MSL's min, got:\n%s", src)
	}
	if strings.Contains(src, "_accel_fmin") {
		t.Errorf("an integer minimum reached the float helper:\n%s", src)
	}
}

// Signed 32-bit add, subtract and multiply are emitted through unsigned
// arithmetic, so they wrap as specs/008-numerics.md section 3 requires.
//
// MSL's int overflow is undefined and the compiler folds what it may assume
// never happens: `a + 1 > a` becomes true, and the oracle's int32 says false at
// MaxInt32. Shifts and division are left as they are, because the spec makes
// their excluded cases errors rather than wrapping, and neither error exists on
// this target yet.
func TestSignedArithmeticWrapsInMSL(t *testing.T) {
	i32 := &ir.Type{Kind: ir.I32}
	u32 := &ir.Type{Kind: ir.U32}
	a := ir.NewLocal(0, i32, 0, "a", nil)
	b := ir.NewLocal(0, i32, 1, "b", nil)
	emitted := func(t *testing.T, typ *ir.Type, op token.Token) string {
		t.Helper()
		x := ir.NewLocal(0, typ, 2, "x", nil)
		src, err := emit.MSL(kernelWith(ir.NewBlock(0,
			ir.NewDeclare(0, a, ir.NewConst(0, i32, constant.MakeInt64(1))),
			ir.NewDeclare(0, b, ir.NewConst(0, i32, constant.MakeInt64(2))),
			ir.NewDeclare(0, x, ir.NewBinary(0, typ, op, a, b)))))
		if err != nil {
			t.Fatalf("MSL: %v", err)
		}
		return src
	}
	for _, op := range []token.Token{token.ADD, token.SUB, token.MUL} {
		want := "int(uint(a) " + op.String() + " uint(b))"
		if src := emitted(t, i32, op); !strings.Contains(src, want) {
			t.Errorf("signed %v should be emitted as %s, got:\n%s", op, want, src)
		}
	}
	// Shifts and division are not wrapping operations: they go through the
	// helpers that spell Go's result where MSL's is undefined, and record a
	// fault where Go's is a panic (specs/008-numerics.md section 3).
	for op, want := range map[token.Token]string{
		token.SHL: "_accel_shl_i32(a, uint(b))",
		token.SHR: "_accel_shr_i32(a, uint(b))",
		token.QUO: "_accel_div_i32(a, b, _fault)",
		token.REM: "_accel_rem_i32(a, b, _fault)",
	} {
		if src := emitted(t, i32, op); !strings.Contains(src, want) {
			t.Errorf("signed %v should be emitted as %s, got:\n%s", op, want, src)
		}
	}
	for op, want := range map[token.Token]string{
		token.SHL: "_accel_shl_u32(a, uint(b))",
		token.QUO: "_accel_div_u32(a, b, _fault)",
	} {
		if src := emitted(t, u32, op); !strings.Contains(src, want) {
			t.Errorf("unsigned %v should be emitted as %s, got:\n%s", op, want, src)
		}
	}
	// Unsigned arithmetic already wraps and needs no cast.
	if src := emitted(t, u32, token.ADD); !strings.Contains(src, "(a + b)") {
		t.Errorf("unsigned add should stay a plain add, got:\n%s", src)
	}
}
