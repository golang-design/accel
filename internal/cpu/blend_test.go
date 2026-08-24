// Copyright 2026 The golang.design Initiative Authors.
// All rights reserved. Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

package cpu

import (
	"testing"

	"golang.design/x/accel/internal/driver"
	"golang.design/x/accel/internal/raster"
)

// Every blend factor and operation maps to the one that means the same thing.
//
// The two enumerations agree today, so a numeric conversion would work today.
// It would also mistranslate silently the day either gains a value in the
// middle: every factor after the insertion would shift by one, and the result
// is a plausible image rather than an error. This walks every value by name, so
// a new one in either list is a compile error here rather than a wrong picture
// somewhere else.
func TestBlendFactorsMapOneToOne(t *testing.T) {
	for _, c := range []struct {
		from driver.BlendFactor
		to   raster.BlendFactor
	}{
		{driver.FactorZero, raster.FactorZero},
		{driver.FactorOne, raster.FactorOne},
		{driver.FactorSrcColor, raster.FactorSrcColor},
		{driver.FactorOneMinusSrcColor, raster.FactorOneMinusSrcColor},
		{driver.FactorSrcAlpha, raster.FactorSrcAlpha},
		{driver.FactorOneMinusSrcAlpha, raster.FactorOneMinusSrcAlpha},
		{driver.FactorDstColor, raster.FactorDstColor},
		{driver.FactorOneMinusDstColor, raster.FactorOneMinusDstColor},
		{driver.FactorDstAlpha, raster.FactorDstAlpha},
		{driver.FactorOneMinusDstAlpha, raster.FactorOneMinusDstAlpha},
	} {
		if got := rasterFactor(c.from); got != c.to {
			t.Errorf("driver factor %d maps to raster %d, want %d", c.from, got, c.to)
		}
	}
	// And the count matches, so a factor added to one list and not mapped is
	// caught here rather than falling through to zero.
	const lastDriver = driver.FactorOneMinusDstAlpha
	const lastRaster = raster.FactorOneMinusDstAlpha
	if int(lastDriver) != int(lastRaster) {
		t.Errorf("the two factor lists have different lengths, %d and %d; the mapping "+
			"above needs the new value", lastDriver+1, lastRaster+1)
	}
}

func TestBlendOpsMapOneToOne(t *testing.T) {
	for _, c := range []struct {
		from driver.BlendOp
		to   raster.BlendOp
	}{
		{driver.BlendAdd, raster.BlendAdd},
		{driver.BlendSubtract, raster.BlendSubtract},
		{driver.BlendReverseSubtract, raster.BlendReverseSubtract},
		{driver.BlendMin, raster.BlendMin},
		{driver.BlendMax, raster.BlendMax},
	} {
		if got := rasterOp(c.from); got != c.to {
			t.Errorf("driver op %d maps to raster %d, want %d", c.from, got, c.to)
		}
	}
	if int(driver.BlendMax) != int(raster.BlendMax) {
		t.Errorf("the two op lists have different lengths; the mapping above needs the " +
			"new value")
	}
}
