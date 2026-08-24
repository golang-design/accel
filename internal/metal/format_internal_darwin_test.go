// Copyright 2026 The golang.design Initiative Authors.
// All rights reserved. Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

package metal

import (
	"strings"
	"testing"

	"golang.design/x/accel/internal/driver"
	"golang.design/x/accel/internal/mtl"
)

// Every format a plan may name is either mapped or refused by name.
//
// This path hardcoded RGBA32Float for every colour attachment until
// specs/045-texture-attachments.md, so a caller who declared RGBA8Unorm got a
// pipeline that compiled and rendered sixteen bytes per pixel where they asked
// for four. The mapping is the fix and the refusal is the other half of it: a
// format this backend cannot spell must say so rather than default, because
// defaulting is how that defect worked.
//
// Driven by driver.Formats rather than a hand-written list, so a format added
// later fails this test until someone decides which side it is on. That is the
// same shape as the reflection-driven guard in tensor: a new case fails until
// it is classified, rather than being silently absent.
func TestEveryPlanFormatIsMappedOrRefusedByName(t *testing.T) {
	mapped := map[driver.Format]int{
		driver.RGBA32Float:    mtl.PixelFormatRGBA32Float,
		driver.RGBA8Unorm:     mtl.PixelFormatRGBA8Unorm,
		driver.RGBA8UnormSRGB: mtl.PixelFormatRGBA8UnormSRGB,
		driver.BGRA8Unorm:     mtl.PixelFormatBGRA8Unorm,
		driver.Depth32Float:   mtl.PixelFormatDepth32Float,
	}
	for _, f := range driver.Formats {
		t.Run(f.String(), func(t *testing.T) {
			got, err := metalPixelFormat(f)
			want, isMapped := mapped[f]
			switch {
			case isMapped && err != nil:
				t.Fatalf("%v should map: %v", f, err)
			case isMapped && got != want:
				t.Fatalf("%v maps to %d, want %d", f, got, want)
			case !isMapped && err == nil:
				t.Fatalf("%v is not in this test's table and mapped to %d anyway; add it "+
					"to the table or the mapping is untested", f, got)
			case !isMapped:
				// The refusal names the format, so a caller is told what is
				// missing rather than that something is.
				if !strings.Contains(err.Error(), f.String()) {
					t.Errorf("the refusal should name %v, got %v", f, err)
				}
			}
		})
	}
}

// The two members of a family map to different Metal formats.
//
// specs/045-texture-attachments.md section 2.1 puts sRGB on the view rather
// than the texture, so writing linear and presenting sRGB is one texture and
// two views. That only works if the two reach Metal as different pixel formats,
// which is exactly what a mapping keyed on "same bytes per pixel" would get
// wrong.
func TestAFamilysMembersDoNotCollide(t *testing.T) {
	lin, err := metalPixelFormat(driver.RGBA8Unorm)
	if err != nil {
		t.Fatal(err)
	}
	srgb, err := metalPixelFormat(driver.RGBA8UnormSRGB)
	if err != nil {
		t.Fatal(err)
	}
	if lin == srgb {
		t.Fatalf("RGBA8Unorm and RGBA8UnormSRGB both map to %d, so the conversion the "+
			"sRGB view exists for would never happen", lin)
	}
}
