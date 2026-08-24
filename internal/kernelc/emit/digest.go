// Copyright 2026 The golang.design Initiative Authors.
// All rights reserved. Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

package emit

import (
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"strings"

	"golang.design/x/accel/internal/kernel"
	"golang.design/x/accel/internal/kernelc/intrin"
	"golang.design/x/accel/internal/kernelc/ir"
)

// GeneratorVersion is the emitter's own contract. It changes when the shape of
// what this package emits changes, which makes every generated file stale.
const GeneratorVersion = 1

// Digest identifies everything a kernel's generated form depends on.
//
// # Why more than the source
//
// A digest over the kernel's own source alone would say a file is fresh after
// the intrinsic table changed the meaning of one of its calls, or after the
// emitter changed how it spells a rounding point, or after a helper it depends
// on was edited. Each of those changes what the generated file should contain
// while leaving the authored source identical, and each would ship as a
// generated file that no longer matches its inputs.
//
// So the preimage covers the source, the ABI versions of everything between the
// source and the output, the intrinsic table's contents, and the helpers the
// body reaches.
//
// # Why the preimage is line-oriented and versioned
//
// Because it grows. Helpers arrive with spec 013 and uniform codecs with 014,
// and both add inputs. A line-oriented preimage with a section that is empty
// today takes new lines without changing how the existing ones are written,
// which matters because changing the format would reissue every committed
// generated file at the same time as the change being reviewed.
func Digest(k *ir.Func) string {
	return digestOf(preimage(k))
}

// preimage is the exact text a digest is taken over. It is exported through
// [Preimage] so a mismatch can be explained rather than only reported: a
// freshness failure that says only "the digest differs" leaves a reader to
// guess which of half a dozen inputs moved.
func preimage(k *ir.Func) string {
	var b strings.Builder

	// Versions first, so that a diff of two preimages shows the cheapest
	// explanation at the top.
	fmt.Fprintf(&b, "generator/%d\n", GeneratorVersion)
	fmt.Fprintf(&b, "abi/kernel/%d\n", kernel.ABIVersion)
	fmt.Fprintf(&b, "abi/intrin/%s\n", digestOf(intrin.Digest()))

	fmt.Fprintf(&b, "kernel\t%s\n", k.Name)
	fmt.Fprintf(&b, "workgroup\t%d\t%d\t%d\n", k.Workgroup[0], k.Workgroup[1], k.Workgroup[2])
	fmt.Fprintf(&b, "thread\t%d\n", k.Thread)

	// Bindings carry their inferred access, because the access is part of what a
	// backend is told and a body edit that only changes a read into a write must
	// make the file stale.
	for _, bind := range k.Bindings {
		fmt.Fprintf(&b, "binding\t%d\t%s\t%s\t%s\n", bind.Index, bind.Name, bind.Type, accessName(bind))
	}

	// Textures carry their inferred read for the reason bindings carry their
	// access: it is part of what a backend is told, and a body edit that stops
	// fetching from one must make the file stale. The loop emits nothing for a
	// kernel with no texture, so adding it does not reissue the corpus.
	for _, tx := range k.Textures {
		fmt.Fprintf(&b, "texture\t%d\t%s\t%d\t%v\n", tx.Index, tx.Name, tx.Param, tx.Reads)
	}

	// Intrinsics by authored spelling, in first-use order. The authored spelling
	// rather than the resolved path, so that relocating a type does not
	// invalidate every committed digest.
	for _, in := range k.Intrinsics {
		fmt.Fprintf(&b, "intrinsic\t%s\n", in)
	}

	// One line per helper the body reaches, with its own source digest, so that
	// editing a helper without regenerating its callers is caught. That is the
	// case a digest over the kernel's own source alone would miss entirely: the
	// caller is untouched and what it compiles to is not.
	fmt.Fprintf(&b, "helpers\t%d\n", len(k.Helpers))
	for _, h := range k.Helpers {
		fmt.Fprintf(&b, "helper\t%s\t%s\n", h.Name, digestOf(h.Source))
	}

	// The normalized source last, because it is the largest part and the one a
	// reader scrolls past.
	fmt.Fprintf(&b, "source\t%s\n", digestOf(k.Source))
	return b.String()
}

// Preimage returns the text [Digest] hashes, for explaining a mismatch.
func Preimage(k *ir.Func) string { return preimage(k) }

func accessName(b *ir.Binding) string {
	switch {
	case b.Read && b.Write:
		return "read-write"
	case b.Write:
		return "write"
	case b.Read:
		return "read"
	}
	return "none"
}

func digestOf(s string) string {
	sum := sha256.Sum256([]byte(s))
	return hex.EncodeToString(sum[:16])
}
