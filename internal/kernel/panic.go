// Copyright 2026 The golang.design Initiative Authors.
// All rights reserved. Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

package kernel

import (
	"fmt"
	"runtime"
	"runtime/debug"
	"strings"
)

// Panic is a panic raised inside a kernel body, carried out of the goroutine it
// happened on together with where it happened.
//
// # Why the site travels with the value
//
// A panic recovered on one goroutine and re-raised on another has the second
// goroutine's stack, and a panic formatted with %v has no stack at all. Either
// way the line that failed -- the index that overran, the nil that was
// dereferenced -- is gone, and what reaches the caller is "index out of range"
// with the kernel's name. The site is captured where the panic is first
// recovered, while the panicking frames are still on the stack, and it goes
// wherever the value goes.
type Panic struct {
	// Value is what the kernel panicked with.
	Value any

	// Site is the source position of the kernel statement that panicked, as
	// file:line, followed by the function it is in. Empty when the recovery
	// could not find the panicking frame.
	Site string

	// Stack is the panicking goroutine's stack at the recovery, in the format
	// runtime/debug.Stack produces.
	Stack []byte
}

func (p *Panic) Error() string {
	if p.Site == "" {
		return fmt.Sprintf("%v", p.Value)
	}
	return fmt.Sprintf("%v at %s", p.Value, p.Site)
}

// Unwrap exposes a panic value that was itself an error, so a runtime error's
// type is still reachable through errors.As.
func (p *Panic) Unwrap() error {
	err, _ := p.Value.(error)
	return err
}

// Recovered turns a recovered value into a [Panic] carrying the site it was
// raised at.
//
// It must be called from a deferred function on the goroutine that panicked,
// because that is the only place the panicking frames are still on the stack.
// A value that is already a *Panic is returned as it is: the site was captured
// by whoever recovered it first, and a second capture would name the line that
// re-raised it.
func Recovered(r any) *Panic {
	if p, ok := r.(*Panic); ok {
		return p
	}
	return &Panic{Value: r, Site: panicSite(), Stack: debug.Stack()}
}

// panicSite is the innermost frame below the runtime's panic machinery, which
// is the statement that panicked.
//
// A deferred function runs above runtime.gopanic, and gopanic sits above the
// frames that raised the panic. Between gopanic and the kernel there may be
// more runtime frames -- goPanicIndex for a bounds failure, panicmem and
// sigpanic for a nil dereference -- and they are skipped for the same reason
// gopanic is: they are where the runtime noticed, not where the kernel went
// wrong.
func panicSite() string {
	pcs := make([]uintptr, 64)
	n := runtime.Callers(1, pcs)
	frames := runtime.CallersFrames(pcs[:n])
	past := false
	for {
		f, more := frames.Next()
		switch {
		case !past:
			past = f.Function == "runtime.gopanic"
		case !strings.HasPrefix(f.Function, "runtime."):
			return fmt.Sprintf("%s:%d (%s)", f.File, f.Line, f.Function)
		}
		if !more {
			return ""
		}
	}
}
