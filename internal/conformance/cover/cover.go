// Copyright 2026 The golang.design Initiative Authors.
// All rights reserved. Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

// Package cover reports per-package statement coverage under spec 011 section
// 10's checked exclusions.
//
// The gate is per package rather than one repository average, because an
// average lets a well-tested package carry an untested one and reports a number
// nobody can act on.
//
// The one exclusion this implements is section 10.1's design-stage stub rule: a
// function whose body is exactly panic(ErrNotImplemented) does not count. The
// rule is syntactic and self-retiring, and the excluded count is reported
// alongside the percentage, so a package that scores well because most of it
// does not exist yet is visible as exactly that rather than as a pass.
package cover

import (
	"bufio"
	"fmt"
	"go/ast"
	"go/parser"
	"go/token"
	"io"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
)

// Block is one statement block from a coverage profile.
type Block struct {
	File                string
	StartLine, StartCol int
	EndLine, EndCol     int
	Statements          int
	Count               int
}

// PackageReport is one package's coverage after exclusions.
type PackageReport struct {
	Package string

	Covered  int // statements executed at least once
	Total    int // statements counted, after exclusions
	Excluded int // statements in excluded declarations
	Stubs    int // excluded declarations, which is the number that must reach zero
	Percent  float64
}

// Report is every package, in import-path order.
type Report struct {
	Packages []PackageReport
	Missing  []string // packages with no counted statements at all
}

// ParseProfile reads a Go coverage profile.
func ParseProfile(r io.Reader) ([]Block, error) {
	var blocks []Block
	s := bufio.NewScanner(r)
	s.Buffer(make([]byte, 0, 64<<10), 4<<20)

	first := true
	for s.Scan() {
		line := strings.TrimSpace(s.Text())
		if line == "" {
			continue
		}
		if first {
			first = false
			if strings.HasPrefix(line, "mode:") {
				continue
			}
			return nil, fmt.Errorf("cover: profile does not start with a mode line")
		}
		b, err := parseBlock(line)
		if err != nil {
			return nil, err
		}
		blocks = append(blocks, b)
	}
	return blocks, s.Err()
}

// parseBlock reads one "file:startLine.startCol,endLine.endCol stmts count" row.
func parseBlock(line string) (Block, error) {
	malformed := func() (Block, error) {
		return Block{}, fmt.Errorf("cover: malformed profile line %q", line)
	}

	colon := strings.LastIndex(line, ":")
	if colon < 0 {
		return malformed()
	}
	file, rest := line[:colon], line[colon+1:]

	fields := strings.Fields(rest)
	if len(fields) != 3 {
		return malformed()
	}
	spanParts := strings.Split(fields[0], ",")
	if len(spanParts) != 2 {
		return malformed()
	}
	start, err1 := parsePos(spanParts[0])
	end, err2 := parsePos(spanParts[1])
	stmts, err3 := strconv.Atoi(fields[1])
	count, err4 := strconv.Atoi(fields[2])
	if err1 != nil || err2 != nil || err3 != nil || err4 != nil {
		return malformed()
	}

	return Block{
		File: file, StartLine: start[0], StartCol: start[1],
		EndLine: end[0], EndCol: end[1], Statements: stmts, Count: count,
	}, nil
}

func parsePos(s string) ([2]int, error) {
	lineText, colText, ok := strings.Cut(s, ".")
	if !ok {
		return [2]int{}, fmt.Errorf("cover: bad position %q", s)
	}
	line, err := strconv.Atoi(lineText)
	if err != nil {
		return [2]int{}, err
	}
	col, err := strconv.Atoi(colText)
	if err != nil {
		return [2]int{}, err
	}
	return [2]int{line, col}, nil
}

// Span is one excluded declaration's line range, inclusive.
type Span struct{ Start, End int }

// StubSpans finds every design-stage stub in a Go file.
//
// The rule from spec 011 section 10.1 is deliberately narrow: the body must be
// exactly one expression statement calling panic with the single identifier
// ErrNotImplemented. Anything else counts in full, including a function that
// validates its arguments before panicking, because that validation is
// behaviour somebody should test.
func StubSpans(filename string, src []byte) ([]Span, error) {
	fset := token.NewFileSet()
	f, err := parser.ParseFile(fset, filename, src, 0)
	if err != nil {
		return nil, err
	}

	var spans []Span
	for _, decl := range f.Decls {
		fn, ok := decl.(*ast.FuncDecl)
		if !ok || fn.Body == nil || len(fn.Body.List) != 1 {
			continue
		}
		if !isNotImplementedPanic(fn.Body.List[0]) {
			continue
		}
		spans = append(spans, Span{
			Start: fset.Position(fn.Body.Lbrace).Line,
			End:   fset.Position(fn.Body.Rbrace).Line,
		})
	}
	return spans, nil
}

func isNotImplementedPanic(stmt ast.Stmt) bool {
	expr, ok := stmt.(*ast.ExprStmt)
	if !ok {
		return false
	}
	call, ok := expr.X.(*ast.CallExpr)
	if !ok || len(call.Args) != 1 {
		return false
	}
	fn, ok := call.Fun.(*ast.Ident)
	if !ok || fn.Name != "panic" {
		return false
	}
	arg, ok := call.Args[0].(*ast.Ident)
	return ok && arg.Name == "ErrNotImplemented"
}

// Merge folds duplicate blocks together.
//
// Running the suite with -coverpkg means every test binary reports every
// package, so one statement appears once per binary with a different count each
// time. Summing those would count a statement several times and report a block
// covered by one binary and not another as both. A statement is covered when
// any binary executed it, which is the union, so the counts add and the
// statement is counted once.
func Merge(blocks []Block) []Block {
	type key struct {
		file                                 string
		startLine, startCol, endLine, endCol int
	}
	index := map[key]int{}
	out := make([]Block, 0, len(blocks))
	for _, b := range blocks {
		k := key{b.File, b.StartLine, b.StartCol, b.EndLine, b.EndCol}
		if i, ok := index[k]; ok {
			out[i].Count += b.Count
			continue
		}
		index[k] = len(out)
		out = append(out, b)
	}
	return out
}

// Summarize groups blocks by package and applies the exclusions.
//
// root is the directory the profile's file paths are relative to once module is
// stripped from them, which is how a profile written as an import path is read
// back off disk.
func Summarize(blocks []Block, module, root string) (Report, error) {
	blocks = Merge(blocks)

	type acc struct {
		covered, total, excluded, stubs int
	}
	byPkg := map[string]*acc{}
	stubCache := map[string][]Span{}
	countedStub := map[string]map[Span]bool{}

	for _, b := range blocks {
		pkg := path(b.File)
		a := byPkg[pkg]
		if a == nil {
			a = &acc{}
			byPkg[pkg] = a
		}

		spans, ok := stubCache[b.File]
		if !ok {
			rel := strings.TrimPrefix(b.File, module+"/")
			src, err := os.ReadFile(filepath.Join(root, filepath.FromSlash(rel)))
			if err != nil {
				return Report{}, fmt.Errorf("cover: reading %s: %w", b.File, err)
			}
			spans, err = StubSpans(b.File, src)
			if err != nil {
				return Report{}, fmt.Errorf("cover: parsing %s: %w", b.File, err)
			}
			stubCache[b.File] = spans
			countedStub[b.File] = map[Span]bool{}
		}

		if s, inStub := within(spans, b); inStub {
			a.excluded += b.Statements
			if !countedStub[b.File][s] {
				countedStub[b.File][s] = true
				a.stubs++
			}
			continue
		}

		a.total += b.Statements
		if b.Count > 0 {
			a.covered += b.Statements
		}
	}

	var rep Report
	for pkg, a := range byPkg {
		if a.total == 0 {
			rep.Missing = append(rep.Missing, pkg)
			continue
		}
		rep.Packages = append(rep.Packages, PackageReport{
			Package:  pkg,
			Covered:  a.covered,
			Total:    a.total,
			Excluded: a.excluded,
			Stubs:    a.stubs,
			Percent:  100 * float64(a.covered) / float64(a.total),
		})
	}
	sort.Slice(rep.Packages, func(i, j int) bool { return rep.Packages[i].Package < rep.Packages[j].Package })
	sort.Strings(rep.Missing)
	return rep, nil
}

func within(spans []Span, b Block) (Span, bool) {
	for _, s := range spans {
		if b.StartLine >= s.Start && b.EndLine <= s.End {
			return s, true
		}
	}
	return Span{}, false
}

// path is the import path of the package a profile file path belongs to.
func path(file string) string {
	if i := strings.LastIndex(file, "/"); i >= 0 {
		return file[:i]
	}
	return file
}

// Failures returns the packages below the gate, formatted for a CI log.
func (r Report) Failures(minPercent float64) []string {
	var out []string
	for _, p := range r.Packages {
		if p.Percent <= minPercent {
			out = append(out, fmt.Sprintf("%s is at %.1f%% (%d/%d statements), below the %.0f%% gate",
				p.Package, p.Percent, p.Covered, p.Total, minPercent))
		}
	}
	return out
}

// Print writes the report, including what each package excluded.
func (r Report) Print(w io.Writer) {
	for _, p := range r.Packages {
		fmt.Fprintf(w, "%-52s %6.1f%%  %5d/%-5d statements", p.Package, p.Percent, p.Covered, p.Total)
		if p.Stubs > 0 {
			fmt.Fprintf(w, "  (%d design-stage stubs excluded, %d statements)", p.Stubs, p.Excluded)
		}
		fmt.Fprintln(w)
	}
	for _, pkg := range r.Missing {
		fmt.Fprintf(w, "%-52s      -   every counted statement is a design-stage stub\n", pkg)
	}
}
