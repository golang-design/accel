// Copyright 2026 The golang.design Initiative Authors.
// All rights reserved. Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

package tensor

import (
	"fmt"

	"golang.design/x/accel"
	"golang.design/x/accel/internal/kernels"
)

// RowSampling is a sampling policy with one setting per row.
//
// [Sample] shares one [SamplingOptions] across every row of a batch. A server
// admits requests with their own policies into one batch, so each parameter
// here is a [rows] tensor the caller fills per request: a greedy request beside
// a nucleus-sampled one, each with its own truncation, in one dispatch chain.
// specs/064-per-row-sampling.md.
//
// Every field but Factor is optional; nil leaves that stage out of the graph,
// which is the only way to turn it off, as with [SamplingOptions].
type RowSampling struct {
	// Factor is each row's inverse temperature, and zero marks a greedy row:
	// its logits are scaled by one, so their order survives the softmax, and
	// the row takes its argmax rather than a draw. Required.
	Factor *Tensor

	// TopK keeps that many entries per row, u32; zero, or at least the
	// vocabulary, keeps every entry. Above [TopMaxRounds] the bound applies.
	TopK *Tensor

	// TopP keeps each row's nucleus, f32; a value outside (0, 1) keeps every
	// entry.
	TopP *Tensor

	// Bias is a sparse per-row addend applied before everything else.
	Bias *RowBias

	// Penalties are the repetition, presence and frequency penalties with a
	// row axis on their history.
	Penalties *RowPenalties
}

// RowBias is a per-row sparse logit bias: IDs and Values are [rows, slots],
// and an id at or past the vocabulary is an empty slot.
type RowBias struct {
	IDs    *Tensor
	Values *Tensor
}

// RowPenalties are [Sample]'s penalties with a row axis.
//
// History is a [rows, capacity] u32 state holding each row's recent token ids
// and Filled a [rows] u32 tensor saying how much of each row's ring holds;
// Counts is a [rows, vocab] u32 state the graph rebuilds every step. The three
// strengths are [rows] f32 tensors, with [SamplingOptions]'s zero-is-off rule
// applied per row.
type RowPenalties struct {
	History    *State
	Counts     *State
	Filled     *Tensor
	Repetition *Tensor
	Presence   *Tensor
	Frequency  *Tensor
}

// SampleRows draws one token per row of logits under a per-row policy.
//
// The chain is [Sample]'s -- bias, penalties, scale, softmax, top-k, top-p,
// draw -- with each stage reading its parameter per row. draws holds one
// uniform per row and is read only by the stochastic rows; a greedy row's
// draw is ignored rather than refused, since a batch mixes the two.
func SampleRows(b *Builder, logits, draws *Tensor, o RowSampling, prefix string) *Tensor {
	present := []*Tensor{logits}
	for _, t := range []*Tensor{draws, o.Factor, o.TopK, o.TopP} {
		if t != nil {
			present = append(present, t)
		}
	}
	if poisoned(present...) {
		return b.poison()
	}
	if why := checkLogits(logits); why != "" {
		return b.fail(1, "SampleRows", "%s", why)
	}
	rows, vocab, _ := sampleShape(logits)
	if o.Factor == nil {
		return b.fail(1, "SampleRows", "Factor is nil; every row needs its inverse temperature, "+
			"and zero is how a row asks for the argmax")
	}
	for _, p := range []struct {
		name string
		t    *Tensor
		dt   accel.DType
	}{
		{"Factor", o.Factor, accel.F32}, {"TopK", o.TopK, accel.U32}, {"TopP", o.TopP, accel.F32},
		{"draws", draws, accel.F32},
	} {
		if p.t == nil {
			if p.name == "draws" {
				return b.fail(1, "SampleRows", "draws is nil; every stochastic row draws against "+
					"its own uniform, and the greedy rows ignore theirs")
			}
			continue
		}
		if p.t.dtype != p.dt {
			return b.fail(1, "SampleRows", "%s is %v and the kernels read %v", p.name, p.t.dtype, p.dt)
		}
		if got := p.t.shape.Elements(); got != rows {
			return b.fail(1, "SampleRows", "%s holds %d entries and the logits hold %d rows; "+
				"there is exactly one per row", p.name, got, rows)
		}
	}
	dims := func(map[string]ScalarValue) any {
		return kernels.RowSampleDims{Vocab: uint32(vocab), Rows: uint32(rows)}
	}
	perElement := func(k *accel.Kernel) grid {
		return func(*Tensor) accel.WorkgroupCount {
			wg := int(k.WorkgroupSize.X)
			return accel.WorkgroupCount{X: (rows*vocab + wg - 1) / wg}
		}
	}
	perRow := func(*Tensor) accel.WorkgroupCount { return accel.WorkgroupCount{X: rows} }

	x := logits
	if o.Bias != nil {
		var ok bool
		if x, ok = b.biasRows(x, o.Bias, rows, vocab); !ok {
			return b.poison()
		}
	}
	if o.Penalties != nil {
		var ok bool
		if x, ok = b.penaliseRows(x, o.Penalties, rows, vocab, prefix); !ok {
			return b.poison()
		}
	}

	x = b.record(node{
		op: "ScaleRows", inputs: []*Tensor{x, o.Factor}, kernel: &kernels.ScaleRowsKernel,
		uniform: dims, grid: perElement(&kernels.ScaleRowsKernel),
		reason: "each row scaled by its own inverse temperature, a zero scaling by one so a " +
			"greedy row keeps its order",
	}, accel.F32, x.shape)
	x = Softmax(b, x, SoftmaxOptions{Axis: len(x.shape) - 1})
	if o.TopK != nil {
		x = b.record(node{
			op: "TopKMaskRows", inputs: []*Tensor{x, o.TopK}, kernel: &kernels.TopKMaskRowsKernel,
			uniform: dims, grid: perRow,
			reason: "one workgroup per row extracting that row's k largest, the k read per " +
				"row under specs/063's uniform declaration",
		}, accel.F32, x.shape)
	}
	if o.TopP != nil {
		x = b.record(node{
			op: "TopPMaskRows", inputs: []*Tensor{x, o.TopP}, kernel: &kernels.TopPMaskRowsKernel,
			uniform: dims, grid: perRow,
			reason: "one workgroup per row walking that row's nucleus, the p read per row",
		}, accel.F32, x.shape)
	}
	return b.record(node{
		op: "SampleRows", inputs: []*Tensor{x, draws, o.Factor}, kernel: &kernels.SampleRowsKernel,
		uniform: dims,
		grid: func(*Tensor) accel.WorkgroupCount {
			wg := int(kernels.SampleRowsKernel.WorkgroupSize.X)
			return accel.WorkgroupCount{X: (rows + wg - 1) / wg}
		},
		reason: "one invocation per row: a draw against the row's cumulative mass, or the " +
			"argmax where the row's factor is zero",
	}, accel.U32, Shape{rows})
}

// biasRows adds the sparse bias. The caller has checked logits.
func (b *Builder) biasRows(x *Tensor, bias *RowBias, rows, vocab int) (*Tensor, bool) {
	if bias.IDs == nil || bias.Values == nil {
		b.fail(2, "SampleRows", "Bias needs both IDs and Values")
		return nil, false
	}
	if poisoned(bias.IDs, bias.Values) {
		return nil, false
	}
	if bias.IDs.dtype != accel.U32 || bias.Values.dtype != accel.F32 {
		b.fail(2, "SampleRows", "Bias.IDs is %v and Bias.Values is %v; the kernel reads u32 ids "+
			"and f32 values", bias.IDs.dtype, bias.Values.dtype)
		return nil, false
	}
	if bias.IDs.shape.Elements() != bias.Values.shape.Elements() || bias.IDs.shape.Elements()%rows != 0 {
		b.fail(2, "SampleRows", "Bias.IDs holds %d entries and Bias.Values %d over %d rows; both "+
			"are [rows, slots]", bias.IDs.shape.Elements(), bias.Values.shape.Elements(), rows)
		return nil, false
	}
	slots := bias.IDs.shape.Elements() / rows
	dims := func(map[string]ScalarValue) any {
		return kernels.RowSampleDims{Vocab: uint32(vocab), Rows: uint32(rows), Slots: uint32(slots)}
	}
	return b.record(node{
		op: "LogitBias", inputs: []*Tensor{x, bias.IDs, bias.Values}, kernel: &kernels.LogitBiasKernel,
		uniform: dims,
		grid: func(*Tensor) accel.WorkgroupCount {
			wg := int(kernels.LogitBiasKernel.WorkgroupSize.X)
			return accel.WorkgroupCount{X: (rows*vocab + wg - 1) / wg}
		},
		reason: fmt.Sprintf("one invocation per logit scanning the row's %d bias slots", slots),
	}, accel.F32, x.shape), true
}

// penaliseRows is [Builder.penalise] with a row axis.
func (b *Builder) penaliseRows(x *Tensor, p *RowPenalties, rows, vocab int, prefix string) (*Tensor, bool) {
	if p.History == nil || p.Counts == nil || p.Filled == nil ||
		p.Repetition == nil || p.Presence == nil || p.Frequency == nil {
		b.fail(2, "SampleRows", "Penalties needs History, Counts, Filled, Repetition, Presence "+
			"and Frequency; a strength that is off for a row is a zero in its tensor")
		return nil, false
	}
	if poisoned(p.Filled, p.Repetition, p.Presence, p.Frequency) {
		return nil, false
	}
	if p.Counts.desc.DType != accel.U32 || p.History.desc.DType != accel.U32 {
		b.fail(2, "SampleRows", "the history and counts states hold token ids and counts, which "+
			"are u32; got %v and %v", p.History.desc.DType, p.Counts.desc.DType)
		return nil, false
	}
	if got := p.Counts.shape.Elements(); got != rows*vocab {
		b.fail(2, "SampleRows", "the counts state holds %d entries and the batch is %d rows of "+
			"%d; there is one count per row per token id", got, rows, vocab)
		return nil, false
	}
	if p.History.shape.Elements()%rows != 0 || p.History.shape.Elements() == 0 {
		b.fail(2, "SampleRows", "the history state holds %d entries over %d rows; it is "+
			"[rows, capacity]", p.History.shape.Elements(), rows)
		return nil, false
	}
	historyCap := p.History.shape.Elements() / rows
	for _, r := range []struct {
		name string
		t    *Tensor
		dt   accel.DType
	}{{"Filled", p.Filled, accel.U32}, {"Repetition", p.Repetition, accel.F32},
		{"Presence", p.Presence, accel.F32}, {"Frequency", p.Frequency, accel.F32}} {
		if r.t.dtype != r.dt || r.t.shape.Elements() != rows {
			b.fail(2, "SampleRows", "Penalties.%s is %v over %d entries; it is one %v per row",
				r.name, r.t.dtype, r.t.shape.Elements(), r.dt)
			return nil, false
		}
	}
	dims := func(map[string]ScalarValue) any {
		return kernels.RowSampleDims{Vocab: uint32(vocab), Rows: uint32(rows), History: uint32(historyCap)}
	}
	over := func(k *accel.Kernel, n int) grid {
		return func(*Tensor) accel.WorkgroupCount {
			wg := int(k.WorkgroupSize.X)
			return accel.WorkgroupCount{X: (n + wg - 1) / wg}
		}
	}
	cleared := b.writeCounts(p.Counts, node{
		op: "PenaltyClearRows", kernel: &kernels.PenaltyClearRowsKernel,
		uniform: dims, grid: over(&kernels.PenaltyClearRowsKernel, rows*vocab),
		reason: "a store of zero over every row's counts",
	})
	counted := b.writeCounts(cleared, node{
		op: "PenaltyCountRows", inputs: []*Tensor{readState(b, p.History), p.Filled},
		kernel:  &kernels.PenaltyCountRowsKernel,
		uniform: dims, grid: over(&kernels.PenaltyCountRowsKernel, rows*historyCap),
		reason: "one invocation per history entry incrementing its row's count with an " +
			"integer atomic, the row's fill read per row",
	})
	return b.record(node{
		op:      "PenaltyApplyRows",
		inputs:  []*Tensor{x, readState(b, counted), p.Repetition, p.Presence, p.Frequency},
		kernel:  &kernels.PenaltyApplyRowsKernel,
		uniform: dims, grid: over(&kernels.PenaltyApplyRowsKernel, rows*vocab),
		reason: "one invocation per logit applying its row's three strengths to a count " +
			"that is already final",
	}, accel.F32, x.shape), true
}
