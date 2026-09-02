// Copyright 2026 The golang.design Initiative Authors.
// All rights reserved. Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

package kernels

import "golang.design/x/accel"

// The per-row sampling kernels, specs/064-per-row-sampling.md.
//
// The batched sampling chain shares one policy across every row of a batch,
// because its parameters are uniform fields: a temperature, a k, a p, three
// penalty strengths. A server admits requests with their own policies into one
// batch, and the consumer's spec (tgo 020 §6) names the per-row k and p, a row
// axis on the penalties, a logit bias, and a greedy row beside a stochastic
// one as what it needs. Each parameter here is a binding indexed by row.
//
// What makes a per-row k or p possible at all is specs/063-uniform-loads.md:
// the extraction loops below hold barriers, so their trip counts have to be
// workgroup-uniform, and a count loaded from a table was not until the table
// could be declared uniform. It is the host's routing data, no invocation of
// the dispatch writes it, and //accel:uniform says so.

// RowSampleDims describes a batch of rows sampled with per-row parameters.
type RowSampleDims struct {
	Vocab uint32
	Rows  uint32

	// History is the per-row ring's capacity for the penalty kernels, and
	// Slots the per-row width of the bias tables. Zero where unused.
	History uint32
	Slots   uint32
}

// TopKMaskRows is [TopKMask] with k read per row: ks[row] entries are kept,
// and a row whose k is zero, or at least the vocabulary, is passed through
// untruncated. Above [TopMaxRounds] the bound applies as it does to the
// uniform form.
//
//accel:uniform ks
//accel:kernel workgroup=128
func TopKMaskRows(t accel.Thread, d RowSampleDims, weights []float32, ks []uint32,
	out []float32, best *[128]float32, at *[128]uint32) {

	lane := t.LocalID().X
	row := t.GroupID().X
	base := row * d.Vocab

	rounds := ks[row]
	if rounds > TopMaxRounds {
		rounds = TopMaxRounds
	}
	if rounds >= d.Vocab {
		rounds = 0
	}

	frontV := float32(3.4e38)
	frontI := uint32(0)

	for r := uint32(0); r < rounds; r++ {
		v := float32(-3.4e38)
		idx := d.Vocab
		for i := lane; i < d.Vocab; i += RowWidth {
			w := weights[base+i]
			below := w < frontV
			if w == frontV && i > frontI {
				below = true
			}
			if below {
				better := w > v
				if w == v && i < idx {
					better = true
				}
				if better {
					v = w
					idx = i
				}
			}
		}
		best[lane] = v
		at[lane] = idx
		t.Barrier()

		for stride := uint32(64); stride > 0; stride /= 2 {
			if lane < stride {
				a := best[lane]
				b := best[lane+stride]
				if b > a {
					best[lane] = b
					at[lane] = at[lane+stride]
				} else if b == a && at[lane+stride] < at[lane] {
					at[lane] = at[lane+stride]
				}
			}
			t.Barrier()
		}
		frontV = best[0]
		frontI = at[0]
		t.Barrier()
	}

	// With no rounds the frontier is still the initial one, above every
	// entry, and every entry is kept: the pass-through.
	for i := lane; i < d.Vocab; i += RowWidth {
		w := weights[base+i]
		keep := w > frontV
		if w == frontV && i <= frontI {
			keep = true
		}
		if rounds == 0 {
			keep = true
		}
		if keep {
			out[base+i] = w
		} else {
			out[base+i] = float32(0)
		}
	}
}

// TopPMaskRows is [TopPMask] with p read per row. A row whose p is not in
// (0, 1) is passed through untruncated.
//
//accel:uniform ps
//accel:kernel workgroup=128
func TopPMaskRows(t accel.Thread, d RowSampleDims, weights []float32, ps []float32,
	out []float32, best *[128]float32, at *[128]uint32) {

	lane := t.LocalID().X
	row := t.GroupID().X
	base := row * d.Vocab

	p := ps[row]
	active := p > 0 && p < 1

	sum := float32(0)
	for i := lane; i < d.Vocab; i += RowWidth {
		sum = sum + weights[base+i]
	}
	best[lane] = sum
	t.Barrier()
	for stride := uint32(64); stride > 0; stride /= 2 {
		if lane < stride {
			best[lane] = best[lane] + best[lane+stride]
		}
		t.Barrier()
	}
	target := best[0] * p
	t.Barrier()

	frontV := float32(3.4e38)
	frontI := uint32(0)
	kept := float32(0)

	// The rounds run for every row of the workgroup's grid alike -- the
	// count is a literal -- and an inactive row simply never advances its
	// frontier, which keeps everything.
	for r := uint32(0); r < TopMaxRounds; r++ {
		v := float32(-3.4e38)
		idx := d.Vocab
		for i := lane; i < d.Vocab; i += RowWidth {
			w := weights[base+i]
			below := w < frontV
			if w == frontV && i > frontI {
				below = true
			}
			if below {
				better := w > v
				if w == v && i < idx {
					better = true
				}
				if better {
					v = w
					idx = i
				}
			}
		}
		best[lane] = v
		at[lane] = idx
		t.Barrier()

		for stride := uint32(64); stride > 0; stride /= 2 {
			if lane < stride {
				a := best[lane]
				b := best[lane+stride]
				if b > a {
					best[lane] = b
					at[lane] = at[lane+stride]
				} else if b == a && at[lane+stride] < at[lane] {
					at[lane] = at[lane+stride]
				}
			}
			t.Barrier()
		}

		if active && kept < target {
			frontV = best[0]
			frontI = at[0]
			kept = kept + frontV
		}
		t.Barrier()
	}

	for i := lane; i < d.Vocab; i += RowWidth {
		w := weights[base+i]
		keep := w > frontV
		if w == frontV && i <= frontI {
			keep = true
		}
		if !active {
			keep = true
		}
		if keep {
			out[base+i] = w
		} else {
			out[base+i] = float32(0)
		}
	}
}

// ScaleRows multiplies each row by its own factor. A factor of zero marks a
// greedy row and scales by one instead, so the row's order survives into the
// softmax and [SampleRows] takes its argmax.
//
//accel:kernel workgroup=128
func ScaleRows(t accel.Thread, d RowSampleDims, x []float32, factors []float32, out []float32) {
	i := t.GlobalID().X
	if i >= d.Rows*d.Vocab {
		return
	}
	f := factors[i/d.Vocab]
	if f == 0 {
		f = 1
	}
	out[i] = x[i] * f
}

// SampleRows draws one token per row, or takes the row's argmax where its
// factor is zero: a greedy row beside a stochastic one in one dispatch.
//
// The categorical half is [SampleCategorical]'s walk; the greedy half is the
// argmax of the probabilities, which is the argmax of the logits since the
// softmax is monotone and the greedy row was scaled by one.
//
//accel:kernel workgroup=64
func SampleRows(t accel.Thread, d RowSampleDims, probs []float32, draws []float32,
	factors []float32, out []uint32) {

	r := t.GlobalID().X
	if r >= d.Rows {
		return
	}
	base := r * d.Vocab

	if factors[r] == 0 {
		v := float32(-3.4e38)
		idx := uint32(0)
		for i := uint32(0); i < d.Vocab; i++ {
			x := probs[base+i]
			if x > v {
				v = x
				idx = i
			}
		}
		out[r] = idx
		return
	}

	draw := draws[r]
	if draw < float32(0) {
		draw = float32(0)
	}
	if draw > float32(0.99999994) {
		draw = float32(0.99999994)
	}

	total := float32(0)
	for i := uint32(0); i < d.Vocab; i++ {
		total = total + probs[base+i]
	}
	target := draw * total

	acc := float32(0)
	chosen := d.Vocab - 1
	found := false
	for i := uint32(0); i < d.Vocab; i++ {
		acc = acc + probs[base+i]
		if !found && acc > target {
			chosen = i
			found = true
		}
	}
	out[r] = chosen
}

// LogitBias adds each row's sparse bias to its logits: ids and values are
// [Rows, Slots], and an id at or past the vocabulary is an empty slot.
//
// One invocation per logit scanning the row's slots, rather than one per slot
// scattering into a copy, because the copy would be a second pass over the
// vocabulary and the slots are few.
//
//accel:kernel workgroup=128
func LogitBias(t accel.Thread, d RowSampleDims, logits []float32, ids []uint32,
	values []float32, out []float32) {

	i := t.GlobalID().X
	if i >= d.Rows*d.Vocab {
		return
	}
	row := i / d.Vocab
	id := i - row*d.Vocab
	x := logits[i]
	for s := uint32(0); s < d.Slots; s++ {
		if ids[row*d.Slots+s] == id {
			x = x + values[row*d.Slots+s]
		}
	}
	out[i] = x
}

// PenaltyClearRows zeroes every row's counts.
//
//accel:kernel workgroup=64
func PenaltyClearRows(t accel.Thread, d RowSampleDims, counts []uint32) {
	i := t.GlobalID().X
	if i < d.Rows*d.Vocab {
		counts[i] = 0
	}
}

// PenaltyCountRows is [PenaltyCount] with a row axis: history is [Rows,
// History] and filled[row] is how much of each row's ring holds.
//
//accel:kernel workgroup=64
func PenaltyCountRows(t accel.Thread, d RowSampleDims, history []uint32, filled []uint32,
	counts []uint32) {

	i := t.GlobalID().X
	if i >= d.Rows*d.History {
		return
	}
	row := i / d.History
	if i-row*d.History >= filled[row] {
		return
	}
	id := history[i]
	if id < d.Vocab {
		accel.AddU32(counts, row*d.Vocab+id, 1)
	}
}

// PenaltyApplyRows is [PenaltyApply] with the three strengths read per row.
//
//accel:kernel workgroup=64
func PenaltyApplyRows(t accel.Thread, d RowSampleDims, logits []float32, counts []uint32,
	repetition []float32, presence []float32, frequency []float32, out []float32) {

	i := t.GlobalID().X
	if i >= d.Rows*d.Vocab {
		return
	}
	row := i / d.Vocab
	l := logits[i]
	c := counts[i]
	if c == 0 {
		out[i] = l
		return
	}
	rep := repetition[row]
	if rep != 0 && rep != 1 {
		if l > 0 {
			l = l / rep
		} else {
			l = l * rep
		}
	}
	out[i] = l - presence[row] - frequency[row]*float32(c)
}
