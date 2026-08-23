// Copyright 2026 The golang.design Initiative Authors.
// All rights reserved. Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

package tensor

// Views: shape changes that move no bytes.
//
// Every operator here returns a tensor sharing its operand's producer and
// storage, with different shape, strides or offset. That is what makes a
// transformer's reshaping free: a head split is a view, not a copy, and the
// only thing that has to be true is that the kernel reading it can index what
// the view describes.
//
// specs/024-tensor-bringup.md's lowering refuses a non-contiguous operand,
// which is where [Contiguous] comes in: it materializes a view into packed
// storage so the corpus kernels can read it. Refusing rather than silently
// copying is the choice specs/007-tensor-layer.md makes -- "v0 requires unit
// stride ... which admits ordinary contiguous row-major operands without
// silently materializing either one" -- because a copy nobody asked for is a
// cost nobody can see.

// view returns a tensor over the same storage with a different layout.
func (t *Tensor) view(shape Shape, strides []int, offset int) *Tensor {
	return &Tensor{
		b: t.b, dtype: t.dtype, shape: shape, strides: strides, offset: offset,
		node: t.node, port: t.port,
	}
}

// normalizeAxis turns a possibly negative axis into an index, NumPy style.
func normalizeAxis(axis, rank int) (int, bool) {
	if axis < 0 {
		axis += rank
	}
	if axis < 0 || axis >= rank {
		return 0, false
	}
	return axis, true
}

// Reshape reinterprets a tensor's extent without moving anything.
//
// Legal only on a contiguous operand, which is not a limitation of this
// implementation but of what reshaping means: a strided view's elements are not
// adjacent, so a different extent over them describes different elements. The
// error says so rather than saying "unsupported", because the fix is
// [Contiguous] and a reader should not have to guess that.
func Reshape(b *Builder, x *Tensor, shape Shape) *Tensor {
	if poisoned(x) {
		return b.poison()
	}
	for i, d := range shape {
		if d <= 0 {
			return b.fail(1, "Reshape", "dimension %d is %d, and every dimension is a "+
				"positive concrete integer", i, d)
		}
	}
	if got, want := shape.Elements(), x.shape.Elements(); got != want {
		return b.fail(1, "Reshape", "%v holds %d elements and %v holds %d; a reshape "+
			"renames axes and never adds or drops values", x.shape, want, shape, got)
	}
	if !x.contiguousLayout() {
		return b.fail(1, "Reshape", "the operand is a strided view, whose elements are not "+
			"adjacent, so a different extent over them names different elements; insert "+
			"Contiguous first")
	}
	return x.view(shape, contiguous(shape), x.offset)
}

// Permute reorders axes.
//
// The strides move with them, so this is bookkeeping: a permuted tensor reads
// the same bytes in a different order, and only becomes a copy if something
// downstream needs it contiguous.
func Permute(b *Builder, x *Tensor, axes ...int) *Tensor {
	if poisoned(x) {
		return b.poison()
	}
	if len(axes) != len(x.shape) {
		return b.fail(1, "Permute", "%d axes for a rank-%d tensor", len(axes), len(x.shape))
	}
	seen := make([]bool, len(x.shape))
	norm := make([]int, len(axes))
	for i, a := range axes {
		n, ok := normalizeAxis(a, len(x.shape))
		if !ok {
			return b.fail(1, "Permute", "axis %d is outside a rank-%d tensor", a, len(x.shape))
		}
		if seen[n] {
			return b.fail(1, "Permute", "axis %d appears twice; a permutation contains each "+
				"axis exactly once", n)
		}
		seen[n] = true
		norm[i] = n
	}
	shape := make(Shape, len(axes))
	strides := make([]int, len(axes))
	for i, a := range norm {
		shape[i], strides[i] = x.shape[a], x.strides[a]
	}
	return x.view(shape, strides, x.offset)
}

// Transpose swaps two axes.
func Transpose(b *Builder, x *Tensor, axisA, axisB int) *Tensor {
	if poisoned(x) {
		return b.poison()
	}
	i, ok := normalizeAxis(axisA, len(x.shape))
	if !ok {
		return b.fail(1, "Transpose", "axis %d is outside a rank-%d tensor", axisA, len(x.shape))
	}
	j, ok := normalizeAxis(axisB, len(x.shape))
	if !ok {
		return b.fail(1, "Transpose", "axis %d is outside a rank-%d tensor", axisB, len(x.shape))
	}
	axes := make([]int, len(x.shape))
	for k := range axes {
		axes[k] = k
	}
	axes[i], axes[j] = j, i
	return Permute(b, x, axes...)
}

// Slice narrows one axis to a half-open range.
//
// Unit step only, which keeps a slice a view: a strided step would still be a
// view, and nothing in v0 needs one, so it is absent rather than untested.
func Slice(b *Builder, x *Tensor, axis, start, end int) *Tensor {
	if poisoned(x) {
		return b.poison()
	}
	a, ok := normalizeAxis(axis, len(x.shape))
	if !ok {
		return b.fail(1, "Slice", "axis %d is outside a rank-%d tensor", axis, len(x.shape))
	}
	if start < 0 || end < start || end > x.shape[a] {
		return b.fail(1, "Slice", "[%d, %d) is not a range within axis %d of extent %d",
			start, end, a, x.shape[a])
	}
	shape := make(Shape, len(x.shape))
	copy(shape, x.shape)
	shape[a] = end - start
	strides := make([]int, len(x.strides))
	copy(strides, x.strides)
	return x.view(shape, strides, x.offset+start*x.strides[a])
}

// Broadcast expands size-one axes to a larger shape.
//
// The expansion is a **zero stride**, which is the whole trick: every index
// along that axis reads the same element, so nothing is materialized and
// nothing is copied. A kernel that indexes contiguously cannot read it, which
// is why lowering refuses a broadcast operand and [Contiguous] exists.
func Broadcast(b *Builder, x *Tensor, shape Shape) *Tensor {
	if poisoned(x) {
		return b.poison()
	}
	if len(shape) < len(x.shape) {
		return b.fail(1, "Broadcast", "%v has lower rank than %v; broadcasting adds leading "+
			"axes and never drops one", shape, x.shape)
	}
	strides := make([]int, len(shape))
	off := len(shape) - len(x.shape)
	for i := range shape {
		if i < off {
			// A new leading axis reads the whole operand for every index.
			strides[i] = 0
			continue
		}
		switch d := x.shape[i-off]; {
		case d == shape[i]:
			strides[i] = x.strides[i-off]
		case d == 1:
			strides[i] = 0
		default:
			return b.fail(1, "Broadcast", "axis %d is %d and cannot expand to %d; only a "+
				"size-one axis expands", i, d, shape[i])
		}
	}
	return x.view(shape, strides, x.offset)
}
