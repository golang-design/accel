//go:build darwin

package metal_test

import (
	"runtime"
	"testing"
)

// Pinned bytes are released: many submissions do not grow the heap.
//
// SetBytes pins the caller's slice for the duration of one message send. A
// Pinner's Unpin releases every object it holds, so a missed Unpin would not
// error -- it would retain one slice per send, forever, and only show as growth.
func TestPinnedBytesDoNotAccumulate(t *testing.T) {
	e := benchExecutable(t, 256, false)

	// Warm up, so the first submission's one-time allocations are not counted.
	for range 20 {
		f, err := e.Submit()
		if err != nil {
			t.Fatalf("submit: %v", err)
		}
		if err := f.Wait(); err != nil {
			t.Fatalf("wait: %v", err)
		}
	}
	runtime.GC()
	var before runtime.MemStats
	runtime.ReadMemStats(&before)

	const runs = 300
	for range runs {
		f, err := e.Submit()
		if err != nil {
			t.Fatalf("submit: %v", err)
		}
		if err := f.Wait(); err != nil {
			t.Fatalf("wait: %v", err)
		}
	}
	runtime.GC()
	var after runtime.MemStats
	runtime.ReadMemStats(&after)

	// 256 nodes x 300 submissions is 76800 sends. A retained slice per send
	// would be tens of megabytes; ordinary churn is far below that.
	grew := int64(after.HeapAlloc) - int64(before.HeapAlloc)
	t.Logf("heap %d -> %d bytes over %d submissions (%+d)",
		before.HeapAlloc, after.HeapAlloc, runs, grew)
	if grew > 4<<20 {
		t.Fatalf("the heap grew by %d bytes over %d submissions, which is the "+
			"shape of a pin that is never released", grew, runs)
	}
}
