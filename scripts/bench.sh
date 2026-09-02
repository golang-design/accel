#!/bin/sh
# Runs the benchmarks whose figures the specs cite and prints them as one
# Markdown table, so a CI job can publish them per commit.
#
# Usage: scripts/bench.sh [cpu|metal]
#
# The figures are published, not gated: a shared runner's timing is a
# measurement of the runner as much as of the code, and a gate on it would go
# red for the runner's reasons. What a reader needs is the number next to the
# commit, and the artifact the job uploads is the raw benchmark output.
set -eu
which=${1:-cpu}
count=${BENCH_COUNT:-3}
case "$which" in
cpu)
  pkgs="./ ./internal/kernel/ ./internal/alloc/"
  pattern='BenchmarkGraphSubmit|BenchmarkGraphRebind|BenchmarkDispatchThroughAGraph|BenchmarkDispatchScale|BenchmarkTransientPacking|BenchmarkTLSF|BenchmarkBump'
  ;;
metal)
  pkgs="./tensor/ ./internal/metal/"
  pattern='OnMetal|BenchmarkSubmitAttribution|BenchmarkRenderSubmit'
  ;;
*)
  echo "usage: $0 [cpu|metal]" >&2
  exit 2
  ;;
esac
out=$(go test -run xxx -bench "$pattern" -benchmem -count "$count" $pkgs 2>&1)
printf '%s\n' "$out" > "bench-$which.txt"
echo "| benchmark | ns/op | B/op | allocs/op |"
echo "| --- | ---: | ---: | ---: |"
# The minimum over the runs, which is the robust statistic on a loaded machine.
printf '%s\n' "$out" | awk '
/^Benchmark/ {
  name=$1; sub(/-[0-9]+$/, "", name)
  ns=$3; b=""; a=""
  for (i=4; i<=NF; i++) { if ($i=="B/op") b=$(i-1); if ($i=="allocs/op") a=$(i-1) }
  if (!(name in min) || ns+0 < min[name]+0) { min[name]=ns; bytes[name]=b; allocs[name]=a; order[++n]=name }
}
END { for (i=1; i<=n; i++) { k=order[i]; if (!(k in seen)) { seen[k]=1; printf "| %s | %s | %s | %s |\n", k, min[k], bytes[k], allocs[k] } } }'
