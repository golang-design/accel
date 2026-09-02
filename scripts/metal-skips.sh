#!/bin/sh
# Lists every skipped test in a `go test -json` log with its reason, and fails
# when a skip is conditional on the device.
#
# ci-metal.yml promises a Metal device, and ACCEL_REQUIRE_METAL turns the
# "no device" skips into failures -- but a test that skips for a narrower
# reason (no CAMetalLayer, a capability the device happens to report, a device
# that enumerates and will not open) still skips, silently, and the job stays
# green while the path it covers goes unexercised. This makes every skip
# visible in the job log and turns the device-conditional ones red.
#
# Usage: go test -json ./... | tee test.json; sh scripts/metal-skips.sh test.json
set -eu
log="${1:?usage: metal-skips.sh <go test -json log>}"
command -v jq >/dev/null || { echo "metal-skips: jq is required" >&2; exit 2; }

# The reason a test skipped is the last "file_test.go:NN: ..." output line
# emitted under the test's own name before its skip event. jq keeps the last
# such line per test and prints it at the skip.
skips=$(jq -r '
	select(.Test != null) |
	if .Action == "output" and (.Output | test("_test\\.go:[0-9]+: ")) then
		{key: (.Package + "." + .Test), reason: (.Output | sub("^\\s+"; "") | rtrimstr("\n"))}
	elif .Action == "skip" then
		{key: (.Package + "." + .Test), skip: true}
	else empty end' "$log" | jq -rs '
	reduce .[] as $e ({};
		if $e.skip then .[$e.key] += {skipped: true}
		else .[$e.key] += {reason: $e.reason} end) |
	to_entries[] | select(.value.skipped) |
	"\(.key): \(.value.reason // "(no reason recorded)")"')

if [ -z "$skips" ]; then
	echo "metal-skips: no test skipped"
	exit 0
fi
echo "metal-skips: skipped tests and their reasons:"
echo "$skips" | sed 's/^/  /'

# Skips whose reason names an absent device or layer, or a device that will
# not open, are the ones a job promising the device must not accept. A skip
# because the device *reports* a capability (the float-atomic refusal test
# has nothing to observe on a device that has it) is listed above and is not
# a failure: it is a fact about the hardware the job runs on.
conditional=$(echo "$skips" | grep -Ei 'no Metal|no CAMetalLayer|cannot open|no device' || true)
if [ -n "$conditional" ]; then
	echo "::error::device-conditional skips under ACCEL_REQUIRE_METAL:"
	echo "$conditional" | sed 's/^/  /'
	exit 1
fi
