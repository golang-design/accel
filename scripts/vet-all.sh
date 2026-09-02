#!/bin/sh
# Builds and vets the module for every platform it ships on.
#
# Metal code is //go:build darwin, so a build on one platform stops proving
# that the others still compile. vet as well as build, because build does not
# compile tests: a helper defined in a _darwin_test.go file and called from a
# portable one builds everywhere and fails to vet anywhere but a Mac. See
# CONTRIBUTING.md. CI runs this script; run it before pushing.
set -eu
cd "$(dirname "$0")/.."
for os in linux windows darwin; do
	echo "GOOS=$os go build ./..."
	GOOS=$os CGO_ENABLED=0 go build ./...
	echo "GOOS=$os go vet ./..."
	GOOS=$os CGO_ENABLED=0 go vet ./...
done
