#!/bin/bash
# Shows which functions use the most instructions. Needs an unstripped build:
# make -C src profile-build NOSTRIP=1. Kernels called through the SIMD table still
# appear under their own names, so their share is visible.
# Usage: [SIMD=level] hotspots.sh path/to/arcubate [eval type, default NNUE] [depth, default 9]
engine=${1:?usage: hotspots.sh path/to/arcubate [type] [depth]}
type=${2:-NNUE}
depth=${3:-9}
out=$(mktemp)
printf 'setoption name SIMD value %s\nbench 16 1 %s default depth %s\nquit\n' "${SIMD:-auto}" "$depth" "$type" |
  valgrind --tool=callgrind --callgrind-out-file="$out" "$engine" > /dev/null 2>&1
callgrind_annotate --threshold=98 "$out" | grep -E '^ *[0-9,]+ \([ 0-9.]+%\)' | head -30 | cut -c1-110
rm -f "$out"
