#!/bin/bash
# Checks the bench signature at every SIMD level. Levels the CPU lacks fall back,
# so the script reports which kernels each request actually used.
# Usage: check_bench.sh path/to/arcubate
set -u
SIGNATURE=3624569
engine=${1:?usage: check_bench.sh path/to/arcubate}
status=0
tested=""

for level in auto avx512 avx2 sse41 scalar; do
  output=$(printf 'setoption name SIMD value %s\nbench\nquit\n' "$level" | "$engine" 2>&1)
  used=$(printf '%s\n' "$output" | sed -n 's/.*NNUE SIMD: \([a-z0-9]*\).*/\1/p' | tr -d '\r' | tail -1)
  nodes=$(printf '%s\n' "$output" | sed -n 's/^Nodes searched *: *\([0-9]*\).*/\1/p' | tr -d '\r' | tail -1)
  nps=$(printf '%s\n' "$output" | sed -n 's/^Nodes\/second *: *\([0-9]*\).*/\1/p' | tr -d '\r' | tail -1)
  if [ "$nodes" = "$SIGNATURE" ]; then result=ok; else result=FAIL; status=1; fi
  echo "requested=$level used=${used:-none} nodes=${nodes:-none} nps=${nps:-none} $result"
  tested="$tested ${used:-none}"
done

echo "kernels exercised:$(printf '%s\n' $tested | sort -u | tr '\n' ' ')"
exit $status
