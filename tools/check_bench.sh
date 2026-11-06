#!/bin/bash
# Checks the bench signature at every SIMD level. Levels the CPU lacks fall back,
# so the script reports which kernels each request actually used.
# Usage: check_bench.sh path/to/arcubate
set -u
SIGNATURE=3624569
engine=${1:?usage: check_bench.sh path/to/arcubate}
status=0
tested=""

# Move generation: known perft counts from the start position and Kiwipete
perft=$(printf 'position startpos\ngo perft 6\nposition fen r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1\ngo perft 5\nquit\n' \
  | "$engine" 2>&1 | sed -n 's/^Nodes searched: *\([0-9]*\).*/\1/p' | tr -d '\r' | tr '\n' ' ')
if [ "$perft" = "119060324 193690690 " ]; then echo "perft ok"; else echo "perft FAIL: $perft"; status=1; fi

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
