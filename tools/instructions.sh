#!/bin/bash
# Instructions per searched node under callgrind, for mixed, network-only and
# classical evaluation. Unlike NPS this does not vary with machine load, though it
# does not see cache misses. Startup cost is removed by subtracting a depth 1 run.
# Usage: [SIMD=level] instructions.sh path/to/arcubate [depth, default 9]
engine=${1:?usage: instructions.sh path/to/arcubate [depth]}
depth=${2:-9}

measure() { # eval type, depth; prints "instructions nodes"
  printf 'setoption name SIMD value %s\nbench 16 1 %s default depth %s\nquit\n' "${SIMD:-auto}" "$2" "$1" |
    valgrind --tool=callgrind --callgrind-out-file=/dev/null "$engine" 2>&1 |
    awk '/Collected :/{i=$NF} /^Nodes searched/{n=$NF} END{print i, n}'
}

for type in mixed NNUE classical; do
  read -r i1 n1 < <(measure $type 1)
  read -r i2 n2 < <(measure $type "$depth")
  awk -v t=$type -v a="$i1" -v b="$n1" -v c="$i2" -v d="$n2" \
    'BEGIN{printf "%-10s %10.0f instructions/node over %d nodes, startup %.0fM\n", t, (c-a)/(d-b), d-b, a/1e6}'
done
