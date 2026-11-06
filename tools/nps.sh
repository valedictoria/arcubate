#!/bin/bash
# Measures speed on one pinned core: mixed, network-only and classical-only bench.
# With two binaries the runs alternate, so heat and background load affect both.
# This is the only speed number a change is judged on.
# Usage: [SIMD=level] nps.sh path/to/base [path/to/test] [runs, default 10] [cpu, default 0]
base=${1:?usage: nps.sh base [test] [runs] [cpu]}
test=${2:-}
runs=${3:-10}
cpu=${4:-0}

one() { # binary, eval type
  printf 'setoption name SIMD value %s\nbench 16 1 13 default depth %s\nquit\n' "${SIMD:-auto}" "$2" | taskset -c "$cpu" "$1" 2>&1 |
    sed -n 's/^Nodes\/second *: *\([0-9]*\).*/\1/p'
}
median() { sort -n | awk '{v[NR]=$1} END{print (NR%2)? v[(NR+1)/2] : int((v[NR/2]+v[NR/2+1])/2)}'; }

for type in mixed NNUE classical; do
  a=""; b=""
  for ((i = 0; i < runs; i++)); do
    a+="$(one "$base" $type)"$'\n'
    [ -n "$test" ] && b+="$(one "$test" $type)"$'\n'
  done
  ma=$(printf '%s' "$a" | median)
  if [ -n "$test" ]; then
    mb=$(printf '%s' "$b" | median)
    awk -v t=$type -v a="$ma" -v b="$mb" 'BEGIN{printf "%-10s base %8d  test %8d  %+.2f%%\n", t, a, b, (b-a)*100/a}'
  else
    printf '%-10s %8d\n' "$type" "$ma"
  fi
done
