#!/bin/bash
# Fails if engine code outside the kernels uses instructions newer than AVX.
# Needs an unstripped x86 build: make -C src build ARCH=x86-64-avx NOSTRIP=1
# The static C library is ignored, because it selects its routines by CPU check.
# tzcnt is allowed. Compilers emit "rep bsf", which objdump shows as tzcnt, because
# older CPUs run it as bsf. The two differ only for a zero input, which is undefined
# for the bit scans that produce it. lzcnt has no such property and stays banned.
set -u
binary=${1:?usage: isa_audit.sh path/to/unstripped/arcubate}

newer='(pext|pdep|lzcnt|shlx|shrx|sarx|rorx|mulx|andn|blsr|blsi|blsmsk|bextr|bzhi|movbe|vfmadd[0-9a-z]*|vpbroadcast[a-z]*|vpgather[a-z]*)'
wide_int='(vpermq|vpermd|vinserti128|vextracti128|vpmaddubsw|vpmaddwd|vpacksswb|vpackssdw|vpaddw|vpsubw|vpaddd|vpmaxsb|vpsraw|vpxor|vpxord)'

found=$(objdump -d --no-show-raw-insn "$binary" | awk -v newer="$newer" -v wide="$wide_int" '
  /^[0-9a-f]+ <.*>:$/ { sym = $2; gsub(/[<>:]/, "", sym); next }
  $0 ~ ("\t" newer "[ \t]") || $0 ~ ("\t" wide "[ \t].*%(ymm|zmm)") || /%zmm/ { seen[sym] = 1 }
  END { for (s in seen) print s }')

# C++ symbols and main are ours or libstdc++; everything else is the C library
ours=$(printf '%s\n' "$found" | grep -E '^(_Z|main$)' | grep -vE '^_Z(NSt|St|NKSt|N9__gnu|N10__cxx|N14__gnu)')
kernels=$(printf '%s\n' "$ours" | grep -E 'Kernels.*(avx2|avx512)')
bad=$(printf '%s\n' "$ours" | grep -vE 'Kernels.*(avx2|avx512)' | grep -v '^$')

echo "kernels with newer instructions: $(printf '%s\n' "$kernels" | grep -c .)"
if [ -n "$bad" ]; then
  echo "FAIL: newer instructions outside the kernels:"
  printf '%s\n' "$bad"
  exit 1
fi
if [ -z "$kernels" ]; then
  echo "FAIL: no AVX2 or AVX-512 kernels found, so this audit is not seeing the code"
  exit 1
fi
echo "ok"
