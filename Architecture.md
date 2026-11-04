# Arcubate architecture

This is the reference for what Arcubate is and what must stay the same on every
platform. If this file and the code disagree, the code is right and this file
has a bug. Do not describe anything here as working until a check has shown it.

## What Arcubate is

A UCI chess engine forked from **Stockfish 12** (tag `sf_12`, commit `c306d83`),
licensed under the GPL version 3 or later. The search, classical evaluation,
endgame knowledge and network evaluation are Stockfish 12's.

| Change from Stockfish 12 | Where |
|---|---|
| Syzygy tablebase support removed | `src/search.cpp`, `src/thread.cpp`, `src/position.cpp` |
| Network hot loops moved into runtime-selected kernels | `src/nnue/nnue_kernels.*` and the three NNUE layer headers |
| Memory-mapped opening book | `src/book.*`, `tools/` |
| Defaults: `Threads` 2, `Hash` 64 | `src/ucioption.cpp` |
| New options: `SIMD`, `OwnBook`, `BookFile`, `BookVariety`, `BookDepth` | `src/ucioption.cpp` |
| Default build target `x86-64-avx`, plus `x86-64-avx-amd` | `src/Makefile` |

None of these change what the engine searches, except that the book can replace
a search when the game clock is in charge. That is what makes the bench invariant
below possible.

## The invariants

These hold on every platform and every SIMD level. A port that breaks one is
wrong, however well it seems to play.

1. **Bench signature.** `bench` searches exactly **3624569** nodes. This is the
   number in the official Stockfish 12 commit message. It covers the classical
   and network evaluations, because the default bench uses both.
2. **Same integers everywhere.** Every kernel version computes identical results.
   A kernel may only change speed. Never add an approximation, or an instruction
   that rounds or saturates differently (see "Rules for kernels").
3. **Network file.** `nn-82215d0fd0df.nnue`, 21,022,697 bytes, embedded in the
   binary. The first 12 hex digits of its SHA-256 are in the filename, and
   `make` checks them.
4. **Book file.** The same `arcubate.book` works everywhere. See "Opening book".
5. **Baseline first.** The x86-64 baseline stays a first-class build, not a
   degraded fallback. The reference machine is an i5-3230M (Ivy Bridge, AVX, no
   AVX2 or BMI).

`tools/check_bench.sh` enforces 1 and 2, and `tools/isa_audit.sh` checks that
only kernels use instructions newer than the baseline.

Stockfish 12's bench mixes evaluations, so use
`bench 16 1 13 default depth NNUE` to measure network speed alone.

## Evaluation

The network is **HalfKP 256x2-32-32**: 41,024 input features per side, two 256-wide
accumulators concatenated into 512 values, then two hidden layers of 32 and one
output. Inputs are piece placements relative to each king.

Every layer is **integer** arithmetic: 16-bit accumulators, 8-bit weights in the
affine layers, 32-bit sums. **There is no floating point in the network**, so
256-bit AVX1 floating-point instructions have no role in it.

As in Stockfish 12, evaluation is a hybrid: the classical evaluation handles
positions with a large material imbalance, and the network handles the rest. The
classical evaluation and its endgame knowledge are always present.

## SIMD design

The engine is one binary per CPU family. Everything outside the kernels is built
for the baseline only, so a newer instruction cannot reach an older CPU by accident.

There are five kernels, in `src/nnue/nnue_kernels.cpp`:

| Kernel | Purpose |
|---|---|
| `affine` | Weighted sum of `u8` inputs and `s8` weights into `s32` outputs |
| `relu` | Shift `s32` right by 6 and clip to 0..127 as `u8` |
| `add`, `sub` | Update a 256-wide `s16` accumulator |
| `transform` | Clip the `s16` accumulator to 0..127 as `u8` |

### Levels

| Level | Registers | CPU requirement | How it is enabled |
|---|---|---|---|
| `avx512` | 512-bit | AVX-512 F and BW | Function target attribute |
| `avx2` | 256-bit | AVX2 | Function target attribute |
| `sse41` | 128-bit | SSSE3 and SSE4.1 | Baseline flags (VEX encoded on AVX builds) |
| `neon` | 128-bit | ARM NEON | Compile time, ARM builds only |
| `scalar` | None | Any | Always available |

`relu` and `transform` have no 512-bit version, so the `avx512` level uses their
AVX2 versions. `relu` on ARM uses the scalar version.

### Selection and fallback

`Kernels::select()` in `nnue_kernels.cpp` runs at startup and when the `SIMD` UCI
option changes.

- **x86:** `auto` picks `avx512`, then `avx2`, then `sse41`. The checks use
  `__builtin_cpu_supports`.
- **Unsupported request:** asking for a level the CPU lacks falls back to the best
  level it does have. It never crashes. `avx512` falls to `avx2`, then `sse41`.
- **ARM:** every value except `scalar` selects `neon`.
- **`scalar`:** always honoured, on every platform. Useful for comparing results.

**Limit:** the fallback only covers the kernels. The x86 builds use
`-march=sandybridge`, so a CPU without AVX cannot run them at all.

### Deliberately not used

| Feature | Why |
|---|---|
| MMX | Any CPU with AVX already has SSE4.1, which is twice as wide |
| AVX-512 VNNI | Not needed to match results; left out to keep fewer paths to verify |
| PEXT and BMI2 | Slow on AMD Zen 1 and 2, and unavailable at the baseline |
| AVX1 256-bit | It has no integer instructions, and the network is all integer |

### Rules for kernels

1. AVX2 and AVX-512 code goes **only** in `nnue_kernels.cpp`, in functions marked
   `TARGET_AVX2` or `TARGET_AVX512`.
2. Those functions may call intrinsics only. A shared inline function compiled
   with wider instructions could be picked by the linker and crash older CPUs.
3. Never add per-file flags like `-mavx2`, for the same reason.
4. Use unaligned loads and stores, so an alignment bug cannot crash a kernel.
5. Any new kernel needs a scalar version, and must match the bench signature.

`maddubs` can saturate in general, but cannot here: inputs are at most 127 and
weights fit in `s8`, so a pair sums to at most 32,512, below the `s16` limit of
32,767. This is why the SIMD and scalar results agree. A network with inputs
above 127 would break it.

## Platforms

| Release | Make target | Compiler | Instruction baseline | Best kernel |
|---|---|---|---|---|
| Linux x64 | `x86-64-avx` | gcc | AVX, Sandy Bridge | `avx512` |
| Linux AMD | `x86-64-avx-amd` | gcc | Same, `-mtune=znver2` | `avx512` |
| Linux ARM | `armv8` | gcc | ARMv8 with NEON | `neon` |
| macOS x64 | `x86-64-avx` | clang | AVX, Sandy Bridge | `avx512` |
| macOS ARM | `apple-silicon` | clang | Apple Silicon with NEON | `neon` |
| Windows x64 | `x86-64-avx` | mingw | AVX, Sandy Bridge | `avx512` |
| Windows AMD | `x86-64-avx-amd` | mingw | Same, `-mtune=znver2` | `avx512` |
| Windows ARM | `armv8` | clang | ARMv8 with NEON | `neon` |

"AMD" means tuned for Zen, not a different instruction set: AMD and Intel run the
same x86-64 code, and the AMD builds use the same kernels. The bench signature
must match on both.

Linux and Windows x86 builds are static. macOS builds are not, since Apple does
not support fully static binaries.

### Platform-specific code

Keep this list short. Anything else is shared.

| Concern | Linux | macOS | Windows |
|---|---|---|---|
| Book mapping | `mmap` | `mmap` | `CreateFileMapping` |
| Hash table pages | `madvise(MADV_HUGEPAGE)` | Standard allocation | Large pages if permitted |
| Threads | Stockfish `thread_win32_osx.h` | Same | Same |

## Opening book

Built by `tools/book_tool` from Lichess games (CC0), and memory mapped so only the
touched pages are read.

```
Header    8 bytes "ARCBOOK1", 8 bytes entry count (uint64)
Index     65537 uint32: first entry for each top 16 bits of the key
Entries   8 bytes each, sorted: uint32 key, uint16 move, uint8 weight, uint8 flags
```

All integers are little-endian, which holds on every platform above.

The stored key is bits 47..16 of the position's Zobrist key, and the index supplies
the top 16 bits. Every probed move is also checked for legality.

**The book depends on the Zobrist keys.** Changing how keys are generated silently
disables the book without any error. This has already happened once: Stockfish 12
changed the castling keys, so a book built on Stockfish 11 never matched. Any such
change requires rebuilding the book.

The book is skipped for `go depth`, `go nodes`, `go mate`, `go infinite`, pondering,
analysis mode and Chess960.

## Size budget

The shipped engine, book and licence must total under 35 MB.

| Part | Size |
|---|---|
| Linux x64 binary, including the network | 23.6 MB |
| Opening book | About 0.6 MB |

## Verification

| Check | Command | What it proves |
|---|---|---|
| Bench | `tools/check_bench.sh src/arcubate` | Invariants 1 and 2 at every level this CPU supports |
| Instructions | `tools/isa_audit.sh` | Only kernels and guarded C library routines use newer instructions |
| Move generation | `go perft 6` from the start position | 119,060,324 nodes |

### What has actually been run

Keep this honest and current. A kernel that has not run here is unverified.
Results below are from CI run 41917614057 unless noted. Every listed level matched
the bench signature on that platform.

| Release | Levels that ran | Status |
|---|---|---|
| Linux x64 | `avx2`, `sse41`, `scalar` | Passed |
| Linux AMD | `avx512`, `avx2`, `sse41`, `scalar` | Passed |
| Linux ARM | `neon`, `scalar` | Passed |
| macOS x64 | `avx2`, `sse41`, `scalar` | Passed |
| macOS ARM | `neon`, `scalar` | Passed |
| Windows x64 | `avx512`, `avx2`, `sse41`, `scalar` | Passed |
| Windows AMD | `avx512`, `avx2`, `sse41`, `scalar` | Passed |
| Windows ARM | None | Failed to compile: `_mm_malloc` is x86-only. Fix made, not yet confirmed by CI |

The reference i5-3230M ran `sse41` and `scalar` on the Linux x64 build, and also
fell back correctly when asked for `avx2` and `avx512`.

Which levels run depends on the CPU of the runner assigned, so a level listed for
one run may not run in another. The AMD builds also ran on whatever CPU the runner
had, not necessarily an AMD one. Speeds from CI are too noisy to compare.

Run 41917614057 skipped the CI book check, because `arcubate.book` was not yet in the repository.
