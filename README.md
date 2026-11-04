# Arcubate

A UCI chess engine with a HalfKP network, classical evaluation and an opening book.
It is designed to run well on old hardware: the x86 builds need only AVX, and use
AVX2 or AVX-512 network code when the CPU has it.

Arcubate is a modified version of Stockfish 12 and is licensed under the GNU GPL
version 3 or later; see `Copying.txt`. The Stockfish authors are listed in
`AUTHORS`. The network is Stockfish's `nn-82215d0fd0df.nnue`.

See [Architecture.md](Architecture.md) for the design, including all changes from
Stockfish 12 (made November 2026) and what has been verified on each platform.

## Build

    cd src
    make -j4                 # AVX baseline; see Architecture.md for other targets

The network is embedded in the binary. The engine looks for `arcubate.book` next
to the executable and plays without a book if it is missing.

## UCI options added

- `SIMD`: `auto`, `avx512`, `avx2`, `sse41` or `scalar`. Unsupported levels fall back.
- `OwnBook`, `BookFile`, `BookVariety`, `BookDepth`: control the opening book.

## Book

Game data comes from the [Lichess database](https://database.lichess.org) (CC0).

    cd tools
    make
    ./fetch_games.sh ../bookdata 2200 2019-01 2019-02
    ./book_tool validate 0 2 100000 50000 v0.log ../bookdata/*.agg > /dev/null
    ./book_tool validate 1 2 100000 50000 v1.log ../bookdata/*.agg > /dev/null
    cat v0.log v1.log > drops.txt
    ./book_tool build ../src/arcubate.book --drop drops.txt ../bookdata/*.agg

`validate` is optional. Its arguments are shard, shard count, nodes per search,
number of positions and log file; run the two shards at the same time. The
default size cap is 32 MB.
