/*
  Arcubate book tool. Builds arcubate.book from PGN game data.
  Licensed under the GNU GPL version 3 or later; see Copying.txt.

    zstd -dc games.pgn.zst | book_tool extract out.agg [minElo] [maxPly]
    book_tool validate shard shards nodes topPositions log.txt [--drop f] [opts] a.agg ... > /dev/null
    book_tool build out.book [--drop f] [--min n] [--cap bytes] a.agg ...
*/

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <iostream>
#include <string>
#include <unordered_set>
#include <vector>

#include "bitboard.h"
#include "book.h"
#include "endgame.h"
#include "movegen.h"
#include "position.h"
#include "search.h"
#include "thread.h"
#include "tt.h"
#include "uci.h"
#include "evaluate.h"
#include "misc.h"
#include "nnue/nnue_kernels.h"

namespace PSQT { void init(); }

namespace {

const std::string StartFEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";

constexpr int   MaxMovesPerPos = 6;
constexpr double MinMoverScore = 0.40; // Below this a move is not worth playing
constexpr double MinShare      = 0.03; // Below this share of a position's games, noise

struct Agg {
  uint64_t key;
  uint32_t n, wins, draws;
  uint16_t move;
  uint16_t pad;
};
static_assert(sizeof(Agg) == 24, "Agg is written to disk");

void merge_sorted(std::vector<Agg>& v) {

  std::sort(v.begin(), v.end(), [](const Agg& a, const Agg& b) {
      return a.key != b.key ? a.key < b.key : a.move < b.move; });

  size_t w = 0;
  for (size_t r = 0; r < v.size(); ++r)
      if (w && v[w-1].key == v[r].key && v[w-1].move == v[r].move)
      {
          v[w-1].n += v[r].n, v[w-1].wins += v[r].wins, v[w-1].draws += v[r].draws;
      }
      else
          v[w++] = v[r];
  v.resize(w);
}

// Resolves a SAN token among the legal moves, MOVE_NONE if nothing matches
Move parse_san(const Position& pos, const char* san) {

  char s[16];
  size_t len = 0;
  for (const char* p = san; *p && len < 15; ++p)
      if (!strchr("+#!?", *p))
          s[len++] = *p;
  s[len] = 0;

  if (!strcmp(s, "O-O") || !strcmp(s, "O-O-O"))
  {
      bool king = !strcmp(s, "O-O");
      for (const auto& m : MoveList<LEGAL>(pos))
          if (type_of(m.move) == CASTLING && (to_sq(m.move) > from_sq(m.move)) == king)
              return m.move;
      return MOVE_NONE;
  }

  PieceType promo = NO_PIECE_TYPE;
  if (char* eq = strchr(s, '='))
  {
      promo = eq[1] == 'Q' ? QUEEN : eq[1] == 'R' ? ROOK : eq[1] == 'B' ? BISHOP : KNIGHT;
      *eq = 0, len = size_t(eq - s);
  }
  if (len < 2)
      return MOVE_NONE;

  Square to = make_square(File(s[len-2] - 'a'), Rank(s[len-1] - '1'));
  PieceType pt = PAWN;
  size_t i = 0;
  if (strchr("KQRBN", s[0]))
  {
      pt = s[0] == 'K' ? KING : s[0] == 'Q' ? QUEEN : s[0] == 'R' ? ROOK : s[0] == 'B' ? BISHOP : KNIGHT;
      i = 1;
  }

  int df = -1, dr = -1;
  for ( ; i + 2 < len; ++i)
      if (s[i] >= 'a' && s[i] <= 'h') df = s[i] - 'a';
      else if (s[i] >= '1' && s[i] <= '8') dr = s[i] - '1';

  for (const auto& m : MoveList<LEGAL>(pos))
  {
      Move mv = m.move;
      if (   type_of(mv) != CASTLING && to_sq(mv) == to
          && type_of(pos.moved_piece(mv)) == pt
          && (df < 0 || file_of(from_sq(mv)) == df)
          && (dr < 0 || rank_of(from_sq(mv)) == dr)
          && (type_of(mv) != PROMOTION || promotion_type(mv) == promo))
          return mv;
  }
  return MOVE_NONE;
}

int header_int(const char* line) { const char* q = strchr(line, '"'); return q ? atoi(q + 1) : 0; }

int cmd_extract(int argc, char** argv) {

  if (argc < 3) return 1;
  const int minElo = argc > 3 ? atoi(argv[3]) : 2300;
  const int maxPly = argc > 4 ? atoi(argv[4]) : 36;
  const size_t Flush = 16000000; // About 400 MB between merges

  std::vector<Agg> buf;
  buf.reserve(Flush + 512);
  size_t base = 0;
  uint64_t seen = 0, kept = 0, failed = 0;

  static char line[1 << 20];
  int wElo = 0, bElo = 0, result = -1;
  bool ok = true;

  while (fgets(line, sizeof(line), stdin))
  {
      if (line[0] == '[')
      {
          if      (!strncmp(line, "[WhiteElo ", 10)) wElo = header_int(line);
          else if (!strncmp(line, "[BlackElo ", 10)) bElo = header_int(line);
          else if (!strncmp(line, "[Result ", 8))
              result = !strncmp(line + 9, "1-0", 3) ? 0 : !strncmp(line + 9, "0-1", 3) ? 1
                     : !strncmp(line + 9, "1/2", 3) ? 2 : -1;
          else if (!strncmp(line, "[Termination ", 13))
              ok = !strstr(line, "Abandoned") && !strstr(line, "Rules");
          else if (!strncmp(line, "[TimeControl ", 13))
          {
              const char* q = strchr(line, '"') + 1;
              const char* plus = strchr(q, '+');
              ok = ok && plus && atoi(q) + 40 * atoi(plus + 1) >= 180;
          }
          continue;
      }

      if (line[0] != '1') // Blank separator, or a game without moves
          continue;

      ++seen;
      if ((seen & 0xFFFFF) == 0)
          fprintf(stderr, "games %llu kept %llu unparsed %llu aggregated %zu\n",
                  (unsigned long long)seen, (unsigned long long)kept,
                  (unsigned long long)failed, buf.size());

      if (ok && result >= 0 && wElo >= minElo && bElo >= minElo)
      {
          ++kept;
          std::deque<StateInfo> states(1);
          Position pos;
          pos.set(StartFEN, false, &states.back(), Threads.main());

          char* save = nullptr;
          int depth = 0; // Comment nesting; Lichess adds {[%eval]} and {[%clk]}

          for (char* t = strtok_r(line, " \n", &save); t && pos.game_ply() < maxPly; t = strtok_r(nullptr, " \n", &save))
          {
              if (t[0] == '{') depth = 1;
              if (depth)
              {
                  if (strchr(t, '}')) depth = 0;
                  continue;
              }
              if (t[0] >= '0' && t[0] <= '9') // Move number or the result
              {
                  if (!strchr(t, '.')) break;
                  continue;
              }
              if (t[0] == '*') break;

              Move m = parse_san(pos, t);
              if (!m) { ++failed; break; }

              Color us = pos.side_to_move();
              Agg a;
              a.key = pos.key(), a.move = uint16_t(m), a.pad = 0, a.n = 1;
              a.wins = result == int(us), a.draws = result == 2;
              buf.push_back(a);

              states.emplace_back();
              pos.do_move(m, states.back());
          }
      }
      wElo = bElo = 0, result = -1, ok = true;

      if (buf.size() >= base + Flush)
      {
          merge_sorted(buf);
          base = buf.size();
          buf.reserve(base + Flush + 512);
      }
  }

  merge_sorted(buf);
  FILE* f = fopen(argv[2], "wb");
  if (!f || fwrite(buf.data(), sizeof(Agg), buf.size(), f) != buf.size() || fclose(f))
  { fprintf(stderr, "write failed\n"); return 1; }

  fprintf(stderr, "done: games %llu kept %llu unparsed %llu entries %zu\n",
          (unsigned long long)seen, (unsigned long long)kept, (unsigned long long)failed, buf.size());
  return 0;
}

std::vector<Agg> load(const std::vector<std::string>& files) {

  std::vector<Agg> v;
  for (const auto& name : files)
  {
      FILE* f = fopen(name.c_str(), "rb");
      if (!f) { fprintf(stderr, "cannot read %s\n", name.c_str()); exit(1); }
      fseek(f, 0, SEEK_END); size_t sz = size_t(ftell(f)) / sizeof(Agg); rewind(f);
      size_t at = v.size();
      v.resize(at + sz);
      if (fread(v.data() + at, sizeof(Agg), sz, f) != sz) exit(1);
      fclose(f);
  }
  merge_sorted(v);
  return v;
}

struct Options_ {
  std::string dropFile;
  uint32_t minCount = 3;
  size_t cap = 32000000;
  std::vector<std::string> inputs;
};

uint64_t pair_id(uint64_t key, uint16_t move) { return key * 0x9E3779B97F4A7C15ULL ^ move; }

std::unordered_set<uint64_t> load_drops(const std::string& file) {

  std::unordered_set<uint64_t> drops;
  if (file.empty()) return drops;
  FILE* f = fopen(file.c_str(), "r");
  if (!f) { fprintf(stderr, "cannot read %s\n", file.c_str()); exit(1); }
  char tag[8]; unsigned long long k; unsigned m;
  char line[256];
  while (fgets(line, sizeof(line), f))
      if (sscanf(line, "%7s %llx %x", tag, &k, &m) == 3 && !strcmp(tag, "D"))
          drops.insert(pair_id(k, uint16_t(m)));
  fclose(f);
  return drops;
}

// Applies count, score and share rules; result stays sorted by key
std::vector<Agg> prune(const std::vector<Agg>& all, uint32_t minCount, const std::unordered_set<uint64_t>& drops) {

  std::vector<Agg> out;
  for (size_t i = 0; i < all.size(); )
  {
      size_t j = i;
      uint64_t total = 0;
      for ( ; j < all.size() && all[j].key == all[i].key; ++j) total += all[j].n;

      std::vector<Agg> group;
      for (size_t k = i; k < j; ++k)
      {
          const Agg& a = all[k];
          double score = (a.wins + 0.5 * a.draws) / a.n;
          if (   a.n >= minCount && score >= MinMoverScore && a.n >= MinShare * double(total)
              && !drops.count(pair_id(a.key, a.move)))
              group.push_back(a);
      }
      std::sort(group.begin(), group.end(), [](const Agg& a, const Agg& b) { return a.n > b.n; });
      if (group.size() > size_t(MaxMovesPerPos)) group.resize(MaxMovesPerPos);
      std::sort(group.begin(), group.end(), [](const Agg& a, const Agg& b) { return a.move < b.move; });
      out.insert(out.end(), group.begin(), group.end());
      i = j;
  }
  return out;
}

// Keeps only entries reachable from the start position through kept moves
struct Walker {
  const std::vector<Agg>& v;
  std::vector<bool> seen;
  std::vector<char> legal;
  std::vector<std::pair<uint64_t, std::string>>* positions; // (games, fen)

  explicit Walker(const std::vector<Agg>& a) : v(a), seen(a.size()), legal(a.size()), positions(nullptr) {}

  void walk(Position& pos) {
    auto first = std::lower_bound(v.begin(), v.end(), pos.key(), [](const Agg& a, uint64_t k) { return a.key < k; });
    size_t i = size_t(first - v.begin());
    if (i == v.size() || v[i].key != pos.key() || seen[i])
        return;
    seen[i] = true;

    uint64_t games = 0;
    MoveList<LEGAL> moves(pos);
    size_t j = i;
    for ( ; j < v.size() && v[j].key == pos.key(); ++j)
        if (moves.contains(Move(v[j].move)))
            legal[j] = 1, games += v[j].n;

    if (positions && games)
        positions->emplace_back(games, pos.fen());

    for (size_t k = i; k < j; ++k)
        if (legal[k])
        {
            StateInfo st;
            pos.do_move(Move(v[k].move), st);
            walk(pos);
            pos.undo_move(Move(v[k].move));
        }
  }
};

std::vector<Agg> reachable(const std::vector<Agg>& v, std::vector<std::pair<uint64_t, std::string>>* positions = nullptr) {

  Walker w(v);
  w.positions = positions;
  StateInfo st;
  Position pos;
  pos.set(StartFEN, false, &st, Threads.main());
  w.walk(pos);

  std::vector<Agg> out;
  for (size_t i = 0; i < v.size(); ++i)
      if (w.legal[i])
          out.push_back(v[i]);
  return out;
}

Options_ parse_opts(int first, int argc, char** argv) {

  Options_ o;
  for (int i = first; i < argc; ++i)
  {
      std::string a = argv[i];
      if      (a == "--drop") o.dropFile = argv[++i];
      else if (a == "--min")  o.minCount = uint32_t(atoi(argv[++i]));
      else if (a == "--cap")  o.cap = size_t(atoll(argv[++i]));
      else o.inputs.push_back(a);
  }
  return o;
}

int cmd_build(int argc, char** argv) {

  if (argc < 4) return 1;
  Options_ o = parse_opts(3, argc, argv);
  auto all = load(o.inputs);
  auto drops = load_drops(o.dropFile);
  fprintf(stderr, "merged entries %zu, dropped by validation %zu\n", all.size(), drops.size());

  const size_t maxEntries = (o.cap - Book::HeaderSize) / sizeof(Book::Entry);
  std::vector<Agg> kept;
  uint32_t minCount = o.minCount;

  for ( ; ; ++minCount)
  {
      kept = reachable(prune(all, minCount, drops));
      fprintf(stderr, "min games %u: %zu entries\n", minCount, kept.size());
      if (kept.size() <= maxEntries) break;
  }

  std::vector<uint32_t> index(Book::IndexSize, 0);
  std::vector<Book::Entry> entries(kept.size());

  for (size_t i = 0; i < kept.size(); )
  {
      size_t j = i;
      double top = 0;
      std::vector<double> raw;
      for ( ; j < kept.size() && kept[j].key == kept[i].key; ++j)
      {
          double score = (kept[j].wins + 0.5 * kept[j].draws) / kept[j].n;
          raw.push_back(std::sqrt(double(kept[j].n)) * score * score);
          top = std::max(top, raw.back());
      }
      for (size_t k = i; k < j; ++k)
      {
          entries[k].key    = Book::entry_key(kept[k].key);
          entries[k].move   = kept[k].move;
          entries[k].weight = uint8_t(std::max(1.0, std::round(255.0 * raw[k - i] / top)));
          entries[k].flags  = 0;
      }
      i = j;
  }

  for (const Agg& a : kept) index[Book::bucket(a.key) + 1]++;
  for (size_t i = 1; i < index.size(); ++i) index[i] += index[i - 1];

  FILE* f = fopen(argv[2], "wb");
  uint64_t count = entries.size();
  if (   !f
      || fwrite(Book::Magic, 1, sizeof(Book::Magic), f) != sizeof(Book::Magic)
      || fwrite(&count, sizeof(count), 1, f) != 1
      || fwrite(index.data(), sizeof(uint32_t), index.size(), f) != index.size()
      || fwrite(entries.data(), sizeof(Book::Entry), entries.size(), f) != entries.size()
      || fclose(f))
  { fprintf(stderr, "write failed\n"); return 1; }

  fprintf(stderr, "book: %zu entries, %.1f MB, min games %u\n", entries.size(),
          double(Book::HeaderSize + entries.size() * sizeof(Book::Entry)) / 1e6, minCount);
  return 0;
}

// Searches a position, optionally limited to one root move; score in centipawns
int search_cp(const std::string& fen, uint64_t nodes, Move only) {

  Search::LimitsType limits;
  limits.startTime = now();
  limits.nodes = int64_t(nodes);
  if (only) limits.searchmoves.push_back(only);

  StateListPtr states(new std::deque<StateInfo>(1));
  Position pos;
  pos.set(fen, false, &states->back(), Threads.main());
  Threads.start_thinking(pos, states, limits, false);
  Threads.main()->wait_for_search_finished();
  return int(Threads.main()->rootMoves[0].score) * 100 / int(PawnValueEg);
}

int cmd_validate(int argc, char** argv) {

  if (argc < 7) return 1;
  const size_t shard = size_t(atoi(argv[2])), shards = size_t(atoi(argv[3]));
  const uint64_t nodes = uint64_t(atoll(argv[4]));
  const size_t top = size_t(atoll(argv[5]));
  const std::string log = argv[6];
  Options_ o = parse_opts(7, argc, argv);

  constexpr int MaxLoss = 50, MinScore = -100; // Centipawns, for the side to move

  Options["Threads"] = std::string("1");
  Options["Contempt"] = std::string("0");
  Options["OwnBook"] = std::string("false");

  auto kept = reachable(prune(load(o.inputs), o.minCount, load_drops(o.dropFile)));
  std::vector<std::pair<uint64_t, std::string>> positions;
  reachable(kept, &positions);
  std::stable_sort(positions.begin(), positions.end(), [](const auto& a, const auto& b) { return a.first > b.first; });
  fprintf(stderr, "entries %zu positions %zu, validating top %zu\n", kept.size(), positions.size(), top);

  // Resume after the last finished position recorded in the log
  size_t start = 0;
  if (FILE* f = fopen(log.c_str(), "r"))
  {
      char line[256]; unsigned long long p;
      while (fgets(line, sizeof(line), f))
          if (sscanf(line, "P %llu", &p) == 1) start = size_t(p) + 1;
      fclose(f);
  }

  FILE* out = fopen(log.c_str(), "a");
  if (!out) return 1;

  for (size_t idx = start; idx < std::min(top, positions.size()); ++idx)
  {
      if (idx % shards != shard) continue;

      const std::string& fen = positions[idx].second;
      StateInfo st;
      Position pos;
      pos.set(fen, false, &st, Threads.main());
      const Key key = pos.key();

      int best = search_cp(fen, nodes, MOVE_NONE);

      auto first = std::lower_bound(kept.begin(), kept.end(), key, [](const Agg& a, uint64_t k) { return a.key < k; });
      for (auto it = first; it != kept.end() && it->key == key; ++it)
      {
          int s = search_cp(fen, nodes, Move(it->move));
          if (best - s > MaxLoss || s < MinScore)
              fprintf(out, "D %llx %x %d %d\n", (unsigned long long)key, it->move, best, s);
      }
      fprintf(out, "P %zu\n", idx);
      fflush(out);
      if (idx % 1000 == shard)
          fprintf(stderr, "validated up to %zu\n", idx);
  }
  fclose(out);
  return 0;
}

} // namespace

int main(int argc, char** argv) {

  CommandLine::init(argc, argv);
  UCI::init(Options);
  PSQT::init();
  Bitboards::init();
  Position::init();
  Bitbases::init();
  Endgames::init();
  Threads.set(1);
  Search::clear();
  Eval::NNUE::Kernels::select("auto");
  Eval::init_NNUE();

  int rc = 1;
  std::string cmd = argc > 1 ? argv[1] : "";
  if      (cmd == "extract")  rc = cmd_extract(argc, argv);
  else if (cmd == "build")    rc = cmd_build(argc, argv);
  else if (cmd == "validate") rc = cmd_validate(argc, argv);
  else fprintf(stderr, "usage: book_tool extract|validate|build ...\n");

  Threads.set(0);
  return rc;
}
