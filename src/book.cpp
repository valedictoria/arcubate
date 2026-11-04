/*
  Arcubate, a UCI chess engine derived from Stockfish.
  Licensed under the GNU GPL version 3 or later; see Copying.txt.
*/

#include <algorithm>
#include <chrono>
#include <cstring>
#include <random>

#if defined(_WIN32)
  #ifndef NOMINMAX
    #define NOMINMAX
  #endif
  #include <windows.h>
#else
  #include <fcntl.h>
  #include <sys/mman.h>
  #include <sys/stat.h>
  #include <unistd.h>
#endif

#include "book.h"
#include "misc.h"
#include "movegen.h"
#include "uci.h"

namespace {

  const uint8_t*     Map     = nullptr;
  const uint32_t*    Index   = nullptr;
  const Book::Entry* Entries = nullptr;

#if defined(_WIN32)
  HANDLE File = INVALID_HANDLE_VALUE, Mapping = nullptr;

  void unmap() {
    if (Map)     UnmapViewOfFile(Map);
    if (Mapping) CloseHandle(Mapping);
    if (File != INVALID_HANDLE_VALUE) CloseHandle(File);
    File = INVALID_HANDLE_VALUE, Mapping = nullptr;
  }

  const uint8_t* map_file(const std::string& path, size_t& size) {
    File = CreateFileA(path.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING,
                       FILE_ATTRIBUTE_NORMAL, nullptr);
    LARGE_INTEGER s;
    if (File == INVALID_HANDLE_VALUE || !GetFileSizeEx(File, &s) || !s.QuadPart)
        return nullptr;
    Mapping = CreateFileMappingA(File, nullptr, PAGE_READONLY, 0, 0, nullptr);
    if (!Mapping)
        return nullptr;
    size = size_t(s.QuadPart);
    return static_cast<const uint8_t*>(MapViewOfFile(Mapping, FILE_MAP_READ, 0, 0, 0));
  }
#else
  size_t MapSize = 0;

  void unmap() {
    if (Map)
        munmap(const_cast<uint8_t*>(Map), MapSize);
    MapSize = 0;
  }

  const uint8_t* map_file(const std::string& path, size_t& size) {
    int fd = open(path.c_str(), O_RDONLY);
    if (fd < 0)
        return nullptr;
    struct stat st;
    void* p = MAP_FAILED;
    if (!fstat(fd, &st) && st.st_size > 0)
    {
        size = size_t(st.st_size);
        p = mmap(nullptr, size, PROT_READ, MAP_PRIVATE, fd, 0);
    }
    close(fd);
    if (p == MAP_FAILED)
        return nullptr;
    MapSize = size;
    return static_cast<const uint8_t*>(p);
  }
#endif

  bool absolute_path(const std::string& f) {
    return !f.empty() && (f[0] == '/' || f[0] == '\\' || (f.size() > 1 && f[1] == ':'));
  }

} // namespace


/// Book::init() maps a book file, replacing any previous one. A relative name is
/// looked up next to the executable. A missing or malformed file just leaves the
/// engine without a book.

void Book::init(const std::string& file) {

  unmap();
  Map = nullptr, Index = nullptr, Entries = nullptr;

  size_t size = 0;
  const uint8_t* m = map_file(absolute_path(file) ? file : CommandLine::binaryDirectory + file, size);

  if (!m)
      return;

  Map = m;

  uint64_t count = 0;
  if (size >= HeaderSize)
      std::memcpy(&count, m + 8, sizeof(count));

  if (   size >= HeaderSize
      && !std::memcmp(m, Magic, sizeof(Magic))
      && size == HeaderSize + count * sizeof(Entry))
  {
      Index = reinterpret_cast<const uint32_t*>(m + 16);
      Entries = reinterpret_cast<const Entry*>(m + HeaderSize);
  }
}


/// Book::probe() returns a move for the position, or MOVE_NONE. BookVariety 0
/// plays only the top weighted move, 100 uses every move in proportion to its
/// weight, and values between drop moves too far below the top one.

Move Book::probe(const Position& pos) {

  if (!Entries || pos.game_ply() >= int(Options["BookDepth"]))
      return MOVE_NONE;

  static std::mt19937 rng(uint32_t(std::chrono::steady_clock::now().time_since_epoch().count()));

  const Key k = pos.key();
  const Entry* first = Entries + Index[bucket(k)];
  const Entry* last  = Entries + Index[bucket(k) + 1];
  const uint32_t key = entry_key(k);

  first = std::lower_bound(first, last, key, [](const Entry& e, uint32_t v) { return e.key < v; });

  Move moves[64];
  int weights[64], count = 0, top = 0;

  for ( ; first != last && first->key == key && count < 64; ++first)
  {
      Move m = Move(first->move);

      if (MoveList<LEGAL>(pos).contains(m))
      {
          moves[count] = m;
          weights[count++] = first->weight;
          top = std::max(top, int(first->weight));
      }
  }

  if (!count)
      return MOVE_NONE;

  const int variety = int(Options["BookVariety"]);
  int total = 0;

  for (int i = 0; i < count; ++i)
  {
      if (weights[i] * 100 < top * (100 - variety))
          weights[i] = 0;
      total += weights[i];
  }

  int pick = int(rng() % uint32_t(total));

  for (int i = 0; i < count; ++i)
      if ((pick -= weights[i]) < 0)
          return moves[i];

  return moves[0];
}
