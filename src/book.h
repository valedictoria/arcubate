/*
  Arcubate, a UCI chess engine derived from Stockfish 11.
  Licensed under the GNU GPL version 3 or later; see Copying.txt.
*/

#ifndef BOOK_H_INCLUDED
#define BOOK_H_INCLUDED

#include <cstdint>
#include <string>

#include "position.h"

/// The book file is sorted and memory mapped, so only the pages a probe touches
/// are ever read. Layout:
///
///   Header   magic "ARCBOOK1", then the entry count as a uint64_t
///   Index    65537 uint32_t: first entry for each top 16 bits of the key
///   Entries  sorted by (key16, key32), 8 bytes each
///
/// Bits 47..16 of the position key are stored, which with the index gives 48
/// bits of key per entry. Every move is also checked for legality when probing.

namespace Book {

struct Entry {
  uint32_t key;    // Bits 47..16 of the position key
  uint16_t move;   // A Move as used by the engine
  uint8_t  weight; // 1..255, higher is better
  uint8_t  flags;  // Reserved, 0
};

static_assert(sizeof(Entry) == 8, "Book entry must be 8 bytes");

constexpr char     Magic[8]   = {'A','R','C','B','O','O','K','1'};
constexpr size_t   IndexSize  = 65537;
constexpr size_t   HeaderSize = 16 + IndexSize * sizeof(uint32_t);

inline uint32_t entry_key(Key k) { return uint32_t(k >> 16); }
inline uint32_t bucket(Key k)    { return uint32_t(k >> 48); }

void init(const std::string& file);
Move probe(const Position& pos);

} // namespace Book

#endif // #ifndef BOOK_H_INCLUDED
