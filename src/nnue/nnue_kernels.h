/*
  Arcubate, a UCI chess engine derived from Stockfish.
  Licensed under the GNU GPL version 3 or later; see Copying.txt.
*/

#ifndef NNUE_KERNELS_H_INCLUDED
#define NNUE_KERNELS_H_INCLUDED

#include <cstdint>
#include <string>

/// The hot loops of the network, selected once at runtime. All versions compute
/// exactly the same integers, so the choice affects speed and never the result.
/// The rest of the engine is built for the baseline instruction set only.

namespace Eval::NNUE::Kernels {

  struct Table {
    const char* name;
    void (*affine)(const std::uint8_t* input, const std::int8_t* weights, const std::int32_t* biases,
                   std::uint32_t paddedInputs, std::uint32_t outputs, std::int32_t* output);
    void (*relu)(const std::int32_t* input, std::uint8_t* output, std::uint32_t count);
    void (*add)(std::int16_t* accumulator, const std::int16_t* column, std::uint32_t count);
    void (*sub)(std::int16_t* accumulator, const std::int16_t* column, std::uint32_t count);
    void (*transform)(const std::int16_t* accumulator, std::uint8_t* output, std::uint32_t count);
  };

  extern Table active;

  /// Picks the kernels. "auto" takes the best the CPU and OS support; asking for
  /// a level they do not support falls back to the best one that is available.
  /// Returns the name of the level in use.
  std::string select(const std::string& requested);

} // namespace Eval::NNUE::Kernels

#endif // #ifndef NNUE_KERNELS_H_INCLUDED
