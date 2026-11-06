/*
  Stockfish, a UCI chess playing engine derived from Glaurung 2.1
  Copyright (C) 2004-2020 The Stockfish developers (see AUTHORS file)

  Stockfish is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.

  Stockfish is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

// A class that converts the input features of the NNUE evaluation function

#ifndef NNUE_FEATURE_TRANSFORMER_H_INCLUDED
#define NNUE_FEATURE_TRANSFORMER_H_INCLUDED

#include "nnue_common.h"
#include "nnue_kernels.h"
#include "nnue_architecture.h"
#include "features/index_list.h"

#include <cstring> // std::memset()

namespace Eval::NNUE {

  // Input feature converter
  class FeatureTransformer {

   private:
    // Number of output dimensions for one side
    static constexpr IndexType kHalfDimensions = kTransformedFeatureDimensions;

   public:
    // Output type
    using OutputType = TransformedFeatureType;

    // Number of input/output dimensions
    static constexpr IndexType kInputDimensions = RawFeatures::kDimensions;
    static constexpr IndexType kOutputDimensions = kHalfDimensions * 2;

    // Size of forward propagation buffer
    static constexpr std::size_t kBufferSize =
        kOutputDimensions * sizeof(OutputType);

    // Hash value embedded in the evaluation file
    static constexpr std::uint32_t GetHashValue() {
      return RawFeatures::kHashValue ^ kOutputDimensions;
    }

    // Read network parameters
    bool ReadParameters(std::istream& stream) {
      read_little_endian<BiasType>(stream, biases_, kHalfDimensions);
      read_little_endian<WeightType>(stream, weights_, kHalfDimensions * kInputDimensions);
      return !stream.fail();
    }

    // Bring the accumulator up to date from the nearest earlier position that has
    // one, if applying the changes since then is cheaper than a full refresh. Search
    // often skips evaluating a node, so that position may be several plies back.
    bool UpdateAccumulatorIfPossible(const Position& pos) const {
      const StateInfo* st = pos.state();
      int budget = pos.count<ALL_PIECES>() - 2;
      while (!st->accumulator.computed_accumulation) {
        if (!st->previous || (budget -= st->dirtyPiece.dirty_num + 1) < 0)
          return false;
        st = st->previous;
      }
      if (st != pos.state())
        UpdateAccumulator(pos, st);
      return true;
    }

    // Convert input features
    void Transform(const Position& pos, OutputType* output, bool refresh) const {
      if (refresh || !UpdateAccumulatorIfPossible(pos)) {
        RefreshAccumulator(pos);
      }
      const auto& accumulation = pos.state()->accumulator.accumulation;

      const Color perspectives[2] = {pos.side_to_move(), ~pos.side_to_move()};
      for (IndexType p = 0; p < 2; ++p)
        Kernels::active.transform(accumulation[perspectives[p]][0], &output[kHalfDimensions * p], kHalfDimensions);
    }

   private:
    // Calculate cumulative value without using difference calculation
    void RefreshAccumulator(const Position& pos) const {
      auto& accumulator = pos.state()->accumulator;
      IndexType i = 0;
      Features::IndexList active_indices[2];
      RawFeatures::AppendActiveIndices(pos, kRefreshTriggers[i],
                                       active_indices);
      for (Color perspective : { WHITE, BLACK }) {
        std::memcpy(accumulator.accumulation[perspective][i], biases_,
                   kHalfDimensions * sizeof(BiasType));
        for (const auto index : active_indices[perspective]) {
          Kernels::active.add(accumulator.accumulation[perspective][i],
                              &weights_[kHalfDimensions * index], kHalfDimensions);
        }
      }

      accumulator.computed_accumulation = true;
      accumulator.computed_score = false;
    }

    // Calculate cumulative value from an earlier position's accumulator. The budget
    // in UpdateAccumulatorIfPossible keeps the changes within the index list capacity.
    void UpdateAccumulator(const Position& pos, const StateInfo* base) const {
      auto& accumulator = pos.state()->accumulator;
      IndexType i = 0;
      for (Color perspective : { WHITE, BLACK }) {
        Features::IndexList removed, added;
        bool reset = false;
        for (const StateInfo* st = pos.state(); st != base; st = st->previous)
          reset |= st->dirtyPiece.piece[0] == make_piece(perspective, KING);

        if (reset) {
          std::memcpy(accumulator.accumulation[perspective][i], biases_,
                      kHalfDimensions * sizeof(BiasType));
          RawFeatures::CollectActiveIndices(pos, kRefreshTriggers[i], perspective, &added);
        } else {
          std::memcpy(accumulator.accumulation[perspective][i],
                      base->accumulator.accumulation[perspective][i],
                      kHalfDimensions * sizeof(BiasType));
          for (const StateInfo* st = pos.state(); st != base; st = st->previous)
            RawFeatures::CollectChangedIndices(pos, st->dirtyPiece, kRefreshTriggers[i],
                                               perspective, &removed, &added);
          for (const auto index : removed)
            Kernels::active.sub(accumulator.accumulation[perspective][i],
                                &weights_[kHalfDimensions * index], kHalfDimensions);
        }
        for (const auto index : added)
          Kernels::active.add(accumulator.accumulation[perspective][i],
                              &weights_[kHalfDimensions * index], kHalfDimensions);
      }

      accumulator.computed_accumulation = true;
      accumulator.computed_score = false;
    }

    using BiasType = std::int16_t;
    using WeightType = std::int16_t;

    alignas(kCacheLineSize) BiasType biases_[kHalfDimensions];
    alignas(kCacheLineSize)
        WeightType weights_[kHalfDimensions * kInputDimensions];
  };

}  // namespace Eval::NNUE

#endif // #ifndef NNUE_FEATURE_TRANSFORMER_H_INCLUDED
