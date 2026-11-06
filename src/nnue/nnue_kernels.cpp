/*
  Arcubate, a UCI chess engine derived from Stockfish.
  Licensed under the GNU GPL version 3 or later; see Copying.txt.

  The AVX2 and AVX-512 functions carry target attributes rather than being built
  with extra compiler flags. Only these functions can contain those instructions,
  and they run only after a CPU check, so the binary is safe on an AVX-only CPU.
  They must not call anything but intrinsics.
*/

#include <algorithm>
#include <cstring>

#include "nnue_kernels.h"

#if defined(__x86_64__) || defined(_M_X64)
  #define ARCUBATE_X86
  #include <immintrin.h>
#elif defined(__ARM_NEON) || defined(__aarch64__)
  #define ARCUBATE_NEON
  #include <arm_neon.h>
#endif

namespace Eval::NNUE::Kernels {

namespace {

  constexpr int WeightScaleBits = 6;

  // Portable versions, always available

  void affine_scalar(const std::uint8_t* input, const std::int8_t* weights, const std::int32_t* biases,
                     std::uint32_t paddedInputs, std::uint32_t outputs, std::int32_t* output) {
    for (std::uint32_t i = 0; i < outputs; ++i)
    {
        std::int32_t sum = biases[i];
        const std::int8_t* row = weights + i * paddedInputs;
        for (std::uint32_t j = 0; j < paddedInputs; ++j)
            sum += row[j] * input[j];
        output[i] = sum;
    }
  }

  // The affine layer on inputs that are mostly zero: groups of 4 input bytes that are
  // all zero contribute nothing, so they are skipped. The sum is otherwise the same.

  constexpr std::uint32_t GroupBytes = SparseOutputs * 4;

  inline std::uint32_t read_group(const std::uint8_t* p) {
    std::uint32_t v;
    std::memcpy(&v, p, 4);
    return v;
  }

  void sparse_scalar(const std::uint8_t* input, const std::int8_t* columns, const std::int32_t* biases,
                     std::uint32_t paddedInputs, std::int32_t* output) {
    std::memcpy(output, biases, SparseOutputs * sizeof(std::int32_t));
    for (std::uint32_t j = 0; j < paddedInputs; j += 4)
        if (read_group(input + j))
        {
            const std::int8_t* col = columns + (j / 4) * GroupBytes;
            for (std::uint32_t o = 0; o < SparseOutputs; ++o)
                output[o] += col[o * 4 + 0] * input[j + 0] + col[o * 4 + 1] * input[j + 1]
                           + col[o * 4 + 2] * input[j + 2] + col[o * 4 + 3] * input[j + 3];
        }
  }

  void relu_scalar(const std::int32_t* input, std::uint8_t* output, std::uint32_t count) {
    for (std::uint32_t i = 0; i < count; ++i)
        output[i] = std::uint8_t(std::max(0, std::min(127, input[i] >> WeightScaleBits)));
  }

  void add_scalar(std::int16_t* a, const std::int16_t* c, std::uint32_t count) {
    for (std::uint32_t i = 0; i < count; ++i) a[i] = std::int16_t(a[i] + c[i]);
  }

  void sub_scalar(std::int16_t* a, const std::int16_t* c, std::uint32_t count) {
    for (std::uint32_t i = 0; i < count; ++i) a[i] = std::int16_t(a[i] - c[i]);
  }

  void transform_scalar(const std::int16_t* a, std::uint8_t* output, std::uint32_t count) {
    for (std::uint32_t i = 0; i < count; ++i)
        output[i] = std::uint8_t(std::max<int>(0, std::min<int>(127, a[i])));
  }

#if defined(ARCUBATE_X86)

  // 128-bit SSSE3/SSE4.1 integer code, VEX encoded by the AVX baseline build

  inline std::int32_t hsum128(__m128i s) {
    s = _mm_add_epi32(s, _mm_shuffle_epi32(s, 0x4E));
    s = _mm_add_epi32(s, _mm_shuffle_epi32(s, 0xB1));
    return _mm_cvtsi128_si32(s);
  }

  void affine_sse41(const std::uint8_t* input, const std::int8_t* weights, const std::int32_t* biases,
                    std::uint32_t paddedInputs, std::uint32_t outputs, std::int32_t* output) {
    const __m128i ones = _mm_set1_epi16(1);
    for (std::uint32_t i = 0; i < outputs; ++i)
    {
        const std::int8_t* row = weights + i * paddedInputs;
        __m128i sum = _mm_setzero_si128();
        for (std::uint32_t j = 0; j < paddedInputs; j += 16)
        {
            __m128i p = _mm_maddubs_epi16(_mm_loadu_si128((const __m128i*)(input + j)),
                                          _mm_loadu_si128((const __m128i*)(row + j)));
            sum = _mm_add_epi32(sum, _mm_madd_epi16(p, ones));
        }
        output[i] = hsum128(sum) + biases[i];
    }
  }

  void sparse_sse41(const std::uint8_t* input, const std::int8_t* columns, const std::int32_t* biases,
                    std::uint32_t paddedInputs, std::int32_t* output) {
    constexpr std::uint32_t Regs = SparseOutputs / 4;
    const __m128i ones = _mm_set1_epi16(1);
    const __m128i zero = _mm_setzero_si128();
    __m128i sum[Regs];
    for (std::uint32_t k = 0; k < Regs; ++k)
        sum[k] = _mm_loadu_si128((const __m128i*)(biases + k * 4));

    for (std::uint32_t j = 0; j < paddedInputs; j += 16)
    {
        const __m128i chunk = _mm_loadu_si128((const __m128i*)(input + j));
        unsigned mask = ~_mm_movemask_ps(_mm_castsi128_ps(_mm_cmpeq_epi32(chunk, zero))) & 0xF;
        while (mask)
        {
            const unsigned g = unsigned(__builtin_ctz(mask));
            mask &= mask - 1;
            const std::uint32_t at = j + g * 4;
            const __m128i in = _mm_set1_epi32(int(read_group(input + at)));
            const std::int8_t* col = columns + (at / 4) * GroupBytes;
            for (std::uint32_t k = 0; k < Regs; ++k)
            {
                __m128i p = _mm_maddubs_epi16(in, _mm_loadu_si128((const __m128i*)(col + k * 16)));
                sum[k] = _mm_add_epi32(sum[k], _mm_madd_epi16(p, ones));
            }
        }
    }
    for (std::uint32_t k = 0; k < Regs; ++k)
        _mm_storeu_si128((__m128i*)(output + k * 4), sum[k]);
  }

  void relu_sse41(const std::int32_t* input, std::uint8_t* output, std::uint32_t count) {
    const __m128i zero = _mm_setzero_si128();
    std::uint32_t i = 0;
    for ( ; i + 16 <= count; i += 16)
    {
        const __m128i* in = (const __m128i*)(input + i);
        __m128i w0 = _mm_srai_epi16(_mm_packs_epi32(_mm_loadu_si128(in + 0), _mm_loadu_si128(in + 1)), WeightScaleBits);
        __m128i w1 = _mm_srai_epi16(_mm_packs_epi32(_mm_loadu_si128(in + 2), _mm_loadu_si128(in + 3)), WeightScaleBits);
        _mm_storeu_si128((__m128i*)(output + i), _mm_max_epi8(_mm_packs_epi16(w0, w1), zero));
    }
    relu_scalar(input + i, output + i, count - i);
  }

  void add_sse41(std::int16_t* a, const std::int16_t* c, std::uint32_t count) {
    for (std::uint32_t i = 0; i < count; i += 8)
        _mm_storeu_si128((__m128i*)(a + i), _mm_add_epi16(_mm_loadu_si128((const __m128i*)(a + i)),
                                                          _mm_loadu_si128((const __m128i*)(c + i))));
  }

  void sub_sse41(std::int16_t* a, const std::int16_t* c, std::uint32_t count) {
    for (std::uint32_t i = 0; i < count; i += 8)
        _mm_storeu_si128((__m128i*)(a + i), _mm_sub_epi16(_mm_loadu_si128((const __m128i*)(a + i)),
                                                          _mm_loadu_si128((const __m128i*)(c + i))));
  }

  void transform_sse41(const std::int16_t* a, std::uint8_t* output, std::uint32_t count) {
    const __m128i zero = _mm_setzero_si128();
    for (std::uint32_t i = 0; i < count; i += 16)
    {
        __m128i p = _mm_packs_epi16(_mm_loadu_si128((const __m128i*)(a + i)),
                                    _mm_loadu_si128((const __m128i*)(a + i + 8)));
        _mm_storeu_si128((__m128i*)(output + i), _mm_max_epi8(p, zero));
    }
  }

  // AVX2

  #define TARGET_AVX2 __attribute__((target("avx2")))

  TARGET_AVX2 void affine_avx2(const std::uint8_t* input, const std::int8_t* weights, const std::int32_t* biases,
                               std::uint32_t paddedInputs, std::uint32_t outputs, std::int32_t* output) {
    const __m256i ones = _mm256_set1_epi16(1);
    for (std::uint32_t i = 0; i < outputs; ++i)
    {
        const std::int8_t* row = weights + i * paddedInputs;
        __m256i sum = _mm256_setzero_si256();
        for (std::uint32_t j = 0; j < paddedInputs; j += 32)
        {
            __m256i p = _mm256_maddubs_epi16(_mm256_loadu_si256((const __m256i*)(input + j)),
                                             _mm256_loadu_si256((const __m256i*)(row + j)));
            sum = _mm256_add_epi32(sum, _mm256_madd_epi16(p, ones));
        }
        __m128i s = _mm_add_epi32(_mm256_castsi256_si128(sum), _mm256_extracti128_si256(sum, 1));
        s = _mm_add_epi32(s, _mm_shuffle_epi32(s, 0x4E));
        s = _mm_add_epi32(s, _mm_shuffle_epi32(s, 0xB1));
        output[i] = _mm_cvtsi128_si32(s) + biases[i];
    }
  }

  TARGET_AVX2 void sparse_avx2(const std::uint8_t* input, const std::int8_t* columns, const std::int32_t* biases,
                               std::uint32_t paddedInputs, std::int32_t* output) {
    constexpr std::uint32_t Regs = SparseOutputs / 8;
    const __m256i ones = _mm256_set1_epi16(1);
    const __m256i zero = _mm256_setzero_si256();
    __m256i sum[Regs];
    for (std::uint32_t k = 0; k < Regs; ++k)
        sum[k] = _mm256_loadu_si256((const __m256i*)(biases + k * 8));

    for (std::uint32_t j = 0; j < paddedInputs; j += 32)
    {
        const __m256i chunk = _mm256_loadu_si256((const __m256i*)(input + j));
        unsigned mask = ~_mm256_movemask_ps(_mm256_castsi256_ps(_mm256_cmpeq_epi32(chunk, zero))) & 0xFF;
        while (mask)
        {
            const unsigned g = unsigned(__builtin_ctz(mask));
            mask &= mask - 1;
            const std::uint32_t at = j + g * 4;
            std::uint32_t group;
            __builtin_memcpy(&group, input + at, 4);
            const __m256i in = _mm256_set1_epi32(int(group));
            const std::int8_t* col = columns + (at / 4) * GroupBytes;
            for (std::uint32_t k = 0; k < Regs; ++k)
            {
                __m256i p = _mm256_maddubs_epi16(in, _mm256_loadu_si256((const __m256i*)(col + k * 32)));
                sum[k] = _mm256_add_epi32(sum[k], _mm256_madd_epi16(p, ones));
            }
        }
    }
    for (std::uint32_t k = 0; k < Regs; ++k)
        _mm256_storeu_si256((__m256i*)(output + k * 8), sum[k]);
  }

  TARGET_AVX2 void relu_avx2(const std::int32_t* input, std::uint8_t* output, std::uint32_t count) {
    const __m256i zero = _mm256_setzero_si256();
    const __m256i order = _mm256_set_epi32(7, 3, 6, 2, 5, 1, 4, 0);
    std::uint32_t i = 0;
    for ( ; i + 32 <= count; i += 32)
    {
        const __m256i* in = (const __m256i*)(input + i);
        __m256i w0 = _mm256_srai_epi16(_mm256_packs_epi32(_mm256_loadu_si256(in + 0), _mm256_loadu_si256(in + 1)), WeightScaleBits);
        __m256i w1 = _mm256_srai_epi16(_mm256_packs_epi32(_mm256_loadu_si256(in + 2), _mm256_loadu_si256(in + 3)), WeightScaleBits);
        _mm256_storeu_si256((__m256i*)(output + i),
            _mm256_permutevar8x32_epi32(_mm256_max_epi8(_mm256_packs_epi16(w0, w1), zero), order));
    }
    for ( ; i < count; ++i)
    {
        int v = input[i] >> WeightScaleBits;
        output[i] = std::uint8_t(v < 0 ? 0 : v > 127 ? 127 : v);
    }
  }

  TARGET_AVX2 void add_avx2(std::int16_t* a, const std::int16_t* c, std::uint32_t count) {
    for (std::uint32_t i = 0; i < count; i += 16)
        _mm256_storeu_si256((__m256i*)(a + i), _mm256_add_epi16(_mm256_loadu_si256((const __m256i*)(a + i)),
                                                                _mm256_loadu_si256((const __m256i*)(c + i))));
  }

  TARGET_AVX2 void sub_avx2(std::int16_t* a, const std::int16_t* c, std::uint32_t count) {
    for (std::uint32_t i = 0; i < count; i += 16)
        _mm256_storeu_si256((__m256i*)(a + i), _mm256_sub_epi16(_mm256_loadu_si256((const __m256i*)(a + i)),
                                                                _mm256_loadu_si256((const __m256i*)(c + i))));
  }

  TARGET_AVX2 void transform_avx2(const std::int16_t* a, std::uint8_t* output, std::uint32_t count) {
    const __m256i zero = _mm256_setzero_si256();
    for (std::uint32_t i = 0; i < count; i += 32)
    {
        __m256i p = _mm256_packs_epi16(_mm256_loadu_si256((const __m256i*)(a + i)),
                                       _mm256_loadu_si256((const __m256i*)(a + i + 16)));
        _mm256_storeu_si256((__m256i*)(output + i), _mm256_permute4x64_epi64(_mm256_max_epi8(p, zero), 0xD8));
    }
  }

  // AVX-512 (F and BW). The narrow loops gain nothing from it, so only the affine
  // layer and the accumulator updates have their own versions.

  #define TARGET_AVX512 __attribute__((target("avx512f,avx512bw,avx2")))

  TARGET_AVX512 void affine_avx512(const std::uint8_t* input, const std::int8_t* weights, const std::int32_t* biases,
                                   std::uint32_t paddedInputs, std::uint32_t outputs, std::int32_t* output) {
    const __m512i ones = _mm512_set1_epi16(1);
    const __m256i ones256 = _mm256_set1_epi16(1);
    for (std::uint32_t i = 0; i < outputs; ++i)
    {
        const std::int8_t* row = weights + i * paddedInputs;
        __m512i sum = _mm512_setzero_si512();
        std::uint32_t j = 0;
        for ( ; j + 64 <= paddedInputs; j += 64)
        {
            __m512i p = _mm512_maddubs_epi16(_mm512_loadu_si512(input + j), _mm512_loadu_si512(row + j));
            sum = _mm512_add_epi32(sum, _mm512_madd_epi16(p, ones));
        }
        std::int32_t total = _mm512_reduce_add_epi32(sum);
        if (j < paddedInputs) // Inputs are padded to 32 bytes, so at most one 256-bit chunk
        {
            __m256i p = _mm256_maddubs_epi16(_mm256_loadu_si256((const __m256i*)(input + j)),
                                             _mm256_loadu_si256((const __m256i*)(row + j)));
            p = _mm256_madd_epi16(p, ones256);
            __m128i s = _mm_add_epi32(_mm256_castsi256_si128(p), _mm256_extracti128_si256(p, 1));
            s = _mm_add_epi32(s, _mm_shuffle_epi32(s, 0x4E));
            s = _mm_add_epi32(s, _mm_shuffle_epi32(s, 0xB1));
            total += _mm_cvtsi128_si32(s);
        }
        output[i] = total + biases[i];
    }
  }

  TARGET_AVX512 void sparse_avx512(const std::uint8_t* input, const std::int8_t* columns, const std::int32_t* biases,
                                   std::uint32_t paddedInputs, std::int32_t* output) {
    constexpr std::uint32_t Regs = SparseOutputs / 16;
    const __m512i ones = _mm512_set1_epi16(1);
    const __m256i zero = _mm256_setzero_si256();
    __m512i sum[Regs];
    for (std::uint32_t k = 0; k < Regs; ++k)
        sum[k] = _mm512_loadu_si512(biases + k * 16);

    for (std::uint32_t j = 0; j < paddedInputs; j += 32)
    {
        const __m256i chunk = _mm256_loadu_si256((const __m256i*)(input + j));
        unsigned mask = ~_mm256_movemask_ps(_mm256_castsi256_ps(_mm256_cmpeq_epi32(chunk, zero))) & 0xFF;
        while (mask)
        {
            const unsigned g = unsigned(__builtin_ctz(mask));
            mask &= mask - 1;
            const std::uint32_t at = j + g * 4;
            std::uint32_t group;
            __builtin_memcpy(&group, input + at, 4);
            const __m512i in = _mm512_set1_epi32(int(group));
            const std::int8_t* col = columns + (at / 4) * GroupBytes;
            for (std::uint32_t k = 0; k < Regs; ++k)
            {
                __m512i p = _mm512_maddubs_epi16(in, _mm512_loadu_si512(col + k * 64));
                sum[k] = _mm512_add_epi32(sum[k], _mm512_madd_epi16(p, ones));
            }
        }
    }
    for (std::uint32_t k = 0; k < Regs; ++k)
        _mm512_storeu_si512(output + k * 16, sum[k]);
  }

  TARGET_AVX512 void add_avx512(std::int16_t* a, const std::int16_t* c, std::uint32_t count) {
    for (std::uint32_t i = 0; i < count; i += 32)
        _mm512_storeu_si512(a + i, _mm512_add_epi16(_mm512_loadu_si512(a + i), _mm512_loadu_si512(c + i)));
  }

  TARGET_AVX512 void sub_avx512(std::int16_t* a, const std::int16_t* c, std::uint32_t count) {
    for (std::uint32_t i = 0; i < count; i += 32)
        _mm512_storeu_si512(a + i, _mm512_sub_epi16(_mm512_loadu_si512(a + i), _mm512_loadu_si512(c + i)));
  }

#elif defined(ARCUBATE_NEON)

  void affine_neon(const std::uint8_t* input, const std::int8_t* weights, const std::int32_t* biases,
                   std::uint32_t paddedInputs, std::uint32_t outputs, std::int32_t* output) {
    for (std::uint32_t i = 0; i < outputs; ++i)
    {
        const std::int8_t* row = weights + i * paddedInputs;
        int32x4_t sum = vdupq_n_s32(0);
        for (std::uint32_t j = 0; j < paddedInputs; j += 16)
        {
            int16x8_t lo = vmulq_s16(vreinterpretq_s16_u16(vmovl_u8(vld1_u8(input + j))), vmovl_s8(vld1_s8(row + j)));
            int16x8_t hi = vmulq_s16(vreinterpretq_s16_u16(vmovl_u8(vld1_u8(input + j + 8))), vmovl_s8(vld1_s8(row + j + 8)));
            sum = vpadalq_s16(sum, lo);
            sum = vpadalq_s16(sum, hi);
        }
        output[i] = vgetq_lane_s32(sum, 0) + vgetq_lane_s32(sum, 1)
                  + vgetq_lane_s32(sum, 2) + vgetq_lane_s32(sum, 3) + biases[i];
    }
  }

  void sparse_neon(const std::uint8_t* input, const std::int8_t* columns, const std::int32_t* biases,
                   std::uint32_t paddedInputs, std::int32_t* output) {
    constexpr std::uint32_t Regs = SparseOutputs / 4;
    int32x4_t sum[Regs];
    for (std::uint32_t k = 0; k < Regs; ++k)
        sum[k] = vld1q_s32(biases + k * 4);

    for (std::uint32_t j = 0; j < paddedInputs; j += 4)
    {
        const std::uint32_t group = read_group(input + j);
        if (!group)
            continue;
        // Inputs are at most 127, so they are safe to treat as signed
        const int8x16_t in = vreinterpretq_s8_u32(vdupq_n_u32(group));
        const std::int8_t* col = columns + (j / 4) * GroupBytes;
        for (std::uint32_t k = 0; k < Regs; ++k)
        {
            const int8x16_t w = vld1q_s8(col + k * 16);
            const int16x8_t lo = vmull_s8(vget_low_s8(in), vget_low_s8(w));
            const int16x8_t hi = vmull_s8(vget_high_s8(in), vget_high_s8(w));
            sum[k] = vpadalq_s16(sum[k], vcombine_s16(vpadd_s16(vget_low_s16(lo), vget_high_s16(lo)),
                                                       vpadd_s16(vget_low_s16(hi), vget_high_s16(hi))));
        }
    }
    for (std::uint32_t k = 0; k < Regs; ++k)
        vst1q_s32(output + k * 4, sum[k]);
  }

  void relu_neon(const std::int32_t* input, std::uint8_t* output, std::uint32_t count) {
    const int8x16_t zero = vdupq_n_s8(0);
    std::uint32_t i = 0;
    for ( ; i + 16 <= count; i += 16)
    {
        int16x8_t w0 = vcombine_s16(vqshrn_n_s32(vld1q_s32(input + i), WeightScaleBits),
                                    vqshrn_n_s32(vld1q_s32(input + i + 4), WeightScaleBits));
        int16x8_t w1 = vcombine_s16(vqshrn_n_s32(vld1q_s32(input + i + 8), WeightScaleBits),
                                    vqshrn_n_s32(vld1q_s32(input + i + 12), WeightScaleBits));
        vst1q_u8(output + i, vreinterpretq_u8_s8(vmaxq_s8(vcombine_s8(vqmovn_s16(w0), vqmovn_s16(w1)), zero)));
    }
    relu_scalar(input + i, output + i, count - i);
  }

  void add_neon(std::int16_t* a, const std::int16_t* c, std::uint32_t count) {
    for (std::uint32_t i = 0; i < count; i += 8)
        vst1q_s16(a + i, vaddq_s16(vld1q_s16(a + i), vld1q_s16(c + i)));
  }

  void sub_neon(std::int16_t* a, const std::int16_t* c, std::uint32_t count) {
    for (std::uint32_t i = 0; i < count; i += 8)
        vst1q_s16(a + i, vsubq_s16(vld1q_s16(a + i), vld1q_s16(c + i)));
  }

  void transform_neon(const std::int16_t* a, std::uint8_t* output, std::uint32_t count) {
    const int8x8_t zero = vdup_n_s8(0);
    for (std::uint32_t i = 0; i < count; i += 8)
        vst1_u8(output + i, vreinterpret_u8_s8(vmax_s8(vqmovn_s16(vld1q_s16(a + i)), zero)));
  }

#endif

  const Table Scalar = { "scalar", sparse_scalar, affine_scalar, relu_scalar, add_scalar, sub_scalar, transform_scalar };

#if defined(ARCUBATE_X86)
  const Table SSE41  = { "sse41",  sparse_sse41,  affine_sse41,  relu_sse41,  add_sse41,  sub_sse41,  transform_sse41 };
  const Table AVX2   = { "avx2",   sparse_avx2,   affine_avx2,   relu_avx2,   add_avx2,   sub_avx2,   transform_avx2 };
  const Table AVX512 = { "avx512", sparse_avx512, affine_avx512, relu_avx2,   add_avx512, sub_avx512, transform_avx2 };
#elif defined(ARCUBATE_NEON)
  const Table NEON   = { "neon",   sparse_neon,   affine_neon,   relu_neon,   add_neon,   sub_neon,   transform_neon };
#endif

} // namespace

Table active = Scalar;

std::string select(const std::string& requested) {

  active = Scalar;

  if (requested == "scalar")
      return active.name;

#if defined(ARCUBATE_X86)
  __builtin_cpu_init();

  // These checks also confirm that the OS saves the wider registers
  const bool hasAVX2   = __builtin_cpu_supports("avx2");
  const bool hasAVX512 = hasAVX2 && __builtin_cpu_supports("avx512f") && __builtin_cpu_supports("avx512bw");

  if (hasAVX512 && (requested == "auto" || requested == "avx512"))
      active = AVX512;
  else if (hasAVX2 && (requested == "auto" || requested == "avx512" || requested == "avx2"))
      active = AVX2;
  else
      active = SSE41;
#elif defined(ARCUBATE_NEON)
  active = NEON;
#endif

  return active.name;
}

} // namespace Eval::NNUE::Kernels
