/*******************************************************************************
* Copyright (c) 2018-2024 Cadence Design Systems, Inc.
*
* Permission is hereby granted, free of charge, to any person obtaining
* a copy of this software and associated documentation files (the
* "Software"), to use this Software with Cadence processor cores only and
* not with any other processors and platforms, subject to
* the following conditions:
*
* The above copyright notice and this permission notice shall be included
* in all copies or substantial portions of the Software.
*
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

******************************************************************************/
// Copyright 2018 The Gemmlowp Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// pack_msa.h: optimized MSA specializations of the templates in pack.h.

#ifndef GEMMLOWP_INTERNAL_PACK_MSA_H_
#define GEMMLOWP_INTERNAL_PACK_MSA_H_

#include "pack.h"

#include <msa.h>

namespace gemmlowp {

typedef SideMap<const std::uint8_t, SideMapOrder::WidthMajor>
    WidthMajorUint8SideMap;

template <int Cells>
using DepthMajorSideFormatNCells4x2 = KernelSideFormat<CellFormat<4, 2>, Cells>;

template <int Cells>
class PackingRegisterBlock<
    WidthMajorUint8SideMap,
    PackedSideBlock<DepthMajorSideFormatNCells4x2<Cells>>>
    : public PackingRegisterBlockBase<
          WidthMajorUint8SideMap,
          PackedSideBlock<DepthMajorSideFormatNCells4x2<Cells>>> {
 public:
  typedef DepthMajorSideFormatNCells4x2<Cells> KernelSideFormat;
  typedef typename KernelSideFormat::Cell CellFormat;
  static constexpr int kCells = KernelSideFormat::kCells;
  static const int kCellWidth = CellFormat::kWidth;
  static const int kKernelWidth = CellFormat::kWidth * kCells;
  static const int kCellDepth = CellFormat::kDepth;
  static const int kCellSize = CellFormat::kSize;

  void Pack(PackedSideBlock<KernelSideFormat>* dst, int start_width) {
    std::uint8_t* dst_ptr = dst->current_data();
    const std::uint8_t* const src_ptr = this->complete_src_.data();
    const int stride = this->complete_src_.stride();
    // Load source WidthMajor data
    v16i8 src_lines[4 * kCells];
    for (int i = 0; i < 4 * kCells; i++) {
      src_lines[i] = __builtin_msa_ld_b(
          const_cast<std::uint8_t*>(src_ptr + i * stride), 0);
    }
    // Reorder the data within registers to make DepthMajor 4x2 cells
    v16i8 src_lines_intertwined_2x[2 * kCells][2];
    for (int i = 0; i < kCells; i++) {
      src_lines_intertwined_2x[2 * i][0] =
          __builtin_msa_ilvr_b(src_lines[4 * i + 2], src_lines[4 * i]);
      src_lines_intertwined_2x[2 * i][1] =
          __builtin_msa_ilvl_b(src_lines[4 * i + 2], src_lines[4 * i]);
      src_lines_intertwined_2x[2 * i + 1][0] =
          __builtin_msa_ilvr_b(src_lines[4 * i + 3], src_lines[4 * i + 1]);
      src_lines_intertwined_2x[2 * i + 1][1] =
          __builtin_msa_ilvl_b(src_lines[4 * i + 3], src_lines[4 * i + 1]);
    }
    v16i8 src_lines_intertwined_4x[2 * kCells][2];
    for (int i = 0; i < kCells; i++) {
      src_lines_intertwined_4x[2 * i][0] =
          __builtin_msa_ilvr_b(src_lines_intertwined_2x[2 * i + 1][0],
                               src_lines_intertwined_2x[2 * i][0]);
      src_lines_intertwined_4x[2 * i][1] =
          __builtin_msa_ilvl_b(src_lines_intertwined_2x[2 * i + 1][0],
                               src_lines_intertwined_2x[2 * i][0]);
      src_lines_intertwined_4x[2 * i + 1][0] =
          __builtin_msa_ilvr_b(src_lines_intertwined_2x[2 * i + 1][1],
                               src_lines_intertwined_2x[2 * i][1]);
      src_lines_intertwined_4x[2 * i + 1][1] =
          __builtin_msa_ilvl_b(src_lines_intertwined_2x[2 * i + 1][1],
                               src_lines_intertwined_2x[2 * i][1]);
    }
    // Store the resulting DepthMajor 4x2 cells in the destination packed block
    for (int outer = 0; outer < 2; outer++) {
      for (int inner = 0; inner < 2; inner++) {
        if (kCells % 2 == 0) {
          for (int cell = 0; cell < kCells; cell += 2) {
            v2i64 tmp = __builtin_msa_ilvr_d(
                reinterpret_cast<v2i64>(
                    src_lines_intertwined_4x[2 * (cell + 1) + outer][inner]),
                reinterpret_cast<v2i64>(
                    src_lines_intertwined_4x[2 * cell + outer][inner]));
            __builtin_msa_st_b(reinterpret_cast<v16i8>(tmp), dst_ptr, 0);
            dst_ptr += 16;
          }
          for (int cell = 0; cell < kCells; cell += 2) {
            v2i64 tmp = __builtin_msa_ilvl_d(
                reinterpret_cast<v2i64>(
                    src_lines_intertwined_4x[2 * (cell + 1) + outer][inner]),
                reinterpret_cast<v2i64>(
                    src_lines_intertwined_4x[2 * cell + outer][inner]));
            __builtin_msa_st_b(reinterpret_cast<v16i8>(tmp), dst_ptr, 0);
            dst_ptr += 16;
          }
        } else {
          // Store even number of low vector halves.
          for (int cell = 0; cell < kCells - 1; cell += 2) {
            v2i64 tmp = __builtin_msa_ilvr_d(
                reinterpret_cast<v2i64>(
                    src_lines_intertwined_4x[2 * (cell + 1) + outer][inner]),
                reinterpret_cast<v2i64>(
                    src_lines_intertwined_4x[2 * cell + outer][inner]));
            __builtin_msa_st_b(reinterpret_cast<v16i8>(tmp), dst_ptr, 0);
            dst_ptr += 16;
          }
          // Store last low half and first high half.
          v2i64 tmp = reinterpret_cast<v2i64>(
              src_lines_intertwined_4x[2 * 0 + outer][inner]);
          tmp = __builtin_msa_insve_d(
              tmp, 0,
              reinterpret_cast<v2i64>(
                  src_lines_intertwined_4x[2 * (kCells - 1) + outer][inner]));
          __builtin_msa_st_b(reinterpret_cast<v16i8>(tmp), dst_ptr, 0);
          dst_ptr += 16;
          // Store even number of high vector halves.
          for (int cell = 1; cell < kCells; cell += 2) {
            v2i64 tmp = __builtin_msa_ilvl_d(
                reinterpret_cast<v2i64>(
                    src_lines_intertwined_4x[2 * (cell + 1) + outer][inner]),
                reinterpret_cast<v2i64>(
                    src_lines_intertwined_4x[2 * cell + outer][inner]));
            __builtin_msa_st_b(reinterpret_cast<v16i8>(tmp), dst_ptr, 0);
            dst_ptr += 16;
          }
        }
      }
    }
    // Compute sums across the depth dimension
    v8i16 sums_of_2_cells[kCells][4];
    const v16i8 zeroes = __builtin_msa_ldi_b(0);
    for (int outer = 0; outer < 2; outer++) {
      for (int inner = 0; inner < 2; inner++) {
        int i = 2 * outer + inner;
        for (int cell = 0; cell < kCells; cell++) {
          v8i16 tmp0 = reinterpret_cast<v8i16>(__builtin_msa_ilvr_b(
              zeroes, src_lines_intertwined_4x[2 * cell + outer][inner]));
          v8i16 tmp1 = reinterpret_cast<v8i16>(__builtin_msa_ilvl_b(
              zeroes, src_lines_intertwined_4x[2 * cell + outer][inner]));
          sums_of_2_cells[cell][i] = __builtin_msa_addv_h(tmp0, tmp1);
        }
      }
    }
    v4i32 sums_of_4_cells[kCells][4];
    for (int i = 0; i < 4; i++) {
      for (int cell = 0; cell < kCells; cell++) {
        v4i32 tmp0 = reinterpret_cast<v4i32>(__builtin_msa_ilvr_h(
            reinterpret_cast<v8i16>(zeroes), sums_of_2_cells[cell][i]));
        v4i32 tmp1 = reinterpret_cast<v4i32>(__builtin_msa_ilvl_h(
            reinterpret_cast<v8i16>(zeroes), sums_of_2_cells[cell][i]));
        sums_of_4_cells[cell][i] = __builtin_msa_addv_w(tmp0, tmp1);
      }
    }
    // Update the sums_of_each_slice vector
    for (int cell = 0; cell < kCells; cell++) {
      v4i32 s01 = __builtin_msa_addv_w(sums_of_4_cells[cell][0],
                                       sums_of_4_cells[cell][1]);
      v4i32 s23 = __builtin_msa_addv_w(sums_of_4_cells[cell][2],
                                       sums_of_4_cells[cell][3]);
      v4i32 s = __builtin_msa_addv_w(s01, s23);
      std::int32_t* sums_of_each_slice_ptr =
          dst->sums_of_each_slice() + start_width + 4 * cell;
      v4i32 tmp = __builtin_msa_ld_w(sums_of_each_slice_ptr, 0);
      tmp = __builtin_msa_addv_w(tmp, s);
      __builtin_msa_st_w(tmp, sums_of_each_slice_ptr, 0);
    }
    dst->seek_forward_n_cells(kCells * kRegisterSize / kCellDepth);
  }
};

template <int Cells>
using WidthMajorSideFormatNCells4x2 =
    KernelSideFormat<CellFormat<4, 2, CellOrder::WidthMajor>, Cells>;

template <int Cells>
class PackingRegisterBlock<
    WidthMajorUint8SideMap,
    PackedSideBlock<WidthMajorSideFormatNCells4x2<Cells>>>
    : public PackingRegisterBlockBase<
          WidthMajorUint8SideMap,
          PackedSideBlock<WidthMajorSideFormatNCells4x2<Cells>>> {
 public:
  typedef WidthMajorSideFormatNCells4x2<Cells> KernelSideFormat;
  typedef typename KernelSideFormat::Cell CellFormat;
  static constexpr int kCells = KernelSideFormat::kCells;
  static const int kCellWidth = CellFormat::kWidth;
  static const int kKernelWidth = CellFormat::kWidth * kCells;
  static const int kCellDepth = CellFormat::kDepth;
  static const int kCellSize = CellFormat::kSize;

  void Pack(PackedSideBlock<KernelSideFormat>* dst, int start_width) {
    std::uint8_t* dst_ptr = dst->current_data();
    const std::uint8_t* src_ptr = this->complete_src_.data();
    const int stride = this->complete_src_.stride();
    // Load source WidthMajor data
    v8i16 src_lines[kCells * 4];
    for (int i = 0; i < kCells; i++) {
#define GEMMLOWP_UNROLLED_LOOP_ITER(k)                           \
  src_lines[4 * i + k] =                                         \
      __builtin_msa_ld_h(const_cast<std::uint8_t*>(src_ptr), 0); \
  src_ptr += stride;

      GEMMLOWP_UNROLLED_LOOP_ITER(0)
      GEMMLOWP_UNROLLED_LOOP_ITER(1)
      GEMMLOWP_UNROLLED_LOOP_ITER(2)
      GEMMLOWP_UNROLLED_LOOP_ITER(3)

#undef GEMMLOWP_UNROLLED_LOOP_ITER
    }
    // Reorder the data within registers to make WidthMajor 4x2 cells
    v8i16 src_lines_intertwined_2x[2 * kCells][2];
    for (int i = 0; i < kCells; i++) {
      src_lines_intertwined_2x[2 * i][0] =
          __builtin_msa_ilvr_h(src_lines[4 * i + 2], src_lines[4 * i]);
      src_lines_intertwined_2x[2 * i][1] =
          __builtin_msa_ilvl_h(src_lines[4 * i + 2], src_lines[4 * i]);
      src_lines_intertwined_2x[2 * i + 1][0] =
          __builtin_msa_ilvr_h(src_lines[4 * i + 3], src_lines[4 * i + 1]);
      src_lines_intertwined_2x[2 * i + 1][1] =
          __builtin_msa_ilvl_h(src_lines[4 * i + 3], src_lines[4 * i + 1]);
    }
    v8i16 src_lines_intertwined_4x[2 * kCells][2];
    for (int i = 0; i < kCells; i++) {
      src_lines_intertwined_4x[2 * i][0] =
          __builtin_msa_ilvr_h(src_lines_intertwined_2x[2 * i + 1][0],
                               src_lines_intertwined_2x[2 * i][0]);
      src_lines_intertwined_4x[2 * i][1] =
          __builtin_msa_ilvl_h(src_lines_intertwined_2x[2 * i + 1][0],
                               src_lines_intertwined_2x[2 * i][0]);
      src_lines_intertwined_4x[2 * i + 1][0] =
          __builtin_msa_ilvr_h(src_lines_intertwined_2x[2 * i + 1][1],
                               src_lines_intertwined_2x[2 * i][1]);
      src_lines_intertwined_4x[2 * i + 1][1] =
          __builtin_msa_ilvl_h(src_lines_intertwined_2x[2 * i + 1][1],
                               src_lines_intertwined_2x[2 * i][1]);
    }
    // Store the resulting WidthMajor 4x2 cells in the destination packed block
    for (int outer = 0; outer < 2; outer++) {
      for (int inner = 0; inner < 2; inner++) {
        if (kCells % 2 == 0) {
          for (int cell = 0; cell < kCells; cell += 2) {
            v2i64 tmp = __builtin_msa_ilvr_d(
                reinterpret_cast<v2i64>(
                    src_lines_intertwined_4x[2 * (cell + 1) + outer][inner]),
                reinterpret_cast<v2i64>(
                    src_lines_intertwined_4x[2 * cell + outer][inner]));
            __builtin_msa_st_b(reinterpret_cast<v16i8>(tmp), dst_ptr, 0);
            dst_ptr += 16;
          }
          for (int cell = 0; cell < kCells; cell += 2) {
            v2i64 tmp = __builtin_msa_ilvl_d(
                reinterpret_cast<v2i64>(
                    src_lines_intertwined_4x[2 * (cell + 1) + outer][inner]),
                reinterpret_cast<v2i64>(
                    src_lines_intertwined_4x[2 * cell + outer][inner]));
            __builtin_msa_st_b(reinterpret_cast<v16i8>(tmp), dst_ptr, 0);
            dst_ptr += 16;
          }
        } else {
          // Store even number of low vector halves.
          for (int cell = 0; cell < kCells - 1; cell += 2) {
            v2i64 tmp = __builtin_msa_ilvr_d(
                reinterpret_cast<v2i64>(
                    src_lines_intertwined_4x[2 * (cell + 1) + outer][inner]),
                reinterpret_cast<v2i64>(
                    src_lines_intertwined_4x[2 * cell + outer][inner]));
            __builtin_msa_st_b(reinterpret_cast<v16i8>(tmp), dst_ptr, 0);
            dst_ptr += 16;
          }
          // Store last low half and first high half.
          v2i64 tmp = reinterpret_cast<v2i64>(
              src_lines_intertwined_4x[2 * 0 + outer][inner]);
          tmp = __builtin_msa_insve_d(
              tmp, 0,
              reinterpret_cast<v2i64>(
                  src_lines_intertwined_4x[2 * (kCells - 1) + outer][inner]));
          __builtin_msa_st_b(reinterpret_cast<v16i8>(tmp), dst_ptr, 0);
          dst_ptr += 16;
          // Store even number of high vector halves.
          for (int cell = 1; cell < kCells; cell += 2) {
            v2i64 tmp = __builtin_msa_ilvl_d(
                reinterpret_cast<v2i64>(
                    src_lines_intertwined_4x[2 * (cell + 1) + outer][inner]),
                reinterpret_cast<v2i64>(
                    src_lines_intertwined_4x[2 * cell + outer][inner]));
            __builtin_msa_st_b(reinterpret_cast<v16i8>(tmp), dst_ptr, 0);
            dst_ptr += 16;
          }
        }
      }
    }
    // Compute sums across the depth dimension
    v8i16 sums_of_2[kCells][4];
    for (int outer = 0; outer < 2; outer++) {
      for (int inner = 0; inner < 2; inner++) {
        int i = 2 * outer + inner;
        for (int cell = 0; cell < kCells; cell++) {
          sums_of_2[cell][i] = reinterpret_cast<v8i16>(__builtin_msa_hadd_u_h(
              reinterpret_cast<v16u8>(
                  src_lines_intertwined_4x[2 * cell + outer][inner]),
              reinterpret_cast<v16u8>(
                  src_lines_intertwined_4x[2 * cell + outer][inner])));
        }
      }
    }
    v8i16 sums_of_4[kCells][2];
    for (int i = 0; i < 2; i++) {
      for (int cell = 0; cell < kCells; cell++) {
        sums_of_4[cell][i] = __builtin_msa_addv_h(sums_of_2[cell][2 * i],
                                                  sums_of_2[cell][2 * i + 1]);
      }
    }
    v8i16 sums_of_8[kCells];
    for (int cell = 0; cell < kCells; cell++) {
      sums_of_8[cell] =
          __builtin_msa_addv_h(sums_of_4[cell][0], sums_of_4[cell][1]);
    }

    v4i32 sums_of_16[kCells];
    const v8i16 zeroes = __builtin_msa_ldi_h(0);
    for (int cell = 0; cell < kCells; cell++) {
      sums_of_16[cell] = reinterpret_cast<v4i32>(
          __builtin_msa_ilvr_h(zeroes, sums_of_8[cell]));
      v8i16 tmp = __builtin_msa_ilvl_h(zeroes, sums_of_8[cell]);
      sums_of_16[cell] =
          __builtin_msa_addv_w(sums_of_16[cell], reinterpret_cast<v4i32>(tmp));
    }
    // Update the sums_of_each_slice vector
    for (int cell = 0; cell < kCells; cell++) {
      std::int32_t* sums_of_each_slice_ptr =
          dst->sums_of_each_slice() + start_width + 4 * cell;
      v4i32 tmp = __builtin_msa_ld_w(sums_of_each_slice_ptr, 0);
      tmp = __builtin_msa_addv_w(tmp, sums_of_16[cell]);
      __builtin_msa_st_w(tmp, sums_of_each_slice_ptr, 0);
    }
    dst->seek_forward_n_cells(kCells * kRegisterSize / kCellDepth);
  }
};

}  // namespace gemmlowp

#endif  // GEMMLOWP_INTERNAL_PACK_MSA_H_
