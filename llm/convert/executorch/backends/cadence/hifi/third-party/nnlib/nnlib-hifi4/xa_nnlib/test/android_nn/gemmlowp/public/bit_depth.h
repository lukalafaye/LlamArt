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
// Copyright 2015 The Gemmlowp Authors. All Rights Reserved.
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

// bit_depth.h: defines the settins controlling LHS/RHS bit depth

#ifndef GEMMLOWP_PUBLIC_BIT_DEPTH_H_
#define GEMMLOWP_PUBLIC_BIT_DEPTH_H_

namespace gemmlowp {

// The range of allowed values for an operand.
template <int tMinValue, int tMaxValue>
struct OperandRange {
  static const int kMinValue = tMinValue;
  static const int kMaxValue = tMaxValue;
  static_assert(0 <= kMinValue, "");
  static_assert(kMinValue < kMaxValue, "");
  static_assert(kMaxValue <= 255, "");
};

using Uint8Range = OperandRange<0, 255>;
using Uint8RangeExcludingZero = OperandRange<1, 255>;

template <typename tLhsRange, typename tRhsRange>
struct BitDepthParams {
  using LhsRange = tLhsRange;
  using RhsRange = tRhsRange;
};

// Default: LHS and RHS are 8bit.
using DefaultL8R8BitDepthParams = BitDepthParams<Uint8Range, Uint8Range>;

// Variant: LHS may not take the value 0. This allows using
// faster kernels using signed arithmetic, see
// NEON_64bit_GEMM_Int8Operands_Int32Accumulators_AccumTwoWithin16Bits
using L8R8WithLhsNonzeroBitDepthParams =
    BitDepthParams<Uint8RangeExcludingZero, Uint8Range>;

// Deprecated: when gemmlowp used to allow requantizing 8bit
// inputs to less-than-8-bit depths, the public setting allowing
// that was DefaultL7R5BitDepthParams. That requantization
// feature has been removed, but as the whole point of that
// requantization was to make less-than-8-bit an internal
// optimization without any impact on the API (other than lowering
// accuracy), we can temporarily support users who were using it
// by mapping it to the default 8bit behavior.
using DefaultL7R5BitDepthParams = DefaultL8R8BitDepthParams;

}  // namespace gemmlowp

#endif  // GEMMLOWP_PUBLIC_BIT_DEPTH_H_
