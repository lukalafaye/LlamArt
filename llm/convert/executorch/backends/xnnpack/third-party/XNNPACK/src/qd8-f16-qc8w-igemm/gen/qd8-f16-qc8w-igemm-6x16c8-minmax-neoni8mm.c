// Auto-generated file. Do not edit!
//   Template: src/qs8-igemm/c8-neoni8mm.c.in
//   Generator: tools/xngen
//
// Copyright 2023 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <arm_neon.h>

#include "xnnpack/igemm.h"
#include "xnnpack/math.h"


void xnn_qd8_f16_qc8w_igemm_minmax_ukernel_6x16c8__neoni8mm(
    size_t mr,
    size_t nc,
    size_t kc,
    size_t ks,
    const int8_t** restrict a,
    const void* restrict w,
    xnn_float16* restrict c,
    size_t cm_stride,
    size_t cn_stride,
    size_t a_offset,
    const int8_t* zero,
    const int8_t* zero_data,
    const union xnn_f16_minmax_params params[restrict XNN_MIN_ELEMENTS(1)],
    const struct xnn_qd8_quantization_params quantization_params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(mr != 0);
  assert(mr <= 6);
  assert(nc != 0);
  assert(kc != 0);
  assert(ks != 0);
  assert(ks % (6 * sizeof(void*)) == 0);
  assert(a_offset % sizeof(int8_t) == 0);
  assert(a != NULL);
  assert(w != NULL);
  assert(c != NULL);

  kc = round_up_po2(kc, 8 * sizeof(int8_t));
  uint16_t* c0 = (uint16_t*) c;
  uint16_t* c1 = (uint16_t*) ((uintptr_t) c0 + cm_stride);
  if XNN_UNPREDICTABLE(mr < 2) {
    c1 = c0;
  }
  uint16_t* c2 = (uint16_t*) ((uintptr_t) c1 + cm_stride);
  if XNN_UNPREDICTABLE(mr <= 2) {
    c2 = c1;
  }
  uint16_t* c3 = (uint16_t*) ((uintptr_t) c2 + cm_stride);
  if XNN_UNPREDICTABLE(mr < 4) {
    c3 = c2;
  }
  uint16_t* c4 = (uint16_t*) ((uintptr_t) c3 + cm_stride);
  if XNN_UNPREDICTABLE(mr <= 4) {
    c4 = c3;
  }
  uint16_t* c5 = (uint16_t*) ((uintptr_t) c4 + cm_stride);
  if XNN_UNPREDICTABLE(mr != 6) {
    c5 = c4;
  }

  do {
    const int32x4_t vinput_zero_point = vld1q_dup_s32(&quantization_params->zero_point);
    const int32x4_t vksum0123 = vld1q_s32(w); w = (const int32_t*) w + 4;
    const int32x4_t vksumzp0x0123 = vmulq_s32(vksum0123, vinput_zero_point);
    const int32x4_t vksum4567 = vld1q_s32(w); w = (const int32_t*) w + 4;
    const int32x4_t vksumzp0x4567 = vmulq_s32(vksum4567, vinput_zero_point);
    const int32x4_t vksum89AB = vld1q_s32(w); w = (const int32_t*) w + 4;
    const int32x4_t vksumzp0x89AB = vmulq_s32(vksum89AB, vinput_zero_point);
    const int32x4_t vksumCDEF = vld1q_s32(w); w = (const int32_t*) w + 4;
    const int32x4_t vksumzp0xCDEF = vmulq_s32(vksumCDEF, vinput_zero_point);
    const int32x4_t vksumzp1x0123 = vksumzp0x0123;
    const int32x4_t vksumzp1x4567 = vksumzp0x4567;
    const int32x4_t vksumzp1x89AB = vksumzp0x89AB;
    const int32x4_t vksumzp1xCDEF = vksumzp0xCDEF;
    const int32x4_t vksumzp2x0123 = vksumzp0x0123;
    const int32x4_t vksumzp2x4567 = vksumzp0x4567;
    const int32x4_t vksumzp2x89AB = vksumzp0x89AB;
    const int32x4_t vksumzp2xCDEF = vksumzp0xCDEF;
    const int32x4_t vksumzp3x0123 = vksumzp0x0123;
    const int32x4_t vksumzp3x4567 = vksumzp0x4567;
    const int32x4_t vksumzp3x89AB = vksumzp0x89AB;
    const int32x4_t vksumzp3xCDEF = vksumzp0xCDEF;
    const int32x4_t vksumzp4x0123 = vksumzp0x0123;
    const int32x4_t vksumzp4x4567 = vksumzp0x4567;
    const int32x4_t vksumzp4x89AB = vksumzp0x89AB;
    const int32x4_t vksumzp4xCDEF = vksumzp0xCDEF;
    const int32x4_t vksumzp5x0123 = vksumzp0x0123;
    const int32x4_t vksumzp5x4567 = vksumzp0x4567;
    const int32x4_t vksumzp5x89AB = vksumzp0x89AB;
    const int32x4_t vksumzp5xCDEF = vksumzp0xCDEF;

    int32x4_t vacc01x01 = vreinterpretq_s32_u64(vtrn1q_u64(vreinterpretq_u64_s32(vksumzp0x0123), vreinterpretq_u64_s32(vksumzp1x0123)));
    int32x4_t vacc01x23 = vreinterpretq_s32_u64(vtrn2q_u64(vreinterpretq_u64_s32(vksumzp0x0123), vreinterpretq_u64_s32(vksumzp1x0123)));
    int32x4_t vacc01x45 = vreinterpretq_s32_u64(vtrn1q_u64(vreinterpretq_u64_s32(vksumzp0x4567), vreinterpretq_u64_s32(vksumzp1x4567)));
    int32x4_t vacc01x67 = vreinterpretq_s32_u64(vtrn2q_u64(vreinterpretq_u64_s32(vksumzp0x4567), vreinterpretq_u64_s32(vksumzp1x4567)));
    int32x4_t vacc01x89 = vreinterpretq_s32_u64(vtrn1q_u64(vreinterpretq_u64_s32(vksumzp0x89AB), vreinterpretq_u64_s32(vksumzp1x89AB)));
    int32x4_t vacc01xAB = vreinterpretq_s32_u64(vtrn2q_u64(vreinterpretq_u64_s32(vksumzp0x89AB), vreinterpretq_u64_s32(vksumzp1x89AB)));
    int32x4_t vacc01xCD = vreinterpretq_s32_u64(vtrn1q_u64(vreinterpretq_u64_s32(vksumzp0xCDEF), vreinterpretq_u64_s32(vksumzp1xCDEF)));
    int32x4_t vacc01xEF = vreinterpretq_s32_u64(vtrn2q_u64(vreinterpretq_u64_s32(vksumzp0xCDEF), vreinterpretq_u64_s32(vksumzp1xCDEF)));
    int32x4_t vacc23x01 = vreinterpretq_s32_u64(vtrn1q_u64(vreinterpretq_u64_s32(vksumzp2x0123), vreinterpretq_u64_s32(vksumzp3x0123)));
    int32x4_t vacc23x23 = vreinterpretq_s32_u64(vtrn2q_u64(vreinterpretq_u64_s32(vksumzp2x0123), vreinterpretq_u64_s32(vksumzp3x0123)));
    int32x4_t vacc23x45 = vreinterpretq_s32_u64(vtrn1q_u64(vreinterpretq_u64_s32(vksumzp2x4567), vreinterpretq_u64_s32(vksumzp3x4567)));
    int32x4_t vacc23x67 = vreinterpretq_s32_u64(vtrn2q_u64(vreinterpretq_u64_s32(vksumzp2x4567), vreinterpretq_u64_s32(vksumzp3x4567)));
    int32x4_t vacc23x89 = vreinterpretq_s32_u64(vtrn1q_u64(vreinterpretq_u64_s32(vksumzp2x89AB), vreinterpretq_u64_s32(vksumzp3x89AB)));
    int32x4_t vacc23xAB = vreinterpretq_s32_u64(vtrn2q_u64(vreinterpretq_u64_s32(vksumzp2x89AB), vreinterpretq_u64_s32(vksumzp3x89AB)));
    int32x4_t vacc23xCD = vreinterpretq_s32_u64(vtrn1q_u64(vreinterpretq_u64_s32(vksumzp2xCDEF), vreinterpretq_u64_s32(vksumzp3xCDEF)));
    int32x4_t vacc23xEF = vreinterpretq_s32_u64(vtrn2q_u64(vreinterpretq_u64_s32(vksumzp2xCDEF), vreinterpretq_u64_s32(vksumzp3xCDEF)));
    int32x4_t vacc45x01 = vreinterpretq_s32_u64(vtrn1q_u64(vreinterpretq_u64_s32(vksumzp4x0123), vreinterpretq_u64_s32(vksumzp5x0123)));
    int32x4_t vacc45x23 = vreinterpretq_s32_u64(vtrn2q_u64(vreinterpretq_u64_s32(vksumzp4x0123), vreinterpretq_u64_s32(vksumzp5x0123)));
    int32x4_t vacc45x45 = vreinterpretq_s32_u64(vtrn1q_u64(vreinterpretq_u64_s32(vksumzp4x4567), vreinterpretq_u64_s32(vksumzp5x4567)));
    int32x4_t vacc45x67 = vreinterpretq_s32_u64(vtrn2q_u64(vreinterpretq_u64_s32(vksumzp4x4567), vreinterpretq_u64_s32(vksumzp5x4567)));
    int32x4_t vacc45x89 = vreinterpretq_s32_u64(vtrn1q_u64(vreinterpretq_u64_s32(vksumzp4x89AB), vreinterpretq_u64_s32(vksumzp5x89AB)));
    int32x4_t vacc45xAB = vreinterpretq_s32_u64(vtrn2q_u64(vreinterpretq_u64_s32(vksumzp4x89AB), vreinterpretq_u64_s32(vksumzp5x89AB)));
    int32x4_t vacc45xCD = vreinterpretq_s32_u64(vtrn1q_u64(vreinterpretq_u64_s32(vksumzp4xCDEF), vreinterpretq_u64_s32(vksumzp5xCDEF)));
    int32x4_t vacc45xEF = vreinterpretq_s32_u64(vtrn2q_u64(vreinterpretq_u64_s32(vksumzp4xCDEF), vreinterpretq_u64_s32(vksumzp5xCDEF)));

    size_t p = ks;
    do {
      const int8_t* restrict a0 = a[0];
      if XNN_UNPREDICTABLE(a0 != zero) {
        a0 = (const int8_t*) ((uintptr_t) a0 + a_offset);
      } else {
        a0 = zero_data;
      }
      const int8_t* restrict a1 = a[1];
      if XNN_UNPREDICTABLE(a1 != zero) {
        a1 = (const int8_t*) ((uintptr_t) a1 + a_offset);
      } else {
        a1 = zero_data;
      }
      const int8_t* restrict a2 = a[2];
      if XNN_UNPREDICTABLE(a2 != zero) {
        a2 = (const int8_t*) ((uintptr_t) a2 + a_offset);
      } else {
        a2 = zero_data;
      }
      const int8_t* restrict a3 = a[3];
      if XNN_UNPREDICTABLE(a3 != zero) {
        a3 = (const int8_t*) ((uintptr_t) a3 + a_offset);
      } else {
        a3 = zero_data;
      }
      const int8_t* restrict a4 = a[4];
      if XNN_UNPREDICTABLE(a4 != zero) {
        a4 = (const int8_t*) ((uintptr_t) a4 + a_offset);
      } else {
        a4 = zero_data;
      }
      const int8_t* restrict a5 = a[5];
      if XNN_UNPREDICTABLE(a5 != zero) {
        a5 = (const int8_t*) ((uintptr_t) a5 + a_offset);
      } else {
        a5 = zero_data;
      }
      a += 6;

      uint64x2x2_t va01x0123456789ABCDEF;
      va01x0123456789ABCDEF.val[0] = vdupq_n_u64(0);
      va01x0123456789ABCDEF.val[1] = vdupq_n_u64(0);
      uint64x2x2_t va23x0123456789ABCDEF;
      va23x0123456789ABCDEF.val[0] = vdupq_n_u64(0);
      va23x0123456789ABCDEF.val[1] = vdupq_n_u64(0);
      uint64x2x2_t va45x0123456789ABCDEF;
      va45x0123456789ABCDEF.val[0] = vdupq_n_u64(0);
      va45x0123456789ABCDEF.val[1] = vdupq_n_u64(0);

      // Inner accumulation loop along the 16 columns.
      size_t k = kc;
      // 2x partial unrolled loop to load 8 bytes at a time.
      while (k >= 16 * sizeof(int8_t)) {
        // Load a 6x16 block of activations.
        va01x0123456789ABCDEF = vld2q_lane_u64((const void*) a0, va01x0123456789ABCDEF, 0); a0 += 16;
        va23x0123456789ABCDEF = vld2q_lane_u64((const void*) a2, va23x0123456789ABCDEF, 0); a2 += 16;
        va45x0123456789ABCDEF = vld2q_lane_u64((const void*) a4, va45x0123456789ABCDEF, 0); a4 += 16;
        va01x0123456789ABCDEF = vld2q_lane_u64((const void*) a1, va01x0123456789ABCDEF, 1); a1 += 16;
        va23x0123456789ABCDEF = vld2q_lane_u64((const void*) a3, va23x0123456789ABCDEF, 1); a3 += 16;
        va45x0123456789ABCDEF = vld2q_lane_u64((const void*) a5, va45x0123456789ABCDEF, 1); a5 += 16;

        // Load a 16x16 block of weights.
        const int8x16_t vb01x01234567 = vld1q_s8(w); w = (const int8_t*) w + 16;
        const int8x16_t vb23x01234567 = vld1q_s8(w); w = (const int8_t*) w + 16;
        const int8x16_t vb45x01234567 = vld1q_s8(w); w = (const int8_t*) w + 16;
        const int8x16_t vb67x01234567 = vld1q_s8(w); w = (const int8_t*) w + 16;
        const int8x16_t vb89x01234567 = vld1q_s8(w); w = (const int8_t*) w + 16;
        const int8x16_t vbABx01234567 = vld1q_s8(w); w = (const int8_t*) w + 16;
        const int8x16_t vbCDx01234567 = vld1q_s8(w); w = (const int8_t*) w + 16;
        const int8x16_t vbEFx01234567 = vld1q_s8(w); w = (const int8_t*) w + 16;
        const int8x16_t vb01x89ABCDEF = vld1q_s8(w); w = (const int8_t*) w + 16;
        const int8x16_t vb23x89ABCDEF = vld1q_s8(w); w = (const int8_t*) w + 16;
        const int8x16_t vb45x89ABCDEF = vld1q_s8(w); w = (const int8_t*) w + 16;
        const int8x16_t vb67x89ABCDEF = vld1q_s8(w); w = (const int8_t*) w + 16;
        const int8x16_t vb89x89ABCDEF = vld1q_s8(w); w = (const int8_t*) w + 16;
        const int8x16_t vbABx89ABCDEF = vld1q_s8(w); w = (const int8_t*) w + 16;
        const int8x16_t vbCDx89ABCDEF = vld1q_s8(w); w = (const int8_t*) w + 16;
        const int8x16_t vbEFx89ABCDEF = vld1q_s8(w); w = (const int8_t*) w + 16;

        // Multiply-accumulate: 6x16 * 16x16 --> 6x16.
        vacc01x01 = vmmlaq_s32(vacc01x01, vreinterpretq_s8_u64(va01x0123456789ABCDEF.val[0]), vb01x01234567);
        vacc01x23 = vmmlaq_s32(vacc01x23, vreinterpretq_s8_u64(va01x0123456789ABCDEF.val[0]), vb23x01234567);
        vacc01x45 = vmmlaq_s32(vacc01x45, vreinterpretq_s8_u64(va01x0123456789ABCDEF.val[0]), vb45x01234567);
        vacc01x67 = vmmlaq_s32(vacc01x67, vreinterpretq_s8_u64(va01x0123456789ABCDEF.val[0]), vb67x01234567);
        vacc01x89 = vmmlaq_s32(vacc01x89, vreinterpretq_s8_u64(va01x0123456789ABCDEF.val[0]), vb89x01234567);
        vacc01xAB = vmmlaq_s32(vacc01xAB, vreinterpretq_s8_u64(va01x0123456789ABCDEF.val[0]), vbABx01234567);
        vacc01xCD = vmmlaq_s32(vacc01xCD, vreinterpretq_s8_u64(va01x0123456789ABCDEF.val[0]), vbCDx01234567);
        vacc01xEF = vmmlaq_s32(vacc01xEF, vreinterpretq_s8_u64(va01x0123456789ABCDEF.val[0]), vbEFx01234567);
        vacc23x01 = vmmlaq_s32(vacc23x01, vreinterpretq_s8_u64(va23x0123456789ABCDEF.val[0]), vb01x01234567);
        vacc23x23 = vmmlaq_s32(vacc23x23, vreinterpretq_s8_u64(va23x0123456789ABCDEF.val[0]), vb23x01234567);
        vacc23x45 = vmmlaq_s32(vacc23x45, vreinterpretq_s8_u64(va23x0123456789ABCDEF.val[0]), vb45x01234567);
        vacc23x67 = vmmlaq_s32(vacc23x67, vreinterpretq_s8_u64(va23x0123456789ABCDEF.val[0]), vb67x01234567);
        vacc23x89 = vmmlaq_s32(vacc23x89, vreinterpretq_s8_u64(va23x0123456789ABCDEF.val[0]), vb89x01234567);
        vacc23xAB = vmmlaq_s32(vacc23xAB, vreinterpretq_s8_u64(va23x0123456789ABCDEF.val[0]), vbABx01234567);
        vacc23xCD = vmmlaq_s32(vacc23xCD, vreinterpretq_s8_u64(va23x0123456789ABCDEF.val[0]), vbCDx01234567);
        vacc23xEF = vmmlaq_s32(vacc23xEF, vreinterpretq_s8_u64(va23x0123456789ABCDEF.val[0]), vbEFx01234567);
        vacc45x01 = vmmlaq_s32(vacc45x01, vreinterpretq_s8_u64(va45x0123456789ABCDEF.val[0]), vb01x01234567);
        vacc45x23 = vmmlaq_s32(vacc45x23, vreinterpretq_s8_u64(va45x0123456789ABCDEF.val[0]), vb23x01234567);
        vacc45x45 = vmmlaq_s32(vacc45x45, vreinterpretq_s8_u64(va45x0123456789ABCDEF.val[0]), vb45x01234567);
        vacc45x67 = vmmlaq_s32(vacc45x67, vreinterpretq_s8_u64(va45x0123456789ABCDEF.val[0]), vb67x01234567);
        vacc45x89 = vmmlaq_s32(vacc45x89, vreinterpretq_s8_u64(va45x0123456789ABCDEF.val[0]), vb89x01234567);
        vacc45xAB = vmmlaq_s32(vacc45xAB, vreinterpretq_s8_u64(va45x0123456789ABCDEF.val[0]), vbABx01234567);
        vacc45xCD = vmmlaq_s32(vacc45xCD, vreinterpretq_s8_u64(va45x0123456789ABCDEF.val[0]), vbCDx01234567);
        vacc45xEF = vmmlaq_s32(vacc45xEF, vreinterpretq_s8_u64(va45x0123456789ABCDEF.val[0]), vbEFx01234567);
        vacc01x01 = vmmlaq_s32(vacc01x01, vreinterpretq_s8_u64(va01x0123456789ABCDEF.val[1]), vb01x89ABCDEF);
        vacc01x23 = vmmlaq_s32(vacc01x23, vreinterpretq_s8_u64(va01x0123456789ABCDEF.val[1]), vb23x89ABCDEF);
        vacc01x45 = vmmlaq_s32(vacc01x45, vreinterpretq_s8_u64(va01x0123456789ABCDEF.val[1]), vb45x89ABCDEF);
        vacc01x67 = vmmlaq_s32(vacc01x67, vreinterpretq_s8_u64(va01x0123456789ABCDEF.val[1]), vb67x89ABCDEF);
        vacc01x89 = vmmlaq_s32(vacc01x89, vreinterpretq_s8_u64(va01x0123456789ABCDEF.val[1]), vb89x89ABCDEF);
        vacc01xAB = vmmlaq_s32(vacc01xAB, vreinterpretq_s8_u64(va01x0123456789ABCDEF.val[1]), vbABx89ABCDEF);
        vacc01xCD = vmmlaq_s32(vacc01xCD, vreinterpretq_s8_u64(va01x0123456789ABCDEF.val[1]), vbCDx89ABCDEF);
        vacc01xEF = vmmlaq_s32(vacc01xEF, vreinterpretq_s8_u64(va01x0123456789ABCDEF.val[1]), vbEFx89ABCDEF);
        vacc23x01 = vmmlaq_s32(vacc23x01, vreinterpretq_s8_u64(va23x0123456789ABCDEF.val[1]), vb01x89ABCDEF);
        vacc23x23 = vmmlaq_s32(vacc23x23, vreinterpretq_s8_u64(va23x0123456789ABCDEF.val[1]), vb23x89ABCDEF);
        vacc23x45 = vmmlaq_s32(vacc23x45, vreinterpretq_s8_u64(va23x0123456789ABCDEF.val[1]), vb45x89ABCDEF);
        vacc23x67 = vmmlaq_s32(vacc23x67, vreinterpretq_s8_u64(va23x0123456789ABCDEF.val[1]), vb67x89ABCDEF);
        vacc23x89 = vmmlaq_s32(vacc23x89, vreinterpretq_s8_u64(va23x0123456789ABCDEF.val[1]), vb89x89ABCDEF);
        vacc23xAB = vmmlaq_s32(vacc23xAB, vreinterpretq_s8_u64(va23x0123456789ABCDEF.val[1]), vbABx89ABCDEF);
        vacc23xCD = vmmlaq_s32(vacc23xCD, vreinterpretq_s8_u64(va23x0123456789ABCDEF.val[1]), vbCDx89ABCDEF);
        vacc23xEF = vmmlaq_s32(vacc23xEF, vreinterpretq_s8_u64(va23x0123456789ABCDEF.val[1]), vbEFx89ABCDEF);
        vacc45x01 = vmmlaq_s32(vacc45x01, vreinterpretq_s8_u64(va45x0123456789ABCDEF.val[1]), vb01x89ABCDEF);
        vacc45x23 = vmmlaq_s32(vacc45x23, vreinterpretq_s8_u64(va45x0123456789ABCDEF.val[1]), vb23x89ABCDEF);
        vacc45x45 = vmmlaq_s32(vacc45x45, vreinterpretq_s8_u64(va45x0123456789ABCDEF.val[1]), vb45x89ABCDEF);
        vacc45x67 = vmmlaq_s32(vacc45x67, vreinterpretq_s8_u64(va45x0123456789ABCDEF.val[1]), vb67x89ABCDEF);
        vacc45x89 = vmmlaq_s32(vacc45x89, vreinterpretq_s8_u64(va45x0123456789ABCDEF.val[1]), vb89x89ABCDEF);
        vacc45xAB = vmmlaq_s32(vacc45xAB, vreinterpretq_s8_u64(va45x0123456789ABCDEF.val[1]), vbABx89ABCDEF);
        vacc45xCD = vmmlaq_s32(vacc45xCD, vreinterpretq_s8_u64(va45x0123456789ABCDEF.val[1]), vbCDx89ABCDEF);
        vacc45xEF = vmmlaq_s32(vacc45xEF, vreinterpretq_s8_u64(va45x0123456789ABCDEF.val[1]), vbEFx89ABCDEF);

        k -= 16 * sizeof(int8_t);
      }
      // Handle up to 8 final positions of `k`
      if XNN_UNLIKELY(k != 0) {
        // Load a 6x8 block of activations.
        uint64x2_t va01x01234567 = vld1q_dup_u64((const void*) a0); a0 += 8;
        uint64x2_t va23x01234567 = vld1q_dup_u64((const void*) a2); a2 += 8;
        uint64x2_t va45x01234567 = vld1q_dup_u64((const void*) a4); a4 += 8;
        va01x01234567 = vld1q_lane_u64((const void*) a1, va01x01234567, 1); a1 += 8;
        va23x01234567 = vld1q_lane_u64((const void*) a3, va23x01234567, 1); a3 += 8;
        va45x01234567 = vld1q_lane_u64((const void*) a5, va45x01234567, 1); a5 += 8;

        // Load a 16x16 block of weights.
        const int8x16_t vb01x01234567 = vld1q_s8(w); w = (const int8_t*) w + 16;
        const int8x16_t vb23x01234567 = vld1q_s8(w); w = (const int8_t*) w + 16;
        const int8x16_t vb45x01234567 = vld1q_s8(w); w = (const int8_t*) w + 16;
        const int8x16_t vb67x01234567 = vld1q_s8(w); w = (const int8_t*) w + 16;
        const int8x16_t vb89x01234567 = vld1q_s8(w); w = (const int8_t*) w + 16;
        const int8x16_t vbABx01234567 = vld1q_s8(w); w = (const int8_t*) w + 16;
        const int8x16_t vbCDx01234567 = vld1q_s8(w); w = (const int8_t*) w + 16;
        const int8x16_t vbEFx01234567 = vld1q_s8(w); w = (const int8_t*) w + 16;

        // Multiply-accumulate: 6x4 * 4x16 --> 6x16.
        vacc01x01 = vmmlaq_s32(vacc01x01, vreinterpretq_s8_u64(va01x01234567), vb01x01234567);
        vacc01x23 = vmmlaq_s32(vacc01x23, vreinterpretq_s8_u64(va01x01234567), vb23x01234567);
        vacc01x45 = vmmlaq_s32(vacc01x45, vreinterpretq_s8_u64(va01x01234567), vb45x01234567);
        vacc01x67 = vmmlaq_s32(vacc01x67, vreinterpretq_s8_u64(va01x01234567), vb67x01234567);
        vacc01x89 = vmmlaq_s32(vacc01x89, vreinterpretq_s8_u64(va01x01234567), vb89x01234567);
        vacc01xAB = vmmlaq_s32(vacc01xAB, vreinterpretq_s8_u64(va01x01234567), vbABx01234567);
        vacc01xCD = vmmlaq_s32(vacc01xCD, vreinterpretq_s8_u64(va01x01234567), vbCDx01234567);
        vacc01xEF = vmmlaq_s32(vacc01xEF, vreinterpretq_s8_u64(va01x01234567), vbEFx01234567);
        vacc23x01 = vmmlaq_s32(vacc23x01, vreinterpretq_s8_u64(va23x01234567), vb01x01234567);
        vacc23x23 = vmmlaq_s32(vacc23x23, vreinterpretq_s8_u64(va23x01234567), vb23x01234567);
        vacc23x45 = vmmlaq_s32(vacc23x45, vreinterpretq_s8_u64(va23x01234567), vb45x01234567);
        vacc23x67 = vmmlaq_s32(vacc23x67, vreinterpretq_s8_u64(va23x01234567), vb67x01234567);
        vacc23x89 = vmmlaq_s32(vacc23x89, vreinterpretq_s8_u64(va23x01234567), vb89x01234567);
        vacc23xAB = vmmlaq_s32(vacc23xAB, vreinterpretq_s8_u64(va23x01234567), vbABx01234567);
        vacc23xCD = vmmlaq_s32(vacc23xCD, vreinterpretq_s8_u64(va23x01234567), vbCDx01234567);
        vacc23xEF = vmmlaq_s32(vacc23xEF, vreinterpretq_s8_u64(va23x01234567), vbEFx01234567);
        vacc45x01 = vmmlaq_s32(vacc45x01, vreinterpretq_s8_u64(va45x01234567), vb01x01234567);
        vacc45x23 = vmmlaq_s32(vacc45x23, vreinterpretq_s8_u64(va45x01234567), vb23x01234567);
        vacc45x45 = vmmlaq_s32(vacc45x45, vreinterpretq_s8_u64(va45x01234567), vb45x01234567);
        vacc45x67 = vmmlaq_s32(vacc45x67, vreinterpretq_s8_u64(va45x01234567), vb67x01234567);
        vacc45x89 = vmmlaq_s32(vacc45x89, vreinterpretq_s8_u64(va45x01234567), vb89x01234567);
        vacc45xAB = vmmlaq_s32(vacc45xAB, vreinterpretq_s8_u64(va45x01234567), vbABx01234567);
        vacc45xCD = vmmlaq_s32(vacc45xCD, vreinterpretq_s8_u64(va45x01234567), vbCDx01234567);
        vacc45xEF = vmmlaq_s32(vacc45xEF, vreinterpretq_s8_u64(va45x01234567), vbEFx01234567);
      }

      p -= 6 * sizeof(void*);
    } while (p != 0);

    int32x4_t vacc0x0123 = vcombine_s32(vget_low_s32(vacc01x01), vget_low_s32(vacc01x23));
    int32x4_t vacc1x0123 = vcombine_s32(vget_high_s32(vacc01x01), vget_high_s32(vacc01x23));
    int32x4_t vacc0x4567 = vcombine_s32(vget_low_s32(vacc01x45), vget_low_s32(vacc01x67));
    int32x4_t vacc1x4567 = vcombine_s32(vget_high_s32(vacc01x45), vget_high_s32(vacc01x67));
    int32x4_t vacc0x89AB = vcombine_s32(vget_low_s32(vacc01x89), vget_low_s32(vacc01xAB));
    int32x4_t vacc1x89AB = vcombine_s32(vget_high_s32(vacc01x89), vget_high_s32(vacc01xAB));
    int32x4_t vacc0xCDEF = vcombine_s32(vget_low_s32(vacc01xCD), vget_low_s32(vacc01xEF));
    int32x4_t vacc1xCDEF = vcombine_s32(vget_high_s32(vacc01xCD), vget_high_s32(vacc01xEF));
    int32x4_t vacc2x0123 = vcombine_s32(vget_low_s32(vacc23x01), vget_low_s32(vacc23x23));
    int32x4_t vacc3x0123 = vcombine_s32(vget_high_s32(vacc23x01), vget_high_s32(vacc23x23));
    int32x4_t vacc2x4567 = vcombine_s32(vget_low_s32(vacc23x45), vget_low_s32(vacc23x67));
    int32x4_t vacc3x4567 = vcombine_s32(vget_high_s32(vacc23x45), vget_high_s32(vacc23x67));
    int32x4_t vacc2x89AB = vcombine_s32(vget_low_s32(vacc23x89), vget_low_s32(vacc23xAB));
    int32x4_t vacc3x89AB = vcombine_s32(vget_high_s32(vacc23x89), vget_high_s32(vacc23xAB));
    int32x4_t vacc2xCDEF = vcombine_s32(vget_low_s32(vacc23xCD), vget_low_s32(vacc23xEF));
    int32x4_t vacc3xCDEF = vcombine_s32(vget_high_s32(vacc23xCD), vget_high_s32(vacc23xEF));
    int32x4_t vacc4x0123 = vcombine_s32(vget_low_s32(vacc45x01), vget_low_s32(vacc45x23));
    int32x4_t vacc5x0123 = vcombine_s32(vget_high_s32(vacc45x01), vget_high_s32(vacc45x23));
    int32x4_t vacc4x4567 = vcombine_s32(vget_low_s32(vacc45x45), vget_low_s32(vacc45x67));
    int32x4_t vacc5x4567 = vcombine_s32(vget_high_s32(vacc45x45), vget_high_s32(vacc45x67));
    int32x4_t vacc4x89AB = vcombine_s32(vget_low_s32(vacc45x89), vget_low_s32(vacc45xAB));
    int32x4_t vacc5x89AB = vcombine_s32(vget_high_s32(vacc45x89), vget_high_s32(vacc45xAB));
    int32x4_t vacc4xCDEF = vcombine_s32(vget_low_s32(vacc45xCD), vget_low_s32(vacc45xEF));
    int32x4_t vacc5xCDEF = vcombine_s32(vget_high_s32(vacc45xCD), vget_high_s32(vacc45xEF));

    const float32x4_t vinput_scale = vld1q_dup_f32(&quantization_params->inv_scale);
    float32x4_t vout0x0123 = vcvtq_f32_s32(vacc0x0123);
    float32x4_t vout0x4567 = vcvtq_f32_s32(vacc0x4567);
    float32x4_t vout0x89AB = vcvtq_f32_s32(vacc0x89AB);
    float32x4_t vout0xCDEF = vcvtq_f32_s32(vacc0xCDEF);
    float32x4_t vout1x0123 = vcvtq_f32_s32(vacc1x0123);
    float32x4_t vout1x4567 = vcvtq_f32_s32(vacc1x4567);
    float32x4_t vout1x89AB = vcvtq_f32_s32(vacc1x89AB);
    float32x4_t vout1xCDEF = vcvtq_f32_s32(vacc1xCDEF);
    float32x4_t vout2x0123 = vcvtq_f32_s32(vacc2x0123);
    float32x4_t vout2x4567 = vcvtq_f32_s32(vacc2x4567);
    float32x4_t vout2x89AB = vcvtq_f32_s32(vacc2x89AB);
    float32x4_t vout2xCDEF = vcvtq_f32_s32(vacc2xCDEF);
    float32x4_t vout3x0123 = vcvtq_f32_s32(vacc3x0123);
    float32x4_t vout3x4567 = vcvtq_f32_s32(vacc3x4567);
    float32x4_t vout3x89AB = vcvtq_f32_s32(vacc3x89AB);
    float32x4_t vout3xCDEF = vcvtq_f32_s32(vacc3xCDEF);
    float32x4_t vout4x0123 = vcvtq_f32_s32(vacc4x0123);
    float32x4_t vout4x4567 = vcvtq_f32_s32(vacc4x4567);
    float32x4_t vout4x89AB = vcvtq_f32_s32(vacc4x89AB);
    float32x4_t vout4xCDEF = vcvtq_f32_s32(vacc4xCDEF);
    float32x4_t vout5x0123 = vcvtq_f32_s32(vacc5x0123);
    float32x4_t vout5x4567 = vcvtq_f32_s32(vacc5x4567);
    float32x4_t vout5x89AB = vcvtq_f32_s32(vacc5x89AB);
    float32x4_t vout5xCDEF = vcvtq_f32_s32(vacc5xCDEF);
    vout0x0123 = vmulq_f32(vout0x0123, vinput_scale);
    vout1x0123 = vmulq_f32(vout1x0123, vinput_scale);
    vout0x4567 = vmulq_f32(vout0x4567, vinput_scale);
    vout1x4567 = vmulq_f32(vout1x4567, vinput_scale);
    vout0x89AB = vmulq_f32(vout0x89AB, vinput_scale);
    vout1x89AB = vmulq_f32(vout1x89AB, vinput_scale);
    vout0xCDEF = vmulq_f32(vout0xCDEF, vinput_scale);
    vout1xCDEF = vmulq_f32(vout1xCDEF, vinput_scale);
    vout2x0123 = vmulq_f32(vout2x0123, vinput_scale);
    vout3x0123 = vmulq_f32(vout3x0123, vinput_scale);
    vout2x4567 = vmulq_f32(vout2x4567, vinput_scale);
    vout3x4567 = vmulq_f32(vout3x4567, vinput_scale);
    vout2x89AB = vmulq_f32(vout2x89AB, vinput_scale);
    vout3x89AB = vmulq_f32(vout3x89AB, vinput_scale);
    vout2xCDEF = vmulq_f32(vout2xCDEF, vinput_scale);
    vout3xCDEF = vmulq_f32(vout3xCDEF, vinput_scale);
    vout4x0123 = vmulq_f32(vout4x0123, vinput_scale);
    vout5x0123 = vmulq_f32(vout5x0123, vinput_scale);
    vout4x4567 = vmulq_f32(vout4x4567, vinput_scale);
    vout5x4567 = vmulq_f32(vout5x4567, vinput_scale);
    vout4x89AB = vmulq_f32(vout4x89AB, vinput_scale);
    vout5x89AB = vmulq_f32(vout5x89AB, vinput_scale);
    vout4xCDEF = vmulq_f32(vout4xCDEF, vinput_scale);
    vout5xCDEF = vmulq_f32(vout5xCDEF, vinput_scale);

    const float32x4_t vfilter_output_scale0123 = vld1q_f32(w); w = (const float*) w + 4;
    const float32x4_t vfilter_output_scale4567 = vld1q_f32(w); w = (const float*) w + 4;
    const float32x4_t vfilter_output_scale89AB = vld1q_f32(w); w = (const float*) w + 4;
    const float32x4_t vfilter_output_scaleCDEF = vld1q_f32(w); w = (const float*) w + 4;

    const float32x4_t vbias0123 = vld1q_f32(w); w = (const float*) w + 4;
    vout0x0123 = vfmaq_f32(vbias0123, vout0x0123, vfilter_output_scale0123);
    vout1x0123 = vfmaq_f32(vbias0123, vout1x0123, vfilter_output_scale0123);
    vout2x0123 = vfmaq_f32(vbias0123, vout2x0123, vfilter_output_scale0123);
    vout3x0123 = vfmaq_f32(vbias0123, vout3x0123, vfilter_output_scale0123);
    vout4x0123 = vfmaq_f32(vbias0123, vout4x0123, vfilter_output_scale0123);
    vout5x0123 = vfmaq_f32(vbias0123, vout5x0123, vfilter_output_scale0123);
    const float32x4_t vbias4567 = vld1q_f32(w); w = (const float*) w + 4;
    vout0x4567 = vfmaq_f32(vbias4567, vout0x4567, vfilter_output_scale4567);
    vout1x4567 = vfmaq_f32(vbias4567, vout1x4567, vfilter_output_scale4567);
    vout2x4567 = vfmaq_f32(vbias4567, vout2x4567, vfilter_output_scale4567);
    vout3x4567 = vfmaq_f32(vbias4567, vout3x4567, vfilter_output_scale4567);
    vout4x4567 = vfmaq_f32(vbias4567, vout4x4567, vfilter_output_scale4567);
    vout5x4567 = vfmaq_f32(vbias4567, vout5x4567, vfilter_output_scale4567);
    const float32x4_t vbias89AB = vld1q_f32(w); w = (const float*) w + 4;
    vout0x89AB = vfmaq_f32(vbias89AB, vout0x89AB, vfilter_output_scale89AB);
    vout1x89AB = vfmaq_f32(vbias89AB, vout1x89AB, vfilter_output_scale89AB);
    vout2x89AB = vfmaq_f32(vbias89AB, vout2x89AB, vfilter_output_scale89AB);
    vout3x89AB = vfmaq_f32(vbias89AB, vout3x89AB, vfilter_output_scale89AB);
    vout4x89AB = vfmaq_f32(vbias89AB, vout4x89AB, vfilter_output_scale89AB);
    vout5x89AB = vfmaq_f32(vbias89AB, vout5x89AB, vfilter_output_scale89AB);
    const float32x4_t vbiasCDEF = vld1q_f32(w); w = (const float*) w + 4;
    vout0xCDEF = vfmaq_f32(vbiasCDEF, vout0xCDEF, vfilter_output_scaleCDEF);
    vout1xCDEF = vfmaq_f32(vbiasCDEF, vout1xCDEF, vfilter_output_scaleCDEF);
    vout2xCDEF = vfmaq_f32(vbiasCDEF, vout2xCDEF, vfilter_output_scaleCDEF);
    vout3xCDEF = vfmaq_f32(vbiasCDEF, vout3xCDEF, vfilter_output_scaleCDEF);
    vout4xCDEF = vfmaq_f32(vbiasCDEF, vout4xCDEF, vfilter_output_scaleCDEF);
    vout5xCDEF = vfmaq_f32(vbiasCDEF, vout5xCDEF, vfilter_output_scaleCDEF);

    float16x8_t vfp16out5x01234567 = vcombine_f16(vcvt_f16_f32(vout5x0123), vcvt_f16_f32(vout5x4567));
    float16x8_t vfp16out5x89ABCDEF = vcombine_f16(vcvt_f16_f32(vout5x89AB), vcvt_f16_f32(vout5xCDEF));
    float16x8_t vfp16out4x01234567 = vcombine_f16(vcvt_f16_f32(vout4x0123), vcvt_f16_f32(vout4x4567));
    float16x8_t vfp16out4x89ABCDEF = vcombine_f16(vcvt_f16_f32(vout4x89AB), vcvt_f16_f32(vout4xCDEF));
    float16x8_t vfp16out3x01234567 = vcombine_f16(vcvt_f16_f32(vout3x0123), vcvt_f16_f32(vout3x4567));
    float16x8_t vfp16out3x89ABCDEF = vcombine_f16(vcvt_f16_f32(vout3x89AB), vcvt_f16_f32(vout3xCDEF));
    float16x8_t vfp16out2x01234567 = vcombine_f16(vcvt_f16_f32(vout2x0123), vcvt_f16_f32(vout2x4567));
    float16x8_t vfp16out2x89ABCDEF = vcombine_f16(vcvt_f16_f32(vout2x89AB), vcvt_f16_f32(vout2xCDEF));
    float16x8_t vfp16out1x01234567 = vcombine_f16(vcvt_f16_f32(vout1x0123), vcvt_f16_f32(vout1x4567));
    float16x8_t vfp16out1x89ABCDEF = vcombine_f16(vcvt_f16_f32(vout1x89AB), vcvt_f16_f32(vout1xCDEF));
    float16x8_t vfp16out0x01234567 = vcombine_f16(vcvt_f16_f32(vout0x0123), vcvt_f16_f32(vout0x4567));
    float16x8_t vfp16out0x89ABCDEF = vcombine_f16(vcvt_f16_f32(vout0x89AB), vcvt_f16_f32(vout0xCDEF));
    const float16x8_t voutput_min = vreinterpretq_f16_u16(vld1q_dup_u16((const uint16_t*) &params->scalar.min));
    vfp16out5x01234567 = vmaxq_f16(vfp16out5x01234567, voutput_min);
    vfp16out5x89ABCDEF = vmaxq_f16(vfp16out5x89ABCDEF, voutput_min);
    vfp16out4x01234567 = vmaxq_f16(vfp16out4x01234567, voutput_min);
    vfp16out4x89ABCDEF = vmaxq_f16(vfp16out4x89ABCDEF, voutput_min);
    vfp16out3x01234567 = vmaxq_f16(vfp16out3x01234567, voutput_min);
    vfp16out3x89ABCDEF = vmaxq_f16(vfp16out3x89ABCDEF, voutput_min);
    vfp16out2x01234567 = vmaxq_f16(vfp16out2x01234567, voutput_min);
    vfp16out2x89ABCDEF = vmaxq_f16(vfp16out2x89ABCDEF, voutput_min);
    vfp16out1x01234567 = vmaxq_f16(vfp16out1x01234567, voutput_min);
    vfp16out1x89ABCDEF = vmaxq_f16(vfp16out1x89ABCDEF, voutput_min);
    vfp16out0x01234567 = vmaxq_f16(vfp16out0x01234567, voutput_min);
    vfp16out0x89ABCDEF = vmaxq_f16(vfp16out0x89ABCDEF, voutput_min);
    const float16x8_t voutput_max = vreinterpretq_f16_u16(vld1q_dup_u16((const uint16_t*) &params->scalar.max));
    vfp16out5x01234567 = vminq_f16(vfp16out5x01234567, voutput_max);
    vfp16out5x89ABCDEF = vminq_f16(vfp16out5x89ABCDEF, voutput_max);
    vfp16out4x01234567 = vminq_f16(vfp16out4x01234567, voutput_max);
    vfp16out4x89ABCDEF = vminq_f16(vfp16out4x89ABCDEF, voutput_max);
    vfp16out3x01234567 = vminq_f16(vfp16out3x01234567, voutput_max);
    vfp16out3x89ABCDEF = vminq_f16(vfp16out3x89ABCDEF, voutput_max);
    vfp16out2x01234567 = vminq_f16(vfp16out2x01234567, voutput_max);
    vfp16out2x89ABCDEF = vminq_f16(vfp16out2x89ABCDEF, voutput_max);
    vfp16out1x01234567 = vminq_f16(vfp16out1x01234567, voutput_max);
    vfp16out1x89ABCDEF = vminq_f16(vfp16out1x89ABCDEF, voutput_max);
    vfp16out0x01234567 = vminq_f16(vfp16out0x01234567, voutput_max);
    vfp16out0x89ABCDEF = vminq_f16(vfp16out0x89ABCDEF, voutput_max);

    if XNN_LIKELY(nc >= 16) {
      vst1q_u16(c5, vreinterpretq_u16_f16(vfp16out5x01234567));
      vst1q_u16(c5 + 8, vreinterpretq_u16_f16(vfp16out5x89ABCDEF));
      vst1q_u16(c4, vreinterpretq_u16_f16(vfp16out4x01234567));
      vst1q_u16(c4 + 8, vreinterpretq_u16_f16(vfp16out4x89ABCDEF));
      vst1q_u16(c3, vreinterpretq_u16_f16(vfp16out3x01234567));
      vst1q_u16(c3 + 8, vreinterpretq_u16_f16(vfp16out3x89ABCDEF));
      vst1q_u16(c2, vreinterpretq_u16_f16(vfp16out2x01234567));
      vst1q_u16(c2 + 8, vreinterpretq_u16_f16(vfp16out2x89ABCDEF));
      vst1q_u16(c1, vreinterpretq_u16_f16(vfp16out1x01234567));
      vst1q_u16(c1 + 8, vreinterpretq_u16_f16(vfp16out1x89ABCDEF));
      vst1q_u16(c0, vreinterpretq_u16_f16(vfp16out0x01234567));
      vst1q_u16(c0 + 8, vreinterpretq_u16_f16(vfp16out0x89ABCDEF));

      c0 = (uint16_t*) ((uintptr_t) c0 + cn_stride);
      c1 = (uint16_t*) ((uintptr_t) c1 + cn_stride);
      c2 = (uint16_t*) ((uintptr_t) c2 + cn_stride);
      c3 = (uint16_t*) ((uintptr_t) c3 + cn_stride);
      c4 = (uint16_t*) ((uintptr_t) c4 + cn_stride);
      c5 = (uint16_t*) ((uintptr_t) c5 + cn_stride);

      a = (const int8_t**restrict) ((uintptr_t) a - ks);

      nc -= 16;
    } else {
     if (nc & 8) {
       vst1q_u16(c5, vreinterpretq_u16_f16(vfp16out5x01234567)); c5 += 8;
       vfp16out5x01234567 = vfp16out5x89ABCDEF;
       vst1q_u16(c4, vreinterpretq_u16_f16(vfp16out4x01234567)); c4 += 8;
       vfp16out4x01234567 = vfp16out4x89ABCDEF;
       vst1q_u16(c3, vreinterpretq_u16_f16(vfp16out3x01234567)); c3 += 8;
       vfp16out3x01234567 = vfp16out3x89ABCDEF;
       vst1q_u16(c2, vreinterpretq_u16_f16(vfp16out2x01234567)); c2 += 8;
       vfp16out2x01234567 = vfp16out2x89ABCDEF;
       vst1q_u16(c1, vreinterpretq_u16_f16(vfp16out1x01234567)); c1 += 8;
       vfp16out1x01234567 = vfp16out1x89ABCDEF;
       vst1q_u16(c0, vreinterpretq_u16_f16(vfp16out0x01234567)); c0 += 8;
       vfp16out0x01234567 = vfp16out0x89ABCDEF;
     }
     float16x4_t vfp16out5x0123 = vget_low_f16(vfp16out5x01234567);
     float16x4_t vfp16out4x0123 = vget_low_f16(vfp16out4x01234567);
     float16x4_t vfp16out3x0123 = vget_low_f16(vfp16out3x01234567);
     float16x4_t vfp16out2x0123 = vget_low_f16(vfp16out2x01234567);
     float16x4_t vfp16out1x0123 = vget_low_f16(vfp16out1x01234567);
     float16x4_t vfp16out0x0123 = vget_low_f16(vfp16out0x01234567);
     if (nc & 4) {
       vst1_u16(c5, vreinterpret_u16_f16(vfp16out5x0123)); c5 += 4;
       vst1_u16(c4, vreinterpret_u16_f16(vfp16out4x0123)); c4 += 4;
       vst1_u16(c3, vreinterpret_u16_f16(vfp16out3x0123)); c3 += 4;
       vst1_u16(c2, vreinterpret_u16_f16(vfp16out2x0123)); c2 += 4;
       vst1_u16(c1, vreinterpret_u16_f16(vfp16out1x0123)); c1 += 4;
       vst1_u16(c0, vreinterpret_u16_f16(vfp16out0x0123)); c0 += 4;
       vfp16out5x0123 = vget_high_f16(vfp16out5x01234567);
       vfp16out4x0123 = vget_high_f16(vfp16out4x01234567);
       vfp16out3x0123 = vget_high_f16(vfp16out3x01234567);
       vfp16out2x0123 = vget_high_f16(vfp16out2x01234567);
       vfp16out1x0123 = vget_high_f16(vfp16out1x01234567);
       vfp16out0x0123 = vget_high_f16(vfp16out0x01234567);
     }
     if (nc & 2) {
       vst1_lane_u32((void*) c5, vreinterpret_u32_f16(vfp16out5x0123), 0); c5 += 2;
       vst1_lane_u32((void*) c4, vreinterpret_u32_f16(vfp16out4x0123), 0); c4 += 2;
       vst1_lane_u32((void*) c3, vreinterpret_u32_f16(vfp16out3x0123), 0); c3 += 2;
       vst1_lane_u32((void*) c2, vreinterpret_u32_f16(vfp16out2x0123), 0); c2 += 2;
       vst1_lane_u32((void*) c1, vreinterpret_u32_f16(vfp16out1x0123), 0); c1 += 2;
       vst1_lane_u32((void*) c0, vreinterpret_u32_f16(vfp16out0x0123), 0); c0 += 2;
       vfp16out5x0123 = vext_f16(vfp16out5x0123, vfp16out5x0123, 2);
       vfp16out4x0123 = vext_f16(vfp16out4x0123, vfp16out4x0123, 2);
       vfp16out3x0123 = vext_f16(vfp16out3x0123, vfp16out3x0123, 2);
       vfp16out2x0123 = vext_f16(vfp16out2x0123, vfp16out2x0123, 2);
       vfp16out1x0123 = vext_f16(vfp16out1x0123, vfp16out1x0123, 2);
       vfp16out0x0123 = vext_f16(vfp16out0x0123, vfp16out0x0123, 2);
     }
     if (nc & 1) {
       vst1_lane_u16(c5, vreinterpret_u16_f16(vfp16out5x0123), 0);
       vst1_lane_u16(c4, vreinterpret_u16_f16(vfp16out4x0123), 0);
       vst1_lane_u16(c3, vreinterpret_u16_f16(vfp16out3x0123), 0);
       vst1_lane_u16(c2, vreinterpret_u16_f16(vfp16out2x0123), 0);
       vst1_lane_u16(c1, vreinterpret_u16_f16(vfp16out1x0123), 0);
       vst1_lane_u16(c0, vreinterpret_u16_f16(vfp16out0x0123), 0);
     }
      nc = 0;
    }
  } while (nc != 0);
}