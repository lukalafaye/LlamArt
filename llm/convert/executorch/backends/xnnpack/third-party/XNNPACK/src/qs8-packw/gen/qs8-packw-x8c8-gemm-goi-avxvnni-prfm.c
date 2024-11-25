// Auto-generated file. Do not edit!
//   Template: src/x8-packw/kr-avxvnni.c.in
//   Generator: tools/xngen
//
// Copyright 2024 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.


#include <assert.h>
#include <stddef.h>
#include <stdint.h>

#include <immintrin.h>

#include "xnnpack/packw.h"
#include "xnnpack/unaligned.h"
#include "xnnpack/prefetch.h"


void xnn_qs8_packw_gemm_goi_ukernel_x8c8__avxvnni_prfm(
  size_t g,
  size_t nc,
  size_t kc,
  size_t nr,
  size_t kr,
  size_t sr,
  const int8_t* weights,
  const int32_t* bias,
  const void* scale,
  int8_t* packed_weights,
  size_t extra_bytes,
  const void* params)
{
  assert(g != 0);
  assert(nc != 0);
  assert(kc != 0);
  assert(nr == 8);
  assert(kr == 8);
  assert(sr == 1);
  assert(weights != NULL);
  assert(packed_weights != NULL);

  const __m256i vone = _mm256_set1_epi8(1);
  int8_t* out = (int8_t*) packed_weights;
  const uint32_t* b = (const uint32_t*) bias;
  const uint32_t izp = (uint32_t) (params ? (((const struct xnn_qs8_packw_params*) params)->input_zero_point + 0): 0);
  __m256i vzeropoint = _mm256_set1_epi32((int32_t) izp);

  do {
    // NC main loop multiple of 8
    const int8_t* w0 = (const int8_t*) weights;
    size_t n = nc;
    for (;n >= 8; n -= 8) {
      __m256i vacc0124x8 = _mm256_setzero_si256();
      __m256i vacc4567x8 = _mm256_setzero_si256();

      int32_t* packed_b = (int32_t*) out;
      if XNN_LIKELY(b != NULL) {
        const __m256i vb = _mm256_loadu_si256((const __m256i*) b);
        _mm256_storeu_si256((__m256i*) out, vb);
        b += 8;
      } else {
        _mm256_storeu_si256((__m256i*) out, _mm256_setzero_si256());
      }
      out += 8 * sizeof(uint32_t);

      const int8_t* w1 = w0 + kc;
      const int8_t* w2 = w1 + kc;
      const int8_t* w3 = w2 + kc;
      const int8_t* w4 = w3 + kc;
      const int8_t* w5 = w4 + kc;
      const int8_t* w6 = w5 + kc;
      const int8_t* w7 = w6 + kc;
      xnn_prefetch_to_l1((const int8_t*) w0);
      xnn_prefetch_to_l1((const int8_t*) w0 + 64);
      xnn_prefetch_to_l1((const int8_t*) w1);
      xnn_prefetch_to_l1((const int8_t*) w1 + 64);
      xnn_prefetch_to_l1((const int8_t*) w2);
      xnn_prefetch_to_l1((const int8_t*) w2 + 64);
      xnn_prefetch_to_l1((const int8_t*) w3);
      xnn_prefetch_to_l1((const int8_t*) w3 + 64);
      xnn_prefetch_to_l1((const int8_t*) w4);
      xnn_prefetch_to_l1((const int8_t*) w4 + 64);
      xnn_prefetch_to_l1((const int8_t*) w5);
      xnn_prefetch_to_l1((const int8_t*) w5 + 64);
      xnn_prefetch_to_l1((const int8_t*) w6);
      xnn_prefetch_to_l1((const int8_t*) w6 + 64);
      xnn_prefetch_to_l1((const int8_t*) w7);
      xnn_prefetch_to_l1((const int8_t*) w7 + 64);

      // KC main loop multiple of 8x8
      size_t k = kc;
      for (; k >= 8; k -= 8) {
        __m256i v0123x8 = _mm256_set1_epi64x((int64_t) unaligned_load_u64(w0));
        v0123x8 = _mm256_blend_epi32(v0123x8, _mm256_set1_epi64x((int64_t) unaligned_load_u64(w1)), 0x0C);
        v0123x8 = _mm256_blend_epi32(v0123x8, _mm256_set1_epi64x((int64_t) unaligned_load_u64(w2)), 0x30);
        v0123x8 = _mm256_blend_epi32(v0123x8, _mm256_set1_epi64x((int64_t) unaligned_load_u64(w3)), 0xC0);

        __m256i v4567x8 = _mm256_set1_epi64x((int64_t) unaligned_load_u64(w4));
        v4567x8 = _mm256_blend_epi32(v4567x8, _mm256_set1_epi64x((int64_t) unaligned_load_u64(w5)), 0x0C);
        v4567x8 = _mm256_blend_epi32(v4567x8, _mm256_set1_epi64x((int64_t) unaligned_load_u64(w6)), 0x30);
        v4567x8 = _mm256_blend_epi32(v4567x8, _mm256_set1_epi64x((int64_t) unaligned_load_u64(w7)), 0xC0);
        xnn_prefetch_to_l1((const int8_t*) w0 + 128);
        xnn_prefetch_to_l1((const int8_t*) w1 + 128);
        xnn_prefetch_to_l1((const int8_t*) w2 + 128);
        xnn_prefetch_to_l1((const int8_t*) w3 + 128);
        xnn_prefetch_to_l1((const int8_t*) w4 + 128);
        xnn_prefetch_to_l1((const int8_t*) w5 + 128);
        xnn_prefetch_to_l1((const int8_t*) w6 + 128);
        xnn_prefetch_to_l1((const int8_t*) w7 + 128);

        vacc0124x8 = _mm256_dpbusd_avx_epi32(vacc0124x8, vone, v0123x8);
        vacc4567x8 = _mm256_dpbusd_avx_epi32(vacc4567x8, vone, v4567x8);

        _mm256_storeu_si256((__m256i *)&out[0],  v0123x8);
        _mm256_storeu_si256((__m256i *)&out[32],  v4567x8);

        w0 += 8;
        w1 += 8;
        w2 += 8;
        w3 += 8;
        w4 += 8;
        w5 += 8;
        w6 += 8;
        w7 += 8;
        out += 64;
      }

      // KC remainder 1..KR-1
      if (k != 0) {
        assert(k >= 1 && k <= 7);
        __m256i v0123x8 = _mm256_setzero_si256();
        __m256i v4567x8 = _mm256_setzero_si256();

        if (k & 4) {
          v0123x8 = _mm256_insert_epi32(v0123x8, *(const int32_t *)w0, 0);
          v0123x8 = _mm256_insert_epi32(v0123x8, *(const int32_t *)w1, 2);
          v0123x8 = _mm256_insert_epi32(v0123x8, *(const int32_t *)w2, 4);
          v0123x8 = _mm256_insert_epi32(v0123x8, *(const int32_t *)w3, 6);

          v4567x8 = _mm256_insert_epi32(v4567x8, *(const int32_t *)w4, 0);
          v4567x8 = _mm256_insert_epi32(v4567x8, *(const int32_t *)w5, 2);
          v4567x8 = _mm256_insert_epi32(v4567x8, *(const int32_t *)w6, 4);
          v4567x8 = _mm256_insert_epi32(v4567x8, *(const int32_t *)w7, 6);
          w0 += 4;
          w1 += 4;
          w2 += 4;
          w3 += 4;
          w4 += 4;
          w5 += 4;
          w6 += 4;
          w7 += 4;
        }
        if (k & 2) {
          if (k & 4) {
            v0123x8 = _mm256_insert_epi16(v0123x8, *(const int16_t *)w0, 2);
            v0123x8 = _mm256_insert_epi16(v0123x8, *(const int16_t *)w1, 6);
            v0123x8 = _mm256_insert_epi16(v0123x8, *(const int16_t *)w2, 10);
            v0123x8 = _mm256_insert_epi16(v0123x8, *(const int16_t *)w3, 14);

            v4567x8 = _mm256_insert_epi16(v4567x8, *(const int16_t *)w4, 2);
            v4567x8 = _mm256_insert_epi16(v4567x8, *(const int16_t *)w5, 6);
            v4567x8 = _mm256_insert_epi16(v4567x8, *(const int16_t *)w6, 10);
            v4567x8 = _mm256_insert_epi16(v4567x8, *(const int16_t *)w7, 14);
          } else {
            v0123x8 = _mm256_insert_epi16(v0123x8, *(const int16_t *)w0, 0);
            v0123x8 = _mm256_insert_epi16(v0123x8, *(const int16_t *)w1, 4);
            v0123x8 = _mm256_insert_epi16(v0123x8, *(const int16_t *)w2, 8);
            v0123x8 = _mm256_insert_epi16(v0123x8, *(const int16_t *)w3, 12);

            v4567x8 = _mm256_insert_epi16(v4567x8, *(const int16_t *)w4, 0);
            v4567x8 = _mm256_insert_epi16(v4567x8, *(const int16_t *)w5, 4);
            v4567x8 = _mm256_insert_epi16(v4567x8, *(const int16_t *)w6, 8);
            v4567x8 = _mm256_insert_epi16(v4567x8, *(const int16_t *)w7, 12);
          }

          w0 += 2;
          w1 += 2;
          w2 += 2;
          w3 += 2;
          w4 += 2;
          w5 += 2;
          w6 += 2;
          w7 += 2;
        }
        if (k & 1) {
          if ((k & 4) && (k & 2)) {
            v0123x8 = _mm256_insert_epi8(v0123x8, *(const int8_t *)w0, 6);
            v0123x8 = _mm256_insert_epi8(v0123x8, *(const int8_t *)w1, 14);
            v0123x8 = _mm256_insert_epi8(v0123x8, *(const int8_t *)w2, 22);
            v0123x8 = _mm256_insert_epi8(v0123x8, *(const int8_t *)w3, 30);

            v4567x8 = _mm256_insert_epi8(v4567x8, *(const int8_t *)w4, 6);
            v4567x8 = _mm256_insert_epi8(v4567x8, *(const int8_t *)w5, 14);
            v4567x8 = _mm256_insert_epi8(v4567x8, *(const int8_t *)w6, 22);
            v4567x8 = _mm256_insert_epi8(v4567x8, *(const int8_t *)w7, 30);
          }
          else if (k & 4) {
            v0123x8 = _mm256_insert_epi8(v0123x8, *(const int8_t *)w0, 4);
            v0123x8 = _mm256_insert_epi8(v0123x8, *(const int8_t *)w1, 12);
            v0123x8 = _mm256_insert_epi8(v0123x8, *(const int8_t *)w2, 20);
            v0123x8 = _mm256_insert_epi8(v0123x8, *(const int8_t *)w3, 28);

            v4567x8 = _mm256_insert_epi8(v4567x8, *(const int8_t *)w4, 4);
            v4567x8 = _mm256_insert_epi8(v4567x8, *(const int8_t *)w5, 12);
            v4567x8 = _mm256_insert_epi8(v4567x8, *(const int8_t *)w6, 20);
            v4567x8 = _mm256_insert_epi8(v4567x8, *(const int8_t *)w7, 28);
          }
          else if (k & 2) {
            v0123x8 = _mm256_insert_epi8(v0123x8, *(const int8_t *)w0, 2);
            v0123x8 = _mm256_insert_epi8(v0123x8, *(const int8_t *)w1, 10);
            v0123x8 = _mm256_insert_epi8(v0123x8, *(const int8_t *)w2, 18);
            v0123x8 = _mm256_insert_epi8(v0123x8, *(const int8_t *)w3, 26);

            v4567x8 = _mm256_insert_epi8(v4567x8, *(const int8_t *)w4, 2);
            v4567x8 = _mm256_insert_epi8(v4567x8, *(const int8_t *)w5, 10);
            v4567x8 = _mm256_insert_epi8(v4567x8, *(const int8_t *)w6, 18);
            v4567x8 = _mm256_insert_epi8(v4567x8, *(const int8_t *)w7, 26);
          }
          else {
            v0123x8 = _mm256_insert_epi8(v0123x8, *(const int8_t *)w0, 0);
            v0123x8 = _mm256_insert_epi8(v0123x8, *(const int8_t *)w1, 8);
            v0123x8 = _mm256_insert_epi8(v0123x8, *(const int8_t *)w2, 16);
            v0123x8 = _mm256_insert_epi8(v0123x8, *(const int8_t *)w3, 24);

            v4567x8 = _mm256_insert_epi8(v4567x8, *(const int8_t *)w4, 0);
            v4567x8 = _mm256_insert_epi8(v4567x8, *(const int8_t *)w5, 8);
            v4567x8 = _mm256_insert_epi8(v4567x8, *(const int8_t *)w6, 16);
            v4567x8 = _mm256_insert_epi8(v4567x8, *(const int8_t *)w7, 24);
          }

          w0 += 1;
          w1 += 1;
          w2 += 1;
          w3 += 1;
          w4 += 1;
          w5 += 1;
          w6 += 1;
          w7 += 1;
        }

        vacc0124x8 = _mm256_dpbusd_avx_epi32(vacc0124x8, vone, v0123x8);
        vacc4567x8 = _mm256_dpbusd_avx_epi32(vacc4567x8, vone, v4567x8);

        _mm256_storeu_si256((__m256i *)&out[0],  v0123x8);
        _mm256_storeu_si256((__m256i *)&out[32],  v4567x8);

        out += 64;
      }

      __m256i vksum = _mm256_hadd_epi32(vacc0124x8, vacc4567x8);
      vksum = _mm256_permute4x64_epi64(vksum, _MM_SHUFFLE(3, 1, 2, 0));
      vksum = _mm256_mullo_epi32(vksum, vzeropoint);
      __m256i vpack =  _mm256_loadu_si256((const __m256i*) packed_b);
      _mm256_storeu_si256((__m256i *)packed_b, _mm256_sub_epi32(vpack, vksum));
      out = (int8_t*) ((uintptr_t) out + extra_bytes);
      w0 = w7;
    }

    // NC remainder (1..7)
    if XNN_UNLIKELY(n != 0) {
      int32_t* packed_b = (int32_t*) out;
      if XNN_LIKELY(b != NULL) {
        size_t nb = n;
        do {
          *((uint32_t*) out) = *b++;
          out += sizeof(uint32_t);
        } while (--nb != 0);
      } else {
        size_t nb = n;
        do {
          *((uint32_t*) out) = 0;
          out += sizeof(uint32_t);
        } while (--nb != 0);
      }
      out += (8 - n) * sizeof(uint32_t);

     const int8_t* w1 = w0 + kc;
     if XNN_UNPREDICTABLE(n < 2) {
       w1 = w0;
     }
     const int8_t* w2 = w1 + kc;
     if XNN_UNPREDICTABLE(n <= 2) {
       w2 = w1;
     }
     const int8_t* w3 = w2 + kc;
     if XNN_UNPREDICTABLE(n < 4) {
       w3 = w2;
     }
     const int8_t* w4 = w3 + kc;
     if XNN_UNPREDICTABLE(n <= 4) {
       w4 = w3;
     }
     const int8_t* w5 = w4 + kc;
     if XNN_UNPREDICTABLE(n < 6) {
       w5 = w4;
     }
     const int8_t* w6 = w5 + kc;
     if XNN_UNPREDICTABLE(n <= 6) {
       w6 = w5;
     }
     const int8_t* w7 = w6 + kc;
     if XNN_UNPREDICTABLE(n < 8) {
       w7 = w6;
     }

     __m256i vacc0124x8 = _mm256_setzero_si256();
     __m256i vacc4567x8 = _mm256_setzero_si256();

     // KC main loop multiple of 8x8
     size_t k = kc;
     for (; k >= 8; k -= 8) {
       __m256i v0123x8 = _mm256_set1_epi64x((int64_t) unaligned_load_u64(w0));
       v0123x8 = _mm256_blend_epi32(v0123x8, _mm256_set1_epi64x((int64_t) unaligned_load_u64(w1)), 0x0C);
       v0123x8 = _mm256_blend_epi32(v0123x8, _mm256_set1_epi64x((int64_t) unaligned_load_u64(w2)), 0x30);
       v0123x8 = _mm256_blend_epi32(v0123x8, _mm256_set1_epi64x((int64_t) unaligned_load_u64(w3)), 0xC0);

       __m256i v4567x8 = _mm256_set1_epi64x((int64_t) unaligned_load_u64(w4));
       v4567x8 = _mm256_blend_epi32(v4567x8, _mm256_set1_epi64x((int64_t) unaligned_load_u64(w5)), 0x0C);
       v4567x8 = _mm256_blend_epi32(v4567x8, _mm256_set1_epi64x((int64_t) unaligned_load_u64(w6)), 0x30);
       v4567x8 = _mm256_blend_epi32(v4567x8, _mm256_set1_epi64x((int64_t) unaligned_load_u64(w7)), 0xC0);

       vacc0124x8 = _mm256_dpbusd_avx_epi32(vacc0124x8, vone, v0123x8);
       vacc4567x8 = _mm256_dpbusd_avx_epi32(vacc4567x8, vone, v4567x8);

       _mm256_storeu_si256((__m256i *)&out[0],  v0123x8);
       _mm256_storeu_si256((__m256i *)&out[32],  v4567x8);

       w0 += 8;
       w1 += 8;
       w2 += 8;
       w3 += 8;
       w4 += 8;
       w5 += 8;
       w6 += 8;
       w7 += 8;
       out += 64;
     }

     // KC remainder of 1..7
     if (k != 0) {
       __m256i v0123x8 = _mm256_setzero_si256();
       __m256i v4567x8 = _mm256_setzero_si256();

       if (k & 4) {
         v0123x8 = _mm256_insert_epi32(v0123x8, *(const int32_t *)w0, 0);
         v0123x8 = _mm256_insert_epi32(v0123x8, *(const int32_t *)w1, 2);
         v0123x8 = _mm256_insert_epi32(v0123x8, *(const int32_t *)w2, 4);
         v0123x8 = _mm256_insert_epi32(v0123x8, *(const int32_t *)w3, 6);

         v4567x8 = _mm256_insert_epi32(v4567x8, *(const int32_t *)w4, 0);
         v4567x8 = _mm256_insert_epi32(v4567x8, *(const int32_t *)w5, 2);
         v4567x8 = _mm256_insert_epi32(v4567x8, *(const int32_t *)w6, 4);
         v4567x8 = _mm256_insert_epi32(v4567x8, *(const int32_t *)w7, 6);
         w0 += 4;
         w1 += 4;
         w2 += 4;
         w3 += 4;
         w4 += 4;
         w5 += 4;
         w6 += 4;
         w7 += 4;
       }
       if (k & 2) {
         if (k & 4) {
           v0123x8 = _mm256_insert_epi16(v0123x8, *(const int16_t *)w0, 2);
           v0123x8 = _mm256_insert_epi16(v0123x8, *(const int16_t *)w1, 6);
           v0123x8 = _mm256_insert_epi16(v0123x8, *(const int16_t *)w2, 10);
           v0123x8 = _mm256_insert_epi16(v0123x8, *(const int16_t *)w3, 14);

           v4567x8 = _mm256_insert_epi16(v4567x8, *(const int16_t *)w4, 2);
           v4567x8 = _mm256_insert_epi16(v4567x8, *(const int16_t *)w5, 6);
           v4567x8 = _mm256_insert_epi16(v4567x8, *(const int16_t *)w6, 10);
           v4567x8 = _mm256_insert_epi16(v4567x8, *(const int16_t *)w7, 14);
         } else {
           v0123x8 = _mm256_insert_epi16(v0123x8, *(const int16_t *)w0, 0);
           v0123x8 = _mm256_insert_epi16(v0123x8, *(const int16_t *)w1, 4);
           v0123x8 = _mm256_insert_epi16(v0123x8, *(const int16_t *)w2, 8);
           v0123x8 = _mm256_insert_epi16(v0123x8, *(const int16_t *)w3, 12);

           v4567x8 = _mm256_insert_epi16(v4567x8, *(const int16_t *)w4, 0);
           v4567x8 = _mm256_insert_epi16(v4567x8, *(const int16_t *)w5, 4);
           v4567x8 = _mm256_insert_epi16(v4567x8, *(const int16_t *)w6, 8);
           v4567x8 = _mm256_insert_epi16(v4567x8, *(const int16_t *)w7, 12);
         }

         w0 += 2;
         w1 += 2;
         w2 += 2;
         w3 += 2;
         w4 += 2;
         w5 += 2;
         w6 += 2;
         w7 += 2;
       }
       if (k & 1) {
         if ((k & 4) && (k & 2)) {
           v0123x8 = _mm256_insert_epi8(v0123x8, *(const int8_t *)w0, 6);
           v0123x8 = _mm256_insert_epi8(v0123x8, *(const int8_t *)w1, 14);
           v0123x8 = _mm256_insert_epi8(v0123x8, *(const int8_t *)w2, 22);
           v0123x8 = _mm256_insert_epi8(v0123x8, *(const int8_t *)w3, 30);

           v4567x8 = _mm256_insert_epi8(v4567x8, *(const int8_t *)w4, 6);
           v4567x8 = _mm256_insert_epi8(v4567x8, *(const int8_t *)w5, 14);
           v4567x8 = _mm256_insert_epi8(v4567x8, *(const int8_t *)w6, 22);
           v4567x8 = _mm256_insert_epi8(v4567x8, *(const int8_t *)w7, 30);
         }
         else if (k & 4) {
           v0123x8 = _mm256_insert_epi8(v0123x8, *(const int8_t *)w0, 4);
           v0123x8 = _mm256_insert_epi8(v0123x8, *(const int8_t *)w1, 12);
           v0123x8 = _mm256_insert_epi8(v0123x8, *(const int8_t *)w2, 20);
           v0123x8 = _mm256_insert_epi8(v0123x8, *(const int8_t *)w3, 28);

           v4567x8 = _mm256_insert_epi8(v4567x8, *(const int8_t *)w4, 4);
           v4567x8 = _mm256_insert_epi8(v4567x8, *(const int8_t *)w5, 12);
           v4567x8 = _mm256_insert_epi8(v4567x8, *(const int8_t *)w6, 20);
           v4567x8 = _mm256_insert_epi8(v4567x8, *(const int8_t *)w7, 28);
         }
         else if (k & 2) {
           v0123x8 = _mm256_insert_epi8(v0123x8, *(const int8_t *)w0, 2);
           v0123x8 = _mm256_insert_epi8(v0123x8, *(const int8_t *)w1, 10);
           v0123x8 = _mm256_insert_epi8(v0123x8, *(const int8_t *)w2, 18);
           v0123x8 = _mm256_insert_epi8(v0123x8, *(const int8_t *)w3, 26);

           v4567x8 = _mm256_insert_epi8(v4567x8, *(const int8_t *)w4, 2);
           v4567x8 = _mm256_insert_epi8(v4567x8, *(const int8_t *)w5, 10);
           v4567x8 = _mm256_insert_epi8(v4567x8, *(const int8_t *)w6, 18);
           v4567x8 = _mm256_insert_epi8(v4567x8, *(const int8_t *)w7, 26);
         }
         else {
           v0123x8 = _mm256_insert_epi8(v0123x8, *(const int8_t *)w0, 0);
           v0123x8 = _mm256_insert_epi8(v0123x8, *(const int8_t *)w1, 8);
           v0123x8 = _mm256_insert_epi8(v0123x8, *(const int8_t *)w2, 16);
           v0123x8 = _mm256_insert_epi8(v0123x8, *(const int8_t *)w3, 24);

           v4567x8 = _mm256_insert_epi8(v4567x8, *(const int8_t *)w4, 0);
           v4567x8 = _mm256_insert_epi8(v4567x8, *(const int8_t *)w5, 8);
           v4567x8 = _mm256_insert_epi8(v4567x8, *(const int8_t *)w6, 16);
           v4567x8 = _mm256_insert_epi8(v4567x8, *(const int8_t *)w7, 24);
         }

         w0 += 1;
         w1 += 1;
         w2 += 1;
         w3 += 1;
         w4 += 1;
         w5 += 1;
         w6 += 1;
         w7 += 1;
       }

       vacc0124x8 = _mm256_dpbusd_avx_epi32(vacc0124x8, vone, v0123x8);
       vacc4567x8 = _mm256_dpbusd_avx_epi32(vacc4567x8, vone, v4567x8);

       _mm256_storeu_si256((__m256i *)&out[0],  v0123x8);
       _mm256_storeu_si256((__m256i *)&out[32],  v4567x8);

       out += 64;
     }

     __m256i vksum = _mm256_hadd_epi32(vacc0124x8, vacc4567x8);
     vksum = _mm256_permute4x64_epi64(vksum, _MM_SHUFFLE(3, 1, 2, 0));
     vksum = _mm256_mullo_epi32(vksum, vzeropoint);
     __m256i vpack =  _mm256_loadu_si256((const __m256i*) packed_b);
     _mm256_storeu_si256((__m256i *)packed_b, _mm256_sub_epi32(vpack, vksum));
     out = (int8_t*) ((uintptr_t) out + extra_bytes);
    }
    weights += nc * kc;
  } while (--g != 0);
}
