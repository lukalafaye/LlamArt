// Auto-generated file. Do not edit!
//   Template: src/f32-velu/scalar-rr2-p6.c.in
//   Generator: tools/xngen
//
// Copyright 2020 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
#include <math.h>

#include "xnnpack/common.h"
#include "xnnpack/math.h"
#include "xnnpack/vunary.h"


void xnn_f32_velu_ukernel__wasm_rr2_p6_u2(
    size_t batch,
    const float* input,
    float* output,
    const struct xnn_f32_elu_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input != NULL);
  assert(output != NULL);

  const float vsat_cutoff = -0x1.154246p+4f;
  const float vmagic_bias = 0x1.8000FEp23f;
  const float vlog2e = 0x1.715476p+0f;
  const float vminus_ln2_hi = -0x1.62E440p-1f;
  const float vminus_ln2_lo = 0x1.0105C6p-21f;
  const float vc6 = 0x1.6b7338p-10f;
  const float vc5 = 0x1.12278Ep-7f;
  const float vc4 = 0x1.555716p-5f;
  const float vc3 = 0x1.5554B0p-3f;
  const float vc2 = 0x1.FFFFFEp-2f;
  const float vone = 1.0f;

  const float vprescale = params->scalar.prescale;
  const float valpha = params->scalar.alpha;
  const float vbeta = params->scalar.beta;

  for (; batch >= 2 * sizeof(float); batch -= 2 * sizeof(float)) {
    float vx0 = input[0];
    float vx1 = input[1];
    input += 2;

    const float vz0 = __builtin_wasm_min_f32(__builtin_wasm_max_f32(vx0 * vprescale, vsat_cutoff), 0.0f);
    const float vz1 = __builtin_wasm_min_f32(__builtin_wasm_max_f32(vx1 * vprescale, vsat_cutoff), 0.0f);

    float vn0 = vz0 * vlog2e + vmagic_bias;
    float vn1 = vz1 * vlog2e + vmagic_bias;

    float vs0 = uint32_as_float(float_as_uint32(vn0) << 23);
    vn0 -= vmagic_bias;
    float vs1 = uint32_as_float(float_as_uint32(vn1) << 23);
    vn1 -= vmagic_bias;

    float vt0 = vn0 * vminus_ln2_hi + vz0;
    float vt1 = vn1 * vminus_ln2_hi + vz1;

    vt0 = vn0 * vminus_ln2_lo + vt0;
    vt1 = vn1 * vminus_ln2_lo + vt1;


    float vp0 = vc6 * vt0 + vc5;
    float vp1 = vc6 * vt1 + vc5;

    vp0 = vp0 * vt0 + vc4;
    vp1 = vp1 * vt1 + vc4;

    vp0 = vp0 * vt0 + vc3;
    vp1 = vp1 * vt1 + vc3;

    vp0 = vp0 * vt0 + vc2;
    vp1 = vp1 * vt1 + vc2;

    vp0 *= vt0;
    vp1 *= vt1;

    vt0 *= vs0;
    vs0 -= vone;
    vt1 *= vs1;
    vs1 -= vone;

    vp0 = vp0 * vt0 + vt0;
    vp1 = vp1 * vt1 + vt1;

    const float ve0 = (vp0 + vs0) * valpha;
    float vy0 = __builtin_wasm_max_f32(vx0 * vbeta, 0.0f);
    const float ve1 = (vp1 + vs1) * valpha;
    float vy1 = __builtin_wasm_max_f32(vx1 * vbeta, 0.0f);

    vy0 += __builtin_wasm_min_f32(ve0, 0.0f);
    vy1 += __builtin_wasm_min_f32(ve1, 0.0f);

    output[0] = vy0;
    output[1] = vy1;
    output += 2;
  }
  if XNN_UNLIKELY(batch != 0) {
    float vx = *input;

    const float vz = __builtin_wasm_min_f32(__builtin_wasm_max_f32(vx * vprescale, vsat_cutoff), 0.0f);

    float vn = vz * vlog2e + vmagic_bias;
    float vs = uint32_as_float(float_as_uint32(vn) << 23);
    vn -= vmagic_bias;

    float vt = vn * vminus_ln2_hi + vz;
    vt = vn * vminus_ln2_lo + vt;


    float vp = vc6 * vt + vc5;
    vp = vp * vt + vc4;
    vp = vp * vt + vc3;
    vp = vp * vt + vc2;
    vp *= vt;

    vt *= vs;
    vs -= vone;
    vp = vp * vt + vt;
    const float ve = (vp + vs) * valpha;

    float vy = __builtin_wasm_max_f32(vx * vbeta, 0.0f);
    vy += __builtin_wasm_min_f32(ve, 0.0f);

    *output = vy;
  }
}