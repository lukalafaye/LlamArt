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
/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
#ifndef TENSORFLOW_CONTRIB_LITE_KERNELS_INTERNAL_COMPATIBILITY_H_
#define TENSORFLOW_CONTRIB_LITE_KERNELS_INTERNAL_COMPATIBILITY_H_

#include <cassert>
#include <cstdint>
#include <cstdlib>

#ifndef TFLITE_DCHECK
#define TFLITE_DCHECK(condition) (condition) ? (void)0 : assert(false)
#endif

#ifndef TFLITE_DCHECK_EQ
#define TFLITE_DCHECK_EQ(x, y) ((x) == (y)) ? (void)0 : assert(false)
#endif

#ifndef TFLITE_DCHECK_NE
#define TFLITE_DCHECK_NE(x, y) ((x) != (y)) ? (void)0 : assert(false)
#endif

#ifndef TFLITE_DCHECK_GE
#define TFLITE_DCHECK_GE(x, y) ((x) >= (y)) ? (void)0 : assert(false)
#endif

#ifndef TFLITE_DCHECK_GT
#define TFLITE_DCHECK_GT(x, y) ((x) > (y)) ? (void)0 : assert(false)
#endif

#ifndef TFLITE_DCHECK_LE
#define TFLITE_DCHECK_LE(x, y) ((x) <= (y)) ? (void)0 : assert(false)
#endif

#ifndef TFLITE_DCHECK_LT
#define TFLITE_DCHECK_LT(x, y) ((x) < (y)) ? (void)0 : assert(false)
#endif

// TODO(ahentz): Clean up: We should stick to the DCHECK versions.
#ifndef TFLITE_CHECK
#define TFLITE_CHECK(condition) (condition) ? (void)0 : abort()
#endif

#ifndef TFLITE_CHECK_EQ
#define TFLITE_CHECK_EQ(x, y) ((x) == (y)) ? (void)0 : abort()
#endif

#ifndef TFLITE_CHECK_NE
#define TFLITE_CHECK_NE(x, y) ((x) != (y)) ? (void)0 : abort()
#endif

#ifndef TFLITE_CHECK_GE
#define TFLITE_CHECK_GE(x, y) ((x) >= (y)) ? (void)0 : abort()
#endif

#ifndef TFLITE_CHECK_GT
#define TFLITE_CHECK_GT(x, y) ((x) > (y)) ? (void)0 : abort()
#endif

#ifndef TFLITE_CHECK_LE
#define TFLITE_CHECK_LE(x, y) ((x) <= (y)) ? (void)0 : abort()
#endif

#ifndef TFLITE_CHECK_LT
#define TFLITE_CHECK_LT(x, y) ((x) < (y)) ? (void)0 : abort()
#endif

// TODO(ahentz): Clean up.
using uint8 = std::uint8_t;
using int16 = std::int16_t;
using uint16 = std::uint16_t;
using int32 = std::int32_t;
using uint32 = std::uint32_t;

#endif  // TENSORFLOW_CONTRIB_LITE_KERNELS_INTERNAL_COMPATIBILITY_H_
