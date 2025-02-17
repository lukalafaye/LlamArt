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

// instrumentation.h: contains the definitions needed to
// instrument code for profiling:
//   ScopedProfilingLabel, RegisterCurrentThreadForProfiling.
//
// profiler.h is only needed to drive the profiler:
//   StartProfiling, FinishProfiling.
//
// See the usage example in profiler.h.

#ifndef GEMMLOWP_PROFILING_INSTRUMENTATION_H_
#define GEMMLOWP_PROFILING_INSTRUMENTATION_H_

#ifndef HIFI_BUILD
#include <cstdio>
#endif //HiFi_BUILD

#ifndef GEMMLOWP_USE_STLPORT
#include <cstdint>
#else
#include <stdint.h>
namespace std {
using ::int16_t;
using ::int32_t;
using ::int8_t;
using ::size_t;
using ::uint16_t;
using ::uint32_t;
using ::uint8_t;
using ::uintptr_t;
}  // namespace std
#endif

#include <algorithm>
#include <cassert>
#include <cstdlib>

#ifdef GEMMLOWP_PROFILING
#include <cstring>
#include <set>
#endif

#include "./pthread_everywhere.h"

namespace gemmlowp {

inline void ReleaseBuildAssertion(bool condition, const char* msg) {
  if (!condition) {
    fprintf(stderr, "gemmlowp error: %s\n", msg);
    abort();
  }
}

#ifndef HIFI_BUILD
class Mutex {
 public:
  Mutex(const Mutex&) = delete;
  Mutex& operator=(const Mutex&) = delete;

  Mutex() { pthread_mutex_init(&m, NULL); }
  ~Mutex() { pthread_mutex_destroy(&m); }

  void Lock() { pthread_mutex_lock(&m); }
  void Unlock() { pthread_mutex_unlock(&m); }

 private:
  pthread_mutex_t m;
};

class GlobalMutexes {
 public:
  static Mutex* Profiler() {
    static Mutex m;
    return &m;
  }

  static Mutex* EightBitIntGemm() {
    static Mutex m;
    return &m;
  }
};

// A very simple RAII helper to lock and unlock a Mutex
struct ScopedLock {
  ScopedLock(Mutex* m) : _m(m) { _m->Lock(); }
  ~ScopedLock() { _m->Unlock(); }

 private:
  Mutex* _m;
};
#else
#undef GEMMLOWP_PROFILING
#endif //HIFI_BUILD

// Profiling definitions. Two paths: when profiling is enabled,
// and when profiling is disabled.
#ifdef GEMMLOWP_PROFILING
// This code path is when profiling is enabled.

// A pseudo-call-stack. Contrary to a real call-stack, this only
// contains pointers to literal strings that were manually entered
// in the instrumented code (see ScopedProfilingLabel).
struct ProfilingStack {
  static const std::size_t kMaxSize = 14;
  typedef const char* LabelsArrayType[kMaxSize];
  LabelsArrayType labels;
  std::size_t size;
  Mutex* lock;

  ProfilingStack() { memset(this, 0, sizeof(ProfilingStack)); }

  void Push(const char* label) {
    ScopedLock sl(lock);
    ReleaseBuildAssertion(size < kMaxSize, "ProfilingStack overflow");
    labels[size] = label;
    size++;
  }

  void Pop() {
    ScopedLock sl(lock);
    ReleaseBuildAssertion(size > 0, "ProfilingStack underflow");
    size--;
  }

  void UpdateTop(const char* new_label) {
    ScopedLock sl(lock);
    assert(size);
    labels[size - 1] = new_label;
  }

  ProfilingStack& operator=(const ProfilingStack& other) {
    memcpy(this, &other, sizeof(ProfilingStack));
    return *this;
  }

  bool operator==(const ProfilingStack& other) const {
    return !memcmp(this, &other, sizeof(ProfilingStack));
  }
};

static_assert(
    !(sizeof(ProfilingStack) & (sizeof(ProfilingStack) - 1)),
    "ProfilingStack should have power-of-two size to fit in cache lines");

struct ThreadInfo;

// The global set of threads being profiled.
inline std::set<ThreadInfo*>& ThreadsUnderProfiling() {
  static std::set<ThreadInfo*> v;
  return v;
}

struct ThreadInfo {
  pthread_key_t key;  // used only to get a callback at thread exit.
  ProfilingStack stack;

  ThreadInfo() {
    pthread_key_create(&key, ThreadExitCallback);
    pthread_setspecific(key, this);
    stack.lock = new Mutex();
  }

  static void ThreadExitCallback(void* ptr) {
    ScopedLock sl(GlobalMutexes::Profiler());
    ThreadInfo* self = static_cast<ThreadInfo*>(ptr);
    ThreadsUnderProfiling().erase(self);
    pthread_key_delete(self->key);
    delete self->stack.lock;
  }
};

inline ThreadInfo& ThreadLocalThreadInfo() {
  static pthread_key_t key;
  static auto DeleteThreadInfo = [](void* threadInfoPtr) {
    ThreadInfo* threadInfo = static_cast<ThreadInfo*>(threadInfoPtr);
    if (threadInfo) {
      delete threadInfo;
    }
  };

  pthread_key_create(&key, DeleteThreadInfo);

  ThreadInfo* threadInfo = static_cast<ThreadInfo*>(pthread_getspecific(key));
  if (!threadInfo) {
    threadInfo = new ThreadInfo();
    pthread_setspecific(key, threadInfo);
  }
  return *threadInfo;
}

// ScopedProfilingLabel is how one instruments code for profiling
// with this profiler. Construct local ScopedProfilingLabel variables,
// passing a literal string describing the local code. Profile
// samples will then be annotated with this label, while it is in scope
// (whence the name --- also known as RAII).
// See the example in profiler.h.
class ScopedProfilingLabel {
  ProfilingStack* profiling_stack_;

 public:
  explicit ScopedProfilingLabel(const char* label)
      : profiling_stack_(&ThreadLocalThreadInfo().stack) {
    profiling_stack_->Push(label);
  }

  ~ScopedProfilingLabel() { profiling_stack_->Pop(); }

  void Update(const char* new_label) { profiling_stack_->UpdateTop(new_label); }
};

// To be called once on each thread to be profiled.
inline void RegisterCurrentThreadForProfiling() {
  ScopedLock sl(GlobalMutexes::Profiler());
  ThreadsUnderProfiling().insert(&ThreadLocalThreadInfo());
}

#else  // not GEMMLOWP_PROFILING
// This code path is when profiling is disabled.

// This empty definition of ScopedProfilingLabel ensures that
// it has zero runtime overhead when profiling is disabled.
struct ScopedProfilingLabel {
  explicit ScopedProfilingLabel(const char*) {}
  void Update(const char*) {}
};

inline void RegisterCurrentThreadForProfiling() {}

#endif

}  // end namespace gemmlowp

#endif  // GEMMLOWP_PROFILING_INSTRUMENTATION_H_
