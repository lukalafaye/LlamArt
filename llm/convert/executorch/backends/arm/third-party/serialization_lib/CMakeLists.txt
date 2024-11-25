#TOSA serialization library

# Copyright (c) 2020-2023, ARM Limited.
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#         http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

# Contains TOSA flatbuffer serialization library content.

cmake_minimum_required(VERSION 3.13.4)
project(TosaSerialization)

set(CMAKE_CXX_STANDARD 14 CACHE STRING "C++ standard to conform to")
set(CMAKE_CXX_STANDARD_REQUIRED YES)

set(CMAKE_VERBOSE_MAKEFILE ON)

option(BUILD_TESTS "Build test applications" ON)
option(FLATBUFFERS_ROOT "Location where the flatbuffers 'include' and 'lib' folders to be found" Off)

include_directories(${PROJECT_SOURCE_DIR}/third_party/half/include)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

add_library(tosa_serialization_lib
  src/tosa_serialization_handler.cpp
  src/numpy_utils.cpp
)

# If flatbuffers is built externally just link it
if (TARGET flatbuffers)
  target_link_libraries(tosa_serialization_lib PRIVATE flatbuffers)
else()
  # Verify we have a valid flatbuffers include path.
  # We will explicitly exclude the system include directories and only
  # accept either a user supplied value or the local third_party/flatbuffers.
  find_path(FLATBUFFERS_INCLUDE_PATH flatbuffers/flatbuffers.h
            NO_DEFAULT_PATH
            HINTS ${FLATBUFFERS_ROOT} ./third_party/flatbuffers
            PATH_SUFFIXES include)
  message(STATUS "Flatbuffers include located at: ${FLATBUFFERS_INCLUDE_PATH}")
  include_directories(${FLATBUFFERS_INCLUDE_PATH})

  # Next is the library.
  # We will explicitly exclude the system lib directories and only accept
  # either a user supplied value or the local third_party/flatbuffers.
  find_library(FLATBUFFERS_LIBRARY
              NAMES libflatbuffers.a flatbuffers
              NO_DEFAULT_PATH
              HINTS ${FLATBUFFERS_ROOT} ./third_party/flatbuffers
              PATH_SUFFIXES lib)

  if(FLATBUFFERS_LIBRARY)
      message(STATUS "Flatbuffers library located at: ${FLATBUFFERS_LIBRARY}")
      target_link_libraries(tosa_serialization_lib PRIVATE ${FLATBUFFERS_LIBRARY})
  else()
      # It's not there we treat third_party/flatbuffers as a sub project.
      # In this case we'll need to build the downloaded source.
      # Turn off unnecessary flatbuffers targets
      set(FLATBUFFERS_BUILD_TESTS OFF)
      add_subdirectory(third_party/flatbuffers)
      target_link_libraries(tosa_serialization_lib PRIVATE flatbuffers)
  endif()
endif()

set(public_headers)
list(APPEND public_headers
     include/attribute.h
     include/attribute.def
     include/numpy_utils.h
     include/tosa_generated.h
     include/tosa_serialization_handler.h
)

set_target_properties(tosa_serialization_lib PROPERTIES PUBLIC_HEADER "${public_headers}")

# Optionally build test executables.
if (BUILD_TESTS)
    add_executable(serialization_npy_test
      test/src/serialization_npy_test.cpp
    )

    target_link_libraries(serialization_npy_test
      tosa_serialization_lib
    )
endif()

set(TOSA_SERIALIZATION_LIB_CMAKE_DIR "${CMAKE_INSTALL_LIBDIR}/cmake/tosa_serialization_lib")
# Follow GNU packaging norms for installation directory structure.
include(GNUInstallDirs)
install(
  TARGETS tosa_serialization_lib EXPORT TosaSerializationLibTargets
  PUBLIC_HEADER
  ARCHIVE
)

install(EXPORT TosaSerializationLibTargets
  FILE TosaSerializationLibTargets.cmake
  NAMESPACE tosa::
  DESTINATION ${TOSA_SERIALIZATION_LIB_CMAKE_DIR}
)
