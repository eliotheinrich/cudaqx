# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

cmake_minimum_required(VERSION 3.24)
project(qec_playback_emulator CXX)

set(CMAKE_CXX_STANDARD 20)

# ── Installation prefixes ─────────────────────────────────────────────────────
# Both default to the values the CUDA-QX dev container uses.  Override on the
# cmake command line if your installation differs.

set(CUDAQ_PREFIX   "/usr/local/cudaq" CACHE PATH "CUDA-Q install prefix")
set(CUDAQX_PREFIX  "$ENV{HOME}/.cudaqx" CACHE PATH "CUDA-QX install prefix")

# ── Find dependencies ─────────────────────────────────────────────────────────

find_package(CUDAToolkit REQUIRED)

find_path(CUDAQ_REALTIME_INCLUDE_DIR
  NAMES cudaq/realtime/daemon/dispatcher/cudaq_realtime.h
  HINTS ${CUDAQ_PREFIX}
  PATH_SUFFIXES include
  REQUIRED
)

find_library(CUDAQ_UDP_TRANSPORT_LIB  cudaq-realtime-udp-transport HINTS ${CUDAQ_PREFIX}/lib REQUIRED)
find_library(CUDAQ_LIB                cudaq                         HINTS ${CUDAQ_PREFIX}/lib REQUIRED)
find_library(CUDAQ_COMMON_LIB         cudaq-common                  HINTS ${CUDAQ_PREFIX}/lib REQUIRED)
find_library(CUDAQ_MLIR_RUNTIME_LIB   cudaq-mlir-runtime            HINTS ${CUDAQ_PREFIX}/lib REQUIRED)
find_library(NVQIR_LIB                nvqir                         HINTS ${CUDAQ_PREFIX}/lib REQUIRED)
find_library(NVQIR_STIM_LIB           nvqir-stim                    HINTS ${CUDAQ_PREFIX}/lib REQUIRED)
find_library(CUDAQ_QEC_LIB            cudaq-qec                     HINTS ${CUDAQX_PREFIX}/lib REQUIRED)
find_library(CUDAQ_QEC_DECODERS_LIB   cudaq-qec-decoders            HINTS ${CUDAQX_PREFIX}/lib REQUIRED)
find_library(CUDAQ_QEC_REALTIME_LIB   cudaq-qec-realtime-decoding   HINTS ${CUDAQX_PREFIX}/lib REQUIRED)

set(_LINK_LIBS
  ${CUDAQ_UDP_TRANSPORT_LIB}
  ${CUDAQ_QEC_REALTIME_LIB}
  ${CUDAQ_QEC_DECODERS_LIB}
  ${CUDAQ_QEC_LIB}
  ${CUDAQ_LIB}
  ${CUDAQ_COMMON_LIB}
  ${CUDAQ_MLIR_RUNTIME_LIB}
  ${NVQIR_LIB}
  ${NVQIR_STIM_LIB}
  CUDA::cudart
)

set(_INCLUDE_DIRS
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CUDAQ_REALTIME_INCLUDE_DIR}
  ${CUDAQ_PREFIX}/include
  ${CUDAQX_PREFIX}/include
  ${CUDAToolkit_INCLUDE_DIRS}
)

# ── CLI binary ────────────────────────────────────────────────────────────────

add_executable(qec_playback_emulator
  main.cpp
  playback_emulator.cpp
  session_sink.cpp
)
target_include_directories(qec_playback_emulator PRIVATE ${_INCLUDE_DIRS})
target_compile_definitions(qec_playback_emulator PRIVATE CUDAQ_REALTIME_ROOT)
target_link_directories(qec_playback_emulator PRIVATE ${CUDAQ_PREFIX}/lib ${CUDAQX_PREFIX}/lib)
target_link_libraries(qec_playback_emulator PRIVATE ${_LINK_LIBS})
set_target_properties(qec_playback_emulator PROPERTIES
  BUILD_RPATH  "${CUDAQX_PREFIX}/lib;${CUDAQ_PREFIX}/lib"
  INSTALL_RPATH "$ORIGIN/../lib"
)

# ── Python module ─────────────────────────────────────────────────────────────

find_package(Python REQUIRED COMPONENTS Interpreter Development)
execute_process(
  COMMAND ${Python_EXECUTABLE} -c "import nanobind; print(nanobind.cmake_dir())"
  OUTPUT_VARIABLE _NB_CMAKE_DIR OUTPUT_STRIP_TRAILING_WHITESPACE
)
list(APPEND CMAKE_PREFIX_PATH "${_NB_CMAKE_DIR}")
find_package(nanobind REQUIRED)

nanobind_add_module(qec_playback_emulator_py
  python/py_playback_emulator.cpp
  playback_emulator.cpp
  session_sink.cpp
)
set_target_properties(qec_playback_emulator_py PROPERTIES
  OUTPUT_NAME qec_playback_emulator
)
target_include_directories(qec_playback_emulator_py PRIVATE ${_INCLUDE_DIRS})
target_compile_definitions(qec_playback_emulator_py PRIVATE CUDAQ_REALTIME_ROOT)
target_link_directories(qec_playback_emulator_py PRIVATE ${CUDAQ_PREFIX}/lib ${CUDAQX_PREFIX}/lib)
target_link_libraries(qec_playback_emulator_py PRIVATE ${_LINK_LIBS})
set_target_properties(qec_playback_emulator_py PROPERTIES
  BUILD_RPATH  "${CUDAQX_PREFIX}/lib;${CUDAQ_PREFIX}/lib;${CUDAQ_PREFIX}/lib"
  INSTALL_RPATH "$ORIGIN/../lib"
)
