# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# qec_playback_emulator: CLI binary + Python module that replay pre-recorded
# syndrome bits into the realtime decoding API on a precise schedule.
#
# Mirrors the decoding-server guard: both tools need the CUDA-Q realtime
# headers and the in-process session library.  The Python module additionally
# requires nanobind; it is built only when nanobind_add_module is already
# available (i.e. when the full Python package is being built).
#
# The installed binary lives under <prefix>/bin alongside decoding_server and
# belongs to the same qec-tools component (not included in the Python wheel).

if(NOT CUDAQ_REALTIME_INCLUDE_DIR)
  message(STATUS "qec_playback_emulator: CUDAQ_REALTIME_INCLUDE_DIR not set; "
                 "skipping playback-emulator build.")
  return()
endif()

find_library(QEC_UDP_TRANSPORT_LIBRARY cudaq-realtime-udp-transport
  HINTS ${CUDAQ_REALTIME_ROOT} ${CUDAQ_INSTALL_DIR}
  PATH_SUFFIXES lib)

if(NOT QEC_UDP_TRANSPORT_LIBRARY)
  message(STATUS "qec_playback_emulator: cudaq-realtime-udp-transport not "
                 "found; skipping playback-emulator build.")
  return()
endif()

set(_PE_SOURCES
  ${CMAKE_CURRENT_SOURCE_DIR}/playback_emulator.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/ring_buffer_injector_sink.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/udp_server_sink.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/udp_ring_sink.cpp
)

set(_PE_INCLUDE_DIRS
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CUDAQ_REALTIME_INCLUDE_DIR}
  ${CUDAToolkit_INCLUDE_DIRS}
  # qec_realtime_session.h and rpc_producer.h are private headers in lib/realtime/;
  # reached directly because this tool is in-tree (see ring_buffer_injector_sink.cpp).
  ${CMAKE_CURRENT_SOURCE_DIR}/../../lib/realtime
)

set(_PE_LINK_LIBS
  ${QEC_UDP_TRANSPORT_LIBRARY}
  cudaq-qec
  cudaq-qec-realtime-decoding
  cudaq-qec-decoders
  cudaq::cudaq
  cudaq::cudaq-common
  cudaq-mlir-runtime
  nvqir
  nvqir-stim
  CUDA::cudart
)

# ── CLI binary ────────────────────────────────────────────────────────────────

add_executable(qec_playback_emulator
  ${CMAKE_CURRENT_SOURCE_DIR}/main.cpp
  ${_PE_SOURCES}
)
target_include_directories(qec_playback_emulator PRIVATE ${_PE_INCLUDE_DIRS})
target_compile_definitions(qec_playback_emulator PRIVATE CUDAQ_REALTIME_ROOT)
target_link_directories(qec_playback_emulator
  PRIVATE ${CUDAQ_INSTALL_DIR}/lib)
target_link_libraries(qec_playback_emulator PRIVATE ${_PE_LINK_LIBS})
set_target_properties(qec_playback_emulator PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
  BUILD_RPATH
    "${CMAKE_BINARY_DIR}/lib;${CMAKE_BINARY_DIR}/lib/decoder-plugins;${CUDAQ_INSTALL_DIR}/lib"
  INSTALL_RPATH
    "$ORIGIN/../lib;$ORIGIN/../lib/decoder-plugins;${CUDAQ_INSTALL_DIR}/lib"
)
install(TARGETS qec_playback_emulator
  COMPONENT qec-tools
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

# ── Python module ─────────────────────────────────────────────────────────────
# The CLI binary always builds.  The Python module needs nanobind.
#
# If the qec python subdirectory has already brought it into scope, reuse that.
# Otherwise fetch the SAME pinned version it uses -- never a pip-installed one.
# nanobind keys its internals capsule by ABI tag, so a module built against a
# different nanobind gets its own internals and cannot see the types CUDA-Q
# registered; `cudaq::kraus_channel` then goes missing and every module in the
# build aborts at import.  Pinning here rather than relying on subdirectory
# order is what lets this build regardless of where it is added from.
if(NOT COMMAND nanobind_add_module)
  find_package(Python QUIET COMPONENTS Interpreter Development.Module)
  if(Python_FOUND)
    include(FetchContent)
    FetchContent_Declare(
      nanobind
      GIT_REPOSITORY https://github.com/wjakob/nanobind
      GIT_TAG v2.12.0            # must match libs/qec/python/CMakeLists.txt
      EXCLUDE_FROM_ALL
    )
    FetchContent_MakeAvailable(nanobind)
  endif()
endif()

if(COMMAND nanobind_add_module)
  nanobind_add_module(qec_playback_emulator_py NB_DOMAIN cudaq
    ${CMAKE_CURRENT_SOURCE_DIR}/python/py_playback_emulator.cpp
    ${_PE_SOURCES}
  )
  set_target_properties(qec_playback_emulator_py PROPERTIES
    OUTPUT_NAME qec_playback_emulator
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
  )
  target_include_directories(qec_playback_emulator_py PRIVATE ${_PE_INCLUDE_DIRS})
  target_compile_definitions(qec_playback_emulator_py PRIVATE CUDAQ_REALTIME_ROOT)
  target_link_directories(qec_playback_emulator_py
    PRIVATE ${CUDAQ_INSTALL_DIR}/lib)
  target_link_libraries(qec_playback_emulator_py PRIVATE ${_PE_LINK_LIBS})
  set_target_properties(qec_playback_emulator_py PROPERTIES
    BUILD_RPATH
      "${CMAKE_BINARY_DIR}/lib;${CMAKE_BINARY_DIR}/lib/decoder-plugins;${CUDAQ_INSTALL_DIR}/lib"
    INSTALL_RPATH
      "$ORIGIN/../lib;$ORIGIN/../lib/decoder-plugins;${CUDAQ_INSTALL_DIR}/lib"
  )
endif()
