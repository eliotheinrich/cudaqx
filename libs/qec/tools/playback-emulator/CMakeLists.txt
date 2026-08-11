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
)

set(_PE_INCLUDE_DIRS
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CUDAQ_REALTIME_INCLUDE_DIR}
  ${CUDAToolkit_INCLUDE_DIRS}
  # qec_realtime_session.h and rpc_producer.h are installed to
  # ${CMAKE_INSTALL_PREFIX}/include but are not part of the public CMake
  # INTERFACE_INCLUDE_DIRECTORIES.  Add the install tree directly.
  ${CMAKE_INSTALL_PREFIX}/include
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
# The CLI binary always builds.  The Python module builds when nanobind is
# available -- either because the python subdirectory was already processed
# (full in-tree build) or because nanobind is pip-installed and discoverable
# via its own CMake dir.

if(NOT COMMAND nanobind_add_module)
  find_package(Python COMPONENTS Interpreter Development.Module QUIET)
  if(Python_FOUND)
    execute_process(
      COMMAND ${Python_EXECUTABLE} -c "import nanobind; print(nanobind.cmake_dir())"
      OUTPUT_VARIABLE _NB_CMAKE_DIR OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET
      RESULT_VARIABLE _NB_RESULT)
    if(_NB_RESULT EQUAL 0 AND _NB_CMAKE_DIR)
      list(APPEND CMAKE_PREFIX_PATH "${_NB_CMAKE_DIR}")
      find_package(nanobind CONFIG QUIET)
    endif()
  endif()
endif()

if(COMMAND nanobind_add_module)
  nanobind_add_module(qec_playback_emulator_py
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
