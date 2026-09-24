#!/bin/bash
# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
#
# run_sifl_demo.sh
#
# Builds the per_round_decoder plugin into a temporary directory and runs
# sifl_demo.py against it. Nothing is left behind.
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INSTALL_PREFIX="${CUDAQX_INSTALL_DIR:-${CUDAQX_INSTALL_PREFIX:-}}"
CUDAQ_PREFIX="${CUDA_QUANTUM_PATH:-/usr/local/cudaq}"

print_usage() {
    cat <<'EOF'
Usage: run_sifl_demo.sh [options]

  --install-prefix DIR  CUDA-QX install (default: $CUDAQX_INSTALL_DIR or
                        $CUDAQX_INSTALL_PREFIX)
  --cudaq-prefix DIR    CUDA-Q install (default: $CUDA_QUANTUM_PATH or
                        /usr/local/cudaq)
  -h, --help            Show this help
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --install-prefix) INSTALL_PREFIX="$2"; shift 2 ;;
        --cudaq-prefix)   CUDAQ_PREFIX="$2"; shift 2 ;;
        -h|--help)        print_usage; exit 0 ;;
        *) echo "Unknown option: $1" >&2; print_usage >&2; exit 1 ;;
    esac
done

if [[ ! -f "${INSTALL_PREFIX}/include/cudaq/qec/decoder.h" ]]; then
    echo "ERROR: '${INSTALL_PREFIX}' is not a CUDA-QX install; pass --install-prefix." >&2
    exit 1
fi

BUILD_DIR="$(mktemp -d)"
trap 'rm -rf "${BUILD_DIR}"' EXIT

echo "Building per_round_decoder..."
g++ -std=c++17 -shared -fPIC "${SCRIPT_DIR}/per_round_decoder.cpp" \
    -I"${INSTALL_PREFIX}/include" -L"${INSTALL_PREFIX}/lib" \
    -lcudaq-qec-decoders -o "${BUILD_DIR}/libper_round_decoder.so"

PYTHONPATH="${CUDAQ_PREFIX}:${INSTALL_PREFIX}${PYTHONPATH:+:${PYTHONPATH}}" \
    python3 "${SCRIPT_DIR}/sifl_demo.py" "${BUILD_DIR}/libper_round_decoder.so"
