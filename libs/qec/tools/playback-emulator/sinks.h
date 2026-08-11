/****************************************************************-*- C++ -*-****
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

/// @file sinks.h
/// @brief Factory functions for the two non-null playback sinks.
///
/// `make_ring_buffer_injector_sink` drives a local in-process decoder.
/// `make_udp_server_sink` ships the same RPC frames to a remote
/// `decoding_server` over UDP.  Both produce the same playback file format and
/// the same lateness statistics, so results are directly comparable.

#include "playback_emulator.h"

#include <cstdint>
#include <memory>
#include <string>

namespace cudaq::qec::emulator {

/// In-process streaming producer.  Writes enqueue frames straight into the
/// session ring; reads and resets go through a native round-trip that spins on
/// the TX doorbell rather than sleeping.
///
/// @param config_path  Multi-decoder YAML (same schema as `decoding_server`).
///                     Exactly one decoder is realized.
/// @param decoder_id   Selects which config entry to realize by its YAML `id`.
///                     Not the wire routing key: a one-decoder session always
///                     routes on key 0.
/// @param dem_file     Optional Stim DEM path for decoders that need one
///                     (chromobius).  Empty means build from `H_sparse`.
std::unique_ptr<sink>
make_ring_buffer_injector_sink(const std::string &config_path,
                               std::size_t decoder_id,
                               const std::string &dem_file = "");

/// UDP transport producer.  Ships the same RPC frames to a remote
/// `decoding_server`; owns no local decoder or session.
///
/// Slot geometry must match the server's `--num-slots` / `--slot-size`.
///
/// @param host        Server host, e.g. "127.0.0.1".
/// @param port        The `port=` value from the server's READY line.
/// @param decoder_id  Routing key written into every payload.
/// @param num_slots   Ring slots, matching the server.
/// @param slot_size   Slot stride in bytes, matching the server.
/// @param observables Correction bits to request on each get_corrections.
std::unique_ptr<sink> make_udp_server_sink(const std::string &host,
                                           std::uint16_t port,
                                           std::size_t decoder_id,
                                           std::uint32_t num_slots,
                                           std::uint32_t slot_size,
                                           std::uint64_t observables);

} // namespace cudaq::qec::emulator
