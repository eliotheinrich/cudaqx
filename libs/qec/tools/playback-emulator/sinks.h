/****************************************************************-*- C++ -*-****
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

/// @file sinks.h
/// @brief Factory functions for the non-null playback sinks.
///
/// `make_ring_buffer_injector_sink` drives a local in-process decoder.
/// `make_udp_server_sink` ships the same RPC frames to a remote
/// `decoding_server` over UDP, and is the default for remote work.
///
/// Both produce the same playback file format and the same lateness
/// statistics, so results are directly comparable.
///
/// The ring-buffer-injector and UDP-server sinks share their dispatch logic
/// via `sink`/`rpc_call.h`: `sink::send`/`sink::try_get_corrections` build a
/// generic `rpc_call` (see `build_rpc_call`) and hand it to the sink's
/// `transport()` override, which knows only how to move an already-built
/// frame to and from its destination. Neither sink builds a wire frame or
/// decides retry policy itself.

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

/// UDP transport producer, and the default for a remote server.  Owns no local
/// decoder or session, and no client-side ring: a plain datagram socket whose
/// replies are matched by `request_id`, so nothing depends on a ring depth or
/// on replies arriving in request order.
///
/// @param host        Server host, e.g. "127.0.0.1".
/// @param port        The `port=` value from the server's READY line.
/// @param decoder_id  Routing key written into every payload.
/// @param slot_size   Datagram stride in bytes, matching the server.
/// @param observables Correction bits to request on each get_corrections.
std::unique_ptr<sink> make_udp_server_sink(const std::string &host,
                                           std::uint16_t port,
                                           std::size_t decoder_id,
                                           std::uint32_t slot_size,
                                           std::uint64_t observables);

} // namespace cudaq::qec::emulator
