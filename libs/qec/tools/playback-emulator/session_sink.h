/****************************************************************-*- C++ -*-****
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

/// @file session_sink.h
/// @brief The two sinks that drive an in-process decoder, differing only in how the
/// producer treats the ring.
///
/// Both bring up the same thing: one decoder realized from a YAML config, and
/// one private `qec_realtime_session` (ring + dispatcher thread) inside this
/// process.  They differ in the producer discipline:
///
///   - `make_inproc_rpc_sink`  calls `rpc_producer::enqueue_syndromes`, 
///     sends one request and waits for its ACK before returning.  Ring depth
///     used: 1.  This is the reference path -- simple, strictly ordered -- but 
///     its per-call cost is a full round trip.
///
///   - `make_ring_buffer_injector_sink` writes straight into the
///     session's RX ring and returns immediately.  Ring depth used: all of
///     them.  Slots are reclaimed lazily on reuse, so the producer only stalls
///     when the ring is full. This is a work-alike against a `qec_realtime_session` 
///     ring, NOT that class, which is unavailable here (it needs a CUDA-Q API 
///     removed in PR4770).
///
/// The injector sink deliberately does NOT go through `rpc_producer` -- it
/// reimplements the wire format against the session's public ring accessors,
/// leaving that file untouched.  It therefore also does not inherit
/// `rpc_producer`'s process-global single-producer guard, so the same
/// one-emulator-process-per-source rule still applies.
///
/// ONE SOURCE PER PROCESS.  `qec_realtime_session::initialize()` rejects a
/// second concurrent HOST-mode session, and `rpc_producer` guards its calls
/// with a process-global flag.  To emulate N syndrome sources, run N copies of
/// this tool.

#include "playback_emulator.h"

#include <memory>
#include <string>

namespace cudaq::qec::emulator {

/// Blocking producer: one `enqueue_syndromes` RPC per event.
///
/// @param config_path  A multi-decoder YAML (same schema `decoding_server`
///                     takes).  Exactly one decoder is realized.
/// @param decoder_id   Selects WHICH config entry to realize, by its YAML
///                     `id`.  It is not the wire routing key: a one-decoder
///                     session always routes on key 0.
/// @param dem_file     Optional path to a Stim detector error model.  Decoders
///                     that take a DEM string rather than a parity-check matrix
///                     (chromobius) cannot be built from the config's
///                     `H_sparse`, and the DEM cannot ride in
///                     `decoder_custom_args` -- that map is schema-validated
///                     against the decoder type when the YAML is parsed, so an
///                     unknown key is rejected before this sink sees it.  Pass
///                     the path here instead; empty means build from H.
std::unique_ptr<sink> make_inproc_rpc_sink(const std::string &config_path,
                                         std::size_t decoder_id,
                                         const std::string &dem_file = "");

/// Streaming producer: same frames, written straight into the ring.
/// Arguments as above.
std::unique_ptr<sink> make_ring_buffer_injector_sink(const std::string &config_path,
                                          std::size_t decoder_id,
                                          const std::string &dem_file = "");


/// Transport producer: publishes the same frames to a `decoding_server`
/// over UDP instead of to a session in this process.
///
/// The other two sinks realize a decoder locally and drive a private
/// `qec_realtime_session`; their numbers are the realtime path's floor, with no
/// wire in them.  This one owns no decoder and no session -- it drives a
/// `cpu_udp` transceiver whose rings the transport ships to the server, so the
/// same playback file and the same lateness statistics can run against a
/// deployment-shaped target and be subtracted.
///
/// Slot geometry is part of the wire contract and must match the server's
/// `--num-slots` / `--slot-size` (defaults 8 x 256); a mismatch puts frames in
/// the wrong place rather than failing cleanly.
///
/// @param host        Server host, e.g. "127.0.0.1".
/// @param port        The `port=` value from the server's READY line.
/// @param decoder_id  Routing key written into every payload; must match a
///                    decoder `id` in the SERVER's config.
/// @param num_slots   Ring slots, matching the server.
/// @param slot_size   Slot stride in bytes, matching the server.
/// @param observables Correction bits to request when a record does not say.
std::unique_ptr<sink> make_udp_server_sink(const std::string &host,
                                           std::uint16_t port,
                                           std::size_t decoder_id,
                                           std::uint32_t num_slots,
                                           std::uint32_t slot_size,
                                           std::uint64_t observables);
} // namespace cudaq::qec::emulator
