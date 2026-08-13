/****************************************************************-*- C++ -*-****
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

/// @file rpc_call.h
/// @brief Generic, transport-agnostic description of one decoder-RPC wire
/// call, and the two functions that turn a playback `event` into one and
/// then serialize it onto the wire.
///
/// This is the seam `sink::transport()` implementations are written against:
/// a concrete sink knows only how to move an already-built frame to and from
/// its destination (a ring slot, a UDP datagram, ...), never how to build the
/// frame or interpret the RPC-level retry/consuming semantics. Adding a new
/// RPC to the protocol means teaching `build_rpc_call` one more case, not
/// touching every sink.

#include "playback_emulator.h"

#include "cudaq/qec/realtime/decoder_rpc_wire_format.h"
#include "cudaq/realtime/daemon/dispatcher/dispatch_kernel_launch.h"

#include <array>
#include <cstdint>

namespace cudaq::qec::emulator {

namespace wire = cudaq::qec::decoding::rpc;

/// Largest payload struct on the wire today (`EnqueueRequestPayload`, 32B).
/// `rpc_call` carries its payload inline at this capacity so building one
/// never allocates on the timing thread; bump this if a future RPC needs a
/// bigger fixed payload.
inline constexpr std::size_t kMaxRpcPayloadBytes = 32;

/// A generic, transport-agnostic description of ONE wire RPC. Built once by
/// `build_rpc_call`, consumed by `sink::send`/`sink::try_get_corrections`,
/// and handed to a concrete sink's `transport()` override.
struct rpc_call {
  std::uint32_t function_id = 0;

  /// Fixed-capacity payload storage (the RPCHeader body, before any trailing
  /// bit-packed bits). `build_rpc_call` places the op's POD payload struct
  /// (EnqueueRequestPayload, GetCorrectionsRequestPayload,
  /// ResetRequestPayload, ...) here via memcpy; `payload_len` is that
  /// struct's size.
  std::array<std::uint8_t, kMaxRpcPayloadBytes> payload{};
  std::size_t payload_len = 0;

  /// Optional trailing syndrome bits: one byte per bit (0x00/0x01), same
  /// layout as `event::syndrome_data`. NOT owned -- aliases the event's
  /// arena / syndrome_source buffer, which outlives the call. Null when the
  /// op carries no syndromes.
  const std::uint8_t *bits = nullptr;
  std::uint64_t num_bits = 0;

  /// True if the caller must wait for an RPCResponse before returning
  /// (get_corrections, reset). False for fire-and-forget (enqueue): the
  /// transport may publish and return immediately.
  bool blocking = false;

  /// True if a NOT_READY status should be retried (subject to
  /// `run_config::wait_for_ready`) rather than surfaced as an error on the
  /// first miss. Only meaningful when `blocking` is also true.
  bool retry_on_not_ready = false;

  /// Number of correction bits the reply is expected to carry; 0 if the call
  /// returns no result body (enqueue, reset). Drives how many bit-packed
  /// bytes `transport()` must fill into its result buffer, and how many bits
  /// get unpacked into `sink::corrections()` on success.
  std::size_t expected_result_bits = 0;

  /// Breadcrumb: becomes `RPCHeader::request_id`. For enqueue this is the
  /// caller-supplied event tag; for get_corrections/reset it's an internal
  /// per-sink request counter.
  std::uint64_t tag = 0;
};

/// THE parsing utility: turn one playback `event` into a generic `rpc_call`.
/// Called from `sink::send()` for enqueue/get_corrections/reset, and again
/// (with `consuming=true, retry_on_not_ready=false`) from
/// `sink::try_get_corrections()` for the `stream_until` poll.
///
/// @param op                 `e.op` for a normal send; `operation::
///                           get_corrections` for the stream_until poll
///                           (there is no wire op for stream_until itself --
///                           it is a client-side loop driven by `run()`).
/// @param e                  Source event; only `syndrome_data`/
///                           `num_syndromes` are read (for enqueue). Pass a
///                           default-constructed `event{}` for the poll.
/// @param decoder_id         Wire routing key.
/// @param num_observables    Decoder's declared observable count; becomes
///                           `GetCorrectionsRequestPayload::return_size` and
///                           `rpc_call::expected_result_bits`.
/// @param tag                Breadcrumb -> `rpc_call::tag`.
/// @param consuming          true => `GetCorrectionsRequestPayload::reset=1`
///                           (the stream_until-poll shape); false => reset=0
///                           (the documented, non-consuming, explicit
///                           get_corrections contract).
/// @param retry_on_not_ready Copied into `rpc_call::retry_on_not_ready`.
///
/// Throws `std::logic_error` for `operation::stream_until`, which has no
/// wire RPC of its own.
rpc_call build_rpc_call(operation op, const event &e, std::uint64_t decoder_id,
                        std::uint64_t num_observables, std::uint64_t tag,
                        bool consuming, bool retry_on_not_ready);

/// Serialize `call` into `dst[0, capacity)` as RPCHeader + payload + (if
/// `call.bits`) bit-packed trailing bits, LSB-first. Returns the total frame
/// length. Throws `std::runtime_error` if the frame does not fit `capacity`.
std::size_t serialize_rpc_frame(const rpc_call &call, std::uint32_t request_id,
                                std::uint8_t *dst, std::size_t capacity);

} // namespace cudaq::qec::emulator
