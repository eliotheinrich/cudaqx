/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "rpc_call.h"

#include <cstring>
#include <stdexcept>

namespace cudaq::qec::emulator {

rpc_call build_rpc_call(operation op, const event &e, std::uint64_t decoder_id,
                        std::uint64_t num_observables, std::uint64_t tag,
                        bool consuming, bool retry_on_not_ready) {
  rpc_call call;
  call.tag = tag;
  switch (op) {
  case operation::enqueue: {
    wire::EnqueueRequestPayload p{};
    p.decoder_id = static_cast<std::int64_t>(decoder_id);
    p.counter = static_cast<std::int64_t>(tag);
    p.syndrome_mapping_id = 0;
    p.num_syndromes = static_cast<std::int64_t>(e.num_syndromes);
    call.function_id = wire::kEnqueueSyndromesFunctionId;
    std::memcpy(call.payload.data(), &p, sizeof(p));
    call.payload_len = sizeof(p);
    call.bits = e.syndrome_data;
    call.num_bits = e.num_syndromes;
    call.blocking = false;
    return call;
  }
  case operation::get_corrections: {
    wire::GetCorrectionsRequestPayload p{};
    p.decoder_id = static_cast<std::int64_t>(decoder_id);
    p.return_size = static_cast<std::int64_t>(num_observables);
    p.reset = consuming ? 1 : 0;
    call.function_id = wire::kGetCorrectionsFunctionId;
    std::memcpy(call.payload.data(), &p, sizeof(p));
    call.payload_len = sizeof(p);
    call.blocking = true;
    call.retry_on_not_ready = retry_on_not_ready;
    call.expected_result_bits = num_observables;
    return call;
  }
  case operation::reset: {
    wire::ResetRequestPayload p{};
    p.decoder_id = static_cast<std::int64_t>(decoder_id);
    call.function_id = wire::kResetDecoderFunctionId;
    std::memcpy(call.payload.data(), &p, sizeof(p));
    call.payload_len = sizeof(p);
    call.blocking = true;
    return call;
  }
  case operation::stream_until:
    throw std::logic_error(
        "build_rpc_call: stream_until has no wire RPC of its own; run() "
        "drives it as enqueue + a consuming get_corrections poll");
  }
  throw std::logic_error("build_rpc_call: unknown operation");
}

std::size_t serialize_rpc_frame(const rpc_call &call, std::uint32_t request_id,
                                std::uint8_t *dst, std::size_t capacity) {
  const std::size_t packed =
      call.bits ? wire::bit_packed_bytes(call.num_bits) : 0;
  const std::size_t body = call.payload_len + packed;
  const std::size_t total = sizeof(cudaq::realtime::RPCHeader) + body;
  if (total > capacity)
    throw std::runtime_error("serialize_rpc_frame: frame needs " +
                             std::to_string(total) + " bytes, capacity is " +
                             std::to_string(capacity));
  std::memset(dst, 0, total);

  auto *header = reinterpret_cast<cudaq::realtime::RPCHeader *>(dst);
  header->magic = cudaq::realtime::RPC_MAGIC_REQUEST;
  header->function_id = call.function_id;
  header->arg_len = static_cast<std::uint32_t>(body);
  header->request_id = request_id;
  header->ptp_timestamp = 0;

  std::uint8_t *args = dst + sizeof(cudaq::realtime::RPCHeader);
  std::memcpy(args, call.payload.data(), call.payload_len);
  if (call.bits)
    for (std::uint64_t i = 0; i < call.num_bits; ++i)
      if (call.bits[i] & 0x1u)
        args[call.payload_len + i / 8] |=
            static_cast<std::uint8_t>(1u << (i % 8));
  return total;
}

} // namespace cudaq::qec::emulator
