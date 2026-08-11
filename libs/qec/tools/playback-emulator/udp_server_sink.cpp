/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "sinks.h"

#include "cudaq/qec/realtime/decoder_rpc_wire_format.h"
#include "cudaq/realtime/cpu_transport/udp_wrapper.h"
#include "cudaq/realtime/daemon/dispatcher/cpu_relax.h"

#include <chrono>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <vector>

namespace cudaq::qec::emulator {

namespace {

namespace wire = cudaq::qec::decoding::rpc;

/// The four ring pointers plus the geometry a transport hands to the sink.
/// Direction is from the CLIENT's point of view and is MIRRORED relative to a
/// session: the client writes requests into TX and polls RX for responses,
/// whereas a session dispatcher reads RX and writes TX.
struct transport_rings {
  std::uint8_t *tx_data = nullptr;
  std::uint8_t *rx_data = nullptr;
  std::uint64_t *tx_flags = nullptr;
  std::uint64_t *rx_flags = nullptr;
  std::uint32_t num_slots = 0;
  std::uint32_t slot_size = 0;
};

/// A wire to a remote `decoding_server`.
///
/// Everything ABOVE this interface -- frame layout, publish/poll protocol,
/// credit reclaim -- is identical for every transport, because the RPC contract
/// is the same one the in-process ring uses.  Everything that differs between
/// transports is construction and handshake, which is exactly what an
/// implementation supplies.
///
/// Adding a transport (cpu_roce, or a partner's) means one new subclass:
/// bring up its transceiver, complete whatever handshake it needs, and report
/// its rings.  Note that cpu_roce's handshake is two-phase -- setup(), exchange
/// QP number and rkey with the peer out of band, then connect() -- so an
/// implementation gets a rendezvous step the UDP one does not need.
class remote_transport {
public:
  virtual ~remote_transport() = default;
  /// Bring the wire up.  Throws on failure.
  virtual void connect() = 0;
  /// Valid once connect() has returned.
  virtual transport_rings rings() const = 0;
  virtual const char *name() const = 0;
};

/// UDP: single-phase connect, geometry chosen by the caller.
class udp_transport : public remote_transport {
public:
  udp_transport(std::string host, std::uint16_t port, std::uint32_t num_slots,
                std::uint32_t slot_size)
      : host_(std::move(host)), port_(port), num_slots_(num_slots),
        slot_size_(slot_size) {}

  ~udp_transport() override {
    if (xcvr_) {
      cpu_udp_close(xcvr_);
      cpu_udp_destroy_transceiver(xcvr_);
    }
  }

  void connect() override {
    xcvr_ = cpu_udp_create_transceiver(slot_size_, num_slots_);
    if (!xcvr_)
      throw std::runtime_error("udp_transport: could not create transceiver");
    if (!cpu_udp_connect(xcvr_, host_.c_str(), port_))
      throw std::runtime_error("udp_transport: could not connect to " + host_ +
                               ":" + std::to_string(port_));
    if (!cpu_udp_start(xcvr_))
      throw std::runtime_error("udp_transport: transceiver start failed");
  }

  transport_rings rings() const override {
    transport_rings r;
    r.tx_data = reinterpret_cast<std::uint8_t *>(
        cpu_udp_get_tx_ring_data_addr(xcvr_));
    r.rx_data = reinterpret_cast<std::uint8_t *>(
        cpu_udp_get_rx_ring_data_addr(xcvr_));
    r.tx_flags = reinterpret_cast<std::uint64_t *>(
        cpu_udp_get_tx_ring_flag_addr(xcvr_));
    r.rx_flags = reinterpret_cast<std::uint64_t *>(
        cpu_udp_get_rx_ring_flag_addr(xcvr_));
    r.num_slots = num_slots_;
    r.slot_size = slot_size_;
    return r;
  }

  const char *name() const override { return "udp_server"; }

private:
  std::string host_;
  std::uint16_t port_;
  std::uint32_t num_slots_, slot_size_;
  cpu_udp_transceiver_t xcvr_ = nullptr;
};

/// Publishes the decoding RPCs to a remote server over any `remote_transport`.
///
/// The frame layout is identical to the in-process sinks -- 24-byte RPCHeader
/// plus the same payload structs -- so a playback file and its statistics carry
/// across unchanged.  Enqueues stream (publish and advance, replies drained a
/// lap later); reads and resets round-trip, because they return a value.
class remote_sink : public sink {
public:
  remote_sink(std::unique_ptr<remote_transport> transport,
              std::size_t decoder_id, std::uint64_t observables)
      : transport_(std::move(transport)), decoder_id_(decoder_id),
        observables_(observables) {
    transport_->connect();
    const transport_rings r = transport_->rings();
    tx_data_ = r.tx_data;
    rx_data_ = r.rx_data;
    tx_flags_ = r.tx_flags;
    rx_flags_ = r.rx_flags;
    num_slots_ = r.num_slots;
    slot_size_ = r.slot_size;
    if (!tx_data_ || !rx_data_ || !tx_flags_ || !rx_flags_ || !num_slots_)
      throw std::runtime_error("remote_sink: transport reported no rings");
    for (std::uint32_t s = 0; s < num_slots_; ++s) {
      tx_flags_[s] = 0;
      rx_flags_[s] = 0;
    }
  }

  void send(const event &e, std::uint64_t tag) override {
    switch (e.op) {
    case operation::enqueue:
      enqueue(e, tag);
      return;
    case operation::get_corrections:
      read(e, tag);
      return;
    case operation::reset:
      reset(tag);
      return;
    }
  }

  const char *name() const override { return transport_->name(); }

  const std::vector<std::uint8_t> &corrections() const override {
    return corrections_log_;
  }

  std::size_t correction_width() const override { return observables_; }

  std::uint32_t last_not_ready_retries() const override { return last_not_ready_retries_; }

  void report() const override {
    std::printf("%-17s sent=%llu reads=%llu stalls=%llu ring_depth=%u\n",
                transport_->name(),
                static_cast<unsigned long long>(sent_),
                static_cast<unsigned long long>(reads_),
                static_cast<unsigned long long>(stalls_), num_slots_);
  }

private:
  /// Claim the cursor slot, draining any response still parked in it.  Every
  /// RPC gets a reply -- even the fire-and-forget ones -- so a streamed enqueue
  /// leaves one behind that must be consumed before the slot comes round again.
  ///
  /// Limitation: if the server answers an OLD request AFTER claim() exits (i.e.
  /// rx_flags was 0 when claim() ran, so nothing was drained), the response
  /// arrives after we've already published a NEW request in this slot.
  /// await_reply() will then return that stale response.  read() catches this
  /// via the request_id check.
  void claim(std::uint32_t slot) {
    bool stalled = false;
    const auto deadline =
        std::chrono::steady_clock::now() + std::chrono::milliseconds(kTimeoutMs);
    while (true) {
      if (rx_flags_[slot] != 0) {
        __sync_synchronize();
        rx_flags_[slot] = 0;
        tx_flags_[slot] = 0;
      }
      if (tx_flags_[slot] == 0 && rx_flags_[slot] == 0)
        break;
      if (std::chrono::steady_clock::now() >= deadline)
        throw std::runtime_error(
            "remote_sink: timed out claiming slot " + std::to_string(slot) +
            " after " + std::to_string(kTimeoutMs) + " ms waiting for an "
            "enqueue ACK (" + std::to_string(sent_) + " enqueues sent so far"
            ", ring_depth=" + std::to_string(num_slots_) + "); on a jittery "
            "host try --server-slots=" + std::to_string(num_slots_ * 2));
      stalled = true;
      CUDAQ_REALTIME_CPU_RELAX();
    }
    if (stalled)
      ++stalls_;
  }

  /// Fill the cursor slot with one request and publish it.  Returns the slot.
  std::uint32_t publish(std::uint32_t function_id, const void *payload,
                        std::size_t payload_len, const std::uint8_t *bits,
                        std::uint64_t num_bits, std::uint64_t tag) {
    const std::uint32_t slot = cursor_;
    claim(slot);

    std::uint8_t *tx_slot = tx_data_ + slot * slot_size_;
    const std::size_t packed = bits ? wire::bit_packed_bytes(num_bits) : 0;
    const std::size_t body = payload_len + packed;
    if (sizeof(cudaq::realtime::RPCHeader) + body > slot_size_)
      throw std::runtime_error("remote_sink: frame exceeds the " +
                               std::to_string(slot_size_) + "-byte slot");
    std::memset(tx_slot, 0, sizeof(cudaq::realtime::RPCHeader) + body);

    auto *header = reinterpret_cast<cudaq::realtime::RPCHeader *>(tx_slot);
    header->magic = cudaq::realtime::RPC_MAGIC_REQUEST;
    header->function_id = function_id;
    header->arg_len = static_cast<std::uint32_t>(body);
    header->request_id = static_cast<std::uint32_t>(tag);
    header->ptp_timestamp = 0;
    std::uint8_t *args = tx_slot + sizeof(cudaq::realtime::RPCHeader);
    std::memcpy(args, payload, payload_len);
    if (bits)
      for (std::uint64_t i = 0; i < num_bits; ++i)
        if (bits[i] & 0x1u)
          args[payload_len + i / 8] |=
              static_cast<std::uint8_t>(1u << (i % 8));

    __sync_synchronize();
    tx_flags_[slot] = reinterpret_cast<std::uint64_t>(tx_slot);
    cursor_ = (slot + 1u) % num_slots_;
    return slot;
  }

  /// Spin until the reply for `slot` arrives, then return a pointer to it.
  /// `op` names the operation for the error message (e.g. "get_corrections").
  const std::uint8_t *await_reply(std::uint32_t slot, const char *op) {
    const auto deadline =
        std::chrono::steady_clock::now() + std::chrono::milliseconds(kTimeoutMs);
    while (rx_flags_[slot] == 0) {
      if (std::chrono::steady_clock::now() >= deadline)
        throw std::runtime_error(
            "remote_sink: timed out awaiting " + std::string(op) +
            " response in slot " + std::to_string(slot) +
            " after " + std::to_string(kTimeoutMs) + " ms (" +
            std::to_string(sent_) + " enqueues sent so far"
            ", ring_depth=" + std::to_string(num_slots_) + "); on a jittery "
            "host try --server-slots=" + std::to_string(num_slots_ * 2));
      CUDAQ_REALTIME_CPU_RELAX();
    }
    __sync_synchronize();
    return rx_data_ + slot * slot_size_;
  }

  void enqueue(const event &e, std::uint64_t tag) {
    wire::EnqueueRequestPayload p{};
    p.decoder_id = static_cast<std::int64_t>(decoder_id_);
    p.counter = static_cast<std::int64_t>(tag);
    p.syndrome_mapping_id = 0;
    p.num_syndromes = static_cast<std::int64_t>(e.num_syndromes);
    // Streamed: publish and move on.  The reply is drained by claim() when
    // this slot comes round again -- credit discipline one lap of the ring deep.
    publish(wire::kEnqueueSyndromesFunctionId, &p, sizeof(p), e.syndrome_data,
            e.num_syndromes, tag);
    ++sent_;
  }

  void read(const event &, std::uint64_t tag) {
    const std::uint64_t n = observables_;
    wire::GetCorrectionsRequestPayload p{};
    p.decoder_id = static_cast<std::int64_t>(decoder_id_);
    p.return_size = static_cast<std::int64_t>(n);
    p.reset = 0;

    last_not_ready_retries_ = 0;
    const auto deadline = std::chrono::steady_clock::now() +
                          std::chrono::milliseconds(kTimeoutMs);
    while (true) {
      const std::uint32_t slot = publish(wire::kGetCorrectionsFunctionId, &p,
                                         sizeof(p), nullptr, 0, tag);
      const std::uint8_t *rx_slot = await_reply(slot, "get_corrections");
      const auto *response =
          reinterpret_cast<const cudaq::realtime::RPCResponse *>(rx_slot);
      // Consume the reply regardless of status so the slot is clean for reuse.
      rx_flags_[slot] = 0;
      tx_flags_[slot] = 0;

      // Guard against stale responses.  The server echoes request_id back;
      // a mismatch means the slot received a late response for a PREVIOUS
      // request (the server was delayed long enough for the client to lap the
      // ring, claim() found the slot clean, and we published a new request --
      // then the server answered the old request first).  Without this check
      // the stale correction is silently attributed to the current shot.
      //
      // Behaviour mirrors the NOT_READY paths:
      //   - without wait_for_ready: fail loudly (the caller cannot tell the
      //     difference between stale-OK and genuine-OK from the result alone).
      //   - with wait_for_ready: retry; the server will eventually answer the
      //     current request and its response will carry the matching request_id.
      if (response->request_id != static_cast<std::uint32_t>(tag)) {
        if (!wait_for_ready_)
          throw std::runtime_error(
              "remote_sink: stale response in slot " + std::to_string(slot) +
              " (got request_id=" + std::to_string(response->request_id) +
              ", expected " + std::to_string(tag) +
              "); the host was preempted for more than one ring lap --"
              " increase --server-slots or use --wait-for-ready to recover");
        ++last_not_ready_retries_;
        if (std::chrono::steady_clock::now() >= deadline)
          throw std::runtime_error("remote_sink: wait_for_ready timed out after " +
                                   std::to_string(kTimeoutMs) + " ms");
        CUDAQ_REALTIME_CPU_RELAX();
        continue;
      }

      if (response->status == static_cast<int32_t>(wire::RpcStatus::NOT_READY)) {
        ++last_not_ready_retries_;
        if (!wait_for_ready_)
          throw std::runtime_error(
              "remote_sink: get_corrections returned NOT_READY "
              "(the schedule may be too tight; use --wait-for-ready to retry "
              "until the decoder is done)");
        if (std::chrono::steady_clock::now() >= deadline)
          throw std::runtime_error("remote_sink: wait_for_ready timed out after " +
                                   std::to_string(kTimeoutMs) + " ms");
        CUDAQ_REALTIME_CPU_RELAX();
        continue;
      }
      if (response->status != 0) {
        throw std::runtime_error("remote_sink: get_corrections returned status " +
                                 std::to_string(response->status));
      }
      const std::uint8_t *packed =
          rx_slot + sizeof(cudaq::realtime::RPCResponse);
      for (std::uint64_t i = 0; i < n; ++i)
        corrections_log_.push_back((packed[i / 8] >> (i % 8)) & 0x1u);
      ++reads_;
      return;
    }
  }

  void reset(std::uint64_t tag) {
    wire::ResetRequestPayload p{};
    p.decoder_id = static_cast<std::int64_t>(decoder_id_);
    const std::uint32_t slot =
        publish(wire::kResetDecoderFunctionId, &p, sizeof(p), nullptr, 0, tag);
    (void)await_reply(slot, "reset_decoder");
  }

  static constexpr int kTimeoutMs = 5000;

  std::unique_ptr<remote_transport> transport_;
  std::uint8_t *tx_data_ = nullptr, *rx_data_ = nullptr;
  std::uint64_t *tx_flags_ = nullptr, *rx_flags_ = nullptr;
  std::size_t decoder_id_;
  std::uint64_t observables_;
  std::uint32_t num_slots_ = 0, slot_size_ = 0, cursor_ = 0;
  std::uint64_t sent_ = 0, reads_ = 0, stalls_ = 0;
  std::uint32_t last_not_ready_retries_ = 0;
  std::vector<std::uint8_t> corrections_log_;
};

} // namespace

std::unique_ptr<sink> make_udp_server_sink(const std::string &host,
                                           std::uint16_t port,
                                           std::size_t decoder_id,
                                           std::uint32_t num_slots,
                                           std::uint32_t slot_size,
                                           std::uint64_t observables) {
  return std::make_unique<remote_sink>(
      std::make_unique<udp_transport>(host, port, num_slots, slot_size),
      decoder_id, observables);
}

} // namespace cudaq::qec::emulator
