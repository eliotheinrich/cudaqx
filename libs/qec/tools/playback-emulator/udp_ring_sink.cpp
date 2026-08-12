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

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <thread>
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
    r.tx_data =
        reinterpret_cast<std::uint8_t *>(cpu_udp_get_tx_ring_data_addr(xcvr_));
    r.rx_data =
        reinterpret_cast<std::uint8_t *>(cpu_udp_get_rx_ring_data_addr(xcvr_));
    r.tx_flags =
        reinterpret_cast<std::uint64_t *>(cpu_udp_get_tx_ring_flag_addr(xcvr_));
    r.rx_flags =
        reinterpret_cast<std::uint64_t *>(cpu_udp_get_rx_ring_flag_addr(xcvr_));
    r.num_slots = num_slots_;
    r.slot_size = slot_size_;
    return r;
  }

  const char *name() const override { return "udp_ring"; }

private:
  std::string host_;
  std::uint16_t port_;
  std::uint32_t num_slots_, slot_size_;
  cpu_udp_transceiver_t xcvr_ = nullptr;
};

/// RETIRED REFERENCE.  Superseded by udp_server_sink.cpp, which talks to the
/// same server over a plain socket and no client-side ring.  Kept, and still
/// selectable as `--sink=udp_ring`, because it is the only in-tree example of
/// driving the RoCE-shaped ring API that a real RDMA transport exposes -- and
/// because it is the thing the default sink is measured against.
///
/// Do not reach for this one by default.  The ring buys the client nothing over
/// UDP (both peers own separate rings joined by datagrams, so a slot index is
/// meaningful only locally), it costs a --server-slots that has to be chosen,
/// and it runs about half as fast as the socket it wraps.
///
/// Publishes the decoding RPCs to a remote server over any `remote_transport`.
///
/// The frame layout is identical to the in-process sinks -- 24-byte RPCHeader
/// plus the same payload structs -- so a playback file and its statistics carry
/// across unchanged.  Enqueues stream (publish and advance, replies drained a
/// lap later); reads and resets round-trip, because they return a value.
class remote_sink : public sink {
public:
  remote_sink(std::unique_ptr<remote_transport> transport,
              std::size_t decoder_id, std::uint64_t observables,
              std::uint32_t window)
      : transport_(std::move(transport)), decoder_id_(decoder_id),
        observables_(observables), window_(window) {
    transport_->connect();
    const transport_rings r = transport_->rings();
    tx_data_ = r.tx_data;
    rx_data_ = r.rx_data;
    tx_flags_ = r.tx_flags;
    rx_flags_ = r.rx_flags;
    num_slots_ = r.num_slots;
    slot_size_ = r.slot_size;
    if (!tx_data_ || !rx_data_ || !tx_flags_ || !rx_flags_ || !num_slots_)
      throw std::runtime_error("udp_ring_sink: transport reported no rings");
    // The cursor must walk EVERY slot of the transport ring, in order.  A
    // window narrower than the ring hangs: replies are placed by arrival across
    // the whole ring, so any that land in a slot the cursor never visits are
    // never collected.  Measured directly -- ring 16 with a window of 8 fails
    // where ring 8 with a window of 8 passes.
    if (window_ == 0 || window_ > num_slots_)
      window_ = num_slots_;
    slot_tag_.assign(num_slots_, kNoTag);
    reply_buf_.assign(slot_size_, 0);
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

  std::uint32_t last_not_ready_retries() const override {
    return last_not_ready_retries_;
  }

  void report() const override {
    std::printf("%-17s sent=%llu reads=%llu stalls=%llu retired=%llu "
        "reorders=%llu ring_depth=%u\n",
                transport_->name(), static_cast<unsigned long long>(sent_),
                static_cast<unsigned long long>(reads_),
                static_cast<unsigned long long>(stalls_),
                static_cast<unsigned long long>(retired_),
        static_cast<unsigned long long>(reorders_), num_slots_);
  }

private:
  /// One discipline for every way the server can fall behind the schedule.
  ///
  /// Three symptoms, one cause.  A slot that still holds an unanswered request
  /// means we lapped the ring; a reply carrying a PREVIOUS request_id means we
  /// lapped it and the late answer landed in the recycled slot; NOT_READY means
  /// the decode has not finished.  In each case the emulator got ahead of the
  /// decoder, so each answers to the same policy:
  ///
  ///   wait_for_ready = false -> throw here.  Blocking instead would bury the
  ///     overrun inside the run's wall clock, and nothing in the returned
  ///     corrections distinguishes a stalled read from a prompt one.
  ///   wait_for_ready = true  -> return, so the caller can retry, until the
  ///     shared deadline expires.
  ///
  /// Callers differ only in the remedy -- claim() waits in place, read()
  /// re-publishes -- so this decides policy and leaves the loop to them.
  /// Counters stay with the callers too, since "stall" and "retry" count
  /// different things.
  void behind(const std::string &what, std::uint32_t slot,
              std::chrono::steady_clock::time_point deadline) {
    const std::string where =
        "udp_ring_sink: " + what + " in slot " + std::to_string(slot) + " (" +
        std::to_string(sent_) +
        " enqueues sent, ring_depth=" + std::to_string(num_slots_) + ")";
    if (!wait_for_ready_)
      throw std::runtime_error(
          where +
          "; the schedule is tighter than the decoder -- pass "
          "--wait-for-ready to block until it catches up, or widen the ring "
          "with --server-slots=" +
          std::to_string(num_slots_ * 2));
    if (std::chrono::steady_clock::now() >= deadline)
      throw std::runtime_error(where + "; still behind after " +
                               std::to_string(kTimeoutMs) +
                               " ms of --wait-for-ready");
    // Back off rather than spin.  Every retry re-publishes, and a full-rate
    // retry loop is itself what pushes requests off the wire: enqueue-only
    // replays never lose anything at any ring depth, while a read that retries
    // hard loses them within tens of attempts.  Waiting is also simply correct
    // here -- NOT_READY means the decoder needs time, which spinning denies it.
    if (++consecutive_waits_ <= kSpinBeforeSleep)
      CUDAQ_REALTIME_CPU_RELAX();
    else
      std::this_thread::sleep_for(std::chrono::microseconds(kBackoffUs));
  }

  /// Claim the cursor slot.  Every RPC gets a reply -- even the fire-and-forget
  /// ones -- so a streamed enqueue leaves one behind that must be retired
  /// before the slot comes round again.
  ///
  /// harvest() does the retiring, by id and across the whole ring, so a reply
  /// that landed elsewhere still frees this slot.  Waiting here at all means
  /// our own request is genuinely unanswered, i.e. we lapped the ring -- which
  /// goes through behind() like every other overrun.
  void claim(std::uint32_t slot) {
    bool stalled = false;
    consecutive_waits_ = 0;
    const auto deadline = std::chrono::steady_clock::now() +
                          std::chrono::milliseconds(kTimeoutMs);
    while (true) {
      harvest(kNoMatch);
      if (tx_flags_[slot] == 0)
        break;
      behind("lapped the ring waiting for an enqueue ACK", slot, deadline);
      stalled = true;
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
      throw std::runtime_error("udp_ring_sink: frame exceeds the " +
                               std::to_string(slot_size_) + "-byte slot");
    std::memset(tx_slot, 0, sizeof(cudaq::realtime::RPCHeader) + body);

    auto *header = reinterpret_cast<cudaq::realtime::RPCHeader *>(tx_slot);
    header->magic = cudaq::realtime::RPC_MAGIC_REQUEST;
    header->function_id = function_id;
    header->arg_len = static_cast<std::uint32_t>(body);
    // request_id identifies THIS transmission, not the shot: read() retries by
    // republishing, and reusing the caller's tag made every attempt carry an id
    // the server had already answered -- it does not answer twice, so the retry
    // waited for a reply that was never coming.  The semantic counter travels
    // in the payload (EnqueueRequestPayload::counter), so the header is free to
    // be a plain sequence number.
    const std::uint32_t id = next_id_++;
    header->request_id = id;
    slot_tag_[slot] = id;
    header->ptp_timestamp = 0;
    std::uint8_t *args = tx_slot + sizeof(cudaq::realtime::RPCHeader);
    std::memcpy(args, payload, payload_len);
    if (bits)
      for (std::uint64_t i = 0; i < num_bits; ++i)
        if (bits[i] & 0x1u)
          args[payload_len + i / 8] |= static_cast<std::uint8_t>(1u << (i % 8));

    __sync_synchronize();
    tx_flags_[slot] = reinterpret_cast<std::uint64_t>(tx_slot);
    cursor_ = (slot + 1u) % window_;
    return slot;
  }

  /// Snapshot every slot: which hold a reply, and for which request.  Used to
  /// tell "the reply never came" from "the reply came back in another slot".
  std::string dump_ring() const {
    std::string out;
    for (std::uint32_t s = 0; s < num_slots_; ++s) {
      const auto *resp = reinterpret_cast<const cudaq::realtime::RPCResponse *>(
          rx_data_ + s * slot_size_);
      out += "[" + std::to_string(s) + " rx=" + (rx_flags_[s] ? "1" : "0") +
             " tx=" + (tx_flags_[s] ? "1" : "0") + " want=" +
             std::to_string(static_cast<std::uint32_t>(slot_tag_[s])) +
             " got=" + std::to_string(resp->request_id) + "]";
    }
    return out;
  }

  /// Retire every reply currently parked in the rx ring, and report whether
  /// the one carrying `wanted` was among them (copying it into reply_buf_).
  /// Replies for other requests are drained rather than left to block the rx
  /// ring -- the transport's RX thread refuses to fill a slot until its flag is
  /// cleared, so an uncollected reply back-pressures every later arrival.
  ///
  /// A reply is owned by its request_id, NOT by the slot it lands in.  The
  /// transport fills the rx ring in ARRIVAL order across the whole ring, so the
  /// reply to a request published in slot s routinely turns up somewhere else.
  /// Waiting on rx_flags_[s] therefore hangs whenever the two orders diverge --
  /// which is what made shallow rings fail while depth 1 (one slot, so the
  /// orders cannot diverge) and deep rings (enough slack to stay in step)
  /// worked.  Measured directly: ring 16 with a cursor window of 8 hangs where
  /// ring 8 with a window of 8 does not.
  ///
  /// Retiring by id also subsumes the old stale-response check: a late reply
  /// for an earlier request is handed back to that request's slot instead of
  /// being mistaken for this one's.
  bool harvest(std::uint32_t wanted) {
    bool found = false;
    for (std::uint32_t s = 0; s < num_slots_; ++s) {
      if (rx_flags_[s] == 0)
        continue;
      __sync_synchronize();
      const std::uint8_t *rx_slot = rx_data_ + s * slot_size_;
      const auto *response =
          reinterpret_cast<const cudaq::realtime::RPCResponse *>(rx_slot);
      const std::uint32_t id = response->request_id;
      if (id == wanted) {
        // Copy before releasing: once the flag is cleared the transport may
        // reuse this slot for the next arrival.
        std::memcpy(reply_buf_.data(), rx_slot, slot_size_);
        found = true;
      }
      rx_flags_[s] = 0;
      ++retired_;
      // Ids are issued in publish order, and the transport fills rx slots in
      // arrival order, so a reply whose id goes backwards means the peer
      // answered out of request order -- which is exactly what slips the two
      // FIFO cursors apart.
      if (id < last_retired_id_)
        ++reorders_;
      last_retired_id_ = id;
      // No tx_flag to release: the transport's TX thread clears it once the
      // datagram is on the wire (udp_transceiver.cpp, txLoop), so tx_flags_
      // means "pending send", not "awaiting reply".  Reuse of a slot therefore
      // only ever waits for the send, which is exactly the buffer's lifetime.
    }
    return found;
  }

  /// Block until the reply carrying `id` has been harvested, wherever it lands.
  /// Returns a pointer to the copy in reply_buf_.
  const std::uint8_t *await_id(std::uint32_t id, const char *op,
                               std::chrono::steady_clock::time_point deadline) {
    while (!harvest(id)) {
      if (std::chrono::steady_clock::now() >= deadline)
        throw std::runtime_error(
            "udp_ring_sink: timed out awaiting " + std::string(op) +
            " response for request " + std::to_string(id) + " after " +
            std::to_string(kTimeoutMs) + " ms (" + std::to_string(sent_) +
            " enqueues sent, " + std::to_string(last_not_ready_retries_) +
            " retries on this read, " + std::to_string(retired_) +
            " replies retired; ring: " + dump_ring() +
            ", ring_depth=" + std::to_string(num_slots_) +
            "); on a jittery host try --server-slots=" +
            std::to_string(num_slots_ * 2));
      CUDAQ_REALTIME_CPU_RELAX();
    }
    return reply_buf_.data();
  }

  void enqueue(const event &e, std::uint64_t tag) {
    wire::EnqueueRequestPayload p{};
    p.decoder_id = static_cast<std::int64_t>(decoder_id_);
    p.counter = static_cast<std::int64_t>(tag);
    p.syndrome_mapping_id = 0;
    p.num_syndromes = static_cast<std::int64_t>(e.num_syndromes);
    // Streamed: publish and move on.  The reply is drained by claim() when
    // this slot comes round again -- credit discipline one lap of the ring
    // deep.
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
    consecutive_waits_ = 0;
    const auto deadline = std::chrono::steady_clock::now() +
                          std::chrono::milliseconds(kTimeoutMs);
    while (true) {
      const std::uint32_t slot = publish(wire::kGetCorrectionsFunctionId, &p,
                                         sizeof(p), nullptr, 0, tag);
      // The id this attempt published; a retry gets a fresh one.
      const std::uint32_t id = static_cast<std::uint32_t>(slot_tag_[slot]);
      const std::uint8_t *rx_slot = await_id(id, "get_corrections", deadline);
      const auto *response =
          reinterpret_cast<const cudaq::realtime::RPCResponse *>(rx_slot);

      if (response->status ==
          static_cast<int32_t>(wire::RpcStatus::NOT_READY)) {
        ++last_not_ready_retries_;
        behind("get_corrections returned NOT_READY", slot, deadline);
        continue;
      }
      if (response->status != 0) {
        throw std::runtime_error(
            "udp_ring_sink: get_corrections returned status " +
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
    const auto deadline = std::chrono::steady_clock::now() +
                          std::chrono::milliseconds(kTimeoutMs);
    (void)await_id(static_cast<std::uint32_t>(slot_tag_[slot]), "reset_decoder",
                   deadline);
  }

  static constexpr int kTimeoutMs = 5000;
  /// Spin briefly for the common case (the decoder is about to finish), then
  /// sleep, so a long wait costs the wire nothing.
  static constexpr int kSpinBeforeSleep = 8;
  static constexpr int kBackoffUs = 100;

  std::unique_ptr<remote_transport> transport_;
  std::uint8_t *tx_data_ = nullptr, *rx_data_ = nullptr;
  std::uint64_t *tx_flags_ = nullptr, *rx_flags_ = nullptr;
  std::size_t decoder_id_;
  std::uint64_t observables_;
  std::uint32_t num_slots_ = 0, slot_size_ = 0, cursor_ = 0, window_ = 0;
  std::uint64_t sent_ = 0, reads_ = 0, stalls_ = 0, retired_ = 0,
                reorders_ = 0;
  std::uint32_t last_retired_id_ = 0;
  /// Tag of the request each slot currently holds, so claim() can tell the
  /// reply it is waiting for from a late one.  kNoTag never matches a real
  /// request_id, so a slot that has never been published is claimed freely.
  std::vector<std::uint64_t> slot_tag_;
  static constexpr std::uint64_t kNoTag = ~0ull;
  /// Unique per publish, so a retry never reuses an already-answered id.
  std::uint32_t next_id_ = 1;
  /// An id no request can carry, for "harvest everything, want nothing".
  static constexpr std::uint32_t kNoMatch = 0;
  /// Replies are copied out here before their slot is released.
  std::vector<std::uint8_t> reply_buf_;
  std::uint32_t consecutive_waits_ = 0;
  std::uint32_t last_not_ready_retries_ = 0;
  std::vector<std::uint8_t> corrections_log_;
};

} // namespace

std::unique_ptr<sink>
make_udp_ring_sink(const std::string &host, std::uint16_t port,
                     std::size_t decoder_id, std::uint32_t num_slots,
                     std::uint32_t slot_size, std::uint64_t observables) {
  return std::make_unique<remote_sink>(
      std::make_unique<udp_transport>(host, port, num_slots, slot_size),
      decoder_id, observables, num_slots);
}

} // namespace cudaq::qec::emulator
