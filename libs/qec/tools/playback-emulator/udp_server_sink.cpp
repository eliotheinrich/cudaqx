/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

/// @file udp_server_sink.cpp
/// @brief The default UDP producer: the decoding RPCs over a plain socket.
///
/// The ring in the retired udp_ring_sink is the RoCE interface shape (see udp_wrapper.h):
/// on an RDMA NIC the hardware writes arriving frames straight into registered
/// slots, so a ring is what you poll.  Over UDP the client gets nothing from
/// it.  Both peers own separate rings joined by datagrams, so a slot index is
/// meaningful only locally, and "the reply lands in the slot I published to"
/// degrades from a shared-memory fact into a convention that one out-of-order
/// reply breaks permanently -- with nothing in the frame to re-establish it.
///
/// So this sink drops the client-side ring entirely.  The kernel's socket
/// receive queue is the inbound queue: deeper than any ring we would size, and
/// it never needs recycling, claiming or lap accounting.  There is no
/// --server-slots to get wrong, because there is no ring.
///
/// Reply handling is a FIFO pipeline, not a search for one answer:
///
///   * every request is recorded in `pending_` in publish order;
///   * `pump()` retires exactly one reply, in arrival order, and hands it to
///     `dispatch()`, which acts on it according to the request it answers --
///     enqueue ACKs carry nothing and are retired, get_corrections replies have
///     their correction bits collected;
///   * a caller that needs an answer blocks in `await()`, which pumps the queue
///     rather than sifting it.  NOTHING IS DISCARDED: every reply is dispatched
///     for its own sake, so a reply arriving while another request is
///     outstanding is still acted on.
///
/// That last point is what makes this extensible.  A richer RPC protocol adds
/// a case to `dispatch()` and nothing else; an unrecognized reply raises rather
/// than being dropped, so a protocol addition cannot go silently unhandled.
///
/// The wire format is byte-identical to udp_ring_sink: one datagram carries
/// one full slot stride, which is what the server's transceiver expects (it
/// drops anything longer than its own page_size).  A playback file and its
/// statistics carry across between the two sinks unchanged.

#include "sinks.h"

#include "cudaq/qec/realtime/decoder_rpc_wire_format.h"
#include "cudaq/realtime/daemon/dispatcher/cpu_relax.h"

#include <arpa/inet.h>
#include <cerrno>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <deque>
#include <netinet/in.h>
#include <stdexcept>
#include <string>
#include <sys/socket.h>
#include <unistd.h>
#include <unordered_map>
#include <vector>

namespace cudaq::qec::emulator {

namespace {

namespace wire = cudaq::qec::decoding::rpc;

class udp_server_sink : public sink {
public:
  udp_server_sink(const std::string &host, std::uint16_t port,
                std::size_t decoder_id, std::uint32_t slot_size,
                std::uint64_t observables)
      : decoder_id_(decoder_id), stride_(slot_size), observables_(observables),
        tx_(slot_size, 0), rx_(slot_size, 0) {
    fd_ = ::socket(AF_INET, SOCK_DGRAM, 0);
    if (fd_ < 0)
      throw std::runtime_error("udp_server_sink: socket() failed");

    // A generous receive buffer is this sink's entire flow-control story: it
    // is where replies wait while we stream enqueues, and it is why no ring
    // depth has to be chosen.  The kernel doubles what we ask for.
    const int rcvbuf = 8 << 20;
    ::setsockopt(fd_, SOL_SOCKET, SO_RCVBUF, &rcvbuf, sizeof(rcvbuf));

    sockaddr_in peer{};
    peer.sin_family = AF_INET;
    peer.sin_port = htons(port);
    if (::inet_pton(AF_INET, host.c_str(), &peer.sin_addr) != 1) {
      ::close(fd_);
      throw std::runtime_error("udp_server_sink: bad host " + host);
    }
    // connect() on a datagram socket fixes the peer, so send()/recv() need no
    // address and the kernel filters replies from anyone else.
    if (::connect(fd_, reinterpret_cast<const sockaddr *>(&peer),
                  sizeof(peer)) != 0) {
      ::close(fd_);
      throw std::runtime_error("udp_server_sink: connect to " + host + ":" +
                               std::to_string(port) + " failed");
    }
  }

  ~udp_server_sink() override {
    if (fd_ >= 0)
      ::close(fd_);
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

  const char *name() const override { return "udp_server"; }

  const std::vector<std::uint8_t> &corrections() const override {
    return corrections_log_;
  }

  std::size_t correction_width() const override { return observables_; }

  std::uint32_t last_not_ready_retries() const override {
    return last_not_ready_retries_;
  }

  void report() const override {
    std::printf("%-17s sent=%llu reads=%llu retired=%llu out_of_order=%llu "
                "unmatched=%llu outstanding=%zu\n",
                name(), static_cast<unsigned long long>(sent_),
                static_cast<unsigned long long>(reads_),
                static_cast<unsigned long long>(retired_),
                static_cast<unsigned long long>(out_of_order_),
                static_cast<unsigned long long>(unmatched_), pending_.size());
  }

private:
  /// A request that has gone out and not yet been answered, in publish order.
  /// `awaited` marks the ones whose status a caller is blocking for; the rest
  /// (streamed enqueues) are retired and forgotten.
  struct pending {
    std::uint32_t id;
    std::uint32_t function_id;
    bool awaited;
  };

  /// Serialize one request into tx_, put it on the wire, and record it as
  /// outstanding.  Returns its id.
  std::uint32_t publish(std::uint32_t function_id, const void *payload,
                        std::size_t payload_len, const std::uint8_t *bits,
                        std::uint64_t num_bits, bool awaited) {
    const std::size_t packed = bits ? wire::bit_packed_bytes(num_bits) : 0;
    const std::size_t body = payload_len + packed;
    if (sizeof(cudaq::realtime::RPCHeader) + body > stride_)
      throw std::runtime_error("udp_server_sink: frame exceeds the " +
                               std::to_string(stride_) + "-byte stride");
    std::memset(tx_.data(), 0, stride_);

    const std::uint32_t id = next_id_++;
    auto *header = reinterpret_cast<cudaq::realtime::RPCHeader *>(tx_.data());
    header->magic = cudaq::realtime::RPC_MAGIC_REQUEST;
    header->function_id = function_id;
    header->arg_len = static_cast<std::uint32_t>(body);
    header->request_id = id;
    header->ptp_timestamp = 0;

    std::uint8_t *args = tx_.data() + sizeof(cudaq::realtime::RPCHeader);
    std::memcpy(args, payload, payload_len);
    if (bits)
      for (std::uint64_t i = 0; i < num_bits; ++i)
        if (bits[i] & 0x1u)
          args[payload_len + i / 8] |= static_cast<std::uint8_t>(1u << (i % 8));

    pending_.push_back({id, function_id, awaited});
    // One datagram carries one full stride: the server's transceiver copies
    // `got` bytes into a page_size slot and drops anything longer.
    if (::send(fd_, tx_.data(), stride_, 0) != static_cast<ssize_t>(stride_))
      throw std::runtime_error("udp_server_sink: send failed: " +
                               std::string(std::strerror(errno)));
    return id;
  }

  /// Act on one reply, according to the request it answers.
  ///
  /// This is the whole of the protocol's client-side behaviour, and the single
  /// place a new RPC needs a case.  It runs for EVERY reply, including ones
  /// nobody is blocked on, so a reply is never dropped merely because it
  /// arrived while a different request was outstanding.
  void dispatch(const pending &req,
                const cudaq::realtime::RPCResponse *response,
                const std::uint8_t *frame) {
    switch (req.function_id) {
    case wire::kEnqueueSyndromesFunctionId:
      // No action, deliberately: a streamed enqueue's reply acknowledges
      // receipt and carries no payload.  Retiring it here is the only work it
      // needs -- it keeps the queue in step and the socket drained.
      break;

    case wire::kGetCorrectionsFunctionId:
      // The corrections live in this reply and nowhere else, so they are
      // collected as it is retired rather than by whoever happens to be
      // waiting.  NOT_READY carries no bits; the waiter re-asks.
      if (response->status == 0) {
        const std::uint8_t *packed =
            frame + sizeof(cudaq::realtime::RPCResponse);
        for (std::uint64_t i = 0; i < observables_; ++i)
          corrections_log_.push_back((packed[i / 8] >> (i % 8)) & 0x1u);
        ++reads_;
      }
      break;

    case wire::kResetDecoderFunctionId:
      // No action: the reply only confirms the decoder was reset.
      break;

    default:
      // A protocol addition must be handled here.  Raising keeps that
      // explicit: the alternative is dropping a reply that carried state.
      throw std::runtime_error(
          "udp_server_sink: no handler for a reply to function_id=" +
          std::to_string(req.function_id) +
          "; add a case to dispatch() when extending the RPC protocol");
    }
  }

  /// Retire at most one reply, in arrival order.  false if the queue is dry.
  bool pump() {
    const ssize_t got = ::recv(fd_, rx_.data(), stride_, MSG_DONTWAIT);
    if (got <= 0)
      return false; // EAGAIN: nothing queued
    const auto *response =
        reinterpret_cast<const cudaq::realtime::RPCResponse *>(rx_.data());
    const std::uint32_t id = response->request_id;

    // FIFO: the head of the queue is the reply we expect next.  Anything else
    // is tolerated -- matching is by id, not by position -- but counted, since
    // it means the server did not answer in request order.
    auto it = pending_.begin();
    if (it == pending_.end() || it->id != id) {
      for (it = pending_.begin(); it != pending_.end() && it->id != id; ++it)
        ;
      if (it == pending_.end()) {
        // A duplicate, or an answer to something already retired.  Nothing
        // owns it, so there is nothing to act on.
        ++unmatched_;
        return true;
      }
      ++out_of_order_;
    }

    const pending req = *it;
    pending_.erase(it);
    ++retired_;
    dispatch(req, response, rx_.data());
    if (req.awaited)
      completed_[req.id] = response->status;
    return true;
  }

  /// Pump the queue until `id` has been retired, and return its status.
  std::int32_t await(std::uint32_t id, const char *op,
                     std::chrono::steady_clock::time_point deadline) {
    while (true) {
      auto done = completed_.find(id);
      if (done != completed_.end()) {
        const std::int32_t status = done->second;
        completed_.erase(done);
        return status;
      }
      if (pump())
        continue;
      if (std::chrono::steady_clock::now() >= deadline)
        throw std::runtime_error(
            "udp_server_sink: timed out awaiting " + std::string(op) +
            " response for request " + std::to_string(id) + " after " +
            std::to_string(kTimeoutMs) + " ms (" + std::to_string(sent_) +
            " enqueues sent, " + std::to_string(last_not_ready_retries_) +
            " retries on this read, " + std::to_string(pending_.size()) +
            " requests outstanding); the decoder is not answering");
      CUDAQ_REALTIME_CPU_RELAX();
    }
  }

  void enqueue(const event &e, std::uint64_t tag) {
    wire::EnqueueRequestPayload p{};
    p.decoder_id = static_cast<std::int64_t>(decoder_id_);
    p.counter = static_cast<std::int64_t>(tag);
    p.syndrome_mapping_id = 0;
    p.num_syndromes = static_cast<std::int64_t>(e.num_syndromes);
    publish(wire::kEnqueueSyndromesFunctionId, &p, sizeof(p), e.syndrome_data,
            e.num_syndromes, /*awaited=*/false);
    ++sent_;
    // Streamed: do not wait for it.  Retire whatever has come back in the
    // meantime, so the socket queue does not grow over a long replay.
    while (pump())
      ;
  }

  void read(const event &, std::uint64_t tag) {
    (void)tag;
    wire::GetCorrectionsRequestPayload p{};
    p.decoder_id = static_cast<std::int64_t>(decoder_id_);
    p.return_size = static_cast<std::int64_t>(observables_);
    p.reset = 0;

    last_not_ready_retries_ = 0;
    const auto deadline = std::chrono::steady_clock::now() +
                          std::chrono::milliseconds(kTimeoutMs);
    while (true) {
      const std::uint32_t id =
          publish(wire::kGetCorrectionsFunctionId, &p, sizeof(p), nullptr, 0,
                  /*awaited=*/true);
      const std::int32_t status = await(id, "get_corrections", deadline);

      if (status == static_cast<std::int32_t>(wire::RpcStatus::NOT_READY)) {
        // The decoder has not finished this block.  Without wait_for_ready that
        // is an error -- the schedule outran the decoder -- and with it we ask
        // again, since only a fresh request can produce a fresh answer.
        ++last_not_ready_retries_;
        if (!wait_for_ready_)
          throw std::runtime_error(
              "udp_server_sink: get_corrections returned NOT_READY; the schedule "
              "is tighter than the decoder -- pass --wait-for-ready to block "
              "until it catches up");
        if (std::chrono::steady_clock::now() >= deadline)
          throw std::runtime_error("udp_server_sink: still NOT_READY after " +
                                   std::to_string(kTimeoutMs) +
                                   " ms of --wait-for-ready");
        CUDAQ_REALTIME_CPU_RELAX();
        continue;
      }
      if (status != 0)
        throw std::runtime_error(
            "udp_server_sink: get_corrections returned status " +
            std::to_string(status));
      // The bits were collected by dispatch() as the reply was retired.
      return;
    }
  }

  void reset(std::uint64_t tag) {
    (void)tag;
    wire::ResetRequestPayload p{};
    p.decoder_id = static_cast<std::int64_t>(decoder_id_);
    const auto deadline = std::chrono::steady_clock::now() +
                          std::chrono::milliseconds(kTimeoutMs);
    const std::uint32_t id = publish(wire::kResetDecoderFunctionId, &p,
                                     sizeof(p), nullptr, 0, /*awaited=*/true);
    const std::int32_t status = await(id, "reset_decoder", deadline);
    if (status != 0)
      throw std::runtime_error("udp_server_sink: reset_decoder returned status " +
                               std::to_string(status));
  }

  static constexpr int kTimeoutMs = 5000;

  int fd_ = -1;
  std::size_t decoder_id_;
  std::size_t stride_;
  std::uint64_t observables_;
  std::vector<std::uint8_t> tx_, rx_;
  std::uint32_t next_id_ = 1;
  /// Outstanding requests in publish order, and the statuses of the retired
  /// ones a caller is still blocking for.
  std::deque<pending> pending_;
  std::unordered_map<std::uint32_t, std::int32_t> completed_;
  std::uint64_t sent_ = 0, reads_ = 0, retired_ = 0, out_of_order_ = 0,
                unmatched_ = 0;
  std::uint32_t last_not_ready_retries_ = 0;
  std::vector<std::uint8_t> corrections_log_;
};

} // namespace

std::unique_ptr<sink> make_udp_server_sink(const std::string &host,
                                             std::uint16_t port,
                                             std::size_t decoder_id,
                                             std::uint32_t slot_size,
                                             std::uint64_t observables) {
  return std::make_unique<udp_server_sink>(host, port, decoder_id, slot_size,
                                         observables);
}

} // namespace cudaq::qec::emulator
