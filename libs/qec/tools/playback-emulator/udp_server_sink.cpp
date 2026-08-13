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
/// `transport()` is what makes this sink different from
/// `ring_buffer_injector_sink`: both build the same `rpc_call` (in
/// `sink::send`/`try_get_corrections`); this class only knows how to move
/// those bytes over a UDP socket and wait for the matching reply.
///
/// Reply handling is a FIFO pipeline: every request is recorded in `pending_`
/// in publish order; `pump()` retires exactly one reply, in arrival order,
/// counting it as a read if the request expected a result body; a caller
/// blocks in `await()`, which pumps the queue rather than sifting it and
/// returns the reply's status plus its raw (still bit-packed) result body. A
/// reply arriving while another request is outstanding is still acted on.

#include "sinks.h"

#include "rpc_call.h"

#include "cudaq/realtime/daemon/dispatcher/cpu_relax.h"

#include <algorithm>
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

class udp_server_sink : public sink {
public:
  udp_server_sink(const std::string &host, std::uint16_t port,
                std::size_t decoder_id, std::uint32_t slot_size,
                std::uint64_t observables)
      : sink(decoder_id, observables), stride_(slot_size), tx_(slot_size, 0),
        rx_(slot_size, 0) {
    fd_ = ::socket(AF_INET, SOCK_DGRAM, 0);
    if (fd_ < 0)
      throw std::runtime_error("udp_server_sink: socket() failed");

    // A generous receive buffer is this sink's entire flow-control story: it
    // is where replies wait while we stream enqueues.
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

  const char *name() const override { return "udp_server"; }

  void report() const override {
    std::printf("%-17s sent=%llu reads=%llu retired=%llu out_of_order=%llu "
                "unmatched=%llu outstanding=%zu\n",
                name(), static_cast<unsigned long long>(sent_),
                static_cast<unsigned long long>(reads_),
                static_cast<unsigned long long>(retired_),
                static_cast<unsigned long long>(out_of_order_),
                static_cast<unsigned long long>(unmatched_), pending_.size());
  }

protected:
  std::int32_t transport(const rpc_call &call,
                         std::uint8_t *result_buf) override {
    if (!call.blocking) {
      publish(call, /*awaited=*/false);
      ++sent_;
      // Streamed: do not wait for it.  Retire whatever has come back in the
      // meantime, so the socket queue does not grow over a long replay.
      while (pump())
        ;
      return 0;
    }

    const auto deadline = std::chrono::steady_clock::now() +
                          std::chrono::milliseconds(kTimeoutMs);
    const std::uint32_t id = publish(call, /*awaited=*/true);
    std::vector<std::uint8_t> body;
    const std::int32_t status = await(id, deadline, &body);
    if (status == 0 && call.expected_result_bits > 0 && result_buf)
      std::memcpy(result_buf, body.data(),
                  std::min(body.size(),
                          wire::bit_packed_bytes(call.expected_result_bits)));
    return status;
  }

private:
  /// A request that has gone out and not yet been answered, in publish order.
  /// `awaited` marks the ones whose status a caller is blocking for; the rest
  /// are retired and forgotten.  `has_result` marks the ones whose reply
  /// carries a result body (e.g. get_corrections), so a successful retire
  /// counts as a read regardless of which RPC it was.
  struct pending {
    std::uint32_t id;
    bool awaited;
    bool has_result;
  };

  /// An awaited request's outcome, stashed by `pump()` until `await()`
  /// collects it. `body` holds the raw (still bit-packed) result bytes when
  /// the reply carried one, empty otherwise.
  struct completion {
    std::int32_t status;
    std::vector<std::uint8_t> body;
  };

  /// Serialize `call` into `tx_`, put it on the wire, and record it as
  /// outstanding.  Returns its id.
  std::uint32_t publish(const rpc_call &call, bool awaited) {
    const std::uint32_t id = next_id_++;
    serialize_rpc_frame(call, id, tx_.data(), stride_);
    pending_.push_back({id, awaited, call.expected_result_bits > 0});
    // One datagram carries one full stride: the server's transceiver copies
    // `got` bytes into a page_size slot and drops anything longer.
    if (::send(fd_, tx_.data(), stride_, 0) != static_cast<ssize_t>(stride_))
      throw std::runtime_error("udp_server_sink: send failed: " +
                               std::string(std::strerror(errno)));
    return id;
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
    if (req.has_result && response->status == 0)
      ++reads_;
    if (req.awaited) {
      completion c{response->status, {}};
      if (response->status == 0) {
        const std::uint8_t *packed =
            rx_.data() + sizeof(cudaq::realtime::RPCResponse);
        c.body.assign(packed,
                      packed + wire::bit_packed_bytes(num_observables_));
      }
      completed_.emplace(req.id, std::move(c));
    }
    return true;
  }

  /// Pump the queue until `id` has been retired, and return its status.
  /// When it carried a result body, copies it into `*body_out` (left empty
  /// otherwise, or on a non-zero status).
  std::int32_t await(std::uint32_t id,
                     std::chrono::steady_clock::time_point deadline,
                     std::vector<std::uint8_t> *body_out) {
    while (true) {
      auto done = completed_.find(id);
      if (done != completed_.end()) {
        const std::int32_t status = done->second.status;
        if (body_out)
          *body_out = std::move(done->second.body);
        completed_.erase(done);
        return status;
      }
      if (pump())
        continue;
      if (std::chrono::steady_clock::now() >= deadline)
        throw std::runtime_error(
            "udp_server_sink: timed out awaiting a response for request " +
            std::to_string(id) + " after " + std::to_string(kTimeoutMs) +
            " ms (" + std::to_string(sent_) + " enqueues sent, " +
            std::to_string(pending_.size()) + " requests outstanding); the "
            "decoder is not answering");
      CUDAQ_REALTIME_CPU_RELAX();
    }
  }

  static constexpr int kTimeoutMs = 5000;

  int fd_ = -1;
  std::size_t stride_;
  std::vector<std::uint8_t> tx_, rx_;
  std::uint32_t next_id_ = 1;

  /// Outstanding requests in publish order, and the outcomes of the retired
  /// ones a caller is still blocking for.
  std::deque<pending> pending_;
  std::unordered_map<std::uint32_t, completion> completed_;
  std::uint64_t sent_ = 0, reads_ = 0, retired_ = 0, out_of_order_ = 0,
                unmatched_ = 0;
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
