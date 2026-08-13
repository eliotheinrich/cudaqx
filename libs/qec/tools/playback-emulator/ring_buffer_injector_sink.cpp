/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "sinks.h"

#include "rpc_call.h"

#include "cudaq/qec/decoder.h"
#include "cudaq/qec/pcm_utils.h"
#include "cudaq/qec/realtime/decoding_config.h"
#include "cudaq/realtime/daemon/dispatcher/cpu_relax.h"

// Internal headers: the producer entry point and the session that owns the
// ring live in lib/, not include/.  The tool is in-tree, so reach them
// directly rather than widening the public surface for a prototype.
#include "qec_realtime_session.h"
#include "rpc_producer.h"
#include "../../lib/realtime/realtime_decoding.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <sstream>
#include <stdexcept>

namespace cudaq::qec::emulator {

namespace {

namespace config = cudaq::qec::decoding::config;

/// Publishes fully-serialized RPC frames straight into the session's ring
/// slots.  `transport()` is what makes this sink different from
/// `udp_server_sink`: both build the same `rpc_call` (in `sink::send`/
/// `try_get_corrections`); this class only knows how to get those bytes into
/// a ring slot and, for blocking calls, wait for the reply.
///
/// A non-blocking call (enqueue) publishes and returns without waiting for
/// the ACK: the cursor advances every call and a slot is reclaimed only when
/// it comes back around, so the ring runs at depth `num_slots` and stalls
/// only when the dispatcher is a full lap behind.  A blocking call
/// (get_corrections, reset) publishes into the cursor slot and spins on the
/// TX doorbell exactly as `reclaim()` does, costing the ring turnaround
/// (~2 us) instead of `rpc_producer`'s 200 us poll granularity; on NOT_READY
/// it returns that status rather than retrying, which is the base class's job.
///
/// Writes into the session's public ring accessors directly rather than
/// through `rpc_producer`, so it does not inherit that library's
/// process-global single-producer guard.  The same one-emulator-process-per-
/// source rule still applies: `qec_realtime_session::initialize()` rejects a
/// second concurrent HOST-mode session.
class ring_buffer_injector_sink : public sink {
public:
  ring_buffer_injector_sink(const std::string &config_path,
                            std::size_t decoder_id,
                            const std::string &dem_file = "")
      : sink(kWireDecoderId, /*num_observables=*/0) {
    // ---- Realize one decoder from the YAML config --------------------------
    std::ifstream in(config_path);
    if (!in)
      throw std::runtime_error("ring_buffer_injector_sink: cannot open " +
                               config_path);
    std::stringstream buf;
    buf << in.rdbuf();

    auto multi = config::multi_decoder_config::from_yaml_str(buf.str());
    const auto match = std::find_if(
        multi.decoders.begin(), multi.decoders.end(), [&](const auto &dc) {
          return dc.id >= 0 && static_cast<std::size_t>(dc.id) == decoder_id;
        });
    if (match == multi.decoders.end())
      throw std::runtime_error("ring_buffer_injector_sink: " + config_path +
                               " has no decoder with id " +
                               std::to_string(decoder_id));

    syndrome_size_ = match->syndrome_size;
    // Observables are the -1 column terminators in the sparse O matrix, the
    // same way realtime_decoding.cpp derives it.  The base class only learns
    // this now -- it cannot be passed to the base constructor's member-init
    // list because it comes from parsing this YAML file.
    set_num_observables(static_cast<std::uint64_t>(
        std::count(match->O_sparse.begin(), match->O_sparse.end(), -1)));

    // Mirrors realtime_decoding.cpp::create_realtime_decoder: build the parity
    // check matrix from the sparse config, realize the plugin, and attach the
    // observable/detector maps the decode path expects.  Reuse the library's
    // own parameter preparation rather than re-deriving it, so a DEM-native
    // GLOBAL decoder (chromobius behind a TRT predecoder) gets its model text
    // forwarded correctly as `global_decoder_params.stim_dem`.
    auto params =
        cudaq::qec::decoding::host::prepare_decoder_params(*match);
    if (match->cuda_device_id.has_value())
      params.insert("cuda_device_id", match->cuda_device_id.value());

    // A top-level DEM decoder can be named by the config instead of --dem.
    std::string dem_source = dem_file;
    if (dem_source.empty() && !match->stim_dem_path.empty() &&
        !params.contains("global_decoder"))
      dem_source = match->stim_dem_path;

    // `decoder_init` is a variant, but realtime_decoding.cpp only ever fills
    // the matrix alternative -- which locks out decoders that need a Stim DEM
    // string instead (chromobius).  The session takes a caller-owned decoder
    // vector, so the choice is ours to make here rather than the library's:
    // name a DEM beside the config and we hand that over instead of H.
    //
    // D_sparse below is unaffected either way.  How a decoder was CONSTRUCTED
    // is independent of how accumulated measurements become detectors.
    cudaq::qec::decoder_init init = cudaq::qec::pcm_from_sparse_vec(
        match->H_sparse, match->syndrome_size, match->block_size);
    if (!dem_source.empty()) {
      std::ifstream dem_in(dem_source);
      if (!dem_in)
        throw std::runtime_error(
            "ring_buffer_injector_sink: cannot open dem_file " + dem_source);
      std::stringstream dem_buf;
      dem_buf << dem_in.rdbuf();
      init = dem_buf.str();
    }

    auto decoder = cudaq::qec::get_decoder(match->type, init, params);
    if (!decoder)
      throw std::runtime_error(
          "ring_buffer_injector_sink: get_decoder returned null for type " +
          match->type);
    decoder->set_decoder_id(static_cast<std::uint32_t>(match->id));
    // A DEM-built decoder has already established its own observable mapping:
    // chromobius emits observable FLIPS directly, so its O is the identity over
    // num_observables and its H is detectors x observables.  The config's O
    // lives in error-mechanism space, so setting it would index far past that
    // matrix.  Leave the decoder's own mapping alone.
    if (dem_source.empty())
      decoder->set_O_sparse(match->O_sparse);
    // D is ours either way -- it maps the measurements we enqueue onto
    // detectors, which is a property of the playback, not of the decoder.
    decoder->set_D_sparse(match->D_sparse);
    decoders_.push_back(std::move(decoder));

    // Null launch fn: a HOST-mode (CPU decoder) session does not need the
    // dispatch-kernel symbol that lives in libcudaq-realtime-dispatch.a.  A
    // GPU graph-dispatch decoder would require linking that archive and
    // passing &cudaq_launch_dispatch_kernel_regular here.
    session_ = std::make_unique<cudaq::qec::realtime::qec_realtime_session>(
        decoders_, /*device_launch_fn=*/nullptr);
    session_->initialize();

    // ---- Validate the ring geometry ----------------------------------------
    num_slots_ = session_->num_slots();
    slot_size_ = session_->slot_size();

    // Validate the frame fits a slot ONCE, here, rather than per send: an
    // oversized payload must be a startup error, not something discovered on
    // the timing thread.
    const std::size_t max_frame =
        sizeof(cudaq::realtime::RPCHeader) +
        sizeof(wire::EnqueueRequestPayload) +
        wire::bit_packed_bytes(syndrome_size_);
    if (max_frame > slot_size_)
      throw std::runtime_error(
          "ring_buffer_injector_sink: a " + std::to_string(syndrome_size_) +
          "-bit frame needs " + std::to_string(max_frame) +
          " bytes but the ring slot stride is only " +
          std::to_string(slot_size_));
  }

  ~ring_buffer_injector_sink() override {
    // Drain before finalizing the session, so the dispatcher is not torn down
    // with requests still in flight.
    for (std::uint32_t s = 0; s < num_slots_; ++s)
      (void)reclaim(s, kReclaimTimeoutMs);
    if (session_)
      session_->finalize();
  }

  void warm_up() override {
    std::vector<std::uint8_t> zeros(syndrome_size_, 0);
    cudaq::qec::decoding::rpc_producer::enqueue_syndromes(
        *session_, kWireDecoderId, zeros.data(), zeros.size(), /*tag=*/0);
    cudaq::qec::decoding::rpc_producer::reset_decoder(*session_, kWireDecoderId);
  }

  const char *name() const override { return "ring_buffer_injector"; }

  std::size_t syndrome_size() const override { return syndrome_size_; }

  void report() const override {
    if (corrections_read_) {
      std::string last(num_observables_, '0');
      const std::size_t base = corrections().size() - num_observables_;
      for (std::size_t i = 0; i < num_observables_; ++i)
        last[i] = static_cast<char>('0' + corrections()[base + i]);
      std::printf("corrections       reads=%llu width=%llu last=%s\n",
                  static_cast<unsigned long long>(corrections_read_),
                  static_cast<unsigned long long>(num_observables_),
                  last.c_str());
    }
    std::printf("injector          sent=%llu sync_ops=%llu stalls=%llu (%.2f%%) "
                "rpc_errors=%llu ring_depth=%u\n",
                static_cast<unsigned long long>(sent_),
                static_cast<unsigned long long>(synchronous_ops_),
                static_cast<unsigned long long>(stalls_),
                sent_ ? 100.0 * static_cast<double>(stalls_) /
                            static_cast<double>(sent_)
                      : 0.0,
                static_cast<unsigned long long>(rpc_errors_), num_slots_);
  }

protected:
  std::int32_t transport(const rpc_call &call,
                         std::uint8_t *result_buf) override {
    const std::uint32_t slot = cursor();
    require_slot(slot);
    std::uint8_t *rx_slot = session_->rx_data_host() + slot * slot_size_;
    // request_id is the caller's event tag for enqueue (a useful breadcrumb);
    // for blocking calls it's just `sent_` since nothing matches replies by
    // id here -- the ring is strict-FIFO, one request per slot.
    const std::uint32_t request_id = call.blocking
        ? static_cast<std::uint32_t>(sent_)
        : static_cast<std::uint32_t>(call.tag);
    serialize_rpc_frame(call, request_id, rx_slot, slot_size_);
    publish(slot);
    session_->set_producer_cursor((slot + 1u) % num_slots_);

    if (!call.blocking) {
      ++sent_;
      return 0;
    }

    // Watch the TX doorbell instead of sleeping through it -- see reclaim().
    std::uint8_t *tx_slot = session_->tx_data_host() + slot * slot_size_;
    const auto *response =
        reinterpret_cast<const cudaq::realtime::RPCResponse *>(tx_slot);
    volatile std::uint64_t *tx = session_->tx_flags_host();
    const auto deadline = std::chrono::steady_clock::now() +
                          std::chrono::milliseconds(kReclaimTimeoutMs);
    while (true) {
      __sync_synchronize();
      if (response->magic == cudaq::realtime::RPC_MAGIC_RESPONSE &&
          tx[slot] != 0)
        break;
      if (std::chrono::steady_clock::now() >= deadline)
        throw std::runtime_error(
            "ring_buffer_injector_sink: timed out waiting for a response in "
            "slot " + std::to_string(slot) + " after " +
            std::to_string(kReclaimTimeoutMs) + " ms");
      CUDAQ_REALTIME_CPU_RELAX();
    }

    const int32_t status = response->status;
    if (status == 0) {
      if (call.expected_result_bits > 0) {
        // Result body is bit-packed LSB-first, right after the 24-byte
        // response.  unpack_and_log_corrections (in sink::dispatch_once)
        // reads it back out of result_buf once transport() returns.
        const std::uint8_t *packed =
            tx_slot + sizeof(cudaq::realtime::RPCResponse);
        std::memcpy(result_buf, packed,
                    wire::bit_packed_bytes(call.expected_result_bits));
        ++corrections_read_;
      }
      ++synchronous_ops_;
    }
    release_response(slot);
    return status;
  }

private:
  /// Wire routing key, NOT the YAML `id`: the YAML id selects which config
  /// entry to realize, while `qec_realtime_session::initialize()` stamps
  /// `routing_key = 0` for a one-decoder session, which only ever answers
  /// key 0.
  static constexpr std::size_t kWireDecoderId = 0;

  /// Bound on the credit wait, mirroring rpc_producer's kAcquireSlotTimeoutMs.
  static constexpr int kReclaimTimeoutMs = 5000;

  /// The ring cursor lives in the SESSION, not here, so this sink and
  /// rpc_producer stay in lockstep when a run mixes streamed enqueues with
  /// synchronous reads.  The consumer is strict-FIFO; two cursors would
  /// desynchronize it.
  std::uint32_t cursor() const {
    return static_cast<std::uint32_t>(session_->producer_cursor() % num_slots_);
  }

  void require_slot(std::uint32_t slot) {
    if (!reclaim(slot, kReclaimTimeoutMs))
      throw std::runtime_error(
          "ring_buffer_injector_sink: timed out reclaiming ring slot " +
          std::to_string(slot) + " after " +
          std::to_string(kReclaimTimeoutMs) +
          " ms; the dispatcher has stopped servicing the ring");
  }

  /// Address-as-flag publish, matching rpc_producer::write_and_signal.
  void publish(std::uint32_t slot) {
    __sync_synchronize();
    session_->rx_flags_host()[slot] = reinterpret_cast<std::uint64_t>(
        session_->rx_data_dev() + slot * slot_size_);
  }

  /// Wait until `slot` is reusable, reclaiming it if the dispatcher has already
  /// answered.  Returns false if `timeout_ms` elapses first.
  bool reclaim(std::uint32_t slot, int timeout_ms) {
    volatile std::uint64_t *rx = session_->rx_flags_host();
    volatile std::uint64_t *tx = session_->tx_flags_host();
    const auto deadline =
        std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms);
    bool stalled = false;
    while (true) {
      if (tx[slot] != 0) {
        // Response landed: check it, then release exactly as
        // rpc_producer::release_slot does (zero the TX body so a stale magic
        // cannot be misread on reuse, fence, then clear the flag).
        std::uint8_t *tx_slot = session_->tx_data_host() + slot * slot_size_;
        const auto *resp =
            reinterpret_cast<const cudaq::realtime::RPCResponse *>(tx_slot);
        if (resp->magic == cudaq::realtime::RPC_MAGIC_RESPONSE &&
            resp->status != 0)
          ++rpc_errors_;
        std::memset(tx_slot, 0, slot_size_);
        __sync_synchronize();
        tx[slot] = 0;
      }
      if (rx[slot] == 0 && tx[slot] == 0)
        break;
      if (std::chrono::steady_clock::now() >= deadline)
        return false;
      // Ring is a full lap ahead of the dispatcher: credit-based backpressure.
      stalled = true;
      CUDAQ_REALTIME_CPU_RELAX();
    }
    if (stalled)
      ++stalls_;
    return true;
  }

  /// Mirror of rpc_producer::release_slot: wipe the TX body so a stale magic
  /// cannot be misread on reuse, fence, then drop the flag.
  void release_response(std::uint32_t slot) {
    std::uint8_t *tx_slot = session_->tx_data_host() + slot * slot_size_;
    std::memset(tx_slot, 0, slot_size_);
    __sync_synchronize();
    session_->tx_flags_host()[slot] = 0;
  }

  // ---- decoder + session state ----
  std::size_t syndrome_size_ = 0;
  std::uint64_t corrections_read_ = 0;
  std::vector<std::unique_ptr<cudaq::qec::decoder>> decoders_;
  std::unique_ptr<cudaq::qec::realtime::qec_realtime_session> session_;

  // ---- ring state ----
  std::uint32_t num_slots_ = 0;
  std::size_t slot_size_ = 0;
  std::uint64_t sent_ = 0;
  std::uint64_t synchronous_ops_ = 0;
  std::uint64_t stalls_ = 0;
  std::uint64_t rpc_errors_ = 0;
};

} // namespace

std::unique_ptr<sink>
make_ring_buffer_injector_sink(const std::string &config_path,
                               std::size_t decoder_id,
                               const std::string &dem_file) {
  return std::make_unique<ring_buffer_injector_sink>(config_path, decoder_id,
                                                     dem_file);
}

} // namespace cudaq::qec::emulator
