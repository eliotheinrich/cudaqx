/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "sinks.h"

#include "cudaq/qec/decoder.h"
#include "cudaq/qec/pcm_utils.h"
#include "cudaq/qec/realtime/decoder_rpc_wire_format.h"
#include "cudaq/qec/realtime/decoding_config.h"
#include "cudaq/realtime/daemon/dispatcher/cpu_relax.h"
#include "cudaq/realtime/daemon/dispatcher/dispatch_kernel_launch.h"

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
namespace wire = cudaq::qec::decoding::rpc;

/// Writes `enqueue_syndromes` frames straight into the session's RX ring and
/// returns without waiting for the ACK.
///
/// Wire format is byte-identical to `rpc_producer::enqueue_syndromes` (24-byte
/// RPCHeader, then a 32-byte EnqueueRequestPayload, then the bit-packed
/// syndrome bits, LSB-first).  What differs is the slot discipline: the cursor
/// advances every call, and a slot is reclaimed only when it comes back around
/// for reuse.  So the ring runs at depth `num_slots` instead of depth 1 and the
/// producer stalls only when the dispatcher is genuinely a full lap behind.
///
/// The injector does NOT go through `rpc_producer` -- it reimplements the wire
/// format against the session's public ring accessors, leaving that file
/// untouched, and therefore does not inherit rpc_producer's process-global
/// single-producer guard.  The same one-emulator-process-per-source rule still
/// applies: `qec_realtime_session::initialize()` rejects a second concurrent
/// HOST-mode session.
class ring_buffer_injector_sink : public sink {
public:
  ring_buffer_injector_sink(const std::string &config_path,
                            std::size_t decoder_id,
                            const std::string &dem_file = "") {
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
    // same way realtime_decoding.cpp derives it.
    num_observables_ = static_cast<std::uint64_t>(
        std::count(match->O_sparse.begin(), match->O_sparse.end(), -1));
    // Preallocate the read-back buffer so get_corrections never allocates on
    // the timing thread; the headroom leaves room for oversize test requests.
    corrections_.assign(std::max<std::size_t>(num_observables_, 1024), 0);

    // Mirrors realtime_decoding.cpp::create_realtime_decoder: build the parity
    // check matrix from the sparse config, realize the plugin, and attach the
    // observable / detector maps the decode path expects.
    // Use the library's own parameter preparation rather than re-deriving it:
    // it materializes `O` for trt_decoder, synthesizes an empty
    // `global_decoder_params` when one is missing, and -- the part that matters
    // here -- reads `stim_dem_path` and forwards the model text down as
    // `global_decoder_params.stim_dem` so a DEM-native GLOBAL decoder
    // (chromobius behind a TRT predecoder) can be constructed.  Duplicating
    // that logic is how this sink previously silently dropped the DEM.
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

  void send(const event &e, std::uint64_t tag) override {
    switch (e.op) {
    case operation::enqueue:
      stream_enqueue(e, tag);
      return;
    case operation::get_corrections: {
      // A read must BLOCK for its result, but it must not SLEEP for it.  We
      // publish the request and watch the TX doorbell exactly as reclaim()
      // already watches it, so the wait costs the ring turnaround (~2 us)
      // instead of rpc_producer's 200 us poll granularity.
      last_not_ready_retries_ = 0;
      const auto deadline = std::chrono::steady_clock::now() +
                            std::chrono::milliseconds(kReclaimTimeoutMs);
      while (true) {
        const auto status = native_round_trip(e, /*is_read=*/true);
        if (status == 0)
          return;
        if (status == static_cast<int32_t>(wire::RpcStatus::NOT_READY)) {
          ++last_not_ready_retries_;
          if (!wait_for_ready_)
            throw std::runtime_error(
                "ring_buffer_injector_sink: get_corrections returned NOT_READY "
                "(the schedule may be too tight; use --wait-for-ready to retry "
                "until the decoder is done)");
          if (std::chrono::steady_clock::now() >= deadline)
            throw std::runtime_error(
                "ring_buffer_injector_sink: wait_for_ready timed out after " +
                std::to_string(kReclaimTimeoutMs) + " ms");
          CUDAQ_REALTIME_CPU_RELAX();
          continue;
        }
        throw std::runtime_error(
            "ring_buffer_injector_sink: get_corrections returned status " +
            std::to_string(status));
      }
    }
    case operation::reset:
      native_round_trip(e, /*is_read=*/false);
      return;
    case operation::stream_until:
      return; // handled in run()
    }
  }

  bool try_get_corrections() override {
    const event dummy{};
    const int32_t status = native_round_trip(dummy, /*is_read=*/true);
    if (status == 0) return true;
    if (status == static_cast<int32_t>(wire::RpcStatus::NOT_READY)) return false;
    throw std::runtime_error(
        "ring_buffer_injector_sink::try_get_corrections: status " +
        std::to_string(status));
  }

  const char *name() const override { return "ring_buffer_injector"; }

  const std::vector<std::uint8_t> &corrections() const override {
    return corrections_log_;
  }

  std::size_t correction_width() const override { return num_observables_; }
  std::size_t syndrome_size() const override { return syndrome_size_; }
  std::uint32_t last_not_ready_retries() const override { return last_not_ready_retries_; }

  void report() const override {
    if (corrections_read_) {
      std::string last(num_observables_, '0');
      const std::size_t base = corrections_log_.size() - num_observables_;
      for (std::size_t i = 0; i < num_observables_; ++i)
        last[i] = static_cast<char>('0' + corrections_log_[base + i]);
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

private:
  /// Wire routing key, NOT the YAML `id`.  The two are different things: the
  /// YAML id selects which config entry to realize, while the routing key
  /// indexes the session's own decoder vector.  `qec_realtime_session::
  /// initialize()` stamps `routing_key = 0` into the function table, so a
  /// one-decoder session only ever answers key 0 -- sending the YAML id here
  /// instead gets `non-zero status (1)` back from the dispatcher.
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

  void stream_enqueue(const event &e, std::uint64_t tag) {
    const std::uint32_t slot = cursor();
    require_slot(slot);
    write_frame(slot, e, tag);
    session_->set_producer_cursor((slot + 1u) % num_slots_);
    ++sent_;
  }

  void require_slot(std::uint32_t slot) {
    if (!reclaim(slot, kReclaimTimeoutMs))
      throw std::runtime_error(
          "ring_buffer_injector_sink: timed out reclaiming ring slot " +
          std::to_string(slot) + " after " +
          std::to_string(kReclaimTimeoutMs) +
          " ms; the dispatcher has stopped servicing the ring");
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

  /// Publish a request into the cursor slot and spin until its response lands,
  /// then consume it.  Returns the response status code (0 == OK).  Never
  /// throws on a non-zero status so the send() loop can retry on NOT_READY.
  ///
  /// `is_read` selects get_corrections (17-byte payload, packed result body)
  /// over reset_decoder (8-byte payload, empty ACK).
  int32_t native_round_trip(const event &, bool is_read) {
    const std::uint64_t n = num_observables_;
    const std::uint32_t slot = cursor();
    require_slot(slot);

    std::uint8_t *rx_slot = session_->rx_data_host() + slot * slot_size_;
    std::size_t body = 0;
    std::uint32_t function_id = 0;
    if (is_read) {
      body = sizeof(wire::GetCorrectionsRequestPayload);
      function_id = wire::kGetCorrectionsFunctionId;
    } else {
      body = sizeof(wire::ResetRequestPayload);
      function_id = wire::kResetDecoderFunctionId;
    }
    std::memset(rx_slot, 0, sizeof(cudaq::realtime::RPCHeader) + body);
    auto *header = reinterpret_cast<cudaq::realtime::RPCHeader *>(rx_slot);
    header->magic = cudaq::realtime::RPC_MAGIC_REQUEST;
    header->function_id = function_id;
    header->arg_len = static_cast<std::uint32_t>(body);
    header->request_id = static_cast<std::uint32_t>(sent_);
    header->ptp_timestamp = 0;

    std::uint8_t *args = rx_slot + sizeof(cudaq::realtime::RPCHeader);
    if (is_read) {
      auto *p = reinterpret_cast<wire::GetCorrectionsRequestPayload *>(args);
      p->decoder_id = static_cast<std::int64_t>(kWireDecoderId);
      p->return_size = static_cast<std::int64_t>(n);
      p->reset = 0;
    } else {
      auto *p = reinterpret_cast<wire::ResetRequestPayload *>(args);
      p->decoder_id = static_cast<std::int64_t>(kWireDecoderId);
    }

    __sync_synchronize();
    session_->rx_flags_host()[slot] = reinterpret_cast<std::uint64_t>(
        session_->rx_data_dev() + slot * slot_size_);
    session_->set_producer_cursor((slot + 1u) % num_slots_);

    // Watch for the reply instead of sleeping through it.
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
    if (status != 0) {
      release_response(slot);
      return status;
    }

    if (is_read) {
      // Result body is bit-packed LSB-first, right after the 24-byte response.
      const std::uint8_t *packed =
          tx_slot + sizeof(cudaq::realtime::RPCResponse);
      for (std::uint64_t i = 0; i < n; ++i)
        corrections_log_.push_back((packed[i / 8] >> (i % 8)) & 0x1u);
      ++corrections_read_;
    }
    release_response(slot);
    ++synchronous_ops_;
    return 0;
  }

  /// Mirror of rpc_producer::release_slot: wipe the TX body so a stale magic
  /// cannot be misread on reuse, fence, then drop the flag.
  void release_response(std::uint32_t slot) {
    std::uint8_t *tx_slot = session_->tx_data_host() + slot * slot_size_;
    std::memset(tx_slot, 0, slot_size_);
    __sync_synchronize();
    session_->tx_flags_host()[slot] = 0;
  }

  void write_frame(std::uint32_t slot, const event &e, std::uint64_t tag) {
    std::uint8_t *rx_slot = session_->rx_data_host() + slot * slot_size_;
    const std::size_t bp_bytes = wire::bit_packed_bytes(e.num_syndromes);
    const std::size_t body = sizeof(wire::EnqueueRequestPayload) + bp_bytes;

    std::memset(rx_slot, 0, sizeof(cudaq::realtime::RPCHeader) + body);

    auto *header = reinterpret_cast<cudaq::realtime::RPCHeader *>(rx_slot);
    header->magic = cudaq::realtime::RPC_MAGIC_REQUEST;
    header->function_id = wire::kEnqueueSyndromesFunctionId;
    header->arg_len = static_cast<std::uint32_t>(body);
    header->request_id = static_cast<std::uint32_t>(tag);
    header->ptp_timestamp = 0;

    auto *p = reinterpret_cast<wire::EnqueueRequestPayload *>(
        rx_slot + sizeof(cudaq::realtime::RPCHeader));
    p->decoder_id = static_cast<std::int64_t>(kWireDecoderId);
    p->counter = static_cast<std::int64_t>(tag);
    p->syndrome_mapping_id = 0;
    p->num_syndromes = static_cast<std::int64_t>(e.num_syndromes);

    // Source is one bit per byte; the wire wants them packed LSB-first.
    std::uint8_t *bits = reinterpret_cast<std::uint8_t *>(p) +
                         sizeof(wire::EnqueueRequestPayload);
    for (std::uint64_t i = 0; i < e.num_syndromes; ++i)
      if (e.syndrome_data[i] & 0x1u)
        bits[i / 8] |= static_cast<std::uint8_t>(1u << (i % 8));

    __sync_synchronize();
    // Address-as-flag publish, matching rpc_producer::write_and_signal.
    session_->rx_flags_host()[slot] =
        reinterpret_cast<std::uint64_t>(session_->rx_data_dev() +
                                        slot * slot_size_);
  }

  // ---- decoder + session state ----
  std::size_t syndrome_size_ = 0;
  std::uint64_t num_observables_ = 0;
  std::vector<std::uint8_t> corrections_;
  std::uint64_t corrections_read_ = 0;
  std::vector<std::uint8_t> corrections_log_;
  std::vector<std::unique_ptr<cudaq::qec::decoder>> decoders_;
  std::unique_ptr<cudaq::qec::realtime::qec_realtime_session> session_;

  // ---- ring state ----
  std::uint32_t num_slots_ = 0;
  std::size_t slot_size_ = 0;
  std::uint64_t sent_ = 0;
  std::uint64_t synchronous_ops_ = 0;
  std::uint64_t stalls_ = 0;
  std::uint64_t rpc_errors_ = 0;

  // ---- read outcome counters ----
  std::uint32_t last_not_ready_retries_ = 0;
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
