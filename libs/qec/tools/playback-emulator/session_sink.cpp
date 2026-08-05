/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "session_sink.h"

#include "cudaq/qec/decoder.h"
#include "cudaq/qec/pcm_utils.h"
#include "cudaq/qec/realtime/decoder_rpc_wire_format.h"
#include "cudaq/qec/realtime/decoding_config.h"
#include "cudaq/realtime/cpu_transport/udp_wrapper.h"
#include "cudaq/realtime/daemon/dispatcher/cpu_relax.h"
#include "cudaq/realtime/daemon/dispatcher/dispatch_kernel_launch.h"

// Internal headers: the producer entry point and the session that owns the
// ring live in lib/, not include/.  The tool is in-tree, so reach them
// directly rather than widening the public surface for a prototype.
#include "cudaq/qec/realtime/qec_realtime_session.h"
#include "cudaq/qec/realtime/rpc_producer.h"

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

/// Shared setup for both inproc sinks: realize one decoder from the YAML and bring up
/// a private session (ring + dispatcher thread) around it.
class session_sink : public sink {
public:
  session_sink(const std::string &config_path, std::size_t decoder_id,
               const std::string &dem_file = "") {
    std::ifstream in(config_path);
    if (!in)
      throw std::runtime_error("rpc_sink: cannot open " + config_path);
    std::stringstream buffer;
    buffer << in.rdbuf();

    auto multi = config::multi_decoder_config::from_yaml_str(buffer.str());
    const auto match = std::find_if(
        multi.decoders.begin(), multi.decoders.end(), [&](const auto &dc) {
          return dc.id >= 0 && static_cast<std::size_t>(dc.id) == decoder_id;
        });
    if (match == multi.decoders.end())
      throw std::runtime_error("rpc_sink: " + config_path +
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
    auto params = match->decoder_custom_args_to_heterogeneous_map();
    if (match->cuda_device_id.has_value())
      params.insert("cuda_device_id", match->cuda_device_id.value());

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
    if (!dem_file.empty()) {
      std::ifstream dem_in(dem_file);
      if (!dem_in)
        throw std::runtime_error("rpc_sink: cannot open dem_file " + dem_file);
      std::stringstream dem_buf;
      dem_buf << dem_in.rdbuf();
      init = dem_buf.str();
    }

    auto decoder = cudaq::qec::get_decoder(match->type, init, params);
    if (!decoder)
      throw std::runtime_error("rpc_sink: get_decoder returned null for type " +
                               match->type);
    decoder->set_decoder_id(static_cast<std::uint32_t>(match->id));
    // A DEM-built decoder has already established its own observable mapping:
    // chromobius emits observable FLIPS directly, so its O is the identity over
    // num_observables and its H is detectors x observables.  The config's O
    // lives in error-mechanism space (473 columns here), so setting it would
    // index far past that matrix.  Leave the decoder's own mapping alone.
    if (dem_file.empty())
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
  }

  ~session_sink() override {
    if (session_)
      session_->finalize();
  }

  const std::vector<std::uint8_t> &corrections() const override {
    return corrections_log_;
  }

  std::size_t correction_width() const override { return num_observables_; }
  std::size_t syndrome_size() const override { return syndrome_size_; }

  void report() const override {
    if (!corrections_read_)
      return;
    // Format only for display, and only the newest read.
    std::string last(num_observables_, '0');
    const std::size_t base = corrections_log_.size() - num_observables_;
    for (std::size_t i = 0; i < num_observables_; ++i)
      last[i] = static_cast<char>('0' + corrections_log_[base + i]);
    std::printf("corrections       reads=%llu width=%llu last=%s\n",
                static_cast<unsigned long long>(corrections_read_),
                static_cast<unsigned long long>(num_observables_),
                last.c_str());
  }

protected:
  /// Blocking enqueue via rpc_producer.  Also used by the injector sink for
  /// its warm-up, before its own cursor discipline takes over.
  void enqueue(const event &e, std::uint64_t tag) {
    cudaq::qec::decoding::rpc_producer::enqueue_syndromes(
        *session_, kWireDecoderId, e.data, e.num_bits, tag);
  }

  /// Read corrections back.  Always the decoder's own observable count -- the
  /// only width it accepts -- and never resetting: a shot boundary that needs
  /// the accumulator cleared says so with an explicit `reset` record.
  void fetch(const event &) {
    const std::uint64_t n = num_observables_;
    cudaq::qec::decoding::rpc_producer::get_corrections(
        *session_, kWireDecoderId, corrections_.data(), n, /*reset=*/0);
    ++corrections_read_;
    for (std::uint64_t i = 0; i < n; ++i)
      corrections_log_.push_back(corrections_[i] & 0x1u);
  }

  void reset_decoder() {
    cudaq::qec::decoding::rpc_producer::reset_decoder(*session_,
                                                      kWireDecoderId);
  }

  /// Wire routing key, NOT the YAML `id`.  The two are different things: the
  /// YAML id selects which config entry to realize, while the routing key
  /// indexes the session's own decoder vector.  `qec_realtime_session::
  /// initialize()` stamps `routing_key = 0` into the function table, so a
  /// one-decoder session only ever answers key 0 -- sending the YAML id here
  /// instead gets `non-zero status (1)` back from the dispatcher.
  static constexpr std::size_t kWireDecoderId = 0;

  std::size_t syndrome_size_ = 0;
  std::uint64_t num_observables_ = 0;
  std::vector<std::uint8_t> corrections_;
  std::uint64_t corrections_read_ = 0;
  std::vector<std::uint8_t> corrections_log_;
  std::vector<std::unique_ptr<cudaq::qec::decoder>> decoders_;
  std::unique_ptr<cudaq::qec::realtime::qec_realtime_session> session_;
};

//===----------------------------------------------------------------------===//
// Blocking producer
//===----------------------------------------------------------------------===//

class inproc_rpc_sink : public session_sink {
public:
  using session_sink::session_sink;

  /// One dry-run decode during the lead-in, so plugin lazy-init and first-touch
  /// ring pages do not land on event 0.
  void warm_up() override {
    std::vector<std::uint8_t> zeros(syndrome_size_, 0);
    cudaq::qec::decoding::rpc_producer::enqueue_syndromes(
        *session_, kWireDecoderId, zeros.data(), zeros.size(), /*tag=*/0);
    reset_decoder();
  }

  void send(const event &e, std::uint64_t tag) override {
    // Every one of these blocks: enqueue and reset wait for their ACK, and
    // get_corrections waits for its result body.  That latency sits
    // directly in the schedule; when it exceeds the inter-arrival gap the
    // runner reports an overrun rather than hiding it.
    switch (e.op) {
    case operation::enqueue:
      enqueue(e, tag);
      return;
    case operation::get_corrections:
      fetch(e);
      return;
    case operation::reset:
      reset_decoder();
      return;
    }
  }

  const char *name() const override { return "inproc_rpc"; }
};

//===----------------------------------------------------------------------===//
// Streaming producer
//===----------------------------------------------------------------------===//

/// Writes `enqueue_syndromes` frames straight into the session's RX ring and
/// returns without waiting for the ACK.
///
/// Wire format is byte-identical to `rpc_producer::enqueue_syndromes` (24-byte
/// RPCHeader, then a 32-byte EnqueueRequestPayload, then the bit-packed
/// syndrome bits, LSB-first).  What differs is the slot discipline: the cursor
/// advances every call, and a slot is reclaimed only when it comes back around
/// for reuse.  So the ring runs at depth `num_slots` instead of depth 1 and the
/// producer stalls only when the dispatcher is genuinely a full lap behind.
class ring_buffer_injector_sink : public session_sink {
public:
  ring_buffer_injector_sink(const std::string &config_path,
                            std::size_t decoder_id,
                            const std::string &dem_file = "")
      : session_sink(config_path, decoder_id, dem_file) {
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
    // Drain before the base class finalizes the session, so the dispatcher is
    // not torn down with requests still in flight.  Ignore the return: a
    // destructor must not throw, and a stuck dispatcher at teardown is already
    // being reported by whatever threw on the way out.
    for (std::uint32_t s = 0; s < num_slots_; ++s)
      (void)reclaim(s, kReclaimTimeoutMs);
  }

  void warm_up() override {
    std::vector<std::uint8_t> zeros(syndrome_size_, 0);
    cudaq::qec::decoding::rpc_producer::enqueue_syndromes(
        *session_, kWireDecoderId, zeros.data(), zeros.size(), /*tag=*/0);
    reset_decoder();
  }

  void send(const event &e, std::uint64_t tag) override {
    switch (e.op) {
    case operation::enqueue:
      stream_enqueue(e, tag);
      return;
    case operation::get_corrections:
      // A read must BLOCK for its result, but it must not SLEEP for it.  We
      // publish the request and watch the TX doorbell exactly as reclaim()
      // already watches it, so the wait costs the ring turnaround (~2 us)
      // instead of rpc_producer's 200 us poll granularity.
      native_round_trip(e, /*is_read=*/true);
      return;
    case operation::reset:
      native_round_trip(e, /*is_read=*/false);
      return;
    }
  }

  const char *name() const override { return "ring_buffer_injector"; }

  void report() const override {
    session_sink::report();
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

  /// Run one blocking rpc_producer call in the middle of an injected run.  Its
  /// acquire_slot waits for `rx == 0 && tx == 0` at the cursor, so the slot it
  /// is about to take must be reclaimed first -- an answered-but-unreclaimed
  /// slot would otherwise spin it out to its 5 s timeout.  rpc_producer
  /// advances the shared cursor itself.
  template <typename Fn> void synchronous(Fn &&fn) {
    require_slot(cursor());
    fn();
    ++synchronous_ops_;
  }

  void require_slot(std::uint32_t slot) {
    if (!reclaim(slot, kReclaimTimeoutMs))
      throw std::runtime_error(
          "ring_buffer_injector_sink: timed out reclaiming ring slot " +
          std::to_string(slot) + " after " +
          std::to_string(kReclaimTimeoutMs) +
          " ms; the dispatcher has stopped servicing the ring");
  }

  /// Bound on the credit wait, mirroring rpc_producer's kAcquireSlotTimeoutMs.
  /// An unbounded spin here would turn a wedged dispatcher into a wedged
  /// emulator with no diagnostic.
  static constexpr int kReclaimTimeoutMs = 5000;

  /// Wait until `slot` is reusable, reclaiming it if the dispatcher has already
  /// answered.  A slot is free once the dispatcher has consumed the request
  /// (rx_flags cleared by the host loop) and we have consumed the response.
  /// Returns false if `timeout_ms` elapses first.
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
      // Ring is a full lap ahead of the dispatcher: this is the credit-based
      // backpressure, and it is the only place the streaming producer blocks.
      stalled = true;
      CUDAQ_REALTIME_CPU_RELAX();
    }
    if (stalled)
      ++stalls_;
    return true;
  }

  /// Build the request in place in the ring slot -- no intermediate buffer, no
  /// allocation on the timing thread.
  /// Publish a request into the cursor slot and spin until its response lands,
  /// then consume it.  Everything here is machinery this sink already owns:
  /// require_slot() to claim the slot, write_frame()'s address-as-flag publish
  /// to hand it over, and reclaim()'s doorbell spin to wait.  The ONLY thing it
  /// does differently from rpc_producer is watch instead of usleep.
  ///
  /// `is_read` selects get_corrections (17-byte payload, packed result body)
  /// over reset_decoder (8-byte payload, empty ACK).
  void native_round_trip(const event &, bool is_read) {
    const std::uint64_t n = num_observables_;

    const std::uint32_t slot = cursor();
    require_slot(slot);

    // --- build the request in place, same envelope as write_frame ---
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

    // --- watch for the reply instead of sleeping through it ---
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
    if (response->status != 0) {
      release_response(slot);
      throw std::runtime_error(
          "ring_buffer_injector_sink: non-zero status (" +
          std::to_string(response->status) + ")");
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
    const std::size_t bp_bytes = wire::bit_packed_bytes(e.num_bits);
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
    p->num_syndromes = static_cast<std::int64_t>(e.num_bits);

    // Source is one bit per byte; the wire wants them packed LSB-first.
    std::uint8_t *bits = reinterpret_cast<std::uint8_t *>(p) +
                         sizeof(wire::EnqueueRequestPayload);
    for (std::uint64_t i = 0; i < e.num_bits; ++i)
      if (e.data[i] & 0x1u)
        bits[i / 8] |= static_cast<std::uint8_t>(1u << (i % 8));

    __sync_synchronize();
    // Address-as-flag publish, matching rpc_producer::write_and_signal.
    session_->rx_flags_host()[slot] =
        reinterpret_cast<std::uint64_t>(session_->rx_data_dev() +
                                        slot * slot_size_);
  }

  std::uint32_t num_slots_ = 0;
  std::size_t slot_size_ = 0;
  std::uint64_t sent_ = 0;
  std::uint64_t synchronous_ops_ = 0;
  std::uint64_t stalls_ = 0;
  std::uint64_t rpc_errors_ = 0;
};


//===----------------------------------------------------------------------===//
// Transport producer (remote decoding_server)
//===----------------------------------------------------------------------===//

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
        throw std::runtime_error("remote_sink: timed out claiming slot " +
                                 std::to_string(slot));
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

  /// Spin until the reply for `slot` arrives, then hand back its RX slot.
  const std::uint8_t *await_reply(std::uint32_t slot) {
    const auto deadline =
        std::chrono::steady_clock::now() + std::chrono::milliseconds(kTimeoutMs);
    while (rx_flags_[slot] == 0) {
      if (std::chrono::steady_clock::now() >= deadline)
        throw std::runtime_error("remote_sink: timed out awaiting a response "
                                 "in slot " + std::to_string(slot));
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
    p.num_syndromes = static_cast<std::int64_t>(e.num_bits);
    // Streamed: publish and move on.  The reply is drained by claim() when this
    // slot comes round again -- the credit discipline the in-process injector
    // uses, one lap of the ring deep.
    publish(wire::kEnqueueSyndromesFunctionId, &p, sizeof(p), e.data,
            e.num_bits, tag);
    ++sent_;
  }

  void read(const event &, std::uint64_t tag) {
    const std::uint64_t n = observables_;
    wire::GetCorrectionsRequestPayload p{};
    p.decoder_id = static_cast<std::int64_t>(decoder_id_);
    p.return_size = static_cast<std::int64_t>(n);
    p.reset = 0;

    const std::uint32_t slot =
        publish(wire::kGetCorrectionsFunctionId, &p, sizeof(p), nullptr, 0, tag);
    const std::uint8_t *rx_slot = await_reply(slot);
    const auto *response =
        reinterpret_cast<const cudaq::realtime::RPCResponse *>(rx_slot);
    if (response->status != 0)
      throw std::runtime_error("remote_sink: get_corrections returned status " +
                               std::to_string(response->status));

    const std::uint8_t *packed = rx_slot + sizeof(cudaq::realtime::RPCResponse);
    for (std::uint64_t i = 0; i < n; ++i)
      corrections_log_.push_back((packed[i / 8] >> (i % 8)) & 0x1u);
    ++reads_;
  }

  void reset(std::uint64_t tag) {
    wire::ResetRequestPayload p{};
    p.decoder_id = static_cast<std::int64_t>(decoder_id_);
    const std::uint32_t slot =
        publish(wire::kResetDecoderFunctionId, &p, sizeof(p), nullptr, 0, tag);
    (void)await_reply(slot);
  }

  static constexpr int kTimeoutMs = 5000;

  std::unique_ptr<remote_transport> transport_;
  std::uint8_t *tx_data_ = nullptr, *rx_data_ = nullptr;
  std::uint64_t *tx_flags_ = nullptr, *rx_flags_ = nullptr;
  std::size_t decoder_id_;
  std::uint64_t observables_;
  std::uint32_t num_slots_ = 0, slot_size_ = 0, cursor_ = 0;
  std::uint64_t sent_ = 0, reads_ = 0, stalls_ = 0;
  std::vector<std::uint8_t> corrections_log_;
};


} // namespace

std::unique_ptr<sink> make_inproc_rpc_sink(const std::string &config_path,
                                         std::size_t decoder_id,
                                         const std::string &dem_file) {
  return std::make_unique<inproc_rpc_sink>(config_path, decoder_id, dem_file);
}

std::unique_ptr<sink> make_ring_buffer_injector_sink(const std::string &config_path,
                                          std::size_t decoder_id,
                                          const std::string &dem_file) {
  return std::make_unique<ring_buffer_injector_sink>(config_path, decoder_id,
                                                     dem_file);
}

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
