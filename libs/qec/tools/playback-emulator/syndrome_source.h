/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

/// @file syndrome_source.h
/// @brief Abstract syndrome source and built-in implementations.
///
/// A syndrome_source provides syndrome bits on demand, one round at a time.
/// Playback records reference one by `source_id=N`; `build_events` resolves
/// the ID against a `source_registry` and stores the pointer in `event::source`.
/// `enqueue` pulls a single round from it, `stream_until` pulls repeatedly.

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace cudaq::qec::emulator {

/// Abstract base: streams syndrome bits one round at a time.
/// Each next_round() call returns one round's bits in one-byte-per-bit layout
/// (0x00/0x01), matching the layout rpc_producer::enqueue_syndromes expects.
/// Round widths may vary (e.g. a final data-qubit round differs from syndrome
/// rounds).  Returns an empty vector when the source is exhausted.
class syndrome_source {
public:
  virtual ~syndrome_source() = default;
  virtual std::vector<std::uint8_t> next_round() = 0;
  /// Maximum bit width across all rounds, or 0 when unknown.
  virtual std::size_t syndrome_size() const = 0;
  /// Number of enqueue events emitted per shot.  Default is 1.
  virtual std::size_t enqueues_per_shot() const { return 1; }
  /// Reconfigure the source to stream `r` syndrome rounds per shot (plus any
  /// mandatory non-syndrome rounds such as a data-qubit readout).  No-op for
  /// sources that do not support runtime reconfiguration.
  virtual void set_rounds(std::size_t) {}
};

// ---------------------------------------------------------------------------
// static_syndrome_source
// ---------------------------------------------------------------------------

/// Syndrome source backed by pre-sourced bits arranged into rounds.
/// Rounds may have different widths.  Bits are one-per-byte (0x00/0x01).
class static_syndrome_source : public syndrome_source {
public:
  /// @param rounds  Each element is one round's bits (one byte per bit).
  explicit static_syndrome_source(
      std::vector<std::vector<std::uint8_t>> rounds);
  std::vector<std::uint8_t> next_round() override;
  std::size_t syndrome_size() const override;

private:
  std::vector<std::vector<std::uint8_t>> rounds_;
  std::size_t cursor_ = 0;
};

// ---------------------------------------------------------------------------
// stim_circuit_source
// ---------------------------------------------------------------------------

/// Syndrome source backed by stim circuit sampling.
///
/// Samples the circuit once at construction time.  Each next_round() call
/// returns one shot's full measurement record as a flat list of 0/1 bytes.
/// All shots have the same width; enqueues_per_shot() is always 1.
///
/// `circuit_text` must be a valid stim circuit string.
class stim_circuit_source : public syndrome_source {
public:
  stim_circuit_source(const std::string &circuit_text, std::size_t shots,
                      std::uint64_t seed = 42);
  std::vector<std::uint8_t> next_round() override;
  std::size_t syndrome_size() const override { return meas_count_; }

private:
  std::vector<std::vector<std::uint8_t>> data_; ///< [shots][measurements]
  std::size_t meas_count_ = 0;
  std::size_t cursor_ = 0;
};

// ---------------------------------------------------------------------------
// stim_memory_source
// ---------------------------------------------------------------------------

/// Syndrome source for stim memory circuits.
///
/// Streams r rounds of syndrome extraction followed by one round of data
/// qubit measurement results per shot.  r is set at construction time or
/// updated via set_rounds(); the source re-samples when r changes.
///
/// Round widths may differ between syndrome and data-qubit rounds.
/// enqueues_per_shot() returns r + (number of data-qubit groups).
class stim_memory_source : public syndrome_source {
public:
  stim_memory_source(const std::string &circuit_text, std::size_t shots,
                     std::uint64_t seed = 42);

  /// Set the number of syndrome extraction rounds to stream per shot.
  /// Triggers a re-sample of the circuit with the new round count selected
  /// from the circuit's measurement groups.
  void set_rounds(std::size_t r) override;

  std::vector<std::uint8_t> next_round() override;
  std::size_t syndrome_size() const override;
  std::size_t enqueues_per_shot() const override;

private:
  struct Group {
    std::size_t offset; ///< First measurement index in the full shot record.
    std::size_t width;  ///< Number of measurements in this group.
  };

  void sample();

  std::string circuit_text_;
  std::size_t shots_;
  std::uint64_t seed_;

  std::vector<Group> syndrome_groups_; ///< All syndrome groups in circuit order.
  std::vector<Group> data_groups_;     ///< All data groups.
  std::vector<Group> active_;          ///< Groups streamed per shot (set by set_rounds).

  std::vector<std::vector<std::uint8_t>> data_; ///< [shots][measurements]
  std::size_t shot_idx_ = 0;
  std::size_t round_idx_ = 0;
};

} // namespace cudaq::qec::emulator
