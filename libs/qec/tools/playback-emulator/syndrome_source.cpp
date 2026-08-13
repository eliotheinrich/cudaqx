/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "syndrome_source.h"

#include "stim.h"

#include <algorithm>
#include <random>

namespace cudaq::qec::emulator {

//===----------------------------------------------------------------------===//
// static_syndrome_source
//===----------------------------------------------------------------------===//

static_syndrome_source::static_syndrome_source(
    std::vector<std::vector<std::uint8_t>> rounds)
    : rounds_(std::move(rounds)) {}

std::vector<std::uint8_t> static_syndrome_source::next_round() {
  if (cursor_ >= rounds_.size())
    return {};
  return rounds_[cursor_++];
}

std::size_t static_syndrome_source::syndrome_size() const {
  std::size_t mx = 0;
  for (const auto &r : rounds_)
    mx = std::max(mx, r.size());
  return mx;
}

//===----------------------------------------------------------------------===//
// Shared stim helpers
//===----------------------------------------------------------------------===//

namespace {

constexpr std::size_t W = stim::MAX_BITWORD_WIDTH;

/// Walk a flat circuit and extract measurement groups in record order.
/// A group is one measurement instruction; syndrome groups are MR* (reset +
/// measure), data groups are bare M*.  The last group in the circuit is always
/// treated as the data readout regardless of its gate type.
struct MeasGroup {
  std::size_t offset; ///< First measurement index in a full shot record.
  std::size_t width;  ///< Number of measurements in this instruction.
  bool is_data;       ///< True for the final readout.
};

std::vector<MeasGroup> measurement_schedule(const stim::Circuit &flat) {
  std::vector<MeasGroup> groups;
  std::size_t offset = 0;
  flat.for_each_operation([&](const stim::CircuitInstruction &inst) {
    auto flags = stim::GATE_DATA[inst.gate_type].flags;
    if (!(flags & stim::GateFlags::GATE_PRODUCES_RESULTS))
      return;
    // Skip gates that only target the measurement record (e.g. feedback ops).
    if (flags & stim::GateFlags::GATE_ONLY_TARGETS_MEASUREMENT_RECORD)
      return;
    const std::size_t width = inst.targets.size();
    if (width == 0)
      return;
    // MR* is both GATE_IS_RESET and GATE_PRODUCES_RESULTS → syndrome round.
    const bool is_syndrome = (flags & stim::GateFlags::GATE_IS_RESET) != 0;
    groups.push_back({offset, width, !is_syndrome});
    offset += width;
  });
  if (!groups.empty())
    groups.back().is_data = true; // last group is always the data readout
  return groups;
}

/// Sample `shots` measurement records from `circuit_text` and return them as a
/// [shots × measurements] array of 0/1 bytes.
std::vector<std::vector<std::uint8_t>>
sample_circuit(const std::string &circuit_text, std::size_t shots,
               std::uint64_t seed) {
  stim::Circuit circ(circuit_text);
  const std::size_t num_meas = circ.count_measurements();
  auto ref = stim::TableauSimulator<W>::reference_sample_circuit(circ);
  std::mt19937_64 rng(seed);
  // transposed=true → result[shot][measurement] layout.
  auto table =
      stim::sample_batch_measurements<W>(circ, ref, shots, rng, /*transposed=*/true);

  std::vector<std::vector<std::uint8_t>> out(shots,
                                             std::vector<std::uint8_t>(num_meas));
  for (std::size_t s = 0; s < shots; ++s)
    for (std::size_t m = 0; m < num_meas; ++m)
      out[s][m] = table[s][m] ? 1u : 0u;
  return out;
}

} // namespace

//===----------------------------------------------------------------------===//
// stim_circuit_source
//===----------------------------------------------------------------------===//

stim_circuit_source::stim_circuit_source(const std::string &circuit_text,
                                         std::size_t shots, std::uint64_t seed)
    : data_(sample_circuit(circuit_text, shots, seed)) {
  meas_count_ = data_.empty() ? 0 : data_[0].size();
}

std::vector<std::uint8_t> stim_circuit_source::next_round() {
  if (cursor_ >= data_.size())
    return {};
  return data_[cursor_++];
}

//===----------------------------------------------------------------------===//
// stim_memory_source
//===----------------------------------------------------------------------===//

stim_memory_source::stim_memory_source(const std::string &circuit_text,
                                       std::size_t shots, std::uint64_t seed)
    : circuit_text_(circuit_text), shots_(shots), seed_(seed) {
  // Parse once to build the full group schedule.
  stim::Circuit flat = stim::Circuit(circuit_text_).flattened();
  for (const auto &g : measurement_schedule(flat)) {
    if (g.is_data)
      data_groups_.push_back({g.offset, g.width});
    else
      syndrome_groups_.push_back({g.offset, g.width});
  }
  // Default: use all syndrome groups.
  active_.insert(active_.end(), syndrome_groups_.begin(),
                 syndrome_groups_.end());
  active_.insert(active_.end(), data_groups_.begin(), data_groups_.end());
  sample();
}

void stim_memory_source::set_rounds(std::size_t r) {
  const std::size_t clamped = std::min(r, syndrome_groups_.size());
  active_.clear();
  active_.insert(active_.end(), syndrome_groups_.begin(),
                 syndrome_groups_.begin() + clamped);
  active_.insert(active_.end(), data_groups_.begin(), data_groups_.end());
  // Re-sample with updated group set (data is already sampled; just reset cursors).
  shot_idx_ = 0;
  round_idx_ = 0;
}

void stim_memory_source::sample() {
  data_ = sample_circuit(circuit_text_, shots_, seed_);
  shot_idx_ = 0;
  round_idx_ = 0;
}

std::vector<std::uint8_t> stim_memory_source::next_round() {
  if (active_.empty() || shot_idx_ >= data_.size())
    return {};
  const auto &g = active_[round_idx_];
  const auto &shot = data_[shot_idx_];
  std::vector<std::uint8_t> result(shot.begin() + g.offset,
                                   shot.begin() + g.offset + g.width);
  if (++round_idx_ >= active_.size()) {
    round_idx_ = 0;
    ++shot_idx_;
  }
  return result;
}

std::size_t stim_memory_source::syndrome_size() const {
  std::size_t mx = 0;
  for (const auto &g : active_)
    mx = std::max(mx, g.width);
  return mx;
}

std::size_t stim_memory_source::enqueues_per_shot() const {
  return active_.size();
}

} // namespace cudaq::qec::emulator
