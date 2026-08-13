# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# Derive a playback file from a Stim circuit, paced by the circuit's tick
# structure.
#
# Two pieces, usable separately:
#
#   measurement_schedule(circuit)
#       Walk the circuit and report one entry per measurement op: which tick it
#       lands on, whether it is an ancilla read or the terminal data readout,
#       and how wide it is.  This is the extraction on its own -- import it if
#       you only want the schedule.
#
#   write_playback_from_circuit(circuit, path, shots=...)
#       Sample the circuit with Stim and write a playback file whose tick
#       column comes from that schedule.

MEASURE_OPS = ("M", "MZ", "MX", "MY", "MR", "MRX", "MRY", "MRZ")


class MeasurementGroup:
    """One measurement instruction: where it lands and what it carries."""

    __slots__ = ("tick", "boundary", "kind", "width", "offset")

    def __init__(self, tick, boundary, kind, width, offset):
        self.tick = tick          # segment the instruction sits in
        self.boundary = boundary  # tick at which its result is visible
        self.kind = kind          # "syndrome" | "data"
        self.width = width        # measurements in this group
        self.offset = offset      # index of its first bit in the shot record

    def __repr__(self):
        return ("MeasurementGroup(tick=%d, boundary=%d, kind=%r, width=%d, "
                "offset=%d)" % (self.tick, self.boundary, self.kind,
                                self.width, self.offset))


def measurement_schedule(circuit):
    """Per-measurement-op schedule derived from the circuit's TICK structure.

    Returns (groups, total_ticks).  `groups` is in measurement-record order, so
    concatenating their widths reproduces one shot's full record.
    """
    groups, tick, offset = [], 0, 0
    for inst in circuit.flattened():
        if inst.name == "TICK":
            tick += 1
            continue
        if inst.name not in MEASURE_OPS:
            continue
        width = len(inst.targets_copy())
        if width == 0:
            raise ValueError("circuit has an empty measurement at tick %d"
                             % tick)
        # A measurement that also resets is an ancilla read; a bare measurement
        # is the destructive readout.  The final group is retagged below, which
        # is what makes a mid-circuit bare M behave sensibly.
        kind = "syndrome" if inst.name.startswith("MR") else "data"
        groups.append(MeasurementGroup(tick, tick, kind, width, offset))
        offset += width
    if not groups:
        raise ValueError("circuit contains no measurements")

    total_ticks = tick
    for g in groups:
        g.boundary = min(g.tick + 1, total_ticks) if total_ticks else 0
    groups[-1].kind = "data"
    # If the data readout shares a boundary with the preceding syndrome round
    # (both clipped to total_ticks), give it the next tick so the two enqueues
    # land at different deadlines and neither overruns the other by construction.
    if len(groups) >= 2 and groups[-1].boundary == groups[-2].boundary:
        groups[-1].boundary = total_ticks + 1
    return groups, total_ticks


def sample_measurements(circuit, shots, seed=None):
    """Raw measurement records -- NOT detectors.  One row per shot."""
    sampler = (circuit.compile_sampler(seed=seed) if seed is not None
               else circuit.compile_sampler())
    return sampler.sample(shots=shots)


def write_playback_from_circuit(circuit, path, shots=100, decoder_id=0,
                                shot_gap_ticks=None, seed=None,
                                reset_each_shot=True, read_corrections=True,
                                comment=None, decoder_deadline_ticks=1,
                                expected_corrections=None):
    """Sample `circuit` and write a playback file paced by its own ticks.

    decoder_deadline_ticks: ticks budgeted for the decoder to return a result,
        counted from the last measurement boundary.  get_corrections is placed
        at last_boundary + decoder_deadline_ticks.  Default is 1, which gives
        the minimum one-tick separation needed to avoid same-deadline collisions
        with the preceding enqueue.

    expected_corrections: optional sequence of length `shots`, each element
        being a sequence of 0/1 ints (one per observable).  When provided,
        the expected bits are appended to each get_corrections line so the
        emulator can report mismatches without an out-of-band diff.

    Returns the (groups, total_ticks) schedule that was used, so a caller can
    build a matching decoder config without re-deriving it.
    """
    if decoder_deadline_ticks < 1:
        raise ValueError("decoder_deadline_ticks must be >= 1")
    groups, total_ticks = measurement_schedule(circuit)
    record = sum(g.width for g in groups)
    read_tick = max(g.boundary for g in groups) + decoder_deadline_ticks
    if shot_gap_ticks is None:
        # Leave a shot's worth of idle between shots by default: the reset and
        # the corrections read are request/response and block, and crowding
        # them against the next shot's stream is the usual cause of overruns
        # that have nothing to do with the cadence under test.
        shot_gap_ticks = max(total_ticks, 1)

    samples = sample_measurements(circuit, shots, seed)
    span = read_tick + shot_gap_ticks

    with open(path, "w") as f:
        f.write("# GENERATED by playback_from_circuit.py -- ticks come from "
                "the circuit's own TICK structure\n")
        f.write("#   <tick> <operation> <decoder_id> [operands...]\n")
        f.write("# PLAYBACK_META decoder_id=%d\n" % decoder_id)
        f.write("# %d measurements/shot in %d groups over %d ticks: %s\n" %
                (record, len(groups), total_ticks,
                 ", ".join("%s:%d@%d" % (g.kind, g.width, g.boundary)
                           for g in groups)))
        if comment:
            f.write("# %s\n" % comment)
        for shot, row in enumerate(samples):
            base = shot * span
            if reset_each_shot:
                f.write("%-8d reset            %d\n" % (base, decoder_id))
            for g in groups:
                bits = "".join("1" if row[g.offset + i] else "0"
                               for i in range(g.width))
                f.write("%-8d enqueue          %d    %s\n" %
                        (base + g.boundary, decoder_id, bits))
            if read_corrections:
                expected_str = ""
                if expected_corrections is not None and shot < len(expected_corrections):
                    expected_str = "  " + "".join(
                        "1" if b else "0" for b in expected_corrections[shot])
                f.write("%-8d get_corrections  %d%s\n" %
                        (base + read_tick, decoder_id, expected_str))
    return groups, total_ticks


def write_source_playback(path, shots, source_id, enqueues_per_shot,
                          decoder_id=0, decoder_deadline_ticks=1,
                          shot_gap_ticks=1):
    """Write a playback file that references a syndrome source by ID.

    Each shot is laid out as:
        reset
        enqueue  source_id=N    (repeated enqueues_per_shot times)
        get_corrections

    The matching source must be registered under source_id when calling
    run_playback(..., sources={source_id: source}).

    decoder_deadline_ticks: ticks between the last enqueue and get_corrections.
    shot_gap_ticks: idle ticks between get_corrections and the next reset.
    """
    gc_tick = enqueues_per_shot + decoder_deadline_ticks
    shot_span = gc_tick + shot_gap_ticks
    with open(path, "w") as f:
        f.write("# GENERATED by write_source_playback\n")
        f.write("# PLAYBACK_META decoder_id=%d\n" % decoder_id)
        for shot in range(shots):
            base = shot * shot_span
            f.write("%-8d reset            %d\n" % (base, decoder_id))
            for r in range(enqueues_per_shot):
                f.write("%-8d enqueue          %d    source_id=%d\n" %
                        (base + r + 1, decoder_id, source_id))
            f.write("%-8d get_corrections  %d\n" % (base + gc_tick, decoder_id))