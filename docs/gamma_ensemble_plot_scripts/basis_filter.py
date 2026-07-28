# Detector basis filter, copied from /workspaces/extract_bb_dem.py.
# filter_detectors_by_basis(circuit, "Z") returns a circuit with only the
# detectors sensitive to X errors (Z stabilizers) -> the Z-component DEM.
import numpy as np
import stim


def _detect_data_qubits(circuit: stim.Circuit) -> list[int]:
    """Identify data qubits as those measured exactly once (final readout only)."""
    times_measured = [0] * circuit.num_qubits
    for inst in circuit:
        if inst.name.startswith("M") and not inst.gate_args_copy():
            for t in inst.targets_copy():
                times_measured[t.qubit_value] += 1
    return [q for q, n in enumerate(times_measured) if n == 1]


def filter_detectors_by_basis(
    circuit: stim.Circuit,
    basis: str,
    qubits: list[int] | None = None,
) -> stim.Circuit:
    """Return a copy of the circuit with detectors insensitive to *basis* removed.

    basis="X" keeps detectors that fire when Z errors occur (X stabilizers).
    basis="Z" keeps detectors that fire when X errors occur (Z stabilizers).
    """
    assert basis in ("X", "Z"), f"basis must be 'X' or 'Z', got {basis!r}"
    pauli_error = "Z" if basis == "X" else "X"

    circuit = circuit.flattened()
    noiseless = circuit.without_noise()

    sampler = noiseless.compile_detector_sampler()
    ref_det, _ = sampler.sample(1, separate_observables=True)
    ref_det = ref_det[0]
    num_det = len(ref_det)

    sensitive = np.zeros(num_det, dtype=bool)
    data_qubits = list(_detect_data_qubits(noiseless)) if qubits is None else list(qubits)
    to_test = list(data_qubits)
    inst_idx = 0

    while to_test:
        for q in to_test:
            injected = stim.Circuit()
            injected += noiseless
            injected.insert(inst_idx, stim.CircuitInstruction(f"{pauli_error}_ERROR", [q], [1.0]))
            inj_det, _ = injected.compile_detector_sampler().sample(1, separate_observables=True)
            sensitive[np.where(ref_det != inj_det[0])] = True

        to_test = []
        for inst in noiseless[inst_idx:]:
            inst_idx += 1
            if inst.name.startswith("R") or inst.name.startswith("M"):
                to_test = list(data_qubits)
                break

    filtered = stim.Circuit()
    det_idx = 0
    for inst in circuit:
        if inst.name == "DETECTOR":
            if not sensitive[det_idx]:
                det_idx += 1
                continue
            det_idx += 1
        filtered.append(inst)
    return filtered
