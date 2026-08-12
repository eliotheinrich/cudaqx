# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# Write the artifacts run_server.py replays: a color-code memory experiment
# staged as an Ising predecoder in front of Chromobius.
#
#   playback.txt      timed reset / enqueue / get_corrections records
#   trt_decoder.yaml  trt_decoder(onnx) with global_decoder=chromobius
#   dem.txt           Stim DEM -- Chromobius is DEM-native and cannot use H
#   predecoder.onnx   exported from the released HuggingFace safetensors
#   truth.npy         observable flips, for scoring
#
# Two things that fail SILENTLY if changed:
#   * gidney_style_noise=True -- the checkpoint is trained on that noise
#     structure; a hand-built MemoryCircuit still "works" but costs most of the
#     predecoder's benefit.
#   * the ONNX is re-exported every run.  A stale one decodes at chance.
#
# Env: ISING_DECODING (repo clone), ISING_MODEL (safetensors checkpoint)

import os, sys, warnings
from types import SimpleNamespace

ISING = os.environ.get("ISING_DECODING", "/workspaces/Ising-Decoding")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.join(ISING, "code"), os.path.join(HERE, "..")]

import numpy as np, torch, cudaq_qec as qec
from qec.color_code.color_code import ColorCode
from qec.color_code.detector_input import ColorDetectorInputTransform
from qec.color_code.reference_superdense_noise import build_color_memory_circuit
from benchmarks.export_detector_input_trtexec import DetectorInputColorEval, DetectorInputModel
from evaluation.logical_error_rate_color import PreDecoderColorEvalModule, _build_color_code_parity_maps
from export.safetensors_utils import load_safetensors
from playback_from_circuit import write_playback_from_circuit

D, ROUNDS, P, SHOTS, SEED = 13, 13, 4e-3, 200, 42
WEIGHTS = os.environ.get("ISING_MODEL", "/workspaces/models/"
                         "ising_decoder_color_code_1_fast_r13_v1.0.400_fp16.safetensors")
MEAS = {"M", "MZ", "MX", "MY", "MR", "MRX", "MRY", "MRZ"}
out = lambda n: os.path.join(HERE, n)

# The circuit the checkpoint was trained on.
circuit = build_color_memory_circuit(
    distance=D, n_rounds=ROUNDS, basis="Z", p_error=P, noise_model_family="legacy",
    noise_instruction_semantics="current", gidney_style_noise=True,
    schedule="nearest-neighbor", add_boundary_detectors=True).stim_circuit

dem_text = str(circuit.detector_error_model(decompose_errors=False))
dem = qec.dem_from_stim_text(dem_text)
H = np.array(dem.detector_error_matrix, np.uint8)
O = np.array(dem.observables_flips_matrix, np.uint8)

# m2d maps measurement -> detector: the wire carries raw MEASUREMENTS and the
# decoder applies D itself.  obs_support is the observable's data-qubit support
# (the Ising benchmark exporter's `[::2]` placeholder yields a useless pre_L).
data_start = int(circuit.num_measurements) - ColorCode(D).num_data
m2d, support, seen = [], [], 0
for name, targets, _ in circuit.flattened_operations():
    if name in MEAS:
        seen += sum(isinstance(t, int) for t in targets)
    elif name == "DETECTOR":
        m2d.append(sorted(seen + int(t[1]) for t in targets if t[0] == "rec"))
    elif name == "OBSERVABLE_INCLUDE":
        support += [seen + int(t[1]) - data_start for t in targets
                    if t[0] == "rec" and seen + int(t[1]) >= data_start]

# predecoder.onnx: dets[N] -> [pre_L, residual_dets][1+N]
model = load_safetensors(WEIGHTS, device="cpu")[0].float().eval()  # release is fp16
xform = ColorDetectorInputTransform(distance=D, rounds=ROUNDS, basis="Z")
maps = _build_color_code_parity_maps(D)
obs_support = torch.zeros(int(maps["num_data"]))
obs_support[support] = 1.0
cfg = SimpleNamespace(code="color", distance=D, n_rounds=ROUNDS, enable_fp16=False,
    model=SimpleNamespace(version="predecoder_memory_v1", dropout_p=0.01, activation="gelu",
                          num_filters=[256] * 5 + [4], kernel_size=[3] * 6,
                          input_channels=4, out_channels=4),
    test=SimpleNamespace(th_data=0.0, th_syn=0.0, sampling_mode="threshold",
                         temperature=1.0, temperature_data=1.0, temperature_syn=1.0))
pipeline = DetectorInputColorEval(
    DetectorInputModel(xform, model),
    PreDecoderColorEvalModule(model, cfg, maps, basis="Z", obs_support=obs_support,
                              num_boundary_dets=int(xform.num_stabs),
                              enable_delta_s2_correction=False, enable_z_ff=True)).eval()
# dynamo=False on purpose: the torch.export path bakes the batch dim into a
# Concat and silently yields a model that fails for batch > 1.  The tracer then
# warns that `if T > 1` is frozen, which is what we want -- the ONNX is per-(d,T).
with warnings.catch_warnings():
    warnings.simplefilter("ignore", DeprecationWarning)
    warnings.simplefilter("ignore", torch.jit.TracerWarning)
    torch.onnx.export(pipeline, torch.zeros(1, xform.detector_width), out("predecoder.onnx"),
                      opset_version=17, input_names=["dets"], output_names=["L_and_residual_dets"],
                      dynamic_axes={"dets": {0: "batch"}, "L_and_residual_dets": {0: "batch"}},
                      do_constant_folding=True, dynamo=False)

# trt_decoder splits its output as [pre_L, residual]; the width must equal
# num_observables + global_decoder.syndrome_size.  stim_dem_path is what lets
# Chromobius -- which needs a DEM string, not H -- be the global decoder.
dc = qec.decoder_config()
dc.id, dc.type = 0, "trt_decoder"
dc.block_size, dc.syndrome_size = H.shape[1], H.shape[0]
dc.H_sparse, dc.O_sparse = qec.pcm_to_sparse_vec(H), qec.pcm_to_sparse_vec(O)
dc.D_sparse, dc.stim_dem_path = qec.d_sparse(m2d), out("dem.txt")
dc.decoder_custom_args = {"onnx_load_path": out("predecoder.onnx"), "batch_size": 1,
                          "use_cuda_graph": True, "global_decoder": "chromobius",
                          "global_decoder_params": {}}
mdc = qec.multi_decoder_config()
mdc.decoders = [dc]
open(out("trt_decoder.yaml"), "w").write(mdc.to_yaml_str())
open(out("dem.txt"), "w").write(dem_text)

write_playback_from_circuit(circuit, out("playback.txt"), shots=SHOTS, seed=SEED)
np.save(out("truth.npy"), circuit.compile_detector_sampler(seed=SEED).sample(
    SHOTS, append_observables=True).astype(np.uint8)[:, -1])

print("d=%d rounds=%d p=%g %d shots | %d detectors -> onnx width %d | wrote "
      "playback.txt trt_decoder.yaml dem.txt predecoder.onnx truth.npy"
      % (D, ROUNDS, P, SHOTS, H.shape[0], 1 + xform.detector_width))
