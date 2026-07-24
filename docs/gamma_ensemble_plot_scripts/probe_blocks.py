# Trigger the launch-param solver for each code x N so the instrumented
# relay_launch_params.h prints [relay-launch] N=.. syndrome_size=.. blocks_x=..
# capacity=..  Run with:
#   CUDAQX_QEC_PRINT_LAUNCH=1 python3 probe_blocks.py 2> blocks_raw.txt
# Small codes (d=3, d=5) put N*D/C below the saturation knee.
import glob
import numpy as np
import stim
import cudaq
import cudaq_qec as qec

BB = "/workspaces/tesseract-decoder/testdata/bivariatebicyclecodes"


def s2c(dem):
    H = np.zeros((dem.num_detectors, dem.num_errors), np.uint8)
    er = np.zeros(dem.num_errors); e = 0
    for ins in dem.flattened():
        if ins.type == "error":
            er[e] = ins.args_copy()[0]
            for t in ins.targets_copy():
                if t.is_relative_detector_id():
                    H[t.val, e] = 1
            e += 1
    cd = qec.DetectorErrorModel(); cd.detector_error_matrix = H; cd.error_rates = er
    cd.observables_flips_matrix = np.zeros((dem.num_observables, dem.num_errors), np.uint8)
    cd.error_ids = np.arange(dem.num_errors, dtype=np.int32); return cd


def bb(tag, p):
    f = [x for x in glob.glob(BB + "/*") if tag in x and "_Z" in x and f"p={p}," in x][0]
    c = stim.Circuit.from_file(f); cd = s2c(c.detector_error_model())
    cd.canonicalize_for_rounds(1); return cd.detector_error_matrix, cd.error_rates


def surf(d, r, p=0.008):
    cudaq.set_target("stim"); cudaq.set_random_seed(7)
    code = qec.get_code("surface_code", distance=d)
    noise = cudaq.NoiseModel(); noise.add_all_qubit_channel("x", cudaq.Depolarization2(p), 1)
    dem = qec.z_dem_from_memory_circuit(code, qec.operation.prep0, r, noise)
    return dem.detector_error_matrix, np.array(dem.error_rates)


codes = [surf(3, 3), surf(5, 5), surf(9, 9), surf(9, 27),
         bb("72,12,6", "0.003"), bb("144,12,12", "0.001")]
g0s = [0.3, 0.3, 0.3, 0.3, 0.125, 0.125]
for (H, er), g0 in zip(codes, g0s):
    for N in (1, 2, 4, 8):
        d = qec.get_decoder("nv-qldpc-decoder", H, max_iterations=50, error_rate_vec=er,
                            use_sparsity=True, bp_method=3, composition=1, gamma0=g0,
                            gamma_dist=[-0.24, 0.66], clip_value=200.0, repeatable=True,
                            srelay_config={"pre_iter": 1, "num_sets": 64,
                                           "stopping_criterion": "FirstConv"},
                            gamma_ensemble_size=N)
        d.decode(np.zeros(H.shape[0], np.uint8))
