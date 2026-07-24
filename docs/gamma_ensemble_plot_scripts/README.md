# Gamma-ensemble user-guide plot scripts

Scripts used to generate the figures in
`docs/sphinx/examples_rst/qec/nv_qldpc_ensemble_gamma_user_guide.rst`.

## Figure → plot script → data

| Report figure | Plot script | Input data (generator) |
|---|---|---|
| `model_decomposition.png` | `plot_model.py` | `report_data.npz` (`run_report.py`), `blocks_raw.txt` (`probe_blocks.py`) |
| `latency_dist_p50p99_hist.png` | `plot_latency_dist_p50p99_hist.py` | `report_data.npz` (`run_report.py`) |
| `latency_percentiles.png` | `plot_latency_percentiles.py` | `report_data.npz` (`run_report.py`) |
| `deadline_surface.png`, `deadline_bb.png` | `plot_deadline_combined.py` | `deadline_data.npz` (`run_deadline.py`) |

## Data-generation scripts

- `run_report.py` — sweeps ensemble size N ∈ {1,2,4,8} over the surface/BB DEMs,
  recording per-shot latency, iteration count, and per-iteration timing → `report_data.npz`.
- `run_deadline.py` — deadline experiment (FirstConv + NConv(5), 15k shots/config,
  `num_sets=128`) → `deadline_data.npz`. `resume_bb144.py` is a resume helper that
  appends the `[[144,12,12]]` NConv(5) subruns if a run is interrupted (not needed for a
  fresh run).
- `probe_blocks.py` — measures the decoder's `blocks_x` launch geometry via the
  env-gated instrumentation: `CUDAQX_QEC_PRINT_LAUNCH=1 python3 probe_blocks.py 2> blocks_raw.txt`.
  (`blocks_raw.txt` feeds `plot_model.py`'s companion `model_law.png`, which is not used
  in the report.)

## Notes

- The data-generation scripts require a CUDA-Q compatible GPU and the built
  `nv-qldpc-decoder` plugin (`probe_blocks.py` additionally needs the plugin built with the
  env-gated launch-param instrumentation).
- The scripts contain absolute paths to the original testing area
  (`/workspaces/qec-ensemble-test/...`) for their `.npz`/`.txt` inputs and outputs; edit
  those paths (or recreate the layout) to rerun elsewhere.
