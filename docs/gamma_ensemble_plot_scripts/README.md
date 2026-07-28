# Gamma-ensemble user-guide plot scripts

Scripts used to generate the figures in
`docs/sphinx/examples_rst/qec/nv_qldpc_ensemble_gamma_user_guide.rst`.

## Figure → plot script → data

The report uses five figures, produced by four plot scripts from two data files:

| Report figure | Plot script | Input data (generator) |
|---|---|---|
| `model_decomposition.png` | `plot_model.py` | `report_data.npz` (`run_report.py`) |
| `latency_dist_p50p99_hist.png` | `plot_latency_dist_p50p99_hist.py` | `report_data.npz` (`run_report.py`) |
| `latency_percentiles.png` | `plot_latency_percentiles.py` | `report_data.npz` (`run_report.py`) |
| `deadline_surface.png`, `deadline_bb.png` | `plot_deadline_combined.py` | `deadline_data.npz` (`run_deadline.py`) |

## Data-generation scripts

- `run_report.py` — sweeps ensemble size N ∈ {1,2,4,8} over the surface/BB DEMs,
  recording per-shot latency, iteration count, and per-iteration timing → `report_data.npz`.
- `run_deadline.py` — deadline experiment (FirstConv + NConv(5), 15k shots/config,
  `num_sets=128`) → `deadline_data.npz`.

## Running

The data-generation scripts require a CUDA-Q compatible GPU and the built
`nv-qldpc-decoder` plugin. The output/input locations and shot counts are
overridable via environment variables (defaults reproduce the original layout):

- `QEC_DATA_ROOT` — root for the `docs_report/` and `docs_deadline/` data + figures
  (default `/workspaces/qec-ensemble-test`).
- `BB_DIR` — Tesseract bivariate-bicycle code circuits
  (default `/workspaces/tesseract-decoder/testdata/bivariatebicyclecodes`).
- `SHOTS_CAP` (`run_report.py`) / `SHOTS` (`run_deadline.py`) — reduce the shot
  count for a fast smoke run.

Typical flow:

```bash
export PYTHONPATH=/usr/local/cudaq:<cudaqx-build>/python
python3 -u run_report.py
python3 -u run_deadline.py
python3 plot_model.py
python3 plot_latency_dist_p50p99_hist.py
python3 plot_latency_percentiles.py
python3 plot_deadline_combined.py
```
