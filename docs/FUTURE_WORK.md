# aeras — Future Work & Known Limitations

This document tracks scheduled enhancements (Phases K, L) and acknowledged limitations of the v1 release.

Last updated: 2026-05-30

---

## 🔧 Scheduled Enhancements (In Progress / Imminent)

### Phase K — Multi-Pollutant via Per-Pollutant Physics Parameters

**Status:** Scheduled (next)
**Estimated time:** 6 hours (2h code + 4h training)

**Motivation:** Run I attempted joint multi-pollutant training with shared `Dx`/`Dy`/`λ_dep` across all 4 pollutants (PM2.5, NO2, O3, SO2). This caused multi-task interference — PM2.5 MAE degraded by ~21% on Random test (44.4 → 53.89), and all pollutants showed negative R² (model predicted near-mean instead of learning variance).

**Root cause:** Different pollutants have genuinely different physical timescales:

| Pollutant | Realistic `λ_dep` interpretation |
|---|---|
| PM2.5 | Slow deposition (days), gravitational settling |
| NO2 | Photolysis during day (~hours), chemically reactive |
| O3 | Photochemical formation AND destruction (~hours) |
| SO2 | Slow but reactive with water (rain washout) |

Forcing one set of `Dx`/`Dy`/`λ_dep` across all 4 species is physically incorrect.

**Fix design:**
- Convert `log_Dx`, `log_Dy`, `log_lambda_dep` from scalar `nn.Parameter`s to vectors of size `OUTPUT_DIM`
- In `compute_pde_residual`, use `model.Dx[i]`, `model.Dy[i]`, `model.lambda_dep[i]` for the i-th pollutant
- Each pollutant learns its own physics from data
- For `OUTPUT_DIM=1` (single-pollutant), behavior is unchanged (vector of length 1 is functionally a scalar)

**Expected outcome:**
- PM2.5 MAE returns to Run F levels (~45 Random, ~75 Diwali) or beats them
- NO2/O3/SO2 each learn their own physics — values comparable against atmospheric chemistry literature
- 4-pollutant dashboard becomes viable

**Files affected:**
- `src/models/pinn.py` — parameter vectorization + per-pollutant PDE residual
- `src/evaluation/evaluate_forward.py` — print per-pollutant physics params
- `api/routes/physics.py` — return per-pollutant params
- `src/config.py` — bump `CHECKPOINT_PREFIX = "aeras_v9_perpoll_params"`
- Documentation in `docs/RESULTS.md` should compare learned per-pollutant `λ_dep` against published values

---

### Phase L — Inverse Model Rework (Source Localization)

**Status:** Scheduled (after K)
**Estimated time:** 8 hours (4h redesign + 4h training)

**Motivation:** Run G attempted joint forward+inverse training. The SourceNet learned a trivial near-zero output (1e-9 magnitude everywhere) because:

1. **Aggressive L1 sparsity penalty** (`W_SOURCE_SPARSITY=0.001`) pushed S → 0
2. **Forward model's high λ_dep (0.234) absorbed unmodeled physics** instead of forcing SourceNet to learn it
3. **Single Diwali example in training** provided insufficient signal against regularization pressure

**Fix design:**

1. **Reduce sparsity penalty:** `W_SOURCE_SPARSITY = 0.0001` (10× weaker)
2. **Add minimum-activation regularizer:** Penalize `1 / (mean(S) + ε)` to actively discourage zero collapse
3. **Pre-train SourceNet on FIRMS data:** Use stubble-burning fire density as a proxy initial source field before joint training. SourceNet gets a sensible starting point near real emission patterns
4. **Optional λ_dep clipping:** Constrain `lambda_dep < 0.05` to force SourceNet (not deposition) to absorb residuals

**Expected outcome:**
- SourceNet produces non-trivial output with spatial hotspots near known industrial zones (Anand Vihar, Mundka, IGI Airport)
- Diwali MAE drops to ~50-60 range as emission spikes are properly modeled
- `/sources` API endpoint becomes meaningful (currently returns near-zero everywhere)

**Files affected:**
- `src/models/source_net.py` — add `min_activation_loss` method
- `src/models/loss.py` — wire new regularizer into total loss
- `src/training/trainer.py` — optional pre-training step on FIRMS
- `src/config.py` — bump `CHECKPOINT_PREFIX = "aeras_v10_inverse_v2"`

---

### Path C v2 — OpenAQ Yearly Retrain Pipeline

**Status:** Scheduled (during/after engineering phases)
**Estimated time:** 4-5 hours code + 1 day data download

**Motivation:** Training data ends 2020-07-01 due to the rohanrao Kaggle dataset's age. For a 2026 deployment, the model effectively extrapolates 5+ years beyond its training distribution. OpenAQ has CPCB-equivalent data through current dates with hourly resolution and free API access.

**Fix design:**
- New `src/data/download_openaq.py` using OpenAQ API
- Bulk download 2018-2025 for Delhi NCR stations
- Preprocess identically to current pipeline (FIRMS, ERA5, etc. integration)
- `scripts/retrain_yearly.sh` documented script that pulls latest data, re-preprocesses, retrains, evaluates
- Manually run yearly — not automated to a cron

**Expected outcome:**
- 5 more Diwali events in training (2020, 2021, 2022, 2023, 2024) instead of 1
- COVID lockdown period captured (natural experiment in emissions reduction)
- Removes the `t_norm > 1` extrapolation issue
- Demonstrates production engineering mindset (model as living system, not one-off artifact)

**Files affected:**
- `src/data/download_openaq.py` (new)
- `scripts/retrain_yearly.sh` (new)
- `docs/DATA_REFRESH.md` (new)

---

## 🔭 Genuine Future Work (Out of Scope for v1)

### Quantile Regression for Calibrated Uncertainty

Currently using MC Dropout for uncertainty. Quantile regression would give:
- Distribution-free calibrated intervals
- Better-suited for health alerts ("with 90% probability, PM2.5 < X")
- Avoids MC Dropout's known under-confidence in tails

**Cost:** ~6 hours. Could replace MC Dropout in v2.

### MODIS AOD Integration

`pyhdf` install failed on Windows during initial setup. AOD currently zero-filled in splits. Miniconda + `pyhdf` would unlock this. Adds direct satellite measurement signal.

**Cost:** ~3 hours.

### EDGAR Emission Inventory as Inverse Prior

EDGAR provides gridded PM2.5/NO2/SO2 emission estimates at 0.1° resolution. Using these as a soft prior for SourceNet (Phase L) would further improve source localization quality.

**Cost:** ~6 hours (download + format + integrate).

### 3D Vertical PINN

Current model is 2D (lat × lon × time). Real atmospheric chemistry is 3D — vertical mixing matters especially for winter inversions. A 3D PINN would directly model BLH dynamics instead of treating it as an input feature.

**Cost:** Major rework. ~30-40 hours. Outside portfolio scope.

### Coupled Chemistry (NO2 + VOCs → O3)

Real photochemistry: NO2 + sunlight + VOCs → ozone formation. Adding cross-pollutant chemistry terms would dramatically improve O3 modeling. Requires reformulating the PDE system.

**Cost:** Significant. ~40+ hours.

### Operational Deployment

Current setup runs locally. A production deployment would need:
- Containerized API (Docker)
- Cloud hosting (Render free tier or self-hosted)
- Automated retraining (extending Path C v2)
- Monitoring + alerting (Prometheus + Grafana)
- Rate limiting / authentication
- TLS / domain

**Cost:** ~20 hours. Mostly packaging after the API hardening in Phase E is done.

### Mixture-of-Experts for Extreme Events

Train a separate "extreme event" model on Diwali + winter data, a "normal day" model on everything else, and a router that picks which to use at inference. Could push Diwali MAE to 40-50 range.

**Cost:** ~12 hours.

### Adversarial Robustness Testing

Stress-test the model with input perturbations (sensor noise, missing data, adversarial features) to characterize failure modes.

**Cost:** ~6 hours.

---

## ⚠️ Acknowledged Limitations (v1 Honest Disclaimers)

These are NOT scheduled work — they are honest limitations of the v1 model that should be disclaimed prominently in the README, dashboard, and Colab notebook.

1. **Training period: 2018-01-01 to 2020-07-01.** Predictions for dates outside this range use cyclic year-mapping to equivalent training days. Yearly retraining via Path C v2 addresses this.

2. **Single Diwali (2018) in training data.** Limits extreme-event generalization. OpenAQ migration (Path C v2) adds 4-5 more Diwalis.

3. **Forward model cannot represent emission spikes.** Diwali fireworks are an instantaneous source event; forward PINN with `S=0` assumes no emissions. Phase L (inverse model rework) addresses this.

4. **Limited spatial coverage.** 37 CPCB stations in Delhi NCR — sparse compared to a dense observation grid. Predictions in areas far from any station carry higher uncertainty.

5. **No real-time data ingestion.** Live AQICN integration in API is "fetch on user request" — not continuous polling. Users see data refreshed at most every hour.

6. **No multi-pollutant in v1 production model.** Run F is PM2.5-only. Phase K addresses this.

7. **Shared physics across pollutants (Run I).** Run I attempted multi-pollutant with shared `Dx`/`Dy`/`λ_dep` and produced negative R² across all pollutants. Documented negative result. Phase K addresses with per-pollutant parameters.

8. **Inverse model returned trivial null source (Run G).** Documented negative result. Phase L addresses with relaxed sparsity + FIRMS pre-training.

---

## 📊 Progress Tracking

| Phase | Status | Date | Outcome |
|---|---|---|---|
| K — Multi-pollutant per-pollutant params | Scheduled | — | TBD |
| L — Inverse model rework | Scheduled | — | TBD |
| Path C v2 — OpenAQ pipeline | Scheduled | — | TBD |

This document is updated as Phases K, L, and Path C v2 progress.
