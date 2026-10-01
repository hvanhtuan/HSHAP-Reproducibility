# H-SHAP reproducibility package (v1.0.0)

Archived on Zenodo: https://doi.org/10.5281/zenodo.23086206 · Source repository: https://github.com/hvanhtuan/HSHAP-Reproducibility

Code, locked results, and replay files for the paper *The hierarchical Owen value (H-SHAP) as an explanation-auditing mechanism for gradient-boosted models on hierarchical categorical representations in property insurance ratemaking* (manuscript, v1.2).

H-SHAP is the two-tier Owen value applied to the explanation game of an EAGB model (LightGBM, Tweedie) on public NFIP flood-insurance data. This package accompanies, and depends on the model specification of, the HCE/EAGB reproducibility package (https://github.com/hvanhtuan/HCE-EAGB-Reproducibility).

## Contents

| Path | Content |
|---|---|
| `code/hshap.py` | Shapley and Owen values from a coalition table; sampled Owen estimator (Algorithm 2) |
| `code/kb45_hshap.py` | Paper Tables 2–3: efficiency, fidelity, cost, frequency–severity decomposition, H2a stability |
| `code/kb4c_proxy.py` | Paper Table 4: H2b proxy screening (10 repetitions) |
| `code/make_synthetic_owen.py` | Paper Table 6 and Proposition 3 checks (synthetic games; runs in seconds) |
| `code/nfip_lib.py`, `hce.py`, `metrics.py`, `prep_nfip.py`, `dl_*` | Shared library, encoder, metrics, data download and risk-cell construction |
| `code/kb2_h1.py` | Regenerates `results/kb2_test_enc.pkl` and `results/models/` from the locked pre-registration |
| `results/prereg_H1.json` | Locked pre-registration, English translation used by the code (local timestamp 2026-09-18; no independent timestamp) |
| `results/prereg_H1.original-vi.json` | The original Vietnamese file that carried the lock (kept unchanged for audit; content identical apart from language and model labels) |
| `results/kb2_test_enc.pkl`, `results/models/` | Hierarchical-credibility encodings and the eight locked LightGBM models used by the H-SHAP scripts |
| `results/kb45_hshap.json`, `kb4c_proxy.json`, `synthetic_owen_results.json` | Reference outputs behind the paper's tables |
| `data/raw_manifest.json` | Names, sizes and SHA-256 of the OpenFEMA raw files (raw data NOT included) |
| `SHA256SUMS.txt` | Checksums of every file in this package |

## Quick check (seconds, no data needed)

```bash
pip install numpy==2.4.4
cd code && python make_synthetic_owen.py
```

The printed table must match paper Table 6 (e.g. kappa = 2: D = 0.0318, max|Delta_g|/max Phi_g = 0.0392; Algorithm 2 mean max error 0.102 / 0.035 / 0.015 at R = 100 / 1000 / 5000).

## Full replay of the NFIP results

1. `pip install -r code/requirements.txt` (Python 3.11).
2. Download the OpenFEMA files named in `data/raw_manifest.json` into `data/raw/` (`code/dl_all.sh`), then `cd code && python prep_nfip.py` to build `data/nfip_cells.parquet` (2,453,910 risk cells). A fresh download may not be byte-identical to the September 2026 snapshot.
3. `cd code && python kb45_hshap.py` (about 70 min; writes `results/kb45_hshap.json`) and `python kb4c_proxy.py` (writes `results/kb4c_proxy.json`).

The included `results/kb2_test_enc.pkl` and `results/models/` let steps 3 run without retraining; `kb2_h1.py test` regenerates them. Scripts use relative paths and must be run from `code/`.

## Notes and limitations

- The pre-registration file carries a local lock time only; the order "locked before results" rests on the authors' statement.
- Raw and claim-/policy-level NFIP files are not distributed; data keep their original terms.
- The audited EAGB model does not outperform the GLM baseline on the 2021–2023 test window (see the EAGB paper); H-SHAP is evaluated as an explanation mechanism, not as evidence of predictive value.
- Reference outputs were produced on the authors' machine; small numerical differences are expected on other hardware, except for the synthetic experiment, which is deterministic.

## Model labels (English)

The original study used Vietnamese labels for some models. In this package the labels were translated consistently in the code, in `prereg_H1.json` and in `results/models/test_index.json`; numerical values are unchanged. Mapping (original Vietnamese label -> label used here) is given in `LABELS.md`.

## License and citation

MIT License (see `LICENSE`). Cite via `CITATION.cff` (DOI: 10.5281/zenodo.23086206).
