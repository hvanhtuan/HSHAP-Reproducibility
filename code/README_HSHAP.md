# H-SHAP code (hierarchical Owen value paper)

Run all scripts from the `code/` directory (they use relative paths such as `../results/`).

| File | Role | Output |
|---|---|---|
| `hshap.py` | Shapley/Owen values from a coalition-value table; sampled Owen estimator (Algorithm 2) | – |
| `kb45_hshap.py` | Tables 2–3: efficiency, fidelity, cost, frequency–severity decomposition, H2a stability | `results/kb45_hshap.json` |
| `kb4c_proxy.py` | Table 4: H2b proxy-variable screening (10 repetitions) | `results/kb4c_proxy.json` |
| `make_synthetic_owen.py` | Table 6: synthetic games, Proposition 3 checks, Monte Carlo | `synthetic_owen_results.json` |

Requirements: `results/prereg_H1.json`, `results/kb2_test_enc.pkl` and `results/models/` (all included in this package; `kb2_h1.py test` regenerates the last two). `kb45_hshap.py` takes about 70 minutes.
