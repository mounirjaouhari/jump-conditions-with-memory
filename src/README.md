# src/

**The physics and the solvers** — everything that computes lives here, in the
Python package `fading_memory/`. The scripts in `scripts/` only read the
configuration, call this package, plot and record.

| Module | Role |
|---|---|
| `physics.py` | The model: complex moduli, contrast r(p), interface coefficients B, C, S, memory kernels, Prony fitting, homogenized scattering R, T; solver-verification identities; Herglotz/Stieltjes helpers. |
| `cell_fem.py` | Q1 finite elements on the unit cell (mesh aligned with the inclusion, geometric far-field grading); cell coefficients for any complex contrast; sesquilinear Herglotz check. |
| `scattering_fem.py` | Reference solution of the *actual* scattering problem (heterogeneous Helmholtz, quasi-periodic Bloch conditions, mode-zero radiation). |
| `marigo_multimodal.py` | Independent modal-matching cross-check of the cell coefficient. |
| `provenance.py` | `RunRecorder` (manifest writer) and `verify` (the audit command). |
| `config.py` | YAML loading, parameter object, output-directory routing. |
| `plotting.py` | The single publication style of every figure. |
| `export.py` | Aggregates all manifests into `paper/generated/values.tex`. |

See [`docs/ARCHITECTURE.md`](../docs/ARCHITECTURE.md) for the full map and
the design rules.
