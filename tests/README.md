# tests/

**101 automated tests** — run with `python run.py test` (or `pytest`).

| File | What it guards |
|---|---|
| `test_cell_fem.py` | Cell solver invariants: truncation independence, mesh convergence, reciprocity, full-layer closed form. |
| `test_scattering.py` | Reference solver invariants: energy conservation, zero-contrast invisibility, analytic full layer, mesh convergence gate (block 10). |
| `test_marigo.py` | External validation against the published values of the elastic case (coefficients, variational bounds, energy identities, modal cross-check). |
| `test_dette_scientifique.py` | One gate per numerical correction that was expensive to find — so no known defect can silently return (module ratio, C validation, memory regime, Herglotz identity, Stieltjes positivity…). |
| `test_limites_analytiques.py` | Closed-form limits (static, large p, full layer) and the Laplace-inversion conventions. |
| `test_enveloppe_secteur.py` | Envelope formulas and the sector condition on the sampled grid. |
| `test_non_regression.py` | Golden values (`golden.json`): static coefficients, kernel samples, R/T points at tight tolerances. |
| `regenerate_golden.py` | Deliberately regenerates `golden.json` — run it only after a justified physics change. |

`conftest.py` (at the repository root) puts `src/` on `sys.path`; no
installation step is needed to run the tests.
