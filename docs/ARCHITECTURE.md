# Architecture

Map of the pipeline: where the physics lives, what each block computes, and
which rules keep the chain trustworthy.

## Data flow

```
config/default.yaml ──► src/fading_memory/ ──► scripts/blocN_*.py ──► results/<config>/blocN/
      (parameters)          (the physics)        (read, call, plot,        manifest.json + CSV
        SHA-256 ↘                                 record — no physics)   ↙ SHA-256 of every output
                 └────────────────────────────────────────────────────┘
                                        │
                                   figures/
                                        │
                          python run.py export
                                        ▼
                        paper/generated/values.tex
                 (one LaTeX macro per quoted number: \FMBe → 0.584)
```

Two hard rules make this trustworthy:

- **No hard-coded numbers.** Parameters live in `config/default.yaml` only;
  quoted values flow through `run.value(...)` into generated macros.
- **No physics in scripts.** Formulas live in `src/fading_memory/`; the
  `scripts/` layer reads the configuration, calls the library, plots and
  records. A block script that computed anything would break provenance.

## Modules (`src/fading_memory/`)

| Module | Role |
|---|---|
| `physics.py` | The model itself: complex moduli, contrast r(p), interface coefficients B, C, S, memory kernels, Prony fitting, homogenized scattering R, T; solver-verification identities; Herglotz identity and Stieltjes measure helpers. |
| `cell_fem.py` | Q1 finite elements on the unit cell (mesh aligned with the inclusion, geometric far-field grading, corner-aware); cell coefficients for any complex contrast; sesquilinear Herglotz check. |
| `scattering_fem.py` | Reference solution of the *actual* problem: heterogeneous Helmholtz on one period, quasi-periodic (Bloch) conditions for oblique incidence, mode-zero radiation conditions; R and T extraction. |
| `marigo_multimodal.py` | Independent modal-matching computation of the cell coefficient (cross-check against the FEM, two unrelated discretizations). |
| `provenance.py` | `RunRecorder` (the manifest writer: git state, config hash, environment, values, output hashes) and `verify` (the audit). |
| `config.py` | YAML loading, parameter object, output-directory routing (variants write into dedicated subfolders, so reference figures cannot be overwritten by accident). |
| `plotting.py` | The single publication style for every figure (vector PDF, embedded fonts, Computer Modern math). |
| `export.py` | Aggregates all manifests into `paper/generated/values.tex` (collision-checked LaTeX macros). |

## Blocks (`scripts/`)

One script per computation block; each can be run alone
(`python run.py bloc6`).

| Block | What it establishes | Outputs |
|---|---|---|
| `bloc0_geometries` | Schematic figures of the geometry and of the homogenization step | fig01–fig04 |
| `bloc1_coefficients_cellulaires` | B(p), C(p), S(a); variational bounds; the nine solver-verification residuals | fig05, fig06 |
| `bloc2_noyaux_memoire` | Kernels K_B, K_C by Laplace inversion; amplitude/relaxation decoupling | fig07 |
| `bloc3_diffusion_harmonique` | R, T vs kh at normal incidence; local convergence orders | fig09 |
| `bloc4_choix_de_a` | Interface-thickness sweep; superconvergence of a = e; energy positivity | fig08, fig11 |
| `bloc5_propagation_temporelle` | Transient reflected signals (exact transfer-function route); Prony check | fig12, fig13 |
| `bloc6_incidence_oblique` | Validation of C against the Bloch reference (computed / omitted / wrong C) | fig15, fig16 |
| `bloc7_regime_memoire` | Memory gain as a function of ω/λ — when does memory matter? | fig17 |
| `bloc8_champs` | Field maps; near-field reading of a = e and of the role of C | fig18–fig20 |
| `bloc9_passivite_stieltjes` | Herglotz identity residual; Stieltjes density μ′ ≥ 0; reconstruction | fig14 |
| `bloc10_convergence_reference` | Accuracy of the reference solver: measured floor, mesh order, truncation sensitivity (paper S8) | fig21 |

## Design rules inherited by any contribution

1. no number in the scripts or the LaTeX — configuration and macros only;
2. no physics outside `src/fading_memory/`;
3. never fix physics silently — record the rationale and regenerate the golden values;
4. commit before producing figures (manifests record the commit);
5. the test suite stays green — each scientific-debt test guards a past fix;
6. warnings are never masked (a silenced `ComplexWarning` once hid a real bug).
