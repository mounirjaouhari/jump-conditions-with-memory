# scripts/

**One script per computation block** — the orchestration layer. Each script
reads the configuration, calls `src/fading_memory/`, plots, and records its
outputs through a `RunRecorder` (manifest + CSV + figure hashes). **No
physics here**: a block script that computed a formula itself would break
the provenance chain.

Run any block alone with `python run.py bloc4`, or all of them with
`python run.py all`.

| Script | What it establishes | Figures |
|---|---|---|
| `bloc0_geometries.py` | Schematic figures of the geometry | fig01–fig04 |
| `bloc1_coefficients_cellulaires.py` | Cell coefficients B, C, S + solver checks | fig05, fig06 |
| `bloc2_noyaux_memoire.py` | Memory kernels by Laplace inversion | fig07 |
| `bloc3_diffusion_harmonique.py` | R, T vs kh at normal incidence | fig09 |
| `bloc4_choix_de_a.py` | Interface-thickness sweep; superconvergence of a = e | fig08, fig11 |
| `bloc5_propagation_temporelle.py` | Transient signals (exact transfer-function route) | fig12, fig13 |
| `bloc6_incidence_oblique.py` | Validation of C against the Bloch reference | fig15, fig16 |
| `bloc7_regime_memoire.py` | When does memory matter? (gain vs ω/λ) | fig17 |
| `bloc8_champs.py` | Field maps; near-field role of a and C | fig18–fig20 |
| `bloc9_passivite_stieltjes.py` | Herglotz residual; Stieltjes density μ′ ≥ 0 | fig14 |
| `bloc10_convergence_reference.py` | Measured accuracy of the reference solver | fig21 |

Naming note: the block scripts and the identifiers passed to
`run.value(...)`/`run.table(...)` keep the historical (French-rooted) names —
they are code identifiers, and the LaTeX macros quoted by the article are
generated from them. Do not rename them.
