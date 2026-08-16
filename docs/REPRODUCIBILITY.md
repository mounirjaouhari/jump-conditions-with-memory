# Reproducibility

How to reproduce the results of the article, and how the repository
guarantees that what you reproduce is what the article shows.

## 1. Environment

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements-lock.txt     # exact versions (recommended)
# or: pip install -r requirements.txt    # floors only
```

Python ≥ 3.10 (developed with CPython 3.13). The exact versions used for the
published figures are recorded twice: in `requirements-lock.txt`, and in the
`environment` field of every run manifest.

## 2. Full reproduction

```bash
python run.py test      # 1. the tests pass                    (~6 min)
python run.py all       # 2. regenerate all data and figures   (~45 min)
python run.py verify    # 3. audit: everything traceable?
```

`all` runs blocks 0–10 in order, then exports the quoted values and audits
the chain. Outputs land in `figures/` (vector PDF) and
`results/reference/<block>/` (manifest + CSV).

## 3. Reproducing one figure or one number

- **One figure**: find its block in the table of
  [`ARCHITECTURE.md`](ARCHITECTURE.md) (e.g. fig11 → `bloc4`), then
  `python run.py bloc4`.
- **One quoted number**: every number in the article is a LaTeX macro
  (`\FMplancherMesure`, `\FMordreKhPetit`, …). Grep it in
  `paper/generated/values.tex`: the comment line names the block that
  produced it and the manifest records the exact run.
- **Without running anything**: every plotted curve is also a CSV file next
  to its manifest (`results/reference/<block>/*.csv`) — errors, orders and
  ratios can be recomputed from the archived data alone.

## 4. The provenance contract

Each run writes `results/<config>/<block>/manifest.json`:

```json
{
  "block": "bloc4",
  "git":  { "commit": "…", "clean_tree": true },
  "config": { "sha256": "…" },
  "physical_parameters": { "mu_ratio": 6.5, "...": "…" },
  "environment": { "python": "3.13.6", "numpy": "…" },
  "values": { "plancherReference": { "value": 2.5e-4, "...": "…" } },
  "outputs": [ { "file": "figures/fig11_….pdf", "sha256": "…" } ]
}
```

- `git.commit` + `clean_tree`: which commit of this repository produced the
  run, and whether the source tree was clean (a dirty tree is recorded as
  such — the run is then declared non-replayable);
- `config.sha256`: which parameters;
- `outputs[].sha256`: fingerprint of every produced file.

`python run.py verify` re-hashes everything and fails loudly on: a figure
with no recording run (*orphan*), a figure whose hash no longer matches its
manifest (*retouched*), or a run made on a dirty tree.

## 5. Numerical tolerances — what “reproduced” means

Floating-point results depend slightly on platform and library versions. The
repository pins what matters:

- `tests/golden.json` stores reference values (static coefficients, kernel
  samples, R/T points) compared by `pytest` at tight tolerances — the same
  gate we use for non-regression;
- exact identities are verified at solver precision and recorded as values
  (full layer ≈ 3·10⁻¹¹, reciprocity ≈ 10⁻¹⁴, Herglotz identity
  ≈ 1.1·10⁻¹⁰ — see Table S8.1 of the article's Supplemental Material);
- the accuracy floor of the reference solver is *measured* (block 10):
  2.5·10⁻⁴ relative on R at the production settings. Differences between
  your machine and the archived values that stay well below these scales are
  reproduction noise, not disagreement.

## 6. Expected runtimes (laptop, single core)

| Command | Time |
|---|---|
| `python run.py test` | ≈ 6 min |
| `python run.py all` | ≈ 45 min (finite elements dominate) |
| one block alone | 30 s – 5 min |

## 7. Relation to the article

The article's Data availability statement points here. The values quoted in
the article were exported from the manifests archived in
`results/reference/`, at the commit recorded in the header of
`paper/generated/values.tex` — a commit of this repository, which anyone can
check out and rerun.
