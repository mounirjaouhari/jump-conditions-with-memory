# config/

**The single source of truth for every parameter.**

`default.yaml` holds all physical parameters (moduli ratio, density ratio,
viscosity, filling fraction, row thickness) and all numerical settings
(meshes, Laplace inversion, sweep grids), organized in one section per
computation block (`bloc0` … `bloc10`).

Rules:

- no physical or numerical number may be hard-coded in a script — if a value
  is not in this file, it does not exist;
- the SHA-256 of this file is recorded in every run manifest
  (`results/<config>/<block>/manifest.json`), so any output can be traced to
  the exact configuration that produced it;
- to run a variant, copy the file and pass it with
  `python run.py all --config config/my_variant.yaml`: outputs are then
  isolated under `results/<variant>/` and `figures/<variant>/`, and the
  reference outputs cannot be overwritten by accident.
