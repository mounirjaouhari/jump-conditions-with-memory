"""Configuration loading.

A single YAML file describes the WHOLE computation. The scripts contain
no hard-coded parameter: they receive a `Config` object.

The hash of the configuration file is propagated all the way into the run
manifests: every figure can be attached to the exact configuration that
produced it.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from .physics import PhysicalParams

# Repository root = two levels above this file (src/fading_memory/)
ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "config" / "default.yaml"


@dataclass
class Config:
    """Complete configuration of a computation, as read from disk."""

    raw: dict[str, Any]
    path: Path
    sha256: str

    # --- access --------------------------------------------------------------
    @property
    def nom(self) -> str:
        """Name of the configuration — used to name the output directories."""
        return self.raw.get("meta", {}).get("nom", self.path.stem)

    def bloc(self, name: str) -> dict[str, Any]:
        """Parameters of one block, e.g. `cfg.bloc('bloc3')['n_kh']`."""
        if name not in self.raw:
            raise KeyError(
                f"Section '{name}' missing from {self.path.name}. "
                f"Available sections: {sorted(self.raw)}"
            )
        return self.raw[name]

    def laplace_kwargs(self) -> dict[str, Any]:
        """Laplace-inversion parameters — GLOBAL, never per block.

        The kernel K_B(t) of figure 7 and the one used for the time-domain
        propagation of figure 12 must be the SAME object. In the legacy code,
        blocks 2 and 5 each inverted it with their own settings
        (N_omega = 2000 versus 1500): two different kernels within a single
        article. These settings are therefore now global.
        """
        n = self.raw["numerique"]
        return {
            "omega_max": n["laplace_omega_max"],
            "n_quad": n["laplace_n_quad"],
            "n_echantillons": n["laplace_n_echantillons"],
            "eps": n["laplace_epsilon"],
        }

    @property
    def convention_laplace(self) -> str:
        return self.raw["numerique"].get("laplace_convention", "standard")

    def params(self, **overrides: Any) -> PhysicalParams:
        """Build the `PhysicalParams` from the config.

        `overrides` allows parameter sweeps (e.g. `cfg.params(visc_ratio=0.01)`)
        without ever duplicating the other values.
        """
        phy = dict(self.raw["physique"])
        num = self.raw["numerique"]
        phy["N_modes"] = num["n_modes"]
        phy["p_large"] = float(num["p_large"])
        phy["solveur"] = num.get("solveur", "fem")
        phy["fem_L"] = float(num.get("fem_L", 6.0))
        phy["fem_raffinement"] = int(num.get("fem_raffinement", 2))
        phy.update(overrides)
        return PhysicalParams(**phy)

    # --- output directories --------------------------------------------------
    @property
    def results_dir(self) -> Path:
        d = ROOT / "results" / self.nom
        d.mkdir(parents=True, exist_ok=True)
        return d

    @property
    def figures_dir(self) -> Path:
        # The reference configuration writes into figures/ (the path the
        # article expects); any variant writes into a dedicated subfolder,
        # which makes it impossible to overwrite the article figures by
        # accident.
        d = ROOT / "figures" if self.nom == "reference"             else ROOT / "figures" / self.nom
        d.mkdir(parents=True, exist_ok=True)
        return d


def load_config(path: str | Path | None = None) -> Config:
    """Load a YAML configuration (by default `config/default.yaml`)."""
    p = Path(path) if path else DEFAULT_CONFIG
    if not p.is_absolute():
        p = (ROOT / p).resolve()
    if not p.exists():
        raise FileNotFoundError(f"Configuration not found: {p}")

    data = p.read_bytes()
    sha = hashlib.sha256(data).hexdigest()
    raw = yaml.safe_load(data.decode("utf-8"))

    for section in ("physique", "numerique"):
        if section not in raw:
            raise ValueError(f"Mandatory section '{section}' missing from {p.name}")

    return Config(raw=raw, path=p, sha256=sha)
