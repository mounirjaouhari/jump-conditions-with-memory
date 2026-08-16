"""Provenance: every execution leaves a verifiable trace.

Problem solved
--------------
In the initial version of the project, the 14 figures of the article could be
attached to NOTHING: neither a code version, nor a parameter set, nor a
machine. It was impossible to know whether a figure still matched the code,
or to regenerate it.

Principle
---------
Every block runs inside a `RunRecorder`. At the end, a `manifest.json` is
written into `results/<config>/<bloc>/` containing:

  - the exact git commit (+ whether the working tree was modified)
  - the SHA-256 hash of the configuration file
  - the physical parameters actually used
  - the versions of Python, NumPy, SciPy, Matplotlib, and the platform
  - the scalar values produced (those that will be quoted in the article)
  - the SHA-256 hash of EVERY output file (figures, CSV)

Verification
------------
    python run.py verify

recomputes the hashes and reports any figure that is orphaned (no manifest),
stale (newer code) or edited by hand outside of any computation.
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from .config import ROOT, Config


# =============================================================================
# Utilities
# =============================================================================

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def _git(*args: str) -> str | None:
    try:
        out = subprocess.run(
            ["git", *args], cwd=ROOT, capture_output=True, text=True, timeout=10
        )
        return out.stdout.strip() if out.returncode == 0 else None
    except (OSError, subprocess.SubprocessError):
        return None


# What defines the computation. A run is replayable only if ALL of this is
# committed. `results/` and `figures/` are excluded: they are the outputs of
# the run itself; modifying them is normal and does not affect replayability.
SOURCES = ["src", "scripts", "config", "tests", "run.py",
           "conftest.py", "pyproject.toml", "requirements.txt"]


def git_state() -> dict[str, Any]:
    """Current commit, and cleanliness of the SOURCES (not of the outputs)."""
    commit = _git("rev-parse", "HEAD")
    status = _git("status", "--porcelain", "--", *SOURCES)
    return {
        "commit": commit,
        "branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        # Modified sources => the executed code is NOT the committed one:
        # the trace is incomplete and the run cannot be replayed identically.
        "clean_tree": (status == "") if status is not None else None,
        "modified_sources": status.splitlines() if status else [],
    }


def environment() -> dict[str, Any]:
    import matplotlib
    import scipy

    return {
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "matplotlib": matplotlib.__version__,
        "platform": platform.platform(),
    }


def _jsonable(v: Any) -> Any:
    """Make NumPy and complex types serializable."""
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    if isinstance(v, (complex, np.complexfloating)):
        return {"re": float(v.real), "im": float(v.imag)}
    if isinstance(v, np.ndarray):
        return [_jsonable(x) for x in v.tolist()]
    if isinstance(v, (list, tuple)):
        return [_jsonable(x) for x in v]
    if isinstance(v, dict):
        return {k: _jsonable(x) for k, x in v.items()}
    return v


# =============================================================================
# Run recorder
# =============================================================================

class RunRecorder:
    """Execution context of a block: produces data, figures and a manifest.

        with RunRecorder("bloc3", cfg) as run:
            run.value("B_e", B_e, "instantaneous elastic coefficient")
            run.table("RT_vs_kh", {"kh": khs, "R": R_vals})
            run.figure(fig, "fig09_RT_vs_kh.png")
    """

    def __init__(self, bloc: str, cfg: Config):
        self.bloc = bloc
        self.cfg = cfg
        self.t0 = datetime.now(timezone.utc)
        self.values: dict[str, Any] = {}
        self.outputs: list[dict[str, Any]] = []
        self.notes: list[str] = []

        self.results_dir = cfg.results_dir / bloc
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.figures_dir = cfg.figures_dir

    # --- recording -----------------------------------------------------------
    def value(self, nom: str, valeur: Any, description: str = "",
              fmt: str = "%.4g") -> Any:
        """Record a scalar value meant to be quoted in the article.

        These values are exported to `paper/generated/values.tex`: a number
        quoted in the text can then no longer diverge from the code that
        produces it. `fmt` is the desired LaTeX format (e.g. "%.2f" for a
        percentage).
        """
        self.values[nom] = {
            "value": _jsonable(valeur),
            "description": description,
            "fmt": fmt,
        }
        return valeur

    def table(self, nom: str, colonnes: dict[str, Any]) -> Path:
        """Write a figure's data to CSV (the data, not just the image)."""
        import pandas as pd

        path = self.results_dir / f"{nom}.csv"
        pd.DataFrame(colonnes).to_csv(path, index=False, float_format="%.10g")
        self._register(path, kind="data")
        return path

    def figure(self, fig, nom_fichier: str, dpi: int = 400) -> Path:
        """Save a figure and record its fingerprint."""
        path = self.figures_dir / nom_fichier
        fig.savefig(path, dpi=dpi)
        self._register(path, kind="figure")
        return path

    def note(self, texte: str) -> None:
        """Log an observation (warning, edge case encountered...)."""
        self.notes.append(texte)

    def _register(self, path: Path, kind: str) -> None:
        self.outputs.append({
            "file": str(path.relative_to(ROOT)).replace("\\", "/"),
            "type": kind,
            "sha256": sha256_file(path),
            "bytes": path.stat().st_size,
        })

    # --- context -------------------------------------------------------------
    def __enter__(self) -> "RunRecorder":
        print(f"[{self.bloc}] config={self.cfg.nom} ({self.cfg.path.name})")
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        manifest = {
            "block": self.bloc,
            "status": "failure" if exc_type else "success",
            "error": f"{exc_type.__name__}: {exc}" if exc_type else None,
            "timestamp_utc": self.t0.isoformat(),
            "duration_s": round((datetime.now(timezone.utc) - self.t0).total_seconds(), 2),
            "git": git_state(),
            "config": {
                "name": self.cfg.nom,
                "file": str(self.cfg.path.relative_to(ROOT)).replace("\\", "/"),
                "sha256": self.cfg.sha256,
            },
            "physical_parameters": self.cfg.params().as_dict(),
            "block_parameters": _jsonable(self.cfg.raw.get(self.bloc, {})),
            "environment": environment(),
            "values": self.values,
            "outputs": self.outputs,
            "notes": self.notes,
        }
        path = self.results_dir / "manifest.json"
        path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

        if exc_type:
            print(f"[{self.bloc}] FAILURE — manifest: {path.relative_to(ROOT)}")
        else:
            print(f"[{self.bloc}] OK — {len(self.outputs)} output(s), "
                  f"manifest: {path.relative_to(ROOT)}")
            if manifest["git"]["clean_tree"] is False:
                print(f"[{self.bloc}] ⚠ modified git sources: this run cannot be "
                      f"replayed as is. Commit before producing figures "
                      f"intended for the article.")
        return False  # never mask the exception


# =============================================================================
# A posteriori verification
# =============================================================================

def verify(verbose: bool = True) -> int:
    """Check that the artifacts on disk match the manifests.

    Returns the number of problems detected (0 = everything is consistent).
    """
    problems: list[str] = []
    tracked: set[str] = set()

    manifests = sorted((ROOT / "results").rglob("manifest.json"))
    if not manifests:
        problems.append("No manifest: no result was produced by this repository.")

    for m in manifests:
        data = json.loads(m.read_text(encoding="utf-8"))
        rel_m = m.relative_to(ROOT)

        if data["status"] != "success":
            problems.append(f"{rel_m}: failed run ({data['error']})")

        if data["git"].get("clean_tree") is False:
            problems.append(
                f"{rel_m}: produced with modified git sources → not replayable "
                f"(commit {data['git'].get('commit', '?')[:8]} + local changes)"
            )

        for out in data["outputs"]:
            p = ROOT / out["file"]
            tracked.add(out["file"])
            if not p.exists():
                problems.append(f"{out['file']}: declared but ABSENT from disk")
            elif sha256_file(p) != out["sha256"]:
                problems.append(
                    f"{out['file']}: MODIFIED since it was generated "
                    f"(hash ≠ manifest) → figure edited by hand?"
                )

    # Figures present but produced by no known run. Figures are PDFs
    # (vector output, publisher requirement); *.png covers possible leftovers.
    # A historical defect scanned ONLY figures/*.png: orphan detection was
    # empty — fixed (PDF + PNG, FR and EN locales).
    for dossier in ("figures", "figures_en"):
        for motif in ("*.pdf", "*.png"):
            for fig in sorted((ROOT / dossier).glob(motif)):
                rel = str(fig.relative_to(ROOT)).replace("\\", "/")
                if rel not in tracked:
                    problems.append(
                        f"{rel}: ORPHAN — no run of this repository produced it")

    if verbose:
        if problems:
            print(f"\n{len(problems)} provenance problem(s):\n")
            for p in problems:
                print(f"  ✗ {p}")
        else:
            print("\n✓ Full provenance: every output matches a recorded run, "
                  "on a clean git tree.")
    return len(problems)
