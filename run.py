#!/usr/bin/env python
"""SINGLE entry point of the project.

    python run.py all                  # regenerate everything (fig01 to fig21 + data)
    python run.py bloc3                # a single block
    python run.py verify               # check the provenance of the figures
    python run.py export               # computed values -> paper/generated/values.tex
    python run.py test                 # test suite
    python run.py all --config config/my_variant.yaml

Every execution writes a manifest into results/: git commit, hash of the
configuration, library versions, fingerprint of each produced file.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

BLOCS = {
    "bloc0": "bloc0_geometries",
    "bloc1": "bloc1_coefficients_cellulaires",
    "bloc2": "bloc2_noyaux_memoire",
    "bloc3": "bloc3_diffusion_harmonique",
    "bloc4": "bloc4_choix_de_a",
    "bloc5": "bloc5_propagation_temporelle",
    "bloc6": "bloc6_incidence_oblique",
    "bloc7": "bloc7_regime_memoire",
    "bloc8": "bloc8_champs",
    "bloc9": "bloc9_passivite_stieltjes",
    "bloc10": "bloc10_convergence_reference",
}


def _pages(log: Path) -> str:
    """Page count, read from the LaTeX log."""
    import re
    m = re.search(r"Output written on \S+ \((\d+) pages",
                  log.read_text(encoding="utf-8", errors="replace"))
    return m.group(1) if m else "?"


def _run_bloc(nom: str, cfg) -> None:
    import importlib

    module = importlib.import_module(BLOCS[nom])
    t0 = time.time()
    module.main(cfg)
    print(f"[{nom}] finished in {time.time() - t0:.1f} s\n")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Article \"Jump conditions with memory\" — computation driver",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "commande",
        choices=[*BLOCS, "all", "verify", "export", "test"],
    )
    parser.add_argument("--config", default=None,
                        help="YAML file (default: config/default.yaml)")
    args = parser.parse_args()

    from fading_memory import load_config

    # --- commands without computation -----------------------------------------
    if args.commande == "verify":
        from fading_memory.provenance import verify
        return 1 if verify() else 0

    if args.commande == "export":
        from fading_memory.export import export_values_tex
        cfg = load_config(args.config)
        export_values_tex(cfg.nom)
        return 0

    if args.commande == "test":
        return subprocess.call([sys.executable, "-m", "pytest", "-q"], cwd=ROOT)

    # --- computations ----------------------------------------------------------
    cfg = load_config(args.config)
    blocs = list(BLOCS) if args.commande == "all" else [args.commande]

    t0 = time.time()
    for nom in blocs:
        _run_bloc(nom, cfg)

    if args.commande == "all":
        from fading_memory.export import export_values_tex
        from fading_memory.provenance import verify

        export_values_tex(cfg.nom)
        print(f"\nTotal: {time.time() - t0:.1f} s")
        print("\n--- Provenance check ---")
        verify()
    return 0


if __name__ == "__main__":
    sys.exit(main())
