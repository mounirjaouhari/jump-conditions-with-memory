"""Export of computed values to LaTeX.

Problem solved
--------------
The results table of the first version of the manuscript displayed "18.3 %",
"2.1 %", "Prony N=5"... while no script produces these numbers: they were
written by hand. Nothing guarantees that they match the code, and nothing
will update them if a parameter changes.

Principle
---------
Every value quoted in the article is recorded by a block via
`run.value(...)`, then exported here as a LaTeX macro:

    \\FMBe   →  13.03

In the manuscript one then writes `$B^e = \\FMBe$` instead of `$B^e = 13.03$`.
The number in the article becomes impossible to desynchronize from the code.

    python run.py export
"""

from __future__ import annotations

import json
from pathlib import Path

from .config import ROOT

_DIGITS = {
    "0": "Zero", "1": "One", "2": "Two", "3": "Three", "4": "Four",
    "5": "Five", "6": "Six", "7": "Seven", "8": "Eight", "9": "Nine",
}


def _macro_name(nom: str) -> str:
    """`E_R_mem` → `\\FMERmem` (LaTeX accepts only letters in a name)."""
    out = []
    for ch in nom:
        if ch.isalpha():
            out.append(ch)
        elif ch.isdigit():
            out.append(_DIGITS[ch])
        # the other characters (_ - .) are ignored
    return "FM" + "".join(out)


def _format(valeur, fmt: str) -> str:
    """Format a value for LaTeX: decimal notation, powers of ten.

    The coefficients B, C are complex numbers whose imaginary part vanishes
    on the real axis: we quote only the real part rather than an unreadable
    "0.584 + 0i". A genuinely nonzero imaginary part is kept.
    """
    if isinstance(valeur, dict) and "re" in valeur:
        if abs(valeur["im"]) < 1e-10 * max(abs(valeur["re"]), 1.0):
            valeur = valeur["re"]
        else:
            return f"{_latex_nombre(fmt % valeur['re'])} + "\
                   f"{_latex_nombre(fmt % valeur['im'])}\\,i"
    if isinstance(valeur, (int, float)):
        return _latex_nombre(fmt % valeur)
    return str(valeur)


def _latex_nombre(txt: str) -> str:
    """`3.46e-02` -> `3.46 \\cdot 10^{-2}` (English decimal points)."""
    if "e" in txt or "E" in txt:
        mantisse, exposant = txt.lower().split("e")
        exp = int(exposant)
        if mantisse in ("1", "1.0", "1.00"):
            return f"10^{{{exp}}}"
        return f"{mantisse} \\cdot 10^{{{exp}}}"
    return txt


def export_values_tex(config_nom: str = "reference") -> Path:
    """Aggregate the values of all manifests into a LaTeX macro file."""
    results = ROOT / "results" / config_nom
    manifests = sorted(results.rglob("manifest.json"))
    if not manifests:
        raise FileNotFoundError(
            f"No manifest in {results}. First run: python run.py all"
        )

    lignes: list[str] = []
    vus: dict[str, str] = {}   # macro -> block of origin (collision detection)
    commits: set[str] = set()

    for m in manifests:
        data = json.loads(m.read_text(encoding="utf-8"))
        bloc = data["block"]
        commit = (data["git"].get("commit") or "?")[:8]
        propre = data["git"].get("clean_tree")
        commits.add(commit + ("" if propre else "+modified"))

        if not data["values"]:
            continue
        lignes.append(f"% --- {bloc} (commit {commit}) " + "-" * 40)
        for nom, info in sorted(data["values"].items()):
            macro = _macro_name(nom)
            if macro in vus:
                raise ValueError(
                    f"LaTeX macro collision \\{macro}: defined by "
                    f"'{vus[macro]}' and by '{bloc}/{nom}'. Rename one of the values."
                )
            vus[macro] = f"{bloc}/{nom}"
            val = _format(info["value"], info.get("fmt", "%.4g"))
            desc = info.get("description", "")
            commentaire = f"  % {desc}" if desc else ""
            lignes.append(f"\\newcommand{{\\{macro}}}{{{val}}}{commentaire}")
        lignes.append("")

    entete = [
        "% =============================================================",
        "% GENERATED FILE — DO NOT EDIT BY HAND",
        "% =============================================================",
        "% Produced by:  python run.py export",
        f"% Configuration: {config_nom}",
        f"% Commits that produced these values: {', '.join(sorted(commits))}",
        "%",
        "% Every numerical value quoted in the article must come from here.",
        "% In the manuscript:  \\input{generated/values.tex}  then  $B^e = \\FMBe$",
        "% =============================================================",
        "",
    ]

    out_dir = ROOT / "paper" / "generated"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "values.tex"
    out.write_text("\n".join(entete + lignes) + "\n", encoding="utf-8")
    print(f"✓ {len(vus)} macro(s) written to {out.relative_to(ROOT)}")
    return out
