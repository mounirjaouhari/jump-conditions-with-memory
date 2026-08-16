"""Regenerates tests/golden.json (non-regression values).

To be run ONLY after a deliberate modification of the physics, and to be
recorded in CHANGELOG.md together with the justification for the change.

    python tests/regenerate_golden.py
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import fading_memory  # noqa: E402,F401  (forces the console to UTF-8)
import fading_memory.physics as ph  # noqa: E402

p_ = ph.PhysicalParams()
B_e, B_v, K_hat_B = ph.decompose_B(p_)
C_e, C_v, _ = ph.decompose_C(p_)

spectre = []
for omega in [0.05, 0.5, 1.0, 2.0]:
    p = -1j * omega + 1e-6
    B, C = ph.coefficient_B(p, p_), ph.coefficient_C(p, p_)
    spectre.append({"p_re": p.real, "p_im": p.imag,
                    "B_re": B.real, "B_im": B.imag,
                    "C_re": C.real, "C_im": C.imag})

diffusion = []
for kh in [0.01, 0.1, 0.3, 0.5]:
    R, T = ph.compute_R_T_homogenized(kh, 0.0, p_)
    Rr, Tr = ph.compute_R_T_reference(kh, 0.0, p_)
    diffusion.append({"kh": kh,
                      "abs_R_hom": abs(R), "abs_T_hom": abs(T),
                      "abs_R_ref": abs(Rr), "abs_T_ref": abs(Tr)})

K_hat, C_inf, lam = ph.noyau_memoire(p_, quoi="B")
t = np.linspace(0.0, 1.5, 20)
K_B = ph.inverse_laplace(K_hat, t, tail_C=C_inf, tail_lambda=lam,
                         omega_max=20000.0, n_quad=200000, n_echantillons=400)

def _re(z):
    return float(np.real(z))


golden = {
    "_avertissement": "Non-regression values. Any DELIBERATE modification "
                      "of the physics requires regenerating this file AND "
                      "recording it in CHANGELOG.md.",
    "params": p_.as_dict(),
    "statiques": {
        "B_0": _re(ph.coefficient_B(0.0, p_)),
        "C_0": _re(ph.coefficient_C(0.0, p_)),
        "S": _re(ph.coefficient_S_full(p_)),
    },
    "decomposition": {"B_e": _re(B_e), "B_v": _re(B_v),
                      "C_e": _re(C_e), "C_v": _re(C_v)},
    "spectre": spectre,
    "diffusion": diffusion,
    "noyau": {"t": t.tolist(), "K_B": K_B.tolist()},
}

out = Path(__file__).parent / "golden.json"
out.write_text(json.dumps(golden, indent=2), encoding="utf-8")
print(f"✓ {out} regenerated")
print(f"  B(0) = {golden['statiques']['B_0']:.10f}")
print(f"  C(0) = {golden['statiques']['C_0']:.10f}")
print(f"  S    = {golden['statiques']['S']:.10f}")
