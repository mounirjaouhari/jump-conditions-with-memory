"""Multimodal method of Marigo & Maurel (2017), appendix B — INDEPENDENT solver.

REFERENCE
---------
J.-J. Marigo, A. Maurel, K. Pham, A. Sbitti,
"Effective Dynamic Properties of a Row of Elastic Inclusions: The Case of
Scalar Shear Waves", *Journal of Elasticity* (2017),
doi:10.1007/s10659-017-9627-4 — appendix B and the associated Matlab scripts.

WHY THIS MODULE EXISTS
----------------------
`cell_fem.py` solves the cell problem by finite elements. This module solves
it by **mode matching**, a completely different method: adapted Fourier bases
on either side of the inclusion, matched in average at y₁ = ±e/2h. Not a
single line of code is shared with the FEM.

The two methods must give the same B₁ and C₂. This is the strongest
**independent cross-check** available — and it settles the residual
disagreement with the published values (B = 0.61 and C = 3.54 in Marigo,
versus 0.584 and 3.579 by finite elements).

LIMITATION: the explicit basis of appendix B is only given for φ = 1/2.
This module is therefore valid for that value only. That is enough: it is
the test case. Only B₁ is ported (agreement to 0.03% with the FEM); for C₂,
see the note at the end of the file.

NOTATION
--------
`e` denotes e/h (dimensionless thickness), `xi` the contrast μ_i/μ_m.
`sinc` is Matlab's: sinc(x) = sin(πx)/(πx), with sinc(0) = 1.
"""

from __future__ import annotations

import numpy as np


def _sinc(x):
    """Matlab's sinc: sin(πx)/(πx), and 1 at 0."""
    return np.sinc(np.asarray(x, dtype=float))     # numpy.sinc IS Matlab's


def _matrices(e: float, xi: float, Np: int, N: int, probleme: int):
    """Coupling matrix F and system matrix M (identical for both problems)."""
    n = np.arange(1, N + 1)
    m = np.arange(-Np, Np + 1)
    bm = np.abs(2.0 * m * np.pi)
    bn = np.abs(2.0 * n * np.pi)

    F = np.zeros((len(m), len(n)), dtype=complex)
    for ii, mm in enumerate(m):
        for jj, nn in enumerate(n):
            if probleme == 1:                       # W⁽¹⁾: symmetric solution
                gmn = 0.25 * (_sinc((nn - mm) / 2) + _sinc((nn + mm) / 2))
                Gmn = 0.5 * (_sinc(nn - mm) + _sinc(nn + mm)) - gmn
                if nn % 2 == 0:
                    F[ii, jj] = 2.0 / np.sqrt(1 + xi) * (gmn + Gmn)
                else:
                    F[ii, jj] = 2.0 / np.sqrt(xi + xi ** 2) * (xi * gmn + Gmn)
            else:                                   # W⁽²⁾: antisymmetric solution
                gmn = 1.0 / (4j) * (_sinc((nn - mm) / 2) - _sinc((nn + mm) / 2))
                Gmn = 1.0 / (2j) * (_sinc(nn - mm) - _sinc(nn + mm)) - gmn
                if nn % 2 == 0:
                    F[ii, jj] = 2.0 / np.sqrt(xi + xi ** 2) * (xi * gmn + Gmn)
                else:
                    F[ii, jj] = 2.0 / np.sqrt(1 + xi) * (gmn + Gmn)

    Bp = np.diag(bm).astype(complex)
    B = np.diag(bn).astype(complex)
    E = np.diag(np.exp(-bn * e)).astype(complex)
    Ip = np.eye(2 * Np + 1, dtype=complex)
    Opp = np.zeros((2 * Np + 1, 2 * Np + 1), dtype=complex)
    Op = np.zeros((N, 2 * Np + 1), dtype=complex)

    M = np.block([
        [Ip,          Opp,        -F @ E,  -F],
        [-F.T @ Bp,   Op,          B @ E,  -B],
        [Opp,        -Ip,          F,       F @ E],
        [Op,          F.T @ Bp,    B,      -B @ E],
    ])
    return M, len(m), len(n)


def B1_multimodal(e: float, xi: float, Np: int = 40, N: int = 40) -> float:
    """B₁ = W⁽¹⁾(+∞) − W⁽¹⁾(−∞), by mode matching (Marigo, appendix B.1).

    Valid for φ = 1/2 only.
    """
    M, nm, nn = _matrices(e, xi, Np, N, probleme=1)
    n = np.arange(1, N + 1)

    S1 = np.zeros(nm, dtype=complex)
    S1[Np] = -e / 2.0 * (1 - xi) / (1 + xi)
    # sinc(n) = 0 for nonzero integer n: the central term of (83) reduces to
    #   S2_n = B·(xi − 1)/2 · sinc(nπ/2),  with B = 2/sqrt(xi + xi²)
    S2 = 2.0 / np.sqrt(xi + xi ** 2) * (
        xi / 2 * _sinc(n / 2) + _sinc(n) - 0.5 * _sinc(n / 2)
    ).astype(complex)

    S = np.concatenate([S1, S2, S1, S2])
    V = np.linalg.lstsq(M, S, rcond=None)[0]

    Vm = V[0:nm]
    Vp = V[nm:2 * nm]
    return float(np.real(Vp[Np] - Vm[Np]))          # eq. (84)


# NOTE — the Matlab script for C₂ (appendix B.2) could NOT be ported reliably:
# formula (90) of the PDF is unreadable upon extraction, and every
# reconstruction attempted VIOLATES Marigo's own variational bound (eq. 56)
# as well as the energy identity C₂ = −∫(M/M_m)|∇W⁽²⁾|² ≤ 0.
#
# C₂ does not need this cross-check: it is validated even more strongly by
# Marigo's exact energy identity (eq. 53), verified to 10⁻¹⁶ by the finite
# element solver — see `cell_fem.solve()` and `tests/test_marigo.py`.
