"""REFERENCE solution: scattering by the actual row of inclusions.

WHY THIS MODULE EXISTS  (debt D1 — the most serious point)
----------------------------------------------------------
Figures 9, 10 and 11 of the paper compare the homogenized model to what the
code called "the real problem". It was not the real problem:
`compute_R_T_real` replaced the row of inclusions with an equivalent
HOMOGENEOUS LAYER (modulus by Reuss average, density by Voigt average), then
applied the thin-layer reflection formula. That model does not see the
microstructure: it is insensitive to the fine geometry of the inclusion.

Comparing the homogenized model to that "reference" amounts to comparing one
approximation to another approximation. The validation section of the paper
therefore validated nothing.

This module solves the TRUE problem.

FORMULATION
-----------
Anti-plane shear wave, normal incidence, harmonic regime e^{-iωt}.
Over one period (periodicity in y2), we solve the heterogeneous Helmholtz
equation

    div( M(y, p) ∇U ) + ω² ρ(y) U = 0,        p = -iω

with M and ρ constant per phase. Boundary conditions:

  * periodicity in y2;
  * at y1 = ±X, radiation conditions. For kh < 2π, only the Rayleigh order 0
    propagates; higher orders decay as e^{-2π|y1|} and are negligible as soon
    as X ≳ 2 (e^{-4π} ≈ 3·10⁻⁶). Hence the impedance conditions exact at
    order 0:

        y1 = +X (outgoing)  :  ∂₁U = +i k_m U
        y1 = −X (incoming)  :  ∂₁U − i k_m U = −2 i k_m e^{...}   (incident wave)

    where k_m = ω √(ρ_m / M_m(p)) is the complex wavenumber of the matrix.

  R and T are then extracted from the averages of U on the two edges.

VALIDATION (see tests/test_scattering.py)
-----------------------------------------
  * no contrast (inclusion = matrix): R = 0 and |T| = 1 to within 10⁻¹⁰;
  * full layer (φ = 1): R and T coincide with the exact analytical formula
    for the homogeneous layer, to within 10⁻⁵;
  * energy conservation |R|² + |T|² = 1 in the absence of viscosity.
"""

from __future__ import annotations

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import splu

from .cell_fem import _GAUSS, CellMesh

_CACHE: dict = {}


class ScatteringMesh(CellMesh):
    """Reuses the cell geometry; adds mass and radiating boundaries."""

    def _matrices_masse_bord(self):
        """Mass (per phase), Bloch coupling G, and boundary matrices at y1 = ±X."""
        if hasattr(self, "_sc"):
            return self._sc

        ne = len(self.elems)
        Me = np.zeros((ne, 4, 4))
        Ge = np.zeros((ne, 4, 4))     # G_ab = ∫ (∂N_a/∂y₂) · N_b   (Bloch)
        for xi, eta in _GAUSS:
            N = 0.25 * np.array([(1 - xi) * (1 - eta), (1 + xi) * (1 - eta),
                                 (1 + xi) * (1 + eta), (1 - xi) * (1 + eta)])
            dN = np.array([[-(1 - eta), -(1 - xi)], [(1 - eta), -(1 + xi)],
                           [(1 + eta), (1 + xi)], [-(1 + eta), (1 - xi)]]) * 0.25
            J = np.einsum("ak,eaj->ekj", dN, self.elem_xy)
            detJ = J[:, 0, 0] * J[:, 1, 1] - J[:, 0, 1] * J[:, 1, 0]
            Jinv = np.zeros_like(J)
            Jinv[:, 0, 0] = J[:, 1, 1] / detJ
            Jinv[:, 1, 1] = J[:, 0, 0] / detJ
            Jinv[:, 0, 1] = -J[:, 0, 1] / detJ
            Jinv[:, 1, 0] = -J[:, 1, 0] / detJ
            gradN = np.einsum("ak,ekj->eaj", dN, Jinv)     # (ne, 4, 2)
            Me += np.einsum("a,b,e->eab", N, N, detJ)
            Ge += np.einsum("ea,b,e->eab", gradN[:, :, 1], N, detJ)

        rows = np.repeat(self.elems, 4, axis=1).ravel()
        cols = np.tile(self.elems, (1, 4)).ravel()
        inc = self.in_inclusion

        def _M(masque):
            v = (Me * masque[:, None, None]).ravel()
            return coo_matrix((v, (rows, cols)),
                              shape=(self.n_nodes, self.n_nodes)).tocsr()

        def _G(masque):
            v = (Ge * masque[:, None, None]).ravel()
            return coo_matrix((v, (rows, cols)),
                              shape=(self.n_nodes, self.n_nodes)).tocsr()

        # --- boundary matrices (1D mass on the edges y1 = ±X) -----------------
        y2e = np.append(self.y2, 1.0)
        bords = {}
        for nom, i in (("gauche", 0), ("droite", self.n1 - 1)):
            r_, c_, v_ = [], [], []
            for j in range(self.n2):
                dy = y2e[j + 1] - y2e[j]
                na, nb = self.node(i, j), self.node(i, j + 1)
                # 1D edge mass matrix: [[1/3, 1/6], [1/6, 1/3]] * dy
                for (u, w, val) in ((na, na, dy / 3), (nb, nb, dy / 3),
                                    (na, nb, dy / 6), (nb, na, dy / 6)):
                    r_.append(u); c_.append(w); v_.append(val)
            bords[nom] = coo_matrix((v_, (r_, c_)),
                                    shape=(self.n_nodes, self.n_nodes)).tocsr()

        self._sc = {"Mm": _M((~inc).astype(float)), "Mi": _M(inc.astype(float)),
                    "Gm": _G((~inc).astype(float)), "Gi": _G(inc.astype(float)),
                    **bords}
        return self._sc

    def resoudre(self, omega, M_m, M_i, rho_m, rho_i, theta=0.0):
        """Return (R, T) for angular frequency omega and incidence angle theta.

        OBLIQUE INCIDENCE — quasi-periodic Bloch conditions
        ---------------------------------------------------
        We set U(y) = e^{i k₂ y₂} · V(y) with V PERIODIC. The equation becomes

            div_D (M D V) + ω² ρ V = 0,      D = ∇ + i k₂ e₂,

        whose weak form, with periodic test functions, reads

            K_Bloch = K + i k₂ (Gᵀ − G) + k₂² Mass_M ,
            G_ab = ∫ M (∂N_a/∂y₂) N_b .

        In the ELASTIC case (M real), i k₂ (Gᵀ − G) is Hermitian and the
        operator is self-adjoint. In the VISCOELASTIC case, M = μ − iωβ is
        COMPLEX: Gᵀ is not the conjugate adjoint G*, the operator is neither
        self-adjoint nor normal, and invertibility rests on rotated coercivity
        (Re[e^{-i·argp} M] > 0). Solving for V remains worthwhile: the mesh
        stays strictly periodic.

        Radiation: only the Rayleigh order 0 propagates as long as kh < 2π.
        Orders n ≠ 0 have k₁ₙ = √(k² − (k₂ + 2πn)²), so |k₁ₙ| ≈ 2π: they are
        evanescent but NOT zero at the truncated boundary. The impedance
        condition is therefore exact on mode 0 only (with k₁ = √(k² − k₂²)),
        and the truncation error, exponentially small, is measured (bloc3).
        """
        self._precalcul()
        S = self._matrices_masse_bord()
        P = self._pre

        # stiffness + mass (M and ρ per phase). RAW matrices: the diffraction
        # problem has no gauge to fix (cf. cell_fem._precalcul).
        K = (M_m * P["Km_brut"] + M_i * P["Ki_brut"]).astype(complex)
        Mass_rho = (rho_m * S["Mm"] + rho_i * S["Mi"]).astype(complex)

        # Wavenumbers. Branch choice: the wave must DECAY as it propagates
        # (dissipative medium), hence Im(k) ≥ 0.
        k_m = omega * np.sqrt(rho_m / M_m)
        if k_m.imag < 0:
            k_m = -k_m
        k2 = k_m * np.sin(theta)             # Snell invariant: imposed
        k1 = np.sqrt(k_m ** 2 - k2 ** 2)
        if k1.imag < 0:
            k1 = -k1

        A = K - omega ** 2 * Mass_rho

        if abs(theta) > 1e-14:               # Bloch coupling
            G = (M_m * S["Gm"] + M_i * S["Gi"]).astype(complex)
            Mass_M = (M_m * S["Mm"] + M_i * S["Mi"]).astype(complex)
            A = A + 1j * k2 * (G.T - G) + k2 ** 2 * Mass_M

        # Total field V = V_inc + V_scattered, with V_inc = e^{i k₁ (y₁ + X)}.
        #   at y₁ = +X (n = +e₁): outgoing, ∂₁V = i k₁ V
        #   at y₁ = −X (n = −e₁): V = 1 + R  ⇒  ∂₁V = i k₁ (2 − V)
        A = A - 1j * k1 * M_m * (S["gauche"] + S["droite"])
        b = (-2j * k1 * M_m) * (S["gauche"] @ np.ones(self.n_nodes, dtype=complex))

        v = splu(A.tocsc()).solve(b)

        # Averages in y₂: they extract the Rayleigh order 0 (higher orders,
        # periodic with zero mean, cancel out by themselves).
        w = self._poids_y2()
        i0, i1 = self.node(0, 0), self.node(self.n1 - 1, 0)
        v_gauche = np.sum(v[i0:i0 + self.n2] * w)
        v_droite = np.sum(v[i1:i1 + self.n2] * w)

        # R and T are measured at y₁ = ∓X: we refer them back to the interface
        # plane (y₁ = 0). The same factor e^{−2ik₁X} applies to both.
        # In a VISCOUS matrix k₁ is complex: without this shift, |R| would
        # depend on X (the reflected wave attenuates on its way back to the
        # boundary) — one would mistake it for a truncation error.
        phase = np.exp(-2j * k1 * self.L)
        R = (v_gauche - 1.0) * phase
        T = v_droite * phase

        # total field U(y) = e^{i k₂ y₂} · V(y), referred to the plane y₁ = 0
        # like R and T (we divide by the amplitude of the incident wave at
        # y₁ = 0)
        self._last_field = {
            "V": v, "k1": k1, "k2": k2,
            "normalisation": np.exp(1j * k1 * self.L),
        }
        return R, T

    def field(self, n_periodes: int = 2):
        """Total field U(y₁, y₂) from the last solve, on a grid.

        Returns (Y1, Y2, U) — U reconstructed over `n_periodes` periods in y₂,
        with U = e^{i k₂ y₂} · V(y), normalized like R and T (incident wave of
        amplitude 1 at y₁ = 0).

        Used to plot the fields, and to measure the error of the homogenized
        model in H¹ norm — the measure used by Marigo et al. (2017), eq. (64),
        which distinguishes the NEAR field (evanescent, which the interface
        encapsulates without reproducing it) from the FAR field.
        """
        d = getattr(self, "_last_field", None)
        if d is None:
            raise RuntimeError("call resoudre() first")

        V = d["V"].reshape(self.n1, self.n2)
        y2 = self.y2

        # V is PERIODIC: we copy it as is onto each period. All the
        # quasi-periodicity is carried by the factor e^{i k₂ y₂} below.
        # (Applying it to the blocks AS WELL would count it twice — and that
        # is exactly the bug that produced horizontal stripes.)
        Y2 = np.concatenate([y2 + k for k in range(n_periodes)])
        Vtot = np.concatenate([V] * n_periodes, axis=1)

        Y1g, Y2g = np.meshgrid(self.y1, Y2, indexing="ij")
        U = Vtot * np.exp(1j * d["k2"] * Y2g) / d["normalisation"]
        return Y1g, Y2g, U


def R_T_reference(omega, params, theta=0.0, X=3.0, raffinement=2):
    """R and T of the REAL problem (row of inclusions), arbitrary incidence."""
    from .physics import inclusion_modulus, matrix_modulus

    cle = (params.e_over_h, params.phi, X, raffinement)
    if cle not in _CACHE:
        _CACHE[cle] = ScatteringMesh(
            params.e_over_h, params.phi, X,
            n1_inc=12 * raffinement, n1_mat=20 * raffinement,
            n2_inc=12 * raffinement, n2_mat=10 * raffinement,
        )
    p = -1j * omega
    return _CACHE[cle].resoudre(
        omega,
        M_m=complex(matrix_modulus(p, params)),
        M_i=complex(inclusion_modulus(p, params)),
        rho_m=1.0, rho_i=params.rho_ratio,
        theta=theta,
    )
