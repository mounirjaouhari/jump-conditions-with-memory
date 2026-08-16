"""Confrontation with Marigo & Maurel (2017), *J. Elasticity*, doi:10.1007/s10659-017-9627-4.

This is THE founding reference: the same problem, but with an **elastic**
matrix. Our framework — a **viscoelastic** matrix — reduces to it exactly at
p = 0, and follows from it by analytic continuation for arbitrary p, since the
cell problem depends on p only through the contrast r(p) = M_i(p)/M_m(p)
(cf. `physics.contraste`).

These tests are therefore the strongest external validation this work has:
they confront our solver with a published, independent source.
"""

import numpy as np
import pytest

import fading_memory.physics as ph
from fading_memory.cell_fem import cell_coefficients as cc
from fading_memory.marigo_multimodal import B1_multimodal
from fading_memory.scattering_fem import R_T_reference

E_H, PHI = 2.0, 0.5          # Marigo's test case: e/h = 2, phi = 1/2
XI = 6.5                     # steel / concrete: mu_i/mu_m = 78/12
RHO = 3.12                   # 7800 / 2500


def _cell(r, raf=3):
    return cc(r, E_H, PHI, L=6.0, raffinement=raf)


class TestIdentitesEnergie:
    """Marigo's EXACT identities — the strongest check of all.

        eq. (48)+(49):  B₁ = (e/h)·φ·(1 − r) + ∫ (M/M_m)|∇W⁽¹⁾|²
        eq. (53):       C₂ = −∫ (M/M_m)|∇W⁽²⁾|²

    These are not approximations: they are equalities. If the cell solver
    violates them, its solution is wrong.
    """

    @pytest.mark.parametrize("r", [0.2, 0.5, 2.0, 6.5, 20.0])
    def test_identite_B1_cas_elastique(self, r):
        d = _cell(r)
        assert d["residu_B1"] < 1e-8, f"residual = {d['residu_B1']:.2e}"

    @pytest.mark.parametrize("r", [0.2, 0.5, 2.0, 6.5, 20.0])
    def test_identite_C2_cas_elastique(self, r):
        d = _cell(r)
        assert d["residu_C2"] < 1e-10, f"residual = {d['residu_C2']:.2e}"

    @pytest.mark.parametrize("omega", [0.1, 1.0, 10.0, 100.0])
    def test_identites_en_viscoelastique(self, omega):
        """Our framework: r(p) is COMPLEX. The identities remain exact."""
        p_ = ph.PhysicalParams()
        r = complex(ph.contraste(-1j * omega, p_))
        d = _cell(r)
        assert d["residu_B1"] < 1e-8
        assert d["residu_C2"] < 1e-10

    def test_C2_est_toujours_negatif(self):
        """Direct consequence of identity (53): C₂ = −∫(M/M_m)|∇W⁽²⁾|² ≤ 0."""
        for r in (0.1, 0.5, 1.0, 2.0, 6.5, 20.0, 100.0):
            assert _cell(r, raf=2)["C2"].real <= 1e-12, f"C₂ > 0 for r = {r}"


class TestBornesVariationnelles:
    """Marigo's rigorous bounds, eq. (60), for a = e."""

    def test_borne_sur_B(self):
        p_ = ph.PhysicalParams()
        borne = E_H / (PHI * XI + 1 - PHI)                    # 0.5333
        B = ph.coefficient_B(0.0, p_).real                    # 0.5839
        assert B >= borne
        assert B < 1.15 * borne, "the bound should be tight"

    def test_borne_sur_C(self):
        p_ = ph.PhysicalParams()
        borne = E_H * XI / (PHI + (1 - PHI) * XI)             # 3.4667
        C = ph.coefficient_C(0.0, p_).real                    # 3.5789
        assert C >= borne
        assert C < 1.10 * borne

    def test_borne_sur_C2(self):
        borne = -E_H * PHI * (1 - PHI) * (XI - 1) ** 2 / (PHI + (1 - PHI) * XI)
        C2 = _cell(XI)["C2"].real
        assert C2 >= borne

    def test_S_est_exact(self):
        """S = a/h + (e/h)φ(ρ_i/ρ_m − 1): closed-form formula, no approximation."""
        p_ = ph.PhysicalParams()
        assert ph.coefficient_S_full(p_) == pytest.approx(4.12, abs=1e-12)

    def test_B_reste_sous_a_sur_h(self):
        """Structural constraint: μ_i > μ_m ⇒ B₁ < 0 ⇒ B < a/h.

        The legacy code gave B = 13.03 > a/h = 2: structurally impossible.
        """
        p_ = ph.PhysicalParams()
        assert ph.coefficient_B(0.0, p_).real < p_.e_over_h


class TestControleCroiseModal:
    """B₁ by modal matching (Marigo, appendix B) — not one line shared with the FEM."""

    @pytest.mark.parametrize("r", [0.2, 0.5, 2.0, 6.5])
    def test_B1_modal_vs_fem(self, r):
        b_modal = B1_multimodal(E_H, r, Np=40, N=40)
        b_fem = _cell(r)["B1"].real
        assert abs(b_modal - b_fem) < 5e-3 * max(abs(b_fem), 1.0)


class TestEnergieInterfacePositive:
    """Marigo: E_a ≥ 0 ⟺ S, B, C ≥ 0. We extend it to the right half-plane."""

    def test_B_et_C_positifs_dans_le_demi_plan(self):
        p_ = ph.PhysicalParams()
        for re_p in (0.01, 1.0, 100.0):
            for im_p in (0.0, 2.0, 50.0):
                p = re_p + 1j * im_p
                assert ph.coefficient_B(p, p_).real >= 0.0
                assert ph.coefficient_C(p, p_).real >= 0.0


class TestOptimumDeA:
    """Marigo finds a = e optimal. We recover it — and we explain why.

    The interface occupies [x_L, x_L + a]. The choice of x_L is free up to
    O(η), but it changes the model at O(η²). Following Marigo, the left face
    is aligned with that of the row: x_L = −e/2.

    With this choice, a = e is the ONLY superconvergent choice: E_R = O(η²),
    versus O(η) everywhere else.

    ⚠ An earlier version of this work centered the interface (x_L = −a/2)
    and read off an "optimum" at a/e ≈ 0.89 there. That was noise: the error
    there dropped below the accuracy floor of the reference solver (~1e-4),
    which an anomalous convergence order (≈ 1 instead of 2) betrayed.
    """

    def test_le_minimum_est_en_a_egal_e(self):
        p_ = ph.PhysicalParams()
        e = p_.e_over_h
        grille = np.linspace(0.5, 1.6, 45)
        for kh in (0.1, 0.2):
            R_ref, _ = R_T_reference(kh, p_, theta=0.0)
            err = [abs(ph.compute_R_T_homogenized(kh, 0.0, p_, a=ae * e)[0] - R_ref)
                   / abs(R_ref) for ae in grille]
            a_opt = grille[int(np.argmin(err))]
            assert abs(a_opt - 1.0) < 0.03, f"kh={kh}: optimum at a/e = {a_opt:.3f}"

    def test_a_egal_e_est_superconvergent(self):
        """E_R = O(η²) at a = e, versus O(η) elsewhere: a full order gained."""
        p_ = ph.PhysicalParams()
        e = p_.e_over_h
        khs = np.array([0.05, 0.1, 0.2, 0.3])

        def ordre(ae):
            err = []
            for kh in khs:
                R_ref, _ = R_T_reference(kh, p_, theta=0.0)
                R_h, _ = ph.compute_R_T_homogenized(kh, 0.0, p_, a=ae * e)
                err.append(abs(R_h - R_ref) / abs(R_ref))
            return float(np.polyfit(np.log(khs), np.log(err), 1)[0])

        assert ordre(1.0) > 1.8, "a = e should be O(η²)"
        for ae in (0.7, 0.9, 1.2, 1.5):
            assert ordre(ae) < 1.4, f"a/e = {ae} should only be O(η)"

    def test_le_plancher_du_solveur_de_reference(self):
        """The trap we fell into: below ~1e-4, one is measuring noise.

        Marigo warns: "for very small error, the error due to the numerical
        method may become dominant when compared to the error due to the model."
        """
        p_ = ph.PhysicalParams()
        kh = 0.05
        vals = [abs(R_T_reference(kh, p_, theta=0.0, X=X, raffinement=raf)[0])
                for raf, X in ((2, 3.0), (3, 4.0), (4, 4.0))]
        dispersion = (max(vals) - min(vals)) / np.mean(vals)
        assert dispersion < 1e-2, "the reference should be stable to within 1%"
        assert dispersion > 1e-6, ("the reference would be exact: the 1e-4 floor "
                                   "used by block 4 would need revisiting")
