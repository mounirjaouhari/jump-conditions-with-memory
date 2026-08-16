"""Validation of the finite-element cell solver.

These are the tests the legacy code did not have, and which would have
immediately revealed the modulus-ratio error (D11).
"""

import numpy as np
import pytest

from fading_memory.cell_fem import cell_coefficients as cc

E_H = 2.0


class TestCouchePleine:
    """φ = 1: the problem becomes 1D and is solved along a line.

        M(y₁)·V' = const,  V' → 1 at infinity  ⇒  const = M_m
        inside the inclusion: W' = M_m/M_i − 1
        ⇒  B₁ = (M_m/M_i − 1)·e/h        [and NOT (M_i/M_m − 1)·e/h]

    This is THE test that was missing: the legacy code failed it by a
    factor of 42.
    """

    @pytest.mark.parametrize("r", [0.2, 0.5, 2.0, 6.5])
    def test_formule_fermee(self, r):
        obtenu = cc(r, E_H, phi=1.0, L=6.0)["B1"].real
        attendu = (1.0 / r - 1.0) * E_H
        assert obtenu == pytest.approx(attendu, rel=1e-8)

    def test_le_code_herite_aurait_echoue(self):
        """Pins the proof down: the legacy formula gives a radically wrong result."""
        r = 6.5
        correct = (1.0 / r - 1.0) * E_H          # −1.6923
        herite = (r - 1.0) * E_H                 # +11.0
        assert not np.isclose(correct, herite, rtol=0.5)
        assert cc(r, E_H, phi=1.0)["B1"].real == pytest.approx(correct, rel=1e-8)


class TestInvariants:
    def test_contraste_nul(self):
        """r = 1: the row does not exist, every corrector vanishes."""
        d = cc(1.0, E_H, phi=0.5, L=6.0)
        for k in ("B1", "B2", "C1", "C2"):
            assert abs(d[k]) < 1e-9, f"{k} = {d[k]}"

    def test_symetrie_B2_nul(self):
        """Centered inclusion: B₂ = 0 by parity."""
        assert abs(cc(6.5, E_H, phi=0.5, L=6.0)["B2"]) < 1e-9

    def test_reciprocite_B2_plus_C1(self):
        """Lemma 5.3 of the paper: B₂ + C₁ = 0 (Betti reciprocity).

        No code in this project had ever checked it.
        """
        d = cc(6.5, E_H, phi=0.5, L=6.0)
        assert abs(d["B2"] + d["C1"]) < 1e-9


class TestConvergence:
    def test_independance_a_la_troncature(self):
        """The corrector is evanescent: B₁ must not depend on L."""
        vals = [cc(6.5, E_H, phi=0.5, L=L)["B1"].real for L in (4.0, 8.0, 16.0)]
        assert max(vals) - min(vals) < 1e-9

    def test_convergence_en_maillage(self):
        """Convergence limited by the corner singularity: the gaps between
        successive levels decrease, and the production level
        (refinement 2 → 3) stays below 10⁻³. The observed order is measured
        by block 10 (debt A8)."""
        b1 = [cc(6.5, E_H, phi=0.5, L=6.0, raffinement=n)["B1"].real
              for n in (1, 2, 3)]
        assert abs(b1[0] - b1[1]) > abs(b1[1] - b1[2])
        assert abs(b1[1] - b1[2]) < 1e-3


class TestMemoireEstBidimensionnelle:
    """The paper's structural result, never stated: memory is born from 2D.

    For φ = 1, B₁(p) = (M_m(p)/M_i − 1)·e/h is AFFINE in p (because M_m is):
    B(p) = B^e + p·B^v, hence K_B ≡ 0 — no memory at all.
    φ < 1 is required for B₁ = G(r(p)) to be nonlinear in p.
    """

    def test_couche_pleine_affine_donc_sans_memoire(self):
        # r(p) = mu_i / (1 + 0.1 p) with delta = 0
        r = lambda p: 6.5 / (1.0 + 0.1 * p)  # noqa: E731
        B = lambda p: E_H + cc(r(p), E_H, phi=1.0, L=6.0)["B1"]  # noqa: E731
        # affine  =>  the second difference vanishes
        p1, p2, p3 = 1.0, 2.0, 3.0
        d2 = B(p3) - 2 * B(p2) + B(p1)
        assert abs(d2) < 1e-6, f"B(p) is not affine: d² = {d2}"

    def test_inclusion_partielle_non_affine_donc_memoire(self):
        r = lambda p: 6.5 / (1.0 + 0.1 * p)  # noqa: E731
        B = lambda p: E_H + cc(r(p), E_H, phi=0.5, L=6.0)["B1"]  # noqa: E731
        p1, p2, p3 = 1.0, 2.0, 3.0
        d2 = B(p3) - 2 * B(p2) + B(p1)
        assert abs(d2) > 1e-3, "B(p) is affine: there would be no memory"
