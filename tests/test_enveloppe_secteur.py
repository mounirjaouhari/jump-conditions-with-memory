"""Envelope monotonicity and sector condition (S_p) — guards of plan B.

Two results of the "theorems first" overhaul:

1. The envelope formulas (passivity proposition, part (i), proof in S7):

       dB/dr = −∫_{Ω_i} |∇V⁽¹⁾|² < 0,      dC/dr = +∫_{Ω_i} |∇V⁽²⁾|² > 0.

   B decreases strictly with the contrast r (a stiffer inclusion makes the
   interface less compliant), C increases strictly (it stiffens). These
   monotonicities are what yield passivity on the real axis. If they break,
   the proposition of §2 no longer has a proof.

2. The sector condition (S_p) of the error-estimate theorem
   (supplement S9):

       Re[e^{−i arg p} · M_m(p)/B(p, e)] > 0,
       Re[e^{−i arg p} · M_m(p) · C(p, e)] > 0.

   On the real axis it follows from passivity; off the axis it is a
   hypothesis verified numerically (block 2, traced values secteurMinB/C).
   These tests pin down a few sentinel points.
"""

import numpy as np
import pytest

import fading_memory.physics as ph
from fading_memory.cell_fem import cell_coefficients as cc

E_H, PHI = 2.0, 0.5          # test case: e/h = 2, phi = 1/2
CONTRASTES = [0.2, 0.5, 2.0, 6.5, 20.0]


def _cell(r):
    return cc(r, E_H, PHI, L=6.0, raffinement=2)


class TestMonotonieEnveloppe:
    """dB/dr < 0 and dC/dr > 0 on the real axis — the proven part of the
    passivity proposition."""

    def test_B_strictement_decroissant_en_r(self):
        # B(r, a) = a/h + B_1(r): the monotonicity can be read off B_1.
        B1 = [_cell(r)["B1"].real for r in CONTRASTES]
        for b_gauche, b_droit, r in zip(B1, B1[1:], CONTRASTES[1:]):
            assert b_droit < b_gauche, (
                f"B_1 does not decrease at r = {r}: {b_gauche:.6f} → {b_droit:.6f}"
            )

    def test_C_strictement_croissant_en_r(self):
        # C(r, a) = a/h + (e/h)·φ·(r − 1) + C_2(r).
        C = [E_H * PHI * (r - 1.0) + _cell(r)["C2"].real for r in CONTRASTES]
        for c_gauche, c_droit, r in zip(C, C[1:], CONTRASTES[1:]):
            assert c_droit > c_gauche, (
                f"C does not increase at r = {r}: {c_gauche:.6f} → {c_droit:.6f}"
            )

    def test_derivee_B_par_differences_finies(self):
        """The slope is decidedly negative (not a rounding artifact)."""
        r0, dr = 6.5, 0.5
        pente = (_cell(r0 + dr)["B1"].real - _cell(r0 - dr)["B1"].real) / (2 * dr)
        assert pente < -1e-4, f"dB_1/dr = {pente:.3e}: not decidedly negative"


class TestConditionSecteur:
    """(S_p) at a few sentinel points of the right half-plane.

    The full sweep is done by block 2 (traced values); here we freeze fixed
    points so that the CI detects a regression without rerunning the block.
    """

    @pytest.mark.parametrize("p", [1.0, 0.1 + 0.3j, 1.0 + 3.0j, 10.0 + 30.0j])
    def test_secteur_positif(self, p):
        params = ph.PhysicalParams()
        sB, sC = ph.condition_secteur(p, params)
        assert sB > 0.0, f"(S_p) violated for B at p = {p}: {sB:.4e}"
        assert sC > 0.0, f"(S_p) violated for C at p = {p}: {sC:.4e}"

    def test_symetrie_conjugaison(self):
        """The foundation of the scan restricted to Im p ≥ 0: both members of
        (S_p) are even in Im p."""
        params = ph.PhysicalParams()
        haut = ph.condition_secteur(1.0 + 2.0j, params)
        bas = ph.condition_secteur(1.0 - 2.0j, params)
        assert haut == pytest.approx(bas, rel=1e-9)


class TestBornesMarigoFonction:
    """`physics.bornes_marigo` reproduces eq. (60) and stays below the computed
    values — this is the traced version of TestBornesVariationnelles."""

    def test_bornes_respectees_et_serrees(self):
        params = ph.PhysicalParams()
        borne_B, borne_C = ph.bornes_marigo(params)
        B = ph.coefficient_B(0.0, params).real
        C = ph.coefficient_C(0.0, params).real
        assert borne_B <= B <= 1.15 * borne_B
        assert borne_C <= C <= 1.15 * borne_C
