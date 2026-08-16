"""Validation of the REFERENCE solution (scattering by the actual row).

Without a reliable reference, no validation figure means anything. This is
debt D1, the most serious one of the paper: the "reference" was a second
approximate model.
"""

import numpy as np
import pytest

import fading_memory.physics as ph
from fading_memory.scattering_fem import R_T_reference


class TestInvariants:
    @pytest.mark.parametrize("theta_deg", [0, 30, 45, 60])
    def test_sans_contraste_la_rangee_est_invisible(self, theta_deg):
        """Inclusion = matrix: R = 0 and |T| = 1, at every angle."""
        p = ph.PhysicalParams(mu_ratio=1.0, rho_ratio=1.0, visc_ratio=0.0)
        R, T = R_T_reference(0.3, p, theta=np.deg2rad(theta_deg))
        assert abs(R) < 1e-3
        assert abs(T) == pytest.approx(1.0, abs=1e-6)

    @pytest.mark.parametrize("kh", [0.05, 0.2, 0.5])
    def test_conservation_energie(self, kh):
        """Without viscosity, |R|² + |T|² = 1 (normal incidence)."""
        p = ph.PhysicalParams(visc_ratio=0.0)
        R, T = R_T_reference(kh, p)
        assert abs(R) ** 2 + abs(T) ** 2 == pytest.approx(1.0, abs=1e-6)

    @pytest.mark.parametrize("theta_deg", [15, 30, 45, 60])
    def test_conservation_energie_en_oblique(self, theta_deg):
        """Bloch conditions: for kh < 2π, a single Rayleigh order
        propagates, so |R|² + |T|² = 1 still holds at oblique incidence."""
        p = ph.PhysicalParams(visc_ratio=0.0)
        R, T = R_T_reference(0.3, p, theta=np.deg2rad(theta_deg))
        assert abs(R) ** 2 + abs(T) ** 2 == pytest.approx(1.0, abs=1e-6)


class TestCouchePleine:
    """φ = 1: the row becomes a homogeneous layer, with an analytical solution."""

    @pytest.mark.parametrize("kh", [0.1, 0.3])
    def test_contre_formule_analytique(self, kh):
        p = ph.PhysicalParams(phi=1.0, visc_ratio=0.0, mu_ratio=6.5, rho_ratio=3.12)
        R, _ = R_T_reference(kh, p, X=4.0, raffinement=3)

        Z_m, Z_i = 1.0, np.sqrt(6.5 * 3.12)
        k_i = kh * np.sqrt(3.12 / 6.5)
        e = p.e_over_h
        d = 2 * Z_i * Z_m * np.cos(k_i * e) + 1j * (Z_i ** 2 + Z_m ** 2) * np.sin(k_i * e)
        R_exact = 1j * (Z_i ** 2 - Z_m ** 2) * np.sin(k_i * e) / d

        assert abs(R) == pytest.approx(abs(R_exact), abs=1e-3)


class TestConvergenceReference:
    """Debt A8: the accuracy of the reference is MEASURED, not asserted.

    Block 10 separates mesh error from truncation error; this test guards
    the gate: the gap to the finest mesh must decrease with refinement,
    and the production level (refinement 2) must stay below 10⁻³ in
    relative deviation on R.
    """

    def test_ecart_maillage_decroit(self):
        p = ph.PhysicalParams()
        R_fin, _ = R_T_reference(0.2, p, X=3.0, raffinement=4)
        ecarts = [abs(R_T_reference(0.2, p, X=3.0, raffinement=raf)[0] - R_fin)
                  for raf in (1, 2, 3)]
        assert ecarts[0] > ecarts[1] > ecarts[2], (
            f"gaps not decreasing: {[f'{e:.2e}' for e in ecarts]}")
        assert ecarts[1] / abs(R_fin) < 1e-3


class TestConvergence:
    def test_independance_a_la_troncature(self):
        """|R| must not depend on X once the phase is properly carried back.

        In a viscous matrix k is complex: without the phase carry-back, the
        reflected wave would attenuate on its way back to the boundary and
        |R| would depend on X.
        """
        p = ph.PhysicalParams()
        vals = [abs(R_T_reference(0.3, p, X=X)[0]) for X in (2.0, 3.0)]
        assert abs(vals[0] - vals[1]) < 1e-6


class TestValidationDuModeleHomogeneise:
    """THE test the paper claimed without having it: O((kh)²) convergence."""

    def test_ordre_de_convergence(self):
        p = ph.PhysicalParams()
        erreurs = []
        for kh in (0.05, 0.1):
            R_ref, _ = R_T_reference(kh, p)
            R_hom, _ = ph.compute_R_T_homogenized(kh, 0.0, p)
            erreurs.append(abs(R_hom - R_ref) / abs(R_ref))
        ordre = np.log(erreurs[1] / erreurs[0]) / np.log(2.0)
        assert 1.8 < ordre < 2.2, f"observed order = {ordre:.2f}, expected ≈ 2"
