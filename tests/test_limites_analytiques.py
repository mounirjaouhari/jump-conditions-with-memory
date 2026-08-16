"""Physical invariants: what the code MUST satisfy, no matter what.

These tests depend on no measured value: they are analytical limits.
If they break, the physics is wrong.
"""

import numpy as np
import pytest

import fading_memory.physics as ph


@pytest.fixture
def sans_contraste():
    """Inclusion identical to the matrix: the row must be invisible."""
    return ph.PhysicalParams(mu_ratio=1.0, rho_ratio=1.0, delta=0.0)


class TestContrasteNul:
    """Without contrast, the correctors vanish: B = C = S = a."""

    def test_B(self, sans_contraste):
        p = sans_contraste
        assert ph.coefficient_B(0.0, p).real == pytest.approx(p.e_over_h, abs=1e-8)

    def test_C(self, sans_contraste):
        p = sans_contraste
        assert ph.coefficient_C(0.0, p).real == pytest.approx(p.e_over_h, abs=1e-8)

    def test_S(self, sans_contraste):
        p = sans_contraste
        assert ph.coefficient_S_full(p) == pytest.approx(p.e_over_h, abs=1e-12)  # exact: closed-form formula


class TestModeZero:
    """B^e is bounded by the two extreme configurations in phi.

    phi -> 1 (full layer): B_1(0) = (mu_m/mu_i - 1).e/h = -1.692
    phi -> 0 (no inclusion): B_1(0) = 0
    For 0 < phi < 1, B_1(0) lies strictly between the two.
    """

    def test_B_encadre_par_les_cas_extremes(self):
        p = ph.PhysicalParams()          # phi = 0.5
        borne = (1.0 / p.mu_ratio - 1.0) * p.e_over_h     # -1.6923
        B1 = ph.coefficient_B(0.0, p).real - p.e_over_h
        assert borne < B1 < 0.0, f"B_1(0) = {B1} outside ]{borne}, 0["

    def test_S_expression_fermee(self):
        p = ph.PhysicalParams()
        attendu = p.e_over_h + p.e_over_h * p.phi * (p.rho_ratio - 1.0)
        assert ph.coefficient_S_full(p) == pytest.approx(attendu, rel=1e-12)


class TestCouchePleine:
    """At phi=1, only the normal channel loses its memory.

    B is affine in p, hence K_B=0. By contrast C_2=0 but the volume term
    (e/h)(r-1) survives: K_C is a nonzero negative exponential. This test
    guards the 0.5.1 theoretical correction against any new abusive
    generalization of the bypass mechanism.
    """

    @pytest.fixture
    def params(self):
        return ph.PhysicalParams(phi=1.0, fem_raffinement=1)

    def test_B_est_affine_en_p(self, params):
        for q in (0.0, 0.7, 0.4 + 0.3j):
            attendu = (params.e_over_h
                       + params.e_over_h
                       * ((1.0 + q * params.visc_ratio) / params.mu_ratio - 1.0))
            assert ph.coefficient_B(q, params) == pytest.approx(attendu, rel=1e-9)

    def test_K_C_reste_non_nul_et_explicite(self, params):
        _, _, K_hat_C = ph.decompose_C(params)
        taux = 1.0 / params.visc_ratio
        for q in (0.5, 1.0 + 0.2j):
            attendu = -params.e_over_h * params.mu_ratio / (q + taux)
            assert K_hat_C(q) == pytest.approx(attendu, rel=1e-8, abs=1e-10)
            assert abs(K_hat_C(q)) > 0.0


class TestSymetrie:
    """Centered rectangular inclusion: B_2 = 0 by parity."""

    def test_B2_nul(self):
        p = ph.PhysicalParams()
        assert ph.cell_problem_W2(0.5, p) == 0.0


class TestLinearite:
    """B(p, a) = a + B_1(p): the dependence on a is affine with unit slope."""

    def test_pente_unitaire_en_a(self):
        p = ph.PhysicalParams()
        a1, a2 = 1.0, 3.0
        d = ph.coefficient_B(0.3, p, a=a2) - ph.coefficient_B(0.3, p, a=a1)
        assert d == pytest.approx(a2 - a1, rel=1e-9)


class TestInversionLaplace:
    """The inverter must recover a known exponential.

    This is the test that exposed the sign error (D3a): the original code's
    convention failed it with a 99.8% error.
    """

    def test_retrouve_une_exponentielle(self):
        tau = 2.0
        t = np.linspace(0.05, 10.0, 40)
        obtenu = ph.inverse_laplace(lambda p: 1.0 / (p + 1.0 / tau), t,
                                    tail_C=1.0, tail_lambda=1.0,
                                    omega_max=5000.0, n_quad=100000,
                                    n_echantillons=300)
        attendu = np.exp(-t / tau)
        erreur = np.max(np.abs(obtenu - attendu)) / np.max(attendu)
        assert erreur < 1e-3, f"max relative error = {erreur:.2e}"

    def test_la_convention_heritee_etait_fausse(self):
        """Freezes the proof of bug D3a, so that it cannot come back."""
        tau = 2.0
        t = np.linspace(0.5, 10.0, 40)
        obtenu = ph.inverse_laplace_fft(lambda p: 1.0 / (p + 1.0 / tau), t,
                                        omega_max=60.0, N_omega=4000,
                                        convention="legacy")
        erreur = np.max(np.abs(obtenu - np.exp(-t / tau))) / np.max(np.exp(-t / tau))
        assert erreur > 0.5, "the 'legacy' convention seems corrected: update this test"


class TestProny:
    """The Prony fit must be passive by construction (alpha, tau > 0)."""

    def test_positivite(self):
        t = np.linspace(0.01, 20.0, 200)
        K = 0.7 * np.exp(-t / 1.5) + 0.3 * np.exp(-t / 6.0)
        alphas, taus, residu = ph.prony_fit(t, K, n_exp=2)
        assert np.all(alphas > 0)
        assert np.all(taus > 0)
        assert residu < 1e-3

    def test_reconstruit_une_somme_connue(self):
        t = np.linspace(0.01, 20.0, 200)
        K = 1.2 * np.exp(-t / 3.0)
        alphas, taus, _ = ph.prony_fit(t, K, n_exp=1)
        assert alphas[0] == pytest.approx(1.2, rel=0.05)
        assert taus[0] == pytest.approx(3.0, rel=0.05)
