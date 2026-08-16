"""Scientific debt: what has been fixed, and what remains open.

The implementation debts are closed; the analysis problems A1--A8 are
explicitly open in docs/DETTE_SCIENTIFIQUE.md. Every test in this file
must pass: each one keeps the door closed behind a correction that has an
executable invariant.

If one of them starts failing, a regression has reintroduced a bug that
was paid for dearly to find.

    pytest tests/test_dette_scientifique.py -v
"""

import numpy as np
import pytest

import fading_memory.physics as ph


class TestProblemeCellulaire:
    def test_D11_le_rapport_des_modules_est_dans_le_bon_sens(self):
        """B^e = 0.584 and not 13.03. The mother error of the whole article."""
        p = ph.PhysicalParams()
        B_e = ph.coefficient_B(0.0, p).real
        assert B_e == pytest.approx(0.584, abs=0.01)
        assert B_e < 1.0, "B^e ≈ 13: the modulus ratio is inverted again"

    def test_D5_le_terme_de_volume_n_est_compte_qu_une_fois(self):
        """C^e = 3.58 and not 13.0 (volume counted twice in the legacy code)."""
        C_e = ph.coefficient_C(0.0, ph.PhysicalParams()).real
        assert C_e == pytest.approx(3.579, abs=0.02)


class TestPassivite:
    def test_D12_passivite_de_K_B(self):
        """Theorem 7.1: Re[p·K̂_B] ≥ 0. The legacy code gave −11.85."""
        p = ph.PhysicalParams()
        _, _, K_hat_B = ph.decompose_B(p)
        passif, minimum = ph.check_passivity(K_hat_B)
        assert passif, f"passivity violated: min = {minimum}"

    def test_D12bis_passivite_de_moins_K_C(self):
        """C is a STIFFNESS: it is −K_C that is of positive type (critique T3)."""
        p = ph.PhysicalParams()
        _, _, K_hat_C = ph.decompose_C(p)
        passif, minimum = ph.check_passivity(lambda q: -K_hat_C(q))
        assert passif, f"passivity of −K_C violated: min = {minimum}"


class TestIdentiteHerglotz:
    """Exact Herglotz identity — the PROOF of μ ≥ 0 (half-plane passivity).

        Im B(r) = -Im(r)·∫_{Ω_i}|∇V¹|²,   Im C(r) = +Im(r)·∫_{Ω_i}|∇V²|².

    Continuation of the envelope formulas to the complex plane; this is what
    makes the passivity theorem unconditional (article, Herglotz prop.; S7).
    It must hold to the accuracy of the energy identities (~1e-10).
    """

    def test_identite_dans_les_deux_demi_plans(self):
        p = ph.PhysicalParams()
        for r in (6.5 + 0.5j, 2.0 + 1.0j, 0.5 + 2.0j, 3.3 - 2.2j, 6.5 - 0.5j):
            err = ph.identite_herglotz(r, p)
            assert err < 1e-8, f"Herglotz identity fails at r={r}: {err:.2e}"

    def test_signes_de_herglotz(self):
        """-B and C are Herglotz: Im B < 0 and Im C > 0 for Im r > 0."""
        from fading_memory.cell_fem import herglotz_identity
        p = ph.PhysicalParams()
        for r in (6.5 + 0.5j, 0.5 + 2.0j, 10.0 + 3.0j):
            d = herglotz_identity(r, p.e_over_h, p.phi,
                                  L=p.fem_L, raffinement=p.fem_raffinement)
            assert d["im_B"] < 0, f"Im B ≥ 0 at r={r} (upper half-plane)"
            assert d["im_C"] > 0, f"Im C ≤ 0 at r={r} (upper half-plane)"


class TestPassiviteStieltjes:
    """Passivity on the HALF-PLANE (a theorem, no longer a conjecture).

    G(r) = B_1(r) is a Stieltjes function of the contrast (Bergman–Milton):
    G(r) = c + ∫ dμ(z)/(z+r), μ ≥ 0. The positivity of μ, composed with the
    homography r(p), gives Re[p·K̂_B] ≥ 0 on the whole half-plane. We keep
    the door closed behind this result: the measure must remain ≥ 0, and its
    representation must reconstruct G.
    """

    def test_mesure_de_stieltjes_positive(self):
        """μ'(z) = −(1/π) Im G(−z+iη) ≥ 0: the ingredient of passivity."""
        p = ph.PhysicalParams()
        z = np.concatenate([np.linspace(0.02, 6.0, 120),
                            np.linspace(6.1, 40.0, 20)])
        mu = ph.mesure_stieltjes(z, p, eta=0.08)
        assert mu.min() >= -1e-3, f"negative Stieltjes measure: min = {mu.min():.2e}"

    def test_representation_de_stieltjes_reconstruit_G(self):
        """G(r0)−G(r1) reconstructed from μ ≈ direct (up to the O(η) bias)."""
        p = ph.PhysicalParams()
        z = np.concatenate([np.linspace(0.02, 6.0, 240),
                            np.linspace(6.1, 40.0, 35)])
        for r1 in (2.0, 4.0, 10.0):
            direct, recon = ph.reconstruction_stieltjes(p, r1, z, eta=0.08)
            rel = abs(direct - recon) / max(abs(direct), 1e-12)
            assert rel < 0.10, f"reconstruction out of tolerance (r1={r1}): {rel:.1%}"


class TestInversionDeLaplace:
    def test_D3_sur_un_cas_ferme_non_degenere(self):
        """f(t) = e^{−t} + 2e^{−50t}, whose tail is NOT the one subtracted."""
        K_hat = lambda p: 1.0 / (p + 1.0) + 2.0 / (p + 50.0)  # noqa: E731
        t = np.array([0.005, 0.05, 0.3, 1.0, 3.0])
        obtenu = ph.inverse_laplace(K_hat, t, tail_C=3.0, tail_lambda=10.0,
                                    omega_max=20000.0, n_quad=200000,
                                    n_echantillons=400)
        exact = np.exp(-t) + 2 * np.exp(-50 * t)
        erreur = np.max(np.abs(obtenu - exact)) / np.max(np.abs(exact))
        assert erreur < 1e-3, f"error = {erreur:.2e}"


class TestModeles:
    def test_D16_le_modele_instantane_differe_du_modele_a_memoire(self):
        """The with_memory flag was computed and then IGNORED (critique T6).

        The "with memory" and "instantaneous" curves of figures 9, 10 and 12
        were therefore the same computation.
        """
        p = ph.PhysicalParams()
        R_mem, _ = ph.compute_R_T_homogenized(0.3, 0.0, p, with_memory=True)
        R_inst, _ = ph.compute_R_T_homogenized(0.3, 0.0, p, with_memory=False)
        assert abs(R_mem - R_inst) > 1e-6, "the two models are identical"

    def test_prony_laplace_coherent_avec_le_noyau_temporel(self):
        """K̂(p) = Σ αₙτₙ/(1+pτₙ) must be the transform of Σ αₙe^{−t/τₙ}."""
        alphas, taus = np.array([2.0, 0.5]), np.array([0.1, 1.0])
        # K̂(0) = ∫K(t)dt = Σ αₙτₙ
        assert ph.prony_laplace(0.0, alphas, taus) == pytest.approx(
            float(np.sum(alphas * taus)), rel=1e-12)
        # lim p·K̂(p) = Σ αₙ = K(0⁺)
        p = 1e8
        assert p * ph.prony_laplace(p, alphas, taus) == pytest.approx(
            float(np.sum(alphas)), rel=1e-6)


# =============================================================================
# THE TWO DEBTS THAT WERE OPEN — now closed
# =============================================================================

class TestO1_ValidationDeC:
    """Coefficient C was validated by NO figure at all.

    At normal incidence, ∂₂ ≡ 0 and the C term vanishes identically from the
    jump conditions. Validating it requires oblique incidence, hence a
    quasi-periodic (Bloch) reference solution — now implemented.
    """

    def test_C_disparait_a_incidence_normale(self):
        """The fact that made C invisible: at θ = 0, it changes nothing."""
        p = ph.PhysicalParams()
        avec, _ = ph.compute_R_T_homogenized(0.3, 0.0, p, with_C=True)
        sans, _ = ph.compute_R_T_homogenized(0.3, 0.0, p, with_C=False)
        assert abs(avec - sans) < 1e-14, "C contributes at θ = 0: formula suspect"

    def test_C_intervient_a_incidence_oblique(self):
        p = ph.PhysicalParams()
        th = np.deg2rad(45.0)
        avec, _ = ph.compute_R_T_homogenized(0.3, th, p, with_C=True)
        sans, _ = ph.compute_R_T_homogenized(0.3, th, p, with_C=False)
        assert abs(avec - sans) > 1e-3

    def test_C_est_juste_le_modele_converge_en_oblique(self):
        """WITH C, the model converges in O((kh)²) to the Bloch reference."""
        p = ph.PhysicalParams()
        th = np.deg2rad(45.0)
        err = []
        for kh in (0.05, 0.1):
            R_ref, _ = ph.compute_R_T_reference(kh, th, p)
            R_hom, _ = ph.compute_R_T_homogenized(kh, th, p)
            err.append(abs(R_hom - R_ref) / abs(R_ref))
        ordre = float(np.log(err[1] / err[0]) / np.log(2.0))
        assert 1.8 < ordre < 2.2, f"order = {ordre:.2f}, expected ≈ 2"
        assert err[0] < 1e-3

    def test_omettre_C_detruit_la_convergence(self):
        """WITHOUT C, the error is huge and does not converge: C is essential."""
        p = ph.PhysicalParams()
        th = np.deg2rad(45.0)
        R_ref, _ = ph.compute_R_T_reference(0.05, th, p)
        R_avec, _ = ph.compute_R_T_homogenized(0.05, th, p)
        R_sans, _ = ph.compute_R_T_homogenized(0.05, th, p, with_C=False)
        e_avec = abs(R_avec - R_ref) / abs(R_ref)
        e_sans = abs(R_sans - R_ref) / abs(R_ref)
        assert e_sans / e_avec > 100.0, f"gain from C = only {e_sans / e_avec:.0f}×"

    def test_le_C_du_code_herite_aurait_echoue(self):
        """C^e = 13.0 (legacy code) is WORSE than ignoring C: the proof."""
        p = ph.PhysicalParams()
        th = np.deg2rad(45.0)
        R_ref, _ = ph.compute_R_T_reference(0.1, th, p)
        R_bon, _ = ph.compute_R_T_homogenized(0.1, th, p)
        R_faux, _ = ph.compute_R_T_homogenized(0.1, th, p, C_override=13.0)
        e_bon = abs(R_bon - R_ref) / abs(R_ref)
        e_faux = abs(R_faux - R_ref) / abs(R_ref)
        assert e_faux > 1.0, "the legacy C no longer degrades the model: suspect"
        assert e_faux / e_bon > 100.0


class TestO2_RegimeDeLaMemoire:
    """The test case placed the study in a regime where memory is negligible.

    The fact is now established, quantified and plotted (block 7): the gain
    from memory is an increasing function of ω/λ, where λ = μ_m/β_m.
    """

    def test_le_gain_croit_avec_la_viscosite(self):
        kh, th = 0.3, np.deg2rad(45.0)
        gains = []
        for vr in (0.01, 1.0):
            p = ph.PhysicalParams(visc_ratio=vr)
            R_ref, _ = ph.compute_R_T_reference(kh, th, p)
            R_mem, _ = ph.compute_R_T_homogenized(kh, th, p, with_memory=True)
            R_ins, _ = ph.compute_R_T_homogenized(kh, th, p, with_memory=False)
            gains.append((abs(R_ins - R_ref) / abs(R_ref))
                         / (abs(R_mem - R_ref) / abs(R_ref)))
        assert gains[0] < 1.1, "memory should be negligible at ωβ/μ = 0.01"
        assert gains[1] > 2.0, "memory should matter at ωβ/μ = 1"

    def test_l_amplitude_du_noyau_ne_depend_pas_de_la_viscosite(self):
        """Physical fact: β_m sets the relaxation TIME, not the AMPLITUDE.

        K_B(0⁺) = B(+∞) − B^e, and B depends on p only through the contrast
        r(p) = M_i/M_m, which goes from μ_i/μ_m to 0 whatever the viscosity.
        """
        amplitudes = []
        for vr in (0.01, 0.1, 1.0):
            _, C_inf, _ = ph.noyau_memoire(ph.PhysicalParams(visc_ratio=vr))
            amplitudes.append(C_inf)
        assert max(amplitudes) - min(amplitudes) < 1e-6 * abs(amplitudes[0])
