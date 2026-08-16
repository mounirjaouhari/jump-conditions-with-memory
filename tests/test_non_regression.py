"""Non-regression: freezes the current numerical behavior.

These reference values do NOT claim to be correct (see
docs/DETTE_SCIENTIFIQUE.md). They record what the code produces today, so
that no modification silently changes a figure of the paper.

If a test fails:
  - INTENDED modification    → regenerate, then record it in CHANGELOG.md:
        python tests/regenerate_golden.py
  - UNINTENDED modification  → you just avoided a wrong figure.

    pytest tests/test_non_regression.py
"""

import json
from pathlib import Path

import numpy as np
import pytest

import fading_memory.physics as ph

GOLDEN = Path(__file__).parent / "golden.json"
REL = 1e-8  # tolerance: purely numerical, no physical margin


@pytest.fixture(scope="module")
def golden():
    if not GOLDEN.exists():
        pytest.skip("golden.json missing — run: python tests/regenerate_golden.py")
    return json.loads(GOLDEN.read_text(encoding="utf-8"))


def _params(d):
    return ph.PhysicalParams(**d)


def test_coefficients_statiques(golden):
    p_ = _params(golden["params"])
    for nom, fonction in [
        ("B_0", lambda: np.real(ph.coefficient_B(0.0, p_))),
        ("C_0", lambda: np.real(ph.coefficient_C(0.0, p_))),
        ("S", lambda: np.real(ph.coefficient_S_full(p_))),
    ]:
        assert fonction() == pytest.approx(golden["statiques"][nom], rel=REL), nom


def test_decomposition(golden):
    p_ = _params(golden["params"])
    B_e, B_v, _ = ph.decompose_B(p_)
    C_e, C_v, _ = ph.decompose_C(p_)
    g = golden["decomposition"]
    assert np.real(B_e) == pytest.approx(g["B_e"], rel=REL)
    assert np.real(C_e) == pytest.approx(g["C_e"], rel=REL)
    # B^v = C^v = 0 when delta = 0: we test nullity, not a relative value
    assert abs(B_v) < 1e-9 and abs(C_v) < 1e-9


def test_coefficients_complexes(golden):
    p_ = _params(golden["params"])
    for entree in golden["spectre"]:
        p = complex(entree["p_re"], entree["p_im"])
        B = ph.coefficient_B(p, p_)
        C = ph.coefficient_C(p, p_)
        assert B.real == pytest.approx(entree["B_re"], rel=REL)
        assert B.imag == pytest.approx(entree["B_im"], rel=REL)
        assert C.real == pytest.approx(entree["C_re"], rel=REL)
        assert C.imag == pytest.approx(entree["C_im"], rel=REL)


def test_reflexion_transmission(golden):
    p_ = _params(golden["params"])
    for entree in golden["diffusion"]:
        kh = entree["kh"]
        R, T = ph.compute_R_T_homogenized(kh, 0.0, p_)
        Rr, Tr = ph.compute_R_T_reference(kh, 0.0, p_)
        assert abs(R) == pytest.approx(entree["abs_R_hom"], rel=REL)
        assert abs(T) == pytest.approx(entree["abs_T_hom"], rel=REL)
        assert abs(Rr) == pytest.approx(entree["abs_R_ref"], rel=REL)
        assert abs(Tr) == pytest.approx(entree["abs_T_ref"], rel=REL)


def test_noyau_memoire(golden):
    p_ = _params(golden["params"])
    K_hat, C_inf, lam = ph.noyau_memoire(p_, quoi="B")
    t = np.array(golden["noyau"]["t"])
    K = ph.inverse_laplace(K_hat, t, tail_C=C_inf, tail_lambda=lam,
                           omega_max=20000.0, n_quad=200000, n_echantillons=400)
    np.testing.assert_allclose(K, golden["noyau"]["K_B"], rtol=1e-6, atol=1e-8)
