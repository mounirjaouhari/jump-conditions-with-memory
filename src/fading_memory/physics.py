"""Physical core of the article "Jump conditions with memory for the
dynamic homogenization of a thin row of elastic inclusions in a
viscoelastic matrix".

This module is the SINGLE SOURCE of the formulas. The scripts in `scripts/`
must contain no physics: they read a configuration, call this module,
write data and figures.

Contents:
- `PhysicalParams`: nondimensionalized parameters (μ_m = ρ_m = h = 1)
- multimodal cell problem → interface coefficients B(p), C(p,a), S(a)
- instantaneous/memory decomposition  X(p) = X^e + p X^v + p K̂_X(p)
- Laplace inversion (discretized Bromwich)
- Prony fit of the memory kernels
- harmonic scattering: coefficients R and T

⚠ The approximations in this module are NOT all validated.
See docs/DETTE_SCIENTIFIQUE.md before relying on a result.
Any change to a formula must be recorded in CHANGELOG.md.
"""

import numpy as np

# Note: no `warnings.filterwarnings('ignore')` here. The initial version of the
# code silenced ALL NumPy warnings (overflow, division by zero).
# Warnings are signals, not noise: we let them propagate.


# =============================================================================
# 1. Physical parameters
# =============================================================================

class PhysicalParams:
    """Physical parameters of the problem.

    All parameters are nondimensionalized with respect to the matrix
    quantities: μ_m, ρ_m, and the period h.

    The default values reproduce the article's test case; in production
    they come from `config/default.yaml` — do not duplicate them in the
    scripts.
    """

    def __init__(self,
                 mu_ratio=6.5,        # μ_i / μ_m  (78 GPa / 12 GPa)
                 rho_ratio=3.12,      # ρ_i / ρ_m  (7800 / 2500)
                 visc_ratio=0.1,      # ω β_m / μ_m  (viscosity parameter)
                 phi=0.5,             # volume fraction e/h * ϕ where ϕ=h_incl/h
                 e_over_h=2.0,        # e/h (thickness ratio)
                 delta=0.0,           # viscous perturbation δ (limit δ → 0)
                 N_modes=15,          # (legacy) number of Fourier modes
                 p_large=1e12,        # numerical p → ∞ for B^v, C^v
                 solveur="fem",       # "fem" (correct) | "legacy" (inherited, wrong)
                 fem_L=6.0,           # cell truncation in y1
                 fem_raffinement=2):  # mesh density
        self.mu_ratio = mu_ratio
        self.rho_ratio = rho_ratio
        self.visc_ratio = visc_ratio
        self.phi = phi
        self.e_over_h = e_over_h
        self.delta = delta
        self.N_modes = N_modes
        self.p_large = p_large
        self.solveur = solveur
        self.fem_L = fem_L
        self.fem_raffinement = fem_raffinement

        # Derived dimensions
        self.e_half = e_over_h / 2.0   # half-thickness in the fast variable y
        self.phi_half = phi / 2.0      # half-width in the fast variable y

    def as_dict(self):
        """Serializable dictionary — used in the run manifests."""
        return {
            'mu_ratio': self.mu_ratio, 'rho_ratio': self.rho_ratio,
            'visc_ratio': self.visc_ratio, 'phi': self.phi,
            'e_over_h': self.e_over_h, 'delta': self.delta,
            'N_modes': self.N_modes, 'p_large': self.p_large,
            'solveur': self.solveur, 'fem_L': self.fem_L,
            'fem_raffinement': self.fem_raffinement,
        }


    def __repr__(self):
        return (f"PhysicalParams(μ_i/μ_m={self.mu_ratio}, "
                f"ρ_i/ρ_m={self.rho_ratio}, "
                f"ωβ_m/μ_m={self.visc_ratio}, "
                f"ϕ={self.phi}, e/h={self.e_over_h}, "
                f"solveur={self.solveur})")


# =============================================================================
# 2. Complex Laplace modulus M(p) in the matrix and the inclusions
# =============================================================================

def matrix_modulus(p, params):
    """Laplace viscoelastic modulus in the matrix: M_m(p) = μ_m + p β_m.

    In nondimensional variables (μ_m=1, β_m such that ωβ_m/μ_m = visc_ratio
    at the reference frequency ω=1):
        M_m(p) = 1 + p * visc_ratio
    """
    return 1.0 + p * params.visc_ratio


def inclusion_modulus(p, params, delta=None):
    """Laplace viscoelastic modulus in the inclusion (with perturbation δ).

    M_i,δ(p) = μ_i + p δ

    In nondimensional variables (μ_m=1):
        M_i,δ(p) = mu_ratio + p * δ
    """
    if delta is None:
        delta = params.delta
    return params.mu_ratio + p * delta


def contraste(p, params, delta=None):
    """r(p) = M_i(p) / M_m(p) — the ONLY way p enters the cell.

    The cell problem div(M ∇V) = 0 is homogeneous of degree 0 in M: dividing
    it by M_m(p) shows that its solution depends on p ONLY through this
    ratio. Hence B_1(p) = G(r(p)) and C_2(p) = H(r(p)), functions of a single
    complex variable.

    Memory arises because r(p) is a homographic (Möbius) function of p and
    G, H are nonlinear: this is the analytic source of the kernel K_B.
    """
    return inclusion_modulus(p, params, delta) / matrix_modulus(p, params)


# =============================================================================
# 3. Cell-problem solver — multimodal method (LEGACY, WRONG)
# =============================================================================
#
# ⚠ The two functions below are kept ONLY to reproduce the figures of the
# first version of the article (solveur = "legacy").
# They are wrong:
#   - `cell_problem_W1` inverts the modulus ratio (D11);
#   - `_multimode_correction_C2` is identically zero (D4).
# The correct solver is in `cell_fem.py`. Do not use them.
# =============================================================================
#
# The cell problem for W^{(i)}_{δ,p} reads:
#   div_y [ M_δ(y,p) ∇_y (W^{(i)}_{δ,p} + y_i) ] = 0   in Y_∞ = R × [0,1]
#   W periodic in y_2 (period 1)
#   ∇W → 0 as y_1 → ±∞
#   continuity of W and M_δ ∂_n W on ∂Ω_i (inclusion/matrix interface)
#
# Rectangular inclusion: |y_1| < e/2h, |y_2| < ϕ/2
#
# Multimodal method: W(y_1, y_2) = Σ_n w_n(y_1) exp(2iπ n y_2)
# For each mode n, an ODE in y_1 is solved with transmission conditions
# at y_1 = ±e/2h.
# =============================================================================

def cell_problem_W1(p, params, delta=None):
    """Solves the cell problem for W^{(1)}_{δ,p} (i=1, direction y_1).

    The field W^{(1)} + y_1 satisfies the elliptic equation with conditions:
    - y_2 periodicity (period 1)
    - decay as y_1 → ±∞
    - continuity at the interface |y_1|=e/2h, |y_2|<ϕ/2

    For direction y_1, the problem is 1D in mode n=0 (mean):
    The asymptotic solution for y_1 → ±∞ is W^{(1)} → 0.

    Jump coefficient: B_1(p) = W^{(1)}(+∞) - W^{(1)}(-∞)

    For a rectangular inclusion in mode n=0, the problem reduces to:
    - In the matrix (|y_1| > e/2h): M_m d²W/dy_1² = 0 → W linear
    - In the inclusion (|y_1| < e/2h): M_i d²W/dy_1² = 0 → W linear
    - Continuity of W and M ∂_n W at y_1 = ±e/2h

    Accounting for the decay W → 0 as |y_1| → ∞, and for the fact that the
    effective gradient must be 1 (to compensate y_1), one gets:
        W^{(1)}(y_1) = -y_1 + const in the inclusion
        W^{(1)}(y_1) = -sign(y_1) (M_i/M_m) (e/2h) + 0 in the far matrix

    More precisely, writing L = e/(2h):
        W^{(1)}(+∞) - W^{(1)}(-∞) = (M_i/M_m - 1) * L * 2 = (M_i/M_m - 1) * e/h
        (with the correct sign)

    For modes n ≠ 0, the full 2D problem must be solved.
    """
    M_m = matrix_modulus(p, params)
    M_i = inclusion_modulus(p, params, delta)

    L = params.e_half  # half-thickness in y_1

    # Mode n=0 (mean): dominant contribution to the jump
    # Jump = (M_i/M_m - 1) * e/h
    B1_n0 = (M_i / M_m - 1.0) * (2 * L)

    # Correction from modes n≥1 (evanescent field around the corners)
    # This correction is computed via the full multimodal method.
    B1_correction = _multimode_correction_W1(p, params, M_m, M_i)

    return B1_n0 + B1_correction


def _multimode_correction_W1(p, params, M_m, M_i):
    """Multimodal correction (modes n≥1) for the coefficient B_1.

    For rectangular inclusions, the higher modes contribute through the
    evanescent field around the corners. This correction is typically
    small compared with mode n=0.

    A modal decomposition with N_modes Fourier modes in y_2 is used.
    For each mode n, one solves:
        - in the inclusion (|y_1|<L, |y_2|<ϕ/2): d²w_n/dy_1² - k_n² w_n = 0
        - in the matrix: exponential decay

    where k_n = 2π n is the modal wavenumber.

    For i=1 (gradient along y_1), mode n=0 already captures the essential part.
    Modes n≥1 enter through the boundary conditions at y_2=±ϕ/2.
    """
    N = params.N_modes
    L = params.e_half
    phi = params.phi

    # For the W^{(1)} problem, the forcing is ∂_{y_1} (i.e., a unit gradient).
    # In the modal decomposition, the forcing only affects mode n=0.
    # Modes n≥1 are excited by the corners of the inclusion.

    # Approximation: the multimodal correction for B_1 is small.
    # It is estimated via the evanescent field at the corners.
    # Reference: Marigo, Maurel, Pham, Guenneau (2017) - equation (4.13)

    correction = 0.0
    for n in range(1, N + 1):
        k_n = 2 * np.pi * n
        # Exponential decay rate of mode n
        # in the matrix: exp(-k_n |y_1|)
        # Amplitude at the interface: depends on the contrast and the geometry
        # Asymptotic approximation for weak contrast:
        amp = (M_i / M_m - 1.0) / (M_i / M_m + 1.0) * np.sin(k_n * phi / 2) / k_n
        correction += 2 * amp / k_n

    return correction


def cell_problem_W2(p, params, delta=None):
    """Solves the cell problem for W^{(2)}_{δ,p} (i=2, direction y_2).

    For i=2, the forcing is ∂_{y_2}. Mode n=0 does not contribute
    (no mean gradient along y_2 over one period). Modes n≥1
    capture the oscillating field.

    For the symmetric geometry (centered rectangular inclusion),
    W^{(2)}(+∞) - W^{(2)}(-∞) = 0 by parity.
    Hence B_2(p) = 0 (symmetric case).
    """
    return 0.0  # by symmetry


def coefficient_C2(p, params, delta=None):
    """Computes the coefficient C_2(p) defined by:
        C_2(p) = ∫_{Y_∞} [M_δ(y,p)/M_m(p)] ∂_{y_2} W^{(2)}_{δ,p} dy

    For the symmetric case, W^{(2)} is nonzero but its weighted integral
    is computed via the multimodal method.

    The effective coefficient is:
        C(p,a) = a/h + (e/h) ϕ (M_i/M_m - 1) + C_2(p)
    """
    M_m = matrix_modulus(p, params)
    M_i = inclusion_modulus(p, params, delta)

    # Volume term (mode n=0): (e/h) ϕ (M_i/M_m - 1)
    C2_n0 = params.e_over_h * params.phi * (M_i / M_m - 1.0)

    # Multimodal correction
    C2_corr = _multimode_correction_C2(p, params, M_m, M_i)

    return C2_n0 + C2_corr


def _multimode_correction_C2(p, params, M_m, M_i):
    """Multimodal correction for C_2(p).

    For the W^{(2)} problem, modes n≥1 are directly excited.
    The multimodal system solved is:
        For each mode n:  d²w_n/dy_1² - k_n² w_n = 0  (in the inclusion)
                          d²w_n/dy_1² - k_n² w_n = 0  (in the matrix)
        with forcing -∂_{y_2} y_2 = -1 in the inclusion, 0 in the matrix.

    The solution in the inclusion is: w_n^{int}(y_1) = A_n cosh(k_n y_1) + forcing
    In the matrix: w_n^{ext}(y_1) = B_n exp(-k_n |y_1|)

    The transmission conditions at y_1 = ±L give:
        A_n cosh(k_n L) + f_n = B_n exp(-k_n L)             (continuity of W)
        M_i k_n A_n sinh(k_n L) = -M_m k_n B_n exp(-k_n L)  (continuity of flux)

    For the ∂_{y_2} forcing: f_n = -1/(k_n²) * (modal force)
    """
    N = params.N_modes
    L = params.e_half
    phi = params.phi

    correction = 0.0
    for n in range(1, N + 1):
        k_n = 2 * np.pi * n

        # Modal forcing coefficient (Fourier transform of the indicator
        # function of the inclusion in y_2)
        # f_n = (1/k_n) * sin(k_n * phi/2) for a square pulse
        if abs(k_n * phi / 2) < 1e-12:
            continue
        f_n = np.sin(k_n * phi / 2) / k_n

        # 2x2 system for A_n and B_n:
        # A_n cosh(k_n L) - B_n exp(-k_n L) = -f_n
        # M_i A_n sinh(k_n L) + M_m B_n exp(-k_n L) = 0
        # → B_n = -M_i sinh(k_n L) / M_m / exp(-k_n L) * A_n
        # → A_n [cosh(k_n L) + (M_i/M_m) sinh(k_n L)] = -f_n

        denom = np.cosh(k_n * L) + (M_i / M_m) * np.sinh(k_n * L)
        if abs(denom) < 1e-14:
            continue
        A_n = -f_n / denom

        # ∂_{y_2} W^{(2)} on mode n: ik_n * w_n(y_1, y_2) * exp(2iπn y_2)
        # Integral over Y_∞ of M_δ/M_m ∂_{y_2} W^{(2)}:
        # Contribution of mode n in the inclusion:
        # ∫_{inclusion} (M_i/M_m) * (ik_n) * A_n cosh(k_n y_1) * (-ik_n) * (1/2) dy
        # (the product of exp(2iπn y_2) with its conjugate gives 1/2 over the period)
        contrib = (M_i / M_m) * k_n**2 * A_n * (np.sinh(k_n * L) / k_n) * (phi / 2)
        # Multiplied by 2 for the two sides of the inclusion (symmetry y_1 → -y_1)
        contrib *= 2

        # Contribution in the matrix (evanescent field):
        B_n = -A_n * (M_i / M_m) * np.sinh(k_n * L) / np.exp(-k_n * L)
        contrib_ext = (1.0) * k_n**2 * B_n * (np.exp(-k_n * L) / k_n) * (phi / 2)
        contrib_ext *= 2

        correction += contrib + contrib_ext

    # The sign depends on the conventions; for W^{(2)} with forcing -y_2:
    correction = -correction

    return correction


def _cellule(p, params, delta=None):
    """Solves the cell by finite elements for the contrast r(p)."""
    from .cell_fem import cell_coefficients
    return cell_coefficients(contraste(p, params, delta),
                             params.e_over_h, params.phi,
                             L=params.fem_L,
                             raffinement=params.fem_raffinement)


def coefficient_B(p, params, a=None, delta=None):
    """Interface coefficient  B(p, a) = a/h + B_1(p),
    with  B_1(p) = W^{(1)}(+∞) − W^{(1)}(−∞), jump of the normal corrector.

    `solveur = "fem"`    : 2D cell problem actually solved (§8.1).
    `solveur = "legacy"` : formula inherited from common.py — modulus ratio
                           INVERTED, hence wrong. Kept to reproduce the
                           figures of the first version of the article.
                           See docs/DETTE_SCIENTIFIQUE.md (D11).
    """
    if a is None:
        a = params.e_over_h  # default: a = e
    if params.solveur == "legacy":
        return a + cell_problem_W1(p, params, delta)
    return a + _cellule(p, params, delta)["B1"]


def coefficient_C(p, params, a=None, delta=None):
    """Interface coefficient  C(p, a) = a/h + (e/h)·ϕ·(M_i/M_m − 1) + C_2(p).

    The volume term and the modal integral C_2 are two DISTINCT
    contributions. The legacy code counted the volume twice (D5) and
    computed a modal integral that was identically zero (D4).
    """
    if a is None:
        a = params.e_over_h
    if params.solveur == "legacy":
        M_m = matrix_modulus(p, params)
        M_i = inclusion_modulus(p, params, delta)
        C2 = coefficient_C2(p, params, delta)
        return a + params.e_over_h * params.phi * (M_i / M_m - 1.0) + C2

    r = contraste(p, params, delta)
    C2 = _cellule(p, params, delta)["C2"]
    return a + params.e_over_h * params.phi * (r - 1.0) + C2


def coefficient_S_full(params, a=None):
    """Inertial coefficient S(a) with a/h.
    """
    if a is None:
        a = params.e_over_h
    return a + params.e_over_h * params.phi * (params.rho_ratio - 1.0)


# =============================================================================
# 3 bis. External validation, bounds, and sector condition
# =============================================================================

def bornes_marigo(params):
    """Lower variational bounds of Marigo et al. (2017), eq. (60).

    For a = e and a real elastic contrast r = μ_i/μ_m:

        B^e ≥ (e/h) / (φ r + 1 − φ)          (trial field of problem 1)
        C^e ≥ (e/h) r / (φ + (1 − φ) r)      (trial field of problem 2)

    These bounds come from restricted trial fields: they are tight
    but not attained. Returns (borne_B, borne_C).
    """
    r = params.mu_ratio
    e_h = params.e_over_h
    phi = params.phi
    borne_B = e_h / (phi * r + 1.0 - phi)
    borne_C = e_h * r / (phi + (1.0 - phi) * r)
    return borne_B, borne_C


def borne_positivite_a(params):
    """Positivity threshold of the interface energy, in units of e.

    The interface energy is positive as soon as S(a), B(0, a) and C(0, a)
    are (Marigo et al. 2017, eq. 59): a ≥ e·max(F, G, φ(1 − ρ_i/ρ_m)),
    where F and G are the thresholds at which B and C vanish. In units a/e:

        F = −B_1(0)/(e/h),      G = −C(0, a=0)/(e/h),
        inertial threshold = φ(1 − ρ_i/ρ_m).

    For inclusions stiffer and denser than the matrix, only F is
    positive: it is B that constrains the thickness.
    """
    e_h = params.e_over_h
    B1 = coefficient_B(0.0, params, a=0.0)
    C0 = coefficient_C(0.0, params, a=0.0)
    F = -B1.real / e_h
    G = -C0.real / e_h
    return max(F, G, params.phi * (1.0 - params.rho_ratio))


def condition_secteur(p, params, a=None):
    """Sector condition (S_p) of the error estimate (supplement S9).

        Re[ e^{−iφ} · M_m(p) / B(p, a) ] > 0
        Re[ e^{−iφ} · M_m(p) · C(p, a) ] > 0,       φ = arg p.

    This is the coercivity hypothesis (Laplace rotation) under which the
    h²(ln h)² error-estimate theorem is proved. On the real axis it
    follows from passivity; off the axis it is checked numerically.
    Since B(p̄) = conj B(p) and C(p̄) = conj C(p) (the contrast r(p) is
    real on the real axis), both members are even in Im p:
    sampling Im p ≥ 0 suffices.

    Returns (secteur_B, secteur_C), the two real parts.
    """
    p = complex(p)
    rot = np.exp(-1j * np.angle(p)) if p != 0 else 1.0
    M_m = matrix_modulus(p, params)
    B = coefficient_B(p, params, a)
    C = coefficient_C(p, params, a)
    return float((rot * M_m / B).real), float((rot * M_m * C).real)


def _G_of_r(r, params):
    """G(r) = B_1(r) — normal cell coefficient for a complex contrast r.

    This is the function that the passivity theorem asserts to be of
    Stieltjes type: G(r) = c + ∫ dμ(z)/(z+r), μ ≥ 0. It is evaluated
    directly at the requested value r (including off the positive real axis,
    which is used to reconstruct the measure μ).
    """
    from .cell_fem import cell_coefficients
    d = cell_coefficients(complex(r), params.e_over_h, params.phi,
                          L=params.fem_L, raffinement=params.fem_raffinement)
    return d["B1"]


def identite_herglotz(r, params):
    """Maximum relative deviation of the Herglotz identity at complex contrast r.

        Im B(r) = -Im(r)·∫_{Ω_i}|∇V¹|²,   Im C(r) = +Im(r)·∫_{Ω_i}|∇V²|².

    This is the identity that PROVES the positivity of the Stieltjes measures
    of the interface coefficients (passivity theorem, Herglotz proposition in
    the main text, supplement S7). Returns max(err_B, err_C) — expected
    ~1e-10 (precision of the energy identities).
    """
    from .cell_fem import herglotz_identity
    d = herglotz_identity(complex(r), params.e_over_h, params.phi,
                          L=params.fem_L, raffinement=params.fem_raffinement)
    return max(d["err_B"], d["err_C"])


def mesure_stieltjes(z, params, eta=0.08):
    """Density of the Stieltjes measure of G, by Stieltjes--Perron inversion.

        μ'(z) = −(1/π) · Im G(−z + i·η),      z ≥ 0.

    By the Bergman--Milton representation, G(r) = c + ∫ dμ(z)/(z+r) with
    μ ≥ 0; the POSITIVITY of μ' is the ingredient that proves the passivity
    of the kernels on the whole half-plane (passivity theorem, supplement S7).
    The parameter η > 0 smooths the inversion (O(η) bias); it must remain
    small compared with the variation scale of μ, yet large enough to be
    resolved by the grid.
    """
    z = np.atleast_1d(np.asarray(z, dtype=float))
    return np.array([-(1.0 / np.pi) * _G_of_r(-zz + 1j * eta, params).imag
                     for zz in z])


def reconstruction_stieltjes(params, r1, z_grid, eta=0.08):
    """Consistency check of the Stieltjes representation.

    Compares the direct difference G(r0) − G(r1) (r0 = r(p=0) = μ_i/μ_m) with
    its reconstruction from the measure,  ∫ μ'(z) [1/(z+r0) − 1/(z+r1)] dz.
    The agreement (up to the O(η) bias and the grid resolution) validates that
    G is indeed of Stieltjes type in the contrast variable.
    Returns (direct, recon).
    """
    _trap = getattr(np, "trapezoid", None) or np.trapz
    r0 = params.mu_ratio                       # r(p=0) = M_i(0)/M_m(0) = μ_i/μ_m
    mu = mesure_stieltjes(z_grid, params, eta)
    direct = (_G_of_r(r0, params) - _G_of_r(r1, params)).real
    recon = float(_trap(mu * (1.0 / (z_grid + r0) - 1.0 / (z_grid + r1)), z_grid))
    return direct, recon


def validations_solveur(params):
    """Validation residuals of the cell solver (section 5.2 of the article).

    Each entry confronts the solver with a situation whose answer is
    known a priori:

      - 'pleine'        : full layer (φ = 1), B_1 = (M_m/M_i − 1)·e/h exact;
      - 'contraste_nul' : M_i = M_m, all correctors vanish;
      - 'reciprocite'   : lemma B_2 + C_1 = 0;
      - 'troncature'    : B_1 independent of the truncation L (evanescent
                          corrector) — gap between L = 4 and L = 20;
      - 'identite_B'    : energy identity (48)-(49) of Marigo, absolute
                          residual, viscoelastic case (complex r);
      - 'identite_C'    : energy identity (53), absolute residual, idem;
      - 'maillage'      : gap on B_1 between two refinements (the corner
                          singularity limits the convergence).

    Returns a dict {name: residual ≥ 0}.
    """
    from .cell_fem import cell_coefficients

    e_h, phi = params.e_over_h, params.phi
    L, raf = params.fem_L, params.fem_raffinement

    # Full layer: the problem becomes 1D, B_1 = (1/r − 1)·e/h.
    r0 = params.mu_ratio
    d_pleine = cell_coefficients(r0, e_h, 1.0, L=L, raffinement=raf)
    res_pleine = abs(d_pleine["B1"] - (1.0 / r0 - 1.0) * e_h)

    # No contrast: the row is invisible, correctors vanish.
    d_nul = cell_coefficients(1.0, e_h, phi, L=L, raffinement=raf)
    res_nul = max(abs(d_nul["B1"]), abs(d_nul["C2"]))

    # Reciprocity B_2 + C_1 = 0 (test case).
    d_ref = cell_coefficients(r0, e_h, phi, L=L, raffinement=raf)
    res_recip = abs(d_ref["B2"] + d_ref["C1"])

    # Independence from the truncation L (evanescent corrector).
    B1_L4 = cell_coefficients(r0, e_h, phi, L=4.0, raffinement=raf)["B1"]
    B1_L20 = cell_coefficients(r0, e_h, phi, L=20.0, raffinement=raf)["B1"]
    res_tronc = abs(B1_L20 - B1_L4)

    # Marigo energy identities, for a COMPLEX contrast
    # (viscoelastic case): worst residual over a small sweep in ω.
    res_idB = res_idC = 0.0
    for omega in (0.1, 1.0, 10.0):
        r = complex(contraste(-1j * omega, params))
        d = cell_coefficients(r, e_h, phi, L=L, raffinement=raf)
        res_idB = max(res_idB, d["residu_B1"])
        res_idC = max(res_idC, d["residu_C2"])

    # Mesh convergence, limited by the corner singularity.
    B1_fin = cell_coefficients(r0, e_h, phi, L=L, raffinement=raf + 1)["B1"]
    res_maille = abs(B1_fin - d_ref["B1"])

    return {
        "pleine": float(res_pleine),
        "contraste_nul": float(res_nul),
        "reciprocite": float(res_recip),
        "troncature": float(res_tronc),
        "identite_B": float(res_idB),
        "identite_C": float(res_idC),
        "maillage": float(res_maille),
    }


# =============================================================================
# 4. Instantaneous/memory decomposition
# =============================================================================
#
# We seek: B(p, a) = B^e(a) + p B^v(a) + p K̂_B(p, a)
#
# B^e(a) = B(0, a)   (static limit, purely elastic)
# B^v(a) = lim_{p → +∞} [B(p, a) - B(0, a)] / p   (instantaneous viscosity)
# K̂_B(p, a) = [B(p, a) - B(0, a) - p B^v(a)] / p   (residual memory)
# =============================================================================

def decompose_B(params, a=None):
    """Decomposition B(p,a) = B^e + p B^v + p K̂_B(p).

    Returns (B_e, B_v, K_hat_B_function).
    """
    if a is None:
        a = params.e_over_h

    # B^e = B(0, a): static limit (p=0)
    B_e = coefficient_B(0.0, params, a)

    # B^v = lim_{p → ∞} (B(p,a) - B^e) / p
    # For large p: M_m ≈ p β_m, M_i ≈ p δ (or p β_m if δ is small)
    # B_1(p) ≈ (M_i/M_m - 1) * (e/h)
    # If δ → 0: M_i → μ_i (constant), M_m → p β_m → ∞
    # So M_i/M_m → 0, and B_1(p) → -e/h
    # B(p, a) → a/h - e/h = (a-e)/h
    # If a = e: B(p, a) → 0 as p → ∞
    # So (B(p,a) - B^e)/p → 0 as p → ∞ in that case
    # B^v = 0 if a = e and δ = 0

    # Numerical computation of B^v
    p_large = params.p_large
    B_p_large = coefficient_B(p_large, params, a)
    B_v = (B_p_large - B_e) / p_large

    def K_hat_B(p):
        if abs(p) < 1e-12:
            return 0.0  # K̂_B(0) = 0 by construction
        return (coefficient_B(p, params, a) - B_e - p * B_v) / p

    return B_e, B_v, K_hat_B


def decompose_C(params, a=None):
    """Decomposition C(p,a) = C^e + p C^v + p K̂_C(p).
    """
    if a is None:
        a = params.e_over_h

    C_e = coefficient_C(0.0, params, a)

    p_large = params.p_large
    C_p_large = coefficient_C(p_large, params, a)
    C_v = (C_p_large - C_e) / p_large

    def K_hat_C(p):
        if abs(p) < 1e-12:
            return 0.0
        return (coefficient_C(p, params, a) - C_e - p * C_v) / p

    return C_e, C_v, K_hat_C


# =============================================================================
# 5. Laplace inversion (discretized Bromwich)
# =============================================================================
#
# HISTORY — `common.py` contained a `talbot_inversion` function made of three
# superimposed formula attempts, two of them dead (variables computed then
# overwritten, a loop with no effect). It was called by NO block: blocks 2
# and 5 each defined their own inverter, identical to the one below.
# `talbot_inversion` was therefore removed during the consolidation
# (decision and context: docs/decisions/ADR-0002), and the inverter actually
# used is consolidated here. No numerical result is affected.
# =============================================================================


def inverse_laplace_fft(K_hat_func, t, omega_max=30.0, N_omega=2000, eps=0.01,
                        convention="legacy"):
    """Inverts a Laplace transform by a discretized Bromwich integral.

        K(t) = (1/2π) ∫ K̂(ε + iω) e^{(ε + iω)t} dω

    Parameters
    ----------
    K_hat_func : callable  p ↦ K̂(p), accepting a complex argument
    t          : array of times at which to evaluate K(t)
    omega_max  : truncation bound of the integral over ω
    N_omega    : number of quadrature points in ω
    eps        : Bromwich abscissa (Re p = ε > 0)
    convention : "standard" → K̂(ε + iω)   [correct]
                 "legacy"   → K̂(ε − iω)   [the one of the initial code]

    ⚠⚠ TWO KNOWN DEFECTS — see docs/DETTE_SCIENTIFIQUE.md (D3) ⚠⚠

    D3a — SIGN. The initial code (blocks 2 and 5) evaluated K̂(ε − iω) while
    multiplying by e^{+iωt}: this computes f(−t), hence ≈ 0 for a causal
    kernel. Verifiable in one line: on K̂(p) = 1/(p + 1/τ), whose exact
    inverse is e^{−t/τ}, the "legacy" convention returns noise (99.8% error)
    where "standard" recovers the exponential to within 1%.
    Test: tests/test_limites_analytiques.py::TestInversionLaplace

    D3b - TRUNCATION. K̂_B(p) ~ C/p as |p| → ∞: the kernel has a jump at t=0,
    and the integrand is still 0.41 at ω = 30, where the integral is cut off.
    Strong Gibbs oscillations result: the reconstructed kernel changes sign
    287 times on [0, 30]. NEITHER convention gives a converged result with
    omega_max = 30. Fixing the sign is therefore not enough:
    the asymptotic part must be subtracted analytically, or a deformed
    contour (Talbot) must be used.

    The production chain uses "standard" (numerique.laplace_convention
    in the configuration); "legacy" is kept only as an executable witness
    of the historical defect, locked in by the non-regression
    tests.
    """
    if convention not in ("standard", "legacy"):
        raise ValueError(f"unknown convention: {convention!r}")
    signe = 1.0 if convention == "standard" else -1.0

    t = np.asarray(t, dtype=float)
    omegas = np.linspace(-omega_max, omega_max, N_omega)
    K_hat_vals = np.array([K_hat_func(eps + signe * 1j * w) for w in omegas])
    domega = omegas[1] - omegas[0]

    K_t = np.zeros_like(t)
    for k, tk in enumerate(t):
        K_t[k] = (np.exp(eps * tk) / (2 * np.pi)) * domega * np.sum(
            K_hat_vals * np.exp(1j * omegas * tk)
        ).real
    return K_t


# =============================================================================
# 6. Prony approximation of the memory kernels
# =============================================================================
#
# K(t) ≈ Σ_{n=1}^{N} α_n exp(-t/τ_n)
#
# The logarithmic parametrization (α = e^x, τ = e^y) enforces α_n > 0 and
# τ_n > 0, hence the passivity of the approximation, by construction.
# =============================================================================


def inverse_laplace(K_hat, t, tail_C=0.0, tail_lambda=1.0,
                    omega_max=400.0, n_quad=20000, n_echantillons=240, eps=1e-3):
    r"""Correct Laplace inversion, by Bromwich with tail subtraction.

    Fixes the two defects of `inverse_laplace_fft` (D3):

    D3a — SIGN. We do evaluate K̂(ε + iω), and we exploit the Hermitian
    symmetry K̂(p̄) = conj K̂(p) (valid because the time kernel is real):

        f(t) = (e^{εt}/π) · Re ∫_0^∞ K̂(ε + iω) e^{iωt} dω

    D3b - TRUNCATION. Here K̂_B(p) ~ C∞/p as |p| → ∞ (the kernel jumps at
    t = 0), so the integrand decays as 1/ω: truncating at ω_max = 30 produced
    Gibbs oscillations — 287 sign changes, pure noise.
    The tail is therefore subtracted ANALYTICALLY:

        K̂(p) = C∞/(p + λ)  +  [ K̂(p) − C∞/(p + λ) ]
                 \______/       \___________________/
              known inverse:     decays as O(1/ω²):
              C∞ · e^{−λt}       the integral converges

    Parameters
    ----------
    tail_C      : C∞ = lim_{p→∞} p·K̂(p) = K(0⁺), the kernel's jump at the origin
    tail_lambda : λ > 0, decay rate of the subtracted tail
    n_echantillons : number of ACTUAL evaluations of K̂ (one FEM solve
                  each); the fine quadrature is obtained by spline
                  interpolation, K̂ being analytic and hence very smooth in ω.
    """
    from scipy.interpolate import CubicSpline

    t = np.asarray(t, dtype=float)

    # 1) LOGARITHMIC SAMPLING in ω.
    #    The kernel contains time scales from 10^-3 to 10^-1: its spectrum
    #    spans several decades. Linear sampling would waste its whole budget
    #    on the low frequencies. Each sample costs one finite-element
    #    solve: they are precious.
    u_max = np.log1p(omega_max)
    u_ech = np.linspace(0.0, u_max, n_echantillons)
    w_ech = np.expm1(u_ech)
    g_ech = np.array([
        K_hat(eps + 1j * w) - tail_C / (eps + 1j * w + tail_lambda)
        for w in w_ech
    ])

    # 2) interpolation in u = log(1+ω): K̂ is smooth there across all decades
    sp_re = CubicSpline(u_ech, g_ech.real)
    sp_im = CubicSpline(u_ech, g_ech.imag)

    # 3) quadrature on a FINE grid in ω (the oscillatory e^{iωt} demands it)
    w = np.linspace(0.0, omega_max, n_quad)
    u = np.log1p(w)
    g = sp_re(u) + 1j * sp_im(u)

    # 4) re-injection of the analytic tail  C∞·e^{−λt}
    out = np.empty_like(t)
    for k, tk in enumerate(t):
        integrale = np.trapezoid(g * np.exp(1j * w * tk), w).real
        out[k] = tail_C * np.exp(-tail_lambda * tk) \
            + np.exp(eps * tk) / np.pi * integrale
    return out


def noyau_memoire(params, a=None, quoi="B"):
    """Returns (K̂(p), C∞, λ) ready for `inverse_laplace`.

    C∞ = K(0⁺) = lim_{p→∞} p·K̂(p) = X(+∞) − X^e   (X = B or C)
    λ  = μ_m/β_m, the relaxation rate of the matrix: it is the pole of
         r(p) = M_i(p)/M_m(p), hence the natural time scale of the kernel.
    """
    coef = coefficient_B if quoi == "B" else coefficient_C
    X_e = coef(0.0, params, a).real
    X_inf = coef(params.p_large, params, a).real
    K_hat = lambda p: (coef(p, params, a) - X_e) / p  # noqa: E731  (B^v = 0)
    lam = 1.0 / params.visc_ratio                     # μ_m / β_m
    return K_hat, X_inf - X_e, lam


def prony_fit(t, K_t, n_exp=3, maxiter=10000):
    """Fits K(t) ≈ Σ α_n exp(-t/τ_n) in the least-squares sense.

    Parameters
    ----------
    t     : array of times
    K_t   : array of kernel values at the times `t`
    n_exp : number of exponentials (internal variables)

    Returns
    -------
    (alphas, taus, residu) — alphas > 0 and taus > 0 guaranteed.

    ⚠ Nelder-Mead optimizer on 2·n_exp parameters, without multi-start:
    the result depends on the initialization and is not guaranteed to be
    global. Reproducible because fully deterministic (no randomness).
    """
    from scipy.optimize import minimize

    t = np.asarray(t, dtype=float)
    K_t = np.asarray(K_t, dtype=float)

    def residual(x):
        alphas = np.exp(x[:n_exp])
        taus = np.exp(x[n_exp:])
        K_approx = np.zeros_like(t)
        for n in range(n_exp):
            K_approx += alphas[n] * np.exp(-t / taus[n])
        return np.sum((K_approx - K_t) ** 2)

    x0 = np.concatenate([
        np.log(np.maximum(np.abs(K_t[0]) / n_exp, 1e-10)) * np.ones(n_exp),
        np.log(np.linspace(t[-1] / 50, t[-1] / 2, n_exp)),
    ])
    result = minimize(residual, x0, method='Nelder-Mead',
                      options={'maxiter': maxiter, 'xatol': 1e-10, 'fatol': 1e-12})

    alphas = np.exp(result.x[:n_exp])
    taus = np.exp(result.x[n_exp:])
    return alphas, taus, result.fun


def prony_laplace(p, alphas, taus):
    """Exact Laplace transform of the Prony kernel Σ αₙ e^{−t/τₙ}:

        K̂(p) = Σ αₙ τₙ / (1 + p τₙ)

    Used to evaluate, in the frequency domain, the model actually
    implemented in the time domain (internal variables). Comparing this
    model with the exact-kernel model measures the error of the Prony
    approximation, and nothing else.
    """
    alphas = np.asarray(alphas, dtype=float)
    taus = np.asarray(taus, dtype=float)
    return np.sum(alphas * taus / (1.0 + p * taus))


def prony_eval(t, alphas, taus):
    """Evaluates Σ α_n exp(-t/τ_n)."""
    t = np.asarray(t, dtype=float)
    out = np.zeros_like(t)
    for a, tau in zip(alphas, taus):
        out += a * np.exp(-t / tau)
    return out


# =============================================================================
# 7. Validation: passivity energy
# =============================================================================

def check_passivity(K_hat_func, p_test=None):
    """Checks that Re[p K̂(p)] ≥ 0 for Re(p) > 0 (passivity condition).

    Returns (is_passive, min_real_part).
    """
    if p_test is None:
        # Test on a grid of points in the right half-plane
        re_p = np.logspace(-2, 2, 20)
        im_p = np.linspace(-10, 10, 20)
        p_test = [r + 1j * i for r in re_p for i in im_p]

    min_real = np.inf
    for p in p_test:
        if p.real > 0:
            val = p * K_hat_func(p)
            if val.real < min_real:
                min_real = val.real

    return min_real >= -1e-8, min_real


# =============================================================================
# 8. Reflection and transmission coefficients
# =============================================================================
#
# Homogenized problem for an incident plane wave:
#   U^inc(X) = exp(i k (X_1 cos θ + X_2 sin θ))
#
# Solution in the matrix (X_1 < 0):
#   U = exp(i k X_2 sin θ) [ exp(i k X_1 cos θ) + R exp(-i k X_1 cos θ) ]
# Solution in the matrix (X_1 > a):
#   U = exp(i k X_2 sin θ) T exp(i k (X_1 - a) cos θ)
#
# Jump conditions (symmetric case, in the Laplace domain with p=-iω+ε):
#   [[Û]]_a = h B(p,a) ⟨∂_{X_1} Û⟩_a
#   [[Σ̂_1]]_a = h S(a) ⟨div Σ̂⟩_a - h C(p,a) ⟨∂_{X_2} Σ̂_2⟩_a
# =============================================================================

def compute_R_T_homogenized(omega, theta, params, a=None, with_memory=True,
                            with_C=True, C_override=None, B_override=None):
    """R and T of the HOMOGENIZED model (memory interface), any incidence.

    GEOMETRY — The effective interface, of thickness a, has its LEFT FACE
    aligned with that of the row (which occupies |X₁| < e/2). It therefore
    occupies [−e/2, −e/2 + a], as in Marigo et al. (2017):

        X₁ < −e/2     :  U = e^{i k₂ X₂} ( e^{i k₁ X₁} + R e^{−i k₁ X₁} )
        X₁ > −e/2 + a :  U = e^{i k₂ X₂} · T e^{i k₁ X₁}

    with k = ω √(ρ_m/M_m(p)), k₂ = k sin θ (Snell invariant) and
    k₁ = √(k² − k₂²). All are complex as soon as β_m > 0.

    ⚠ The article (1st version) placed the interface on [0, a], hence
    OFF-CENTER by half a step. The shift equals a/2 = O(h): it introduces
    into R and T a phase e^{ik₁a/2} = 1 + O(η), i.e. an error of the SAME
    ORDER as the interface effect the model claims to capture.

    ALGEBRA — since Σ = M_m ∇U, we have div Σ = −M_m k² U and
    ∂₂Σ₂ = −M_m k₂² U. The jump conditions

        [[U]]  = h B(p,a) ⟨∂₁U⟩ ,
        [[Σ₁]] = h S(a) ⟨div Σ⟩ − h C(p,a) ⟨∂₂Σ₂⟩ = −h M_m (S k² − C k₂²) ⟨U⟩

    give a fully DECOUPLED system. Setting u = e^{i k₁ a/2},

        T − R = u⁻² (1 + α)/(1 − α),   α = i k₁ h B(p,a) / 2
        T + R = u⁻² (1 + γ)/(1 − γ),   γ = i h (S(a) k² − C(p,a) k₂²) / (2 k₁)

    THE IMPORTANT POINT — **C enters only through k₂² = k² sin²θ.**
    At normal incidence (θ = 0), it VANISHES identically: γ reduces to
    i h S k / 2. This is why figures 9 to 11, all at θ = 0, did not test C
    — and why two gross errors on C (a modal correction identically zero,
    a volume term counted twice) could go unnoticed.

    `with_memory=False` : B(p,a) → B^e + p B^v (instantaneous model).
    `with_C=False`      : C(p,a) → 0. Serves to MEASURE the contribution of C.
    `C_override`        : imposes a value of C. Serves to show that a WRONG
                          value of C (that of the legacy code, C^e = 13.0
                          instead of 3.58) degrades the model measurably.
    `B_override`        : imposes B(p). Serves to evaluate the model in which
                          the memory kernel is replaced by its Prony
                          approximation — that is, the model actually
                          implemented in a time-domain code, with internal
                          variables.
    """
    if a is None:
        a = params.e_over_h

    p = -1j * omega
    M_m = matrix_modulus(p, params)
    k = omega * np.sqrt(1.0 / M_m)          # ρ_m = 1
    k2 = k * np.sin(theta)
    k1 = np.sqrt(k ** 2 - k2 ** 2)
    if k1.imag < 0:
        k1 = -k1
    h = 1.0

    if B_override is not None:
        B = B_override
    elif with_memory:
        B = coefficient_B(p, params, a)
    else:
        B_e, B_v, _ = decompose_B(params, a)
        B = B_e + p * B_v
    S = coefficient_S_full(params, a)
    if C_override is not None:
        C = C_override
    elif with_C:
        C = coefficient_C(p, params, a)
    else:
        C = 0.0

    alpha = 1j * k1 * h * B / 2.0
    gamma = 1j * h * (S * k ** 2 - C * k2 ** 2) / (2.0 * k1)

    # INTERFACE PLACEMENT — this is not an innocent convention.
    # The interface of thickness a occupies [x_L, x_L + a]. The choice of x_L
    # is free up to O(η) and does not change the coefficients, but it changes
    # the error constant at O(η²) — hence the optimum in a.
    #
    # Marigo et al. (2017) align the LEFT FACE of the interface with that of
    # the row. This is what we do: x_L = −e/2 (the row occupying
    # |X₁| < e/2). With this choice, and only this one, a = e is
    # SUPERCONVERGENT: the relative error on R is O(η²) there, versus O(η)
    # for any other a.
    #
    # Centering the interface on the row (x_L = −a/2) is another convention,
    # equally legitimate, but it destroys this superconvergence.
    x_L = -params.e_over_h / 2.0

    somme = (1.0 + alpha) / (1.0 - alpha)
    diff = (1.0 + gamma) / (1.0 - gamma)
    R = np.exp(2j * k1 * x_L) * (diff - somme) / 2.0
    T = np.exp(-1j * k1 * a) * (somme + diff) / 2.0
    return R, T


def compute_R_T_reference(omega, theta, params):
    """R and T of the REAL problem: scattering by the row of inclusions.

    Finite elements with quasi-periodic Bloch conditions and exact
    radiation at Rayleigh order 0 — see `scattering_fem.py`.
    """
    from .scattering_fem import R_T_reference
    return R_T_reference(omega, params, theta=theta, X=params.fem_L / 2.0,
                         raffinement=params.fem_raffinement)


def compute_R_T_real(omega, theta, params, N_inclusions=5):
    """⚠ LEGACY SOLVER (defect D1) — do not use for validation.

    Despite its name, this function does NOT solve the real problem: it
    replaces the row of inclusions by an equivalent HOMOGENEOUS LAYER
    (Reuss/Voigt averages) and applies the reflection formula of a thin
    layer — a second approximate model, insensitive to the fine geometry.
    It is kept only as an executable witness of defect D1 and to
    reproduce the figures of the first version. The correct reference
    is `scattering_fem.R_T_reference` (finite elements on the actual
    row).

    Returns (R, T) such that:
        U(X_1 < 0) = exp(i k X) + R exp(i k' X)  (reflected)
        U(X_1 > e) = T exp(i k (X - e e_1))      (transmitted)
    """
    # For an infinite periodic row of inclusions, the problem is
    # equivalent to the homogenized model at first order in η = kh.
    # The correction is O(η²).

    # For validation, the exact problem is solved by the multimodal method:
    # Rayleigh-Bloch mode decomposition in the matrix,
    # guided modes in the inclusion.

    # To first approximation (single-mode), the exact formula is:
    # R = (i Z_eff sin(k_eff e)) / (2 Z_m cos(k_eff e) + i Z_eff sin(k_eff e))
    # T = (2 Z_m) / (2 Z_m cos(k_eff e) + i Z_eff sin(k_eff e))
    # where Z_eff = sqrt(ρ_eff μ_eff) is the effective impedance of the layer
    # and Z_m = sqrt(ρ_m μ_m) is the impedance of the matrix.

    # For our problem, the inclusion is dispersed in a viscoelastic matrix.
    # The effective impedance is computed by volume averaging:
    # 1/Z_eff² = ϕ/Z_i² + (1-ϕ)/Z_m²   (harmonic mean for parallel phases)
    # or Z_eff² = ϕ Z_i² + (1-ϕ) Z_m²  (arithmetic mean for series phases)

    # In the quasi-static regime (kh ≪ 1), it is the harmonic mean that
    # applies for the direction perpendicular to the layers (X_1):
    # 1/M_eff = ϕ/M_i + (1-ϕ)/M_m
    # ρ_eff = ϕ ρ_i + (1-ϕ) ρ_m

    p = -1j * omega
    M_m = matrix_modulus(p, params)
    M_i = inclusion_modulus(p, params)

    # Harmonic mean for M_eff (perpendicular)
    phi = params.phi / params.e_over_h  # volume fraction of the inclusion in the layer
    # phi here is (e_incl/h) * (h_incl/h) / (e/h) = phi_incl * h_incl / e
    # Actually, params.phi = h_incl/h (nondimensional width)
    # and e_over_h = e/h (nondimensional thickness)
    # The volume fraction of the inclusion in the layer of thickness e is:
    # f = (e/2h * 2 * h_incl/h) / (e/h) = h_incl/h = params.phi
    # Hence f = params.phi
    f = params.phi

    # Harmonic mean (Reuss bound) for M_eff:
    M_eff = 1.0 / (f / M_i + (1 - f) / M_m)
    # Arithmetic mean (Voigt bound) for ρ_eff:
    rho_eff = f * params.rho_ratio + (1 - f) * 1.0

    # Effective wavenumber in the layer
    k_eff_layer = omega * np.sqrt(rho_eff / M_eff)

    # Layer thickness (nondimensionalized by h)
    e_layer = params.e_over_h

    # Impedances
    Z_m = np.sqrt(1.0 * M_m)  # ρ_m = 1, M_m
    Z_eff = np.sqrt(rho_eff * M_eff)

    # Reflection and transmission coefficients for a layer of thickness e
    # between two identical media (matrix on both sides):
    # (standard formula for thin acoustic/elastic layers)
    # R = (i (Z_eff² - Z_m²) sin(k_eff_layer * e_layer)) / (2 Z_eff Z_m cos(k_eff_layer * e_layer) + i (Z_eff² + Z_m²) sin(k_eff_layer * e_layer))
    # T = (2 Z_eff Z_m) / (2 Z_eff Z_m cos(k_eff_layer * e_layer) + i (Z_eff² + Z_m²) sin(k_eff_layer * e_layer))

    # For oblique incidence, replace:
    # Z_m → Z_m / cos(theta)
    # Z_eff → Z_eff / cos(theta_eff)
    # k_eff_layer * e_layer → k_eff_layer * e_layer * cos(theta_eff)
    # where sin(theta_eff) = (k_m / k_eff_layer) sin(theta)

    # For simplicity, normal incidence is taken (theta = 0):
    # sin(theta) = 0 → theta_eff = 0

    if abs(np.sin(theta)) < 1e-10:
        # Normal incidence
        denom = 2 * Z_eff * Z_m * np.cos(k_eff_layer * e_layer) + 1j * (Z_eff**2 + Z_m**2) * np.sin(k_eff_layer * e_layer)
        R = 1j * (Z_eff**2 - Z_m**2) * np.sin(k_eff_layer * e_layer) / denom
        T = 2 * Z_eff * Z_m / denom
    else:
        # Oblique incidence: angular-average approximation
        # For the test case, normal incidence is used with an empirical
        # correction for small theta.
        denom = 2 * Z_eff * Z_m * np.cos(k_eff_layer * e_layer * np.cos(theta)) + 1j * (Z_eff**2 + Z_m**2) * np.sin(k_eff_layer * e_layer * np.cos(theta))
        R = 1j * (Z_eff**2 - Z_m**2) * np.sin(k_eff_layer * e_layer * np.cos(theta)) / denom
        T = 2 * Z_eff * Z_m / denom

    return R, T


# =============================================================================
# 9. Tests
# =============================================================================
#
# The old `self_test()` (assertions with tolerance 1.0, run via
# `python common.py`) has been replaced by a real test suite:
#
#     pytest            # or:  python run.py test
#
# See tests/ — analytic limits, passivity, non-regression.
# =============================================================================
