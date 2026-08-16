"""Solution of the cell problem with Q1 finite elements.

WHY THIS MODULE EXISTS
-------------------------
The paper (§8.1) describes a finite-element solution of the cell
problem. That solver did not exist: `physics.py` (inherited from `common.py`)
used two closed-form formulas instead, both of them wrong —

  * `cell_problem_W1`: modulus ratio INVERTED.
    The exact derivation of mode n = 0 gives
        M(y1) (V' ) = const,  V = W + y1,  V' -> 1 as |y1| -> inf
        => const = M_m,  and inside the inclusion  W' = M_m/M_i - 1
        => B_1 = (M_m/M_i - 1) . e/h
    whereas the code computed  (M_i/M_m - 1) . e/h.
    Consequence: B(0) = 13.03 instead of 0.308, memory kernel of opposite sign,
    passivity theorem violated (Re[p K_B] -> -13.03).

  * `_multimode_correction_C2`: identically zero (the two contributions cancel
    term by term), and the volume term counted twice.

This module actually solves the 2D cell problem. It is indispensable:
with mode n = 0 alone and delta = 0, B(p) is AFFINE in p, so the memory
kernel vanishes. **Memory arises exclusively from the two-dimensional nature
of the problem** (the inclusion occupies only a fraction phi of the period: the
field goes around the inclusion through the matrix). This is the Francfort--
Murat--Tartar mechanism: two phases "in parallel", only one of which is viscous.

FORMULATION
-----------
Cell  Y = [-L, L] x [0, 1],  periodic in y2,  inclusion
  |y1| <= e/2h  and  |y2 - 1/2| <= phi/2.

Problem W1 (normal direction).  V = W1 + y1 :
    div( M grad V ) = 0
    V' -> 1  as |y1| -> inf   =>   imposed flux  M_m dV/dn = +-M_m  at y1 = +-L
    B_1 = <V>(L) - <V>(-L) - 2L                    (the gauge constant drops out)

Problem W2 (tangential direction).  W2 periodic :
    div( M grad(W2 + y2) ) = 0,   zero flux at y1 = +-L
    C_2 = integral of (M/M_m) d(W2)/dy2  over Y

SCALE INVARIANCE — the key to efficiency and to the theory
-----------------------------------------------------------
The equation is homogeneous of degree 0 in M: dividing by M_m(p) changes
nothing. The solution therefore depends on p ONLY through the complex contrast

    r(p) = M_i(p) / M_m(p) = (mu_i + p.delta) / (mu_m + p.beta_m).

Hence  B_1(p) = G(r(p))  and  C_2(p) = H(r(p)): two functions of ONE complex
variable. Memory stems from r(p) being a Möbius transform of p and from G, H
being nonlinear. This is also what makes the computation fast (caching).
"""

from __future__ import annotations

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import splu

# 2x2 Gauss quadrature on the reference element [-1, 1]^2
_G = 1.0 / np.sqrt(3.0)
_GAUSS = [(-_G, -_G), (_G, -_G), (_G, _G), (-_G, _G)]


class CellMesh:
    """Structured Q1 mesh, aligned with the edges of the inclusion.

    Alignment is essential: M is discontinuous across the interface, and an
    element straddling the interface would introduce a first-order error.
    """

    def __init__(self, e_over_h: float, phi: float, L: float = 5.0,
                 n1_inc: int = 24, n1_mat: int = 40, n2_inc: int = 24,
                 n2_mat: int = 20):
        self.L = L
        self.e_over_h = e_over_h
        self.phi = phi

        a1 = e_over_h / 2.0        # half-thickness of the inclusion in y1
        a2 = phi / 2.0             # half-width in y2, centered at y2 = 1/2

        def _axe(bornes, n_par_segment):
            """Nodes aligned with `bornes`; zero-length segments are skipped
            (case phi = 1: there is no matrix left along y2)."""
            noeuds = []
            for (d, f), n in zip(zip(bornes[:-1], bornes[1:]), n_par_segment):
                if f - d > 1e-12:
                    noeuds.append(np.linspace(d, f, n, endpoint=False))
            return np.concatenate(noeuds)

        # --- y1 axis ----------------------------------------------------------
        # The corrector is evanescent: it decays as exp(-2.pi.|y1|) as soon as
        # one leaves the inclusion, but it has a CORNER SINGULARITY at the four
        # corners of the inclusion. Two conflicting requirements:
        #   - mesh finely next to the inclusion (corners);
        #   - go far in y1 (truncation) without blowing up the node count.
        # Hence a GEOMETRIC progression starting from a first cell of size h0
        # set by the inclusion mesh — therefore INDEPENDENT of L. Otherwise,
        # increasing L coarsens the corners, and a meshing error would wrongly
        # be attributed to truncation.
        h0 = a1 / n1_inc                    # cell size inside the inclusion
        q = 1.18                            # geometric common ratio
        noeuds, y, pas = [], a1, h0
        while y < L - 1e-12:
            y = min(y + pas, L)
            noeuds.append(y)
            pas *= q
        matrice = np.array(noeuds)          # (a1, L]

        y1 = np.concatenate([
            -matrice[::-1],                                  # [-L, -a1)
            np.linspace(-a1, a1, 2 * n1_inc + 1),            # [-a1, a1] WITH the edges
            matrice,                                         # (a1, L]
        ])
        # --- y2 axis: periodic (no duplicated node at y2 = 1) -----------------
        y2 = _axe([0.0, 0.5 - a2, 0.5 + a2, 1.0], [n2_mat, 2 * n2_inc, n2_mat])

        self.y1, self.y2 = y1, y2
        self.n1, self.n2 = len(y1), len(y2)      # y2: periodic, no duplicated node
        self.n_nodes = self.n1 * self.n2

        # --- elements ---------------------------------------------------------
        elems, in_inclusion = [], []
        for i in range(self.n1 - 1):
            for j in range(self.n2):
                jp = (j + 1) % self.n2
                elems.append([self.node(i, j), self.node(i + 1, j),
                              self.node(i + 1, jp), self.node(i, jp)])
                yc1 = 0.5 * (y1[i] + y1[i + 1])
                # center in y2: beware of the last element, which closes the period
                y2b = y2[jp] if jp != 0 else 1.0
                yc2 = 0.5 * (y2[j] + y2b)
                in_inclusion.append(abs(yc1) < a1 - 1e-12
                                    and abs(yc2 - 0.5) < a2 - 1e-12)

        self.elems = np.array(elems, dtype=int)
        self.in_inclusion = np.array(in_inclusion, dtype=bool)

        # physical coordinates of the 4 nodes of each element
        xy = []
        for i in range(self.n1 - 1):
            for j in range(self.n2):
                jp = (j + 1) % self.n2
                b = y2[jp] if jp != 0 else 1.0
                xy.append([[y1[i], y2[j]], [y1[i + 1], y2[j]],
                           [y1[i + 1], b], [y1[i], b]])
        self.elem_xy = np.array(xy)

    def node(self, i: int, j: int) -> int:
        return i * self.n2 + (j % self.n2)

    # ---- elementary operators ----------------------------------------------
    def _element_matrices(self):
        """Elementary stiffness matrices and vectors int(dphi/dy2)."""
        ne = len(self.elems)
        Ke = np.zeros((ne, 4, 4))
        Fe2 = np.zeros((ne, 4))          # integral of d(phi_a)/dy2  over the element
        for a, (xi, eta) in enumerate(_GAUSS):
            dN = np.array([
                [-(1 - eta), -(1 - xi)],
                [(1 - eta), -(1 + xi)],
                [(1 + eta), (1 + xi)],
                [-(1 + eta), (1 - xi)],
            ]) * 0.25
            # Jacobian of the mapping (rectangular element)
            J = np.einsum("ak,eaj->ekj", dN, self.elem_xy)
            detJ = J[:, 0, 0] * J[:, 1, 1] - J[:, 0, 1] * J[:, 1, 0]
            Jinv = np.zeros_like(J)
            Jinv[:, 0, 0] = J[:, 1, 1] / detJ
            Jinv[:, 1, 1] = J[:, 0, 0] / detJ
            Jinv[:, 0, 1] = -J[:, 0, 1] / detJ
            Jinv[:, 1, 0] = -J[:, 1, 0] / detJ
            gradN = np.einsum("ak,ekj->eaj", dN, Jinv)        # (ne, 4, 2)
            Ke += np.einsum("eaj,ebj,e->eab", gradN, gradN, detJ)
            Fe2 += np.einsum("ea,e->ea", gradN[:, :, 1], detJ)
        return Ke, Fe2

    def _precalcul(self):
        """Everything that does NOT depend on r, computed only once.

        The operator is AFFINE in r:
            K(r) = K_matrix + r . K_inclusion
            f2(r) = f2_matrix + r . f2_inclusion
        and f1 does not depend on r at all (flux imposed inside the matrix).
        Solving for a new r therefore costs only one sum and one solve.
        """
        if hasattr(self, "_pre"):
            return
        Ke, Fe2 = self._element_matrices()
        rows = np.repeat(self.elems, 4, axis=1).ravel()
        cols = np.tile(self.elems, (1, 4)).ravel()
        inc = self.in_inclusion

        def _K(masque):
            v = (Ke * masque[:, None, None]).ravel()
            return coo_matrix((v, (rows, cols)),
                              shape=(self.n_nodes, self.n_nodes)).tocsr()

        def _f2(masque):
            f = np.zeros(self.n_nodes)
            np.add.at(f, self.elems.ravel(), -(Fe2 * masque[:, None]).ravel())
            return f

        # imposed flux dV/dn = ±1 at y1 = ±L (only the matrix lives there: M/M_m = 1)
        f1 = np.zeros(self.n_nodes)
        y2e = np.append(self.y2, 1.0)
        for j in range(self.n2):
            dy = y2e[j + 1] - y2e[j]
            jp = (j + 1) % self.n2
            f1[self.node(self.n1 - 1, j)] += dy / 2.0
            f1[self.node(self.n1 - 1, jp)] += dy / 2.0
            f1[self.node(0, j)] -= dy / 2.0
            f1[self.node(0, jp)] -= dy / 2.0

        Km_brut, Ki_brut = _K((~inc).astype(float)), _K(inc.astype(float))
        f2m, f2i = _f2((~inc).astype(float)), _f2(inc.astype(float))

        # Gauge: the CELL problem is pure Neumann + periodic, so its kernel is
        # the space of constants. We pin one node once and for all, directly in
        # Km and Ki: row `pin` becomes e_pin in K(r) = Km + r.Ki, for every r.
        # (Pinning at each call via .tolil() cost 0.5 s.) The computed
        # quantities — jumps, gradient integrals — do not depend on the gauge
        # constant.
        #
        # ⚠ The RAW matrices are kept separately: the SCATTERING problem
        # (scattering_fem.py) has NO gauge to fix — its radiation conditions
        # make the operator invertible. Reinjecting a pinned row there would
        # destroy the equation at node 0.
        pin = 0
        Km, Ki = Km_brut.copy(), Ki_brut.copy()
        for A in (Km, Ki):
            A.data[A.indptr[pin]:A.indptr[pin + 1]] = 0.0
            A.eliminate_zeros()
        Km = Km.tolil()
        Km[pin, pin] = 1.0
        Km = Km.tocsr()

        f1_pin, f2m_pin, f2i_pin = f1.copy(), f2m.copy(), f2i.copy()
        f1_pin[pin] = f2m_pin[pin] = f2i_pin[pin] = 0.0

        self._pre = {"Km": Km, "Ki": Ki,
                     "f2m": f2m_pin, "f2i": f2i_pin, "f1": f1_pin,
                     "Km_brut": Km_brut, "Ki_brut": Ki_brut,
                     "Fe2": Fe2, "pin": pin}

    def assemble(self, r: complex):
        """K(r) and right-hand sides, with M/M_m = 1 (matrix) and r (inclusion)."""
        self._precalcul()
        P = self._pre
        K = (P["Km"] + r * P["Ki"]).astype(complex)
        f1 = P["f1"].astype(complex)
        f2 = (P["f2m"] + r * P["f2i"]).astype(complex)
        return K, f1, f2

    # ---- solution -----------------------------------------------------------
    def solve(self, r: complex) -> dict:
        """Solves the two cell problems for a contrast r = M_i/M_m."""
        K, f1, f2 = self.assemble(r)

        # A single factorization for BOTH cell problems: they differ only by
        # their right-hand side.
        lu = splu(K.tocsc())
        V = lu.solve(f1)        # V = W1 + y1
        W2 = lu.solve(f2)

        # --- B_1 = <V>(+L) - <V>(-L) - 2L ------------------------------------
        w = self._poids_y2()
        VL = np.sum(V[self.node(self.n1 - 1, 0):
                      self.node(self.n1 - 1, 0) + self.n2] * w)
        Vm = np.sum(V[self.node(0, 0):self.node(0, 0) + self.n2] * w)
        B1 = VL - Vm - 2 * self.L

        # --- B_2 = <W2>(+L) - <W2>(-L)  (must vanish by symmetry) -------------
        W2L = np.sum(W2[self.node(self.n1 - 1, 0):
                        self.node(self.n1 - 1, 0) + self.n2] * w)
        W2m = np.sum(W2[self.node(0, 0):self.node(0, 0) + self.n2] * w)
        B2 = W2L - W2m

        # --- C_2 = int (M/M_m) dW2/dy2 ;  C_1 = int (M/M_m) dW1/dy2 -----------
        C2 = self._integrale_grad_y2(W2, r)
        W1 = V - self._y1_nodal()
        C1 = self._integrale_grad_y2(W1, r)

        # --- energy identities of Marigo & Maurel (2017) ----------------------
        # These are EXACT equalities, not approximations. They serve as an
        # internal check: if they are not satisfied to quadrature accuracy,
        # the cell solution is wrong.
        #
        #   eq. (48)+(49):  B₁ = (e/h)·φ·(1 − r) + ∫ (M/M_m)|∇W⁽¹⁾|²
        #   eq. (53)     :  C₂ = −∫ (M/M_m)|∇W⁽²⁾|²        (hence C₂ ≤ 0 always)
        #
        # In the VISCOELASTIC setting, r is complex and these identities remain
        # exact algebraic equalities, with complex values.
        e_W1 = self._energie(W1, r)
        e_W2 = self._energie(W2, r)
        B1_identite = self.e_over_h * self.phi * (1.0 - r) + e_W1
        C2_identite = -e_W2

        return {"B1": B1, "B2": B2, "C1": C1, "C2": C2, "r": r,
                "B1_identite": B1_identite, "C2_identite": C2_identite,
                "residu_B1": abs(B1 - B1_identite),
                "residu_C2": abs(C2 - C2_identite)}

    def _energie(self, u, r: complex) -> complex:
        """∫_Y (M/M_m) |∇u|² dy — the energy appearing in the identities."""
        self._precalcul()
        P = self._pre
        K = (P["Km_brut"] + r * P["Ki_brut"]).astype(complex)
        return complex(u @ (K @ u))

    # ---- utilities -----------------------------------------------------------
    def _poids_y2(self):
        """Periodic trapezoidal weights in y2 (sum = 1)."""
        y2e = np.append(self.y2, 1.0)
        dy = np.diff(y2e)
        w = np.zeros(self.n2)
        for j in range(self.n2):
            w[j] = 0.5 * (dy[j] + dy[j - 1])
        return w

    def _y1_nodal(self):
        return np.repeat(self.y1, self.n2)

    def _integrale_grad_y2(self, u, r: complex) -> complex:
        """Integral over Y of (M/M_m) . du/dy2, by elementary quadrature."""
        Fe2 = self._pre["Fe2"]
        M = np.where(self.in_inclusion, r, 1.0).astype(complex)
        ue = u[self.elems]                      # (ne, 4)
        return complex(np.sum(M * np.sum(Fe2 * ue, axis=1)))


# =============================================================================
# High-level interface, with a cache on the contrast r
# =============================================================================

_MAILLAGES: dict = {}
_SOLUTIONS: dict = {}     # exact cache on (r, mesh)


def cell_coefficients(r: complex, e_over_h: float, phi: float,
                      L: float = 5.0, raffinement: int = 1) -> dict:
    """B_1(r), B_2(r), C_1(r), C_2(r) for a complex contrast r = M_i/M_m.

    Two caches:
      - the mesh and its matrices, independent of r;
      - the SOLUTION, indexed exactly by r.

    The second cache is indispensable: `decompose_B(a)` evaluates B(0) and
    B(+inf) for each value of `a`, whereas the cell does not depend on a at all
    (a only enters through the additive shift a/h). Without this cache, the
    sweep over a in block 4 would rerun the SAME finite-element computation
    100 times.
    """
    cle_maillage = (e_over_h, phi, L, raffinement)
    if cle_maillage not in _MAILLAGES:
        _MAILLAGES[cle_maillage] = CellMesh(
            e_over_h, phi, L,
            n1_inc=12 * raffinement, n1_mat=20 * raffinement,
            n2_inc=12 * raffinement, n2_mat=10 * raffinement,
        )
    r = complex(r)
    cle = (r, cle_maillage)
    if cle not in _SOLUTIONS:
        _SOLUTIONS[cle] = _MAILLAGES[cle_maillage].solve(r)
    return _SOLUTIONS[cle]


def herglotz_identity(r: complex, e_over_h: float, phi: float,
                      L: float = 5.0, raffinement: int = 1) -> dict:
    """Exact Herglotz identity — both sides, and their relative gaps.

        Im B(r) = -Im(r) . int_{Omega_i} |grad V1|^2,    V1 = W1 + y1
        Im C(r) = +Im(r) . int_{Omega_i} |grad V2|^2,    V2 = W2 + y2

    (Full C: volume term (e.phi/h)(r-1) included, real a/h has no effect.)
    This is the continuation to the complex plane of the envelope formulas
    dB/dr = -int|grad V1|^2, dC/dr = +int|grad V2|^2: it proves that -B and C
    are Herglotz, hence that their Stieltjes measures are POSITIVE — the key
    link of the passivity theorem on the half-plane (paper, Herglotz prop.;
    supplement S7). The sesquilinear integrals over the inclusion are evaluated
    with the restricted stiffness matrix Ki (unit coefficient).
    """
    cle_maillage = (e_over_h, phi, L, raffinement)
    if cle_maillage not in _MAILLAGES:
        _MAILLAGES[cle_maillage] = CellMesh(
            e_over_h, phi, L,
            n1_inc=12 * raffinement, n1_mat=20 * raffinement,
            n2_inc=12 * raffinement, n2_mat=10 * raffinement,
        )
    mesh = _MAILLAGES[cle_maillage]
    r = complex(r)

    K, f1, f2 = mesh.assemble(r)
    lu = splu(K.tocsc())
    V1 = lu.solve(f1)                       # V1 = W1 + y1 (cf. solve())
    W2 = lu.solve(f2)
    V2 = W2 + np.tile(mesh.y2, mesh.n1)

    mesh._precalcul()
    Ki = mesh._pre["Ki_brut"]               # stiffness on the inclusion, coeff. 1
    int_V1 = float((np.conj(V1) @ (Ki @ V1)).real)   # int_{Omega_i} |grad V1|^2
    int_V2 = float((np.conj(V2) @ (Ki @ V2)).real)

    d = mesh.solve(r)
    im_B = d["B1"].imag                     # Im B = Im B1 (a/h real)
    im_C = e_over_h * phi * r.imag + d["C2"].imag    # volume + cell terms

    rhs_B = -r.imag * int_V1
    rhs_C = +r.imag * int_V2
    err_B = abs(im_B - rhs_B) / max(abs(im_B), 1e-14)
    err_C = abs(im_C - rhs_C) / max(abs(im_C), 1e-14)
    return {"im_B": im_B, "rhs_B": rhs_B, "err_B": err_B,
            "im_C": im_C, "rhs_C": rhs_C, "err_C": err_C,
            "int_V1": int_V1, "int_V2": int_V2, "r": r}
