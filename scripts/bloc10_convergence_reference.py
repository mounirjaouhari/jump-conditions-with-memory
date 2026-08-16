"""Block 10 — convergence of the REFERENCE solver (debt A8).

The paper cited a "precision floor" of the reference (~1e-4) without having
measured it: a load-bearing claim — the floor is plotted on figure 11 and
explains why the errors saturate — but without proof. This block replaces it
with a MEASUREMENT, separating the two error sources of the scattering
solver:

  (i)  sensitivity to the TRUNCATION, measured on |R| by differences between
       consecutive truncations, at the production refinement;
  (ii) DISCRETIZATION error, measured on complex R AT FIXED X: swept
       refinement, deviation from an overkill mesh; order limited by the
       corner singularities.

TWO METHODOLOGICAL LESSONS, learned while measuring (recorded in the CHANGELOG)
-------------------------------------------------------------------------------
1. The far-field mesh follows a geometric progression (designed for the
   evanescent correctors of the cell problem): the elements GROW with the
   distance to the row. The PHASE of R therefore accumulates a discrete
   dispersion which GROWS with X (~1e-2 at kh = 0.4): comparing complex R
   between two truncations measures this drift, not the evanescent modes.
   Discretization is therefore assessed on complex R between two meshes
   that traverse the SAME far field (identical X).
2. Even on |R|, the e^{-2πX} decay of the evanescent orders is NOT
   observable on this mesh: beyond X = 3, the added elements are so coarse
   that |R| DEGRADES as X increases (measured: up to ~4e-3 between X = 4
   and 6 at kh = 0.4). X = 3 is an optimum: the truncation sensitivity is
   already below the discretization floor there, and pushing X further
   costs more in resolution than it gains in truncation.
   This is the quantitative justification of the production setting.

The measured floor is the deviation of the PRODUCTION settings (those of
blocks 3 to 6) from the overkill solution at the same truncation, in the
SAME metric as figure 11: E_R = |R - R_over| / |R_over|, over the worst
(kh, θ) pair.

Added to this is the analogous table for the CELL solver (B1, C2 as a
function of refinement): the discussion clause "mesh convergence is limited
to ~3e-4 by the re-entrant corners" also becomes a measurement.

Output: fig21_convergence_reference.pdf (supplement, S8)
Data  : results/<config>/bloc10/*.csv + manifest.json

    python run.py bloc10 [--config config/xxx.yaml]
"""

import numpy as np

from fading_memory import RunRecorder, load_config
from fading_memory.cell_fem import cell_coefficients
from fading_memory.plotting import DOUBLE, apply_style, plt
from fading_memory.scattering_fem import R_T_reference


def _E_R(R, R_over):
    """Relative deviation from the overkill solution — metric of figure 11."""
    return abs(R - R_over) / abs(R_over)


def main(cfg=None):
    cfg = cfg or load_config()
    apply_style()
    p_ = cfg.params()
    b = cfg.bloc("bloc10")

    raffs = [int(r) for r in b["raffinements"]]
    Xs = [float(x) for x in b["X_values"]]
    khs = [float(k) for k in b["kh_values"]]
    thetas = [float(t) for t in b["theta_deg"]]
    raf_ref = int(b["raffinement_ref"])
    raf_prod, X_prod = int(b["raffinement_production"]), float(b["X_production"])
    bruit = float(b["bruit_fit"])
    combos = [(kh, th) for kh in khs for th in thetas]

    with RunRecorder("bloc10", cfg) as run:
        # --- overkill solutions: mesh raf_ref, at the PRODUCTION truncation
        # (discretization deviations are measured between two meshes
        # traversing the SAME far field — see docstring).
        R_over = {}
        for kh, th in combos:
            R_over[(kh, th)], _ = R_T_reference(
                kh, p_, theta=np.deg2rad(th), X=X_prod, raffinement=raf_ref)
            print(f"  overkill  kh={kh:g} θ={th:g}° : |R| = {abs(R_over[(kh, th)]):.6f}")

        # --- (ii) mesh sweep, at fixed X = X_prod -----------------------------
        lignes_m = {"kh": [], "theta_deg": [], "raffinement": [], "E_R": []}
        E_maille = {}
        for kh, th in combos:
            for raf in raffs:
                R, _ = R_T_reference(kh, p_, theta=np.deg2rad(th),
                                     X=X_prod, raffinement=raf)
                e = _E_R(R, R_over[(kh, th)])
                E_maille[(kh, th, raf)] = e
                lignes_m["kh"].append(kh)
                lignes_m["theta_deg"].append(th)
                lignes_m["raffinement"].append(raf)
                lignes_m["E_R"].append(e)
        run.table("convergence_maillage", lignes_m)

        # Observed order: log-log fit over refinements 1..n-1.
        # The finest level of the sweep is EXCLUDED from the fit: its
        # deviation from the reference (a single level above it)
        # underestimates its true error and would artificially inflate
        # the slope.
        slopes = []
        for kh, th in combos:
            rs = [r for r in raffs[:-1] if E_maille[(kh, th, r)] > bruit]
            if len(rs) >= 2:
                xs = np.log([float(r) for r in rs])
                ys = np.log([E_maille[(kh, th, r)] for r in rs])
                slopes.append(-np.polyfit(xs, ys, 1)[0])
        ordre_maille = float(np.median(slopes))
        run.value("refConvOrdreMaille", ordre_maille,
                  "observed order of mesh convergence of the reference "
                  "solver (log-log fit, refinements 1 to "
                  f"{raffs[-2]}, median over (kh, θ))", fmt="%.2f")
        print(f"  mesh: observed order (median) = {ordre_maille:.2f}")

        # --- (i) truncation sensitivity, on |R|, at the production mesh:
        # differences between CONSECUTIVE truncations. (A large-X reference
        # is invalid here: its distant elements, geometrically enlarged,
        # carry their own error — see docstring.)
        absR = {}
        for kh, th in combos:
            for X in Xs:
                R, _ = R_T_reference(kh, p_, theta=np.deg2rad(th),
                                     X=X, raffinement=raf_prod)
                absR[(kh, th, X)] = abs(R)
        lignes_t = {"kh": [], "theta_deg": [], "X_de": [], "X_a": [],
                    "ecart_absR": []}
        d_tronc = {}
        for kh, th in combos:
            for X0, X1 in zip(Xs[:-1], Xs[1:]):
                d = (abs(absR[(kh, th, X0)] - absR[(kh, th, X1)])
                     / absR[(kh, th, X1)])
                d_tronc[(kh, th, X0, X1)] = d
                lignes_t["kh"].append(kh)
                lignes_t["theta_deg"].append(th)
                lignes_t["X_de"].append(X0)
                lignes_t["X_a"].append(X1)
                lignes_t["ecart_absR"].append(d)
        run.table("convergence_troncature", lignes_t)

        tronc_prod = float(max(d_tronc[(kh, th, Xs[0], Xs[1])]
                               for kh, th in combos))
        run.value("refTroncProd", tronc_prod,
                  f"worst relative deviation of |R| between X = {Xs[0]:g} and {Xs[1]:g} "
                  "at the production mesh (truncation sensitivity)",
                  fmt="%.1e")
        degrade_X = float(max(d_tronc[(kh, th, Xs[-2], Xs[-1])]
                              for kh, th in combos))
        run.value("refDegradeX", degrade_X,
                  f"worst relative deviation of |R| between X = {Xs[-2]:g} and {Xs[-1]:g} "
                  "(the geometrically coarsened far field degrades |R| "
                  "as X grows: X = 3 is an optimum)", fmt="%.1e")
        print(f"  truncation: worst deviation X={Xs[0]:g}↔{Xs[1]:g}: {tronc_prod:.1e}; "
              f"degradation X={Xs[-2]:g}↔{Xs[-1]:g}: {degrade_X:.1e}")

        # --- MEASURED floor of the production settings ------------------------
        # Same truncation as the overkill: the deviation measures the
        # discretization (mesh + phase dispersion of the shared far field),
        # in the metric of figure 11.
        E_prod = {}
        for kh, th in combos:
            R, _ = R_T_reference(kh, p_, theta=np.deg2rad(th),
                                 X=X_prod, raffinement=raf_prod)
            E_prod[(kh, th)] = _E_R(R, R_over[(kh, th)])
            print(f"  production kh={kh:g} θ={th:g}° : E_R = {E_prod[(kh, th)]:.2e}")
        plancher = float(max(E_prod.values()))
        run.value("plancherMesure", plancher,
                  "MEASURED precision floor of the production reference "
                  "(worst relative deviation of R from the overkill solution)",
                  fmt="%.1e")
        run.table("plancher_production", {
            "kh": [kh for kh, _ in combos],
            "theta_deg": [th for _, th in combos],
            "E_R": [E_prod[c] for c in combos],
        })
        print(f"  measured floor (production, worst case): {plancher:.1e}")

        # --- analogous table for the cell solver ------------------------------
        r_stat = p_.mu_ratio          # elastic limit: r = mu_i / mu_m
        cell_raffs = [int(r) for r in b["cell_raffinements"]]
        B1s, C2s = [], []
        for raf in cell_raffs:
            d = cell_coefficients(r_stat, p_.e_over_h, p_.phi,
                                  L=p_.fem_L, raffinement=raf)
            B1s.append(d["B1"].real)
            C2s.append(d["C2"].real)
        ecarts_B1 = [abs(bb - B1s[-1]) for bb in B1s[:-1]]
        ecarts_C2 = [abs(cc - C2s[-1]) for cc in C2s[:-1]]
        # Observed order on B1: slope between the last two deviations from
        # the finest mesh.
        ordre_cell = float(np.log(ecarts_B1[-2] / ecarts_B1[-1])
                           / np.log(cell_raffs[-2] / cell_raffs[-3]))
        run.value("cellConvOrdre", ordre_cell,
                  "observed order of mesh convergence of the cell problem "
                  "(B1, limited by the corner singularities)", fmt="%.2f")
        run.value("cellConvEcartFin", float(ecarts_B1[-1]),
                  "deviation of B1 between the two finest cell meshes",
                  fmt="%.1e")
        run.table("convergence_cellule", {
            "raffinement": cell_raffs,
            "B1": B1s,
            "C2": C2s,
            "ecart_B1_au_plus_fin": ecarts_B1 + [0.0],
            "ecart_C2_au_plus_fin": ecarts_C2 + [0.0],
        })
        print(f"  cell: B1 = {B1s[-1]:.6f}, observed order = {ordre_cell:.2f}, "
              f"finest deviation = {ecarts_B1[-1]:.1e}")

        # --- fig21 (supplement S8) --------------------------------------------
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=DOUBLE)
        couleurs = {kh: c for kh, c in zip(khs, ("#1f4e79", "#b2182b", "#1b7837"))}
        styles = {thetas[0]: "-", thetas[-1]: "--"}

        mids = [0.5 * (a + b) for a, b in zip(Xs[:-1], Xs[1:])]
        for kh, th in combos:
            ds = [max(d_tronc[(kh, th, a, b)], 1e-17)
                  for a, b in zip(Xs[:-1], Xs[1:])]
            ax1.semilogy(mids, ds, styles[th], color=couleurs[kh], marker="o",
                         ms=4, label=rf"$kh={kh:g}$, $\theta={th:g}^\circ$")
        Xg = np.linspace(mids[0], mids[-1], 50)
        d0 = max(d_tronc[(kh, th, Xs[0], Xs[1])] for kh, th in combos)
        ax1.semilogy(Xg, d0 * np.exp(-2 * np.pi * (Xg - mids[0])), "k:",
                     lw=1.2, label=r"$\propto e^{-2\pi X}$")
        ax1.set_xlabel(r"$X$ (truncation)")
        ax1.set_ylabel(r"relative deviation of $|R|$ between consecutive truncations")
        ax1.set_title("Truncation sensitivity", fontsize=12)
        ax1.legend(fontsize=7.5, ncol=2)

        for kh, th in combos:
            es = [max(E_maille[(kh, th, raf)], 1e-17) for raf in raffs]
            ax2.loglog(raffs, es, styles[th], color=couleurs[kh], marker="s",
                       ms=4, label=rf"$kh={kh:g}$, $\theta={th:g}^\circ$")
        rg = np.array([raffs[0], raffs[-1]], dtype=float)
        e0 = max(E_maille[(khs[0], thetas[0], raffs[0])], 1e-17)
        ax2.loglog(rg, e0 * (rg / rg[0]) ** (-ordre_maille), "k:", lw=1.2,
                   label=rf"slope $-{ordre_maille:.2f}$")
        ax2.set_xlabel("mesh refinement")
        ax2.set_ylabel(r"$E_R$ (deviation from the overkill solution)")
        ax2.set_title("Mesh error", fontsize=12)
        ax2.set_xticks(raffs)
        ax2.set_xticklabels([str(r) for r in raffs])
        ax2.legend(fontsize=7.5, ncol=2)

        fig.tight_layout()
        run.figure(fig, "fig21_convergence_reference.pdf")
        plt.close(fig)


if __name__ == "__main__":
    main()
