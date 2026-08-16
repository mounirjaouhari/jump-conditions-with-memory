"""Block 1 — Cell coefficients B(p), C(p), S(a).

Produces: fig05_coefficient_B.pdf, fig06_coefficient_C.pdf
Data: results/<config>/bloc1/*.csv + manifest.json

    python run.py bloc1 [--config config/xxx.yaml]
"""

import numpy as np

from fading_memory import RunRecorder, load_config
from fading_memory.marigo_multimodal import B1_multimodal
from fading_memory.physics import (borne_positivite_a, bornes_marigo,
                                   coefficient_B, coefficient_C,
                                   coefficient_S_full, decompose_B, decompose_C,
                                   validations_solveur)
from fading_memory.plotting import SIMPLE, apply_style, plt


def main(cfg=None):
    cfg = cfg or load_config()
    apply_style()
    p_ = cfg.params()
    b = cfg.bloc("bloc1")

    with RunRecorder("bloc1", cfg) as run:
        # --- static limits (p = 0) -------------------------------------------
        run.value("BZero", coefficient_B(0.0, p_), "B(0, a=e) — static limit")
        run.value("CZero", coefficient_C(0.0, p_), "C(0, a=e) — static limit")
        run.value("S", coefficient_S_full(p_), "S(a=e) — inertial coefficient")

        # --- instantaneous / memory decomposition ------------------------------
        B_e, B_v, _ = decompose_B(p_)
        C_e, C_v, _ = decompose_C(p_)
        run.value("Be", B_e, "B^e: instantaneous elastic part", fmt="%.3f")
        run.value("Bv", B_v, "B^v: instantaneous viscosity", fmt="%.3e")
        run.value("Ce", C_e, "C^e: instantaneous elastic part", fmt="%.3f")
        run.value("Cv", C_v, "C^v: instantaneous viscosity", fmt="%.3e")

        print(f"  B(0) = {coefficient_B(0.0, p_):.6f}   B^e = {B_e:.6f}   B^v = {B_v:.3e}")
        print(f"  C(0) = {coefficient_C(0.0, p_):.6f}   C^e = {C_e:.6f}   C^v = {C_v:.3e}")
        print(f"  S    = {coefficient_S_full(p_):.6f}")

        # --- positivity of the interface energy (Marigo et al. 2017) ----------
        # E_a >= 0 as soon as S, B, C >= 0. In the viscoelastic case, the
        # condition becomes Re B(p) >= 0 and Re C(p) >= 0 on the right half-plane.
        import itertools
        minB = minC = float("inf")
        for re_p, im_p in itertools.product([0.01, 1.0, 100.0], [0.0, 2.0, 50.0]):
            pp = re_p + 1j * im_p
            minB = min(minB, coefficient_B(pp, p_, ).real)
            minC = min(minC, coefficient_C(pp, p_).real)
        run.value("minReB", minB, "min Re B(p) on the right half-plane", fmt="%.3f")
        run.value("minReC", minC, "min Re C(p) on the right half-plane", fmt="%.3f")
        print(f"  interface energy: min Re B = {minB:+.4f}   min Re C = {minC:+.4f}")
        if minB < 0 or minC < 0:
            run.note("NEGATIVE INTERFACE ENERGY — Marigo condition violated.")

        # --- positivity threshold of the interface energy (Marigo, eq. 59) -----
        borne_a = borne_positivite_a(p_)
        run.value("bornePositiviteA", borne_a,
                  "a/e positivity threshold of the interface energy "
                  "(max of the B, C and S thresholds)", fmt="%.2f")
        print(f"  positivity threshold: a >= {borne_a:.3f} e")

        # --- Marigo (2017) variational bounds, eq. (60) ------------------------
        borne_B, borne_C = bornes_marigo(p_)
        run.value("borneMarigoB", borne_B,
                  "Marigo (2017) lower bound on B^e, a = e", fmt="%.3f")
        run.value("borneMarigoC", borne_C,
                  "Marigo (2017) lower bound on C^e, a = e", fmt="%.3f")
        print(f"  Marigo bounds: B^e >= {borne_B:.4f}   C^e >= {borne_C:.4f}")
        if B_e.real < borne_B or C_e.real < borne_C:
            run.note("MARIGO BOUND VIOLATED — the cell solver is wrong.")

        # --- cross-check by modal matching (Marigo, appendix B.1) --------------
        # The modal method is only written for φ = 1/2 (the test case).
        if abs(p_.phi - 0.5) < 1e-12:
            B1_modal = B1_multimodal(p_.e_over_h, p_.mu_ratio)
            B1_fem = coefficient_B(0.0, p_).real - p_.e_over_h
            ecart_pct = abs(B1_modal - B1_fem) / abs(B1_fem) * 100.0
            run.value("BunModal", B1_modal,
                      "B_1(0) by modal matching (Marigo, appendix B.1)", fmt="%.4f")
            run.value("BunFem", B1_fem, "B_1(0) by finite elements", fmt="%.4f")
            run.value("ecartModalFem", ecart_pct,
                      "deviation (%) between the two methods", fmt="%.2f")
            print(f"  modal check: B_1 = {B1_modal:.4f} (modal) vs "
                  f"{B1_fem:.4f} (FEM), deviation {ecart_pct:.3f} %")
        else:
            run.note("Modal check skipped: the appendix B.1 method "
                     "assumes φ = 1/2.")

        # --- cell solver validations (section 5.2) ------------------------------
        valid = validations_solveur(p_)
        run.value("validPleine", valid["pleine"],
                  "full-layer residual: |B_1 − (M_m/M_i − 1)e/h|, φ = 1", fmt="%.0e")
        run.value("validContrasteNul", valid["contraste_nul"],
                  "no-contrast residual: max(|B_1|, |C_2|), M_i = M_m", fmt="%.0e")
        run.value("validReciprocite", valid["reciprocite"],
                  "reciprocity lemma residual |B_2 + C_1|", fmt="%.0e")
        run.value("validTroncature", valid["troncature"],
                  "deviation on B_1 between truncations L = 4 and L = 20", fmt="%.0e")
        run.value("validIdentiteB", valid["identite_B"],
                  "max residual of the energy identity (48)-(49), complex r", fmt="%.0e")
        run.value("validIdentiteC", valid["identite_C"],
                  "max residual of the energy identity (53), complex r", fmt="%.0e")
        run.value("validMaillage", valid["maillage"],
                  "deviation on B_1 between two mesh refinements", fmt="%.0e")
        print("  solver validations: " +
              "  ".join(f"{k}={v:.1e}" for k, v in valid.items()))

        # --- frequency sweep -------------------------------------------------
        omegas = np.linspace(b["omega_min"], b["omega_max"], b["n_omega"])
        eps = float(b["epsilon_laplace"])
        p_vals = -1j * omegas + eps
        B_vals = np.array([coefficient_B(p, p_) for p in p_vals])
        C_vals = np.array([coefficient_C(p, p_) for p in p_vals])

        run.table("coefficients_vs_omega", {
            "omega": omegas,
            "Re_B": B_vals.real, "Im_B": B_vals.imag,
            "Re_C": C_vals.real, "Im_C": C_vals.imag,
        })

        # --- fig05: B(omega) -------------------------------------------------
        fig, ax = plt.subplots(figsize=SIMPLE)
        ax.plot(omegas, B_vals.real, "b-", lw=2, label=r"$\mathrm{Re}\,B(\omega, a)$")
        ax.plot(omegas, B_vals.imag, "r--", lw=2, label=r"$\mathrm{Im}\,B(\omega, a)$")
        ax.axhline(B_e, color="b", ls=":", alpha=0.5,
                   label=r"$B^e = B(0)$ (static limit)")
        ax.set_xlabel(r"$\omega h / c_m$ (dimensionless frequency)")
        ax.set_ylabel(r"$B(\omega, a)$")
        ax.set_title(r"Interface coefficient $B(\omega, a=e)$", fontsize=12)
        ax.legend(loc="best", fontsize=10)
        fig.tight_layout()
        run.figure(fig, "fig05_coefficient_B.pdf")
        plt.close(fig)

        # --- fig06: C(omega) -------------------------------------------------
        fig, ax = plt.subplots(figsize=SIMPLE)
        ax.plot(omegas, C_vals.real, "b-", lw=2, label=r"$\mathrm{Re}\,C(\omega, a)$")
        ax.plot(omegas, C_vals.imag, "r--", lw=2, label=r"$\mathrm{Im}\,C(\omega, a)$")
        ax.set_xlabel(r"$\omega h / c_m$")
        ax.set_ylabel(r"$C(\omega, a)$")
        ax.set_title(r"Interface coefficient $C(\omega, a=e)$", fontsize=12)
        ax.legend(loc="best", fontsize=10)
        fig.tight_layout()
        run.figure(fig, "fig06_coefficient_C.pdf")
        plt.close(fig)


if __name__ == "__main__":
    main()
