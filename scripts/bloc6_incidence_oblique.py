"""Block 6 — Oblique incidence: the validation of the coefficient C.

Produces: fig15_validation_C.pdf, fig16_RT_vs_theta.pdf
Data    : results/<config>/bloc6/*.csv + manifest.json

WHY THIS BLOCK EXISTS
---------------------
At normal incidence, ∂₂ ≡ 0: the term −h·C·⟨∂₂Σ₂⟩ vanishes identically from
the jump conditions. Figures 9 to 11 therefore test ONLY B and S. The
coefficient C was validated by nothing — and it is precisely on C that the
legacy code made its two grossest errors (modal correction identically zero,
volume term counted twice). Nobody could notice.

C enters the model only through k₂² = k² sin²θ. Validating it therefore
REQUIRES oblique incidence, hence a quasi-periodic reference solution
(Bloch conditions): this is `scattering_fem.ScatteringMesh.resoudre(theta≠0)`.

    python run.py bloc6 [--config config/xxx.yaml]
"""

import numpy as np

from fading_memory import RunRecorder, load_config
from fading_memory.physics import (coefficient_C, compute_R_T_homogenized,
                                   compute_R_T_reference)
from fading_memory.plotting import DOUBLE, apply_style, plt


def main(cfg=None):
    cfg = cfg or load_config()
    apply_style()
    p_ = cfg.params()
    b = cfg.bloc("bloc6")

    with RunRecorder("bloc6", cfg) as run:
        angles = [np.deg2rad(d) for d in b["angles_deg"]]
        khs = np.linspace(b["kh_min"], b["kh_max"], b["n_kh"])
        # C^e from the legacy code (counter-example documented in the config).
        C_HERITE = float(b["c_herite"])

        run.value("Cek", float(np.real(coefficient_C(0.0, p_))),
                  "C^e used (computed by finite elements)", fmt="%.3f")
        run.value("CeHerite", C_HERITE,
                  "C^e from the legacy code (counter-example: a wrong C is "
                  "worse than C = 0)", fmt="%.1f")

        # === fig15: error with / without / with a wrong C ===================
        fig, axes = plt.subplots(1, 2, figsize=DOUBLE)
        colonnes = {"kh": khs}
        resume = {}
        donnees = {}

        for th, color in zip(angles, ["blue", "green", "red"]):
            deg = int(round(np.rad2deg(th)))
            e_avec, e_sans, e_faux = [], [], []
            for kh in khs:
                R_ref, _ = compute_R_T_reference(kh, th, p_)
                R_c, _ = compute_R_T_homogenized(kh, th, p_)
                R_0, _ = compute_R_T_homogenized(kh, th, p_, with_C=False)
                R_f, _ = compute_R_T_homogenized(kh, th, p_, C_override=C_HERITE)
                d = abs(R_ref)
                e_avec.append(abs(R_c - R_ref) / d)
                e_sans.append(abs(R_0 - R_ref) / d)
                e_faux.append(abs(R_f - R_ref) / d)

            e_avec = np.asarray(e_avec)
            e_sans = np.asarray(e_sans)
            e_faux = np.asarray(e_faux)
            donnees[deg] = (e_avec, e_sans, e_faux)
            colonnes[f"E_R_avecC_{deg}deg"] = e_avec
            colonnes[f"E_R_sansC_{deg}deg"] = e_sans
            colonnes[f"E_R_Cfaux_{deg}deg"] = e_faux

            axes[0].loglog(khs, e_avec, color=color, lw=2,
                           label=rf"$\theta = {deg}^\circ$")
            axes[1].loglog(khs, e_sans / e_avec, color=color, lw=2,
                           label=rf"$\theta = {deg}^\circ$")

            # convergence order, measured over the first decade
            i, j = 0, len(khs) // 3
            ordre = float(np.log(e_avec[j] / e_avec[i]) / np.log(khs[j] / khs[i]))
            resume[deg] = (ordre, float(e_sans.max() / e_avec.max()))
            run.note(f"theta={deg} deg: convergence order = {ordre:.2f}; "
                     f"gain from C = {e_sans[0] / e_avec[0]:.0f}x at kh={khs[0]:.2f}")
            print(f"  θ={deg:2d}° : order = {ordre:.2f}   "
                  f"gain from C = {e_sans[0] / e_avec[0]:.0f}× (kh={khs[0]:.2f})")

        # reference slope (kh)² on the left panel
        axes[0].loglog(khs, khs ** 2 * (0.2), "k--", lw=1, alpha=0.6,
                       label=r"slope $(kh)^2$")
        axes[0].set_ylabel(r"$E_R$ (with $C$)")
        axes[0].set_title("The complete model converges as $\\mathcal{O}((kh)^2)$",
                          fontsize=11)
        axes[1].axhline(1.0, color="k", ls=":", lw=1)
        axes[1].set_ylabel(r"$E_R(C=0) \, / \, E_R(C)$")
        axes[1].set_title("Error factor when omitting $C$", fontsize=11)
        for ax in axes:
            ax.set_xlabel(r"$kh$")
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3, which="both")
        fig.tight_layout()
        run.figure(fig, "fig15_validation_C.pdf")
        plt.close(fig)

        run.table("erreur_vs_kh_oblique", colonnes)
        ordre_min = min(o for o, _ in resume.values())
        gain_max = max(g for _, g in resume.values())
        run.value("ordreObliqueMin", ordre_min,
                  "minimal convergence order at oblique incidence", fmt="%.2f")
        run.value("gainCMax", gain_max,
                  "maximal error factor if C is omitted", fmt="%.0f")

        # Citable values: what happens to the error when C is omitted or wrong?
        tous_avec = np.concatenate([d[0] for d in donnees.values()])
        tous_sans = np.concatenate([d[1] for d in donnees.values()])
        run.value("erreurAvecCMediane", float(np.median(tous_avec)),
                  "median error of the complete model at oblique incidence "
                  "(all angles, all kh)", fmt="%.0e")
        run.value("erreurSansCMin", float(tous_sans.min()),
                  "smallest relative error observed with C = 0", fmt="%.1f")
        run.value("erreurSansCMax", float(tous_sans.max()),
                  "largest relative error observed with C = 0", fmt="%.1f")
        if 45 in donnees:
            _, e_s45, e_f45 = donnees[45]
            run.value("erreurCFauxPct", float(e_f45.max() * 100.0),
                      "max error (%) with the wrong legacy C^e, θ = 45°",
                      fmt="%.0f")
            run.value("erreurCNulPct", float(e_s45.max() * 100.0),
                      "max error (%) with C = 0, θ = 45°", fmt="%.0f")
            print(f"  θ=45° : wrong C → {e_f45.max() * 100:.0f} %   "
                  f"C = 0 → {e_s45.max() * 100:.0f} %")

        # === fig16: |R| as a function of the angle ===========================
        kh0 = b["kh_figure_angles"]
        thetas = np.linspace(0.0, np.deg2rad(b["theta_max_deg"]), b["n_theta"])
        R_ref, R_c, R_0, R_f = [], [], [], []
        for th in thetas:
            R_ref.append(abs(compute_R_T_reference(kh0, th, p_)[0]))
            R_c.append(abs(compute_R_T_homogenized(kh0, th, p_)[0]))
            R_0.append(abs(compute_R_T_homogenized(kh0, th, p_, with_C=False)[0]))
            R_f.append(abs(compute_R_T_homogenized(kh0, th, p_,
                                                   C_override=C_HERITE)[0]))

        deg = np.rad2deg(thetas)
        run.table("RT_vs_theta", {
            "theta_deg": deg, "R_reference": R_ref, "R_avec_C": R_c,
            "R_sans_C": R_0, "R_C_herite": R_f,
        })

        fig, ax = plt.subplots(figsize=(6.6, 4.4))
        ax.plot(deg, R_ref, "k-", lw=2.5, label="Actual problem (reference Bloch)")
        ax.plot(deg, R_c, "b--", lw=1.9, label=r"Homogenized, computed $C$")
        ax.plot(deg, R_0, "r:", lw=1.9, label=r"Homogenized, $C = 0$")
        ax.plot(deg, R_f, "-.", color="darkorange", lw=1.6,
                label=rf"Homogenized, $C^e = {C_HERITE:.0f}$ (legacy code)")
        ax.set_xlabel(r"Angle of incidence $\theta$ (degrees)")
        ax.set_ylabel(r"$|R|$")
        ax.set_title(rf"Reflection at oblique incidence ($kh = {kh0}$)", fontsize=12)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        run.figure(fig, "fig16_RT_vs_theta.pdf")
        plt.close(fig)


if __name__ == "__main__":
    main()
