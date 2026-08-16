"""Block 4 — Influence of the parameter a (interface thickness).

Produces: fig08_energie_passivite.pdf, fig11_erreur_vs_a_over_e.pdf
Data: results/<config>/bloc4/*.csv + manifest.json

    python run.py bloc4 [--config config/xxx.yaml]
"""

import numpy as np

from fading_memory import RunRecorder, load_config
from fading_memory.physics import (coefficient_B, coefficient_C,
                                   compute_R_T_homogenized,
                                   compute_R_T_reference, decompose_B,
                                   decompose_C)
from fading_memory.plotting import DOUBLE, SIMPLE_HAUTE, apply_style, plt


def main(cfg=None):
    cfg = cfg or load_config()
    apply_style()
    p_ = cfg.params()
    b = cfg.bloc("bloc4")

    with RunRecorder("bloc4", cfg) as run:
        # === 1. Instantaneous coefficients and passivity vs a/e =================
        a_ratios = np.linspace(b["a_over_e_min"], b["a_over_e_max"], b["n_a_over_e"])
        omega_test = b["omega_test"]

        B_e_vals, C_e_vals, ReB, ReC = [], [], [], []
        for r in a_ratios:
            a = r * p_.e_over_h
            B_e_vals.append(decompose_B(p_, a)[0])
            C_e_vals.append(decompose_C(p_, a)[0])
            p_test = -1j * omega_test + 0.01
            ReB.append(coefficient_B(p_test, p_, a).real)
            ReC.append(coefficient_C(p_test, p_, a).real)

        # B^e and C^e are real (static limit) but the computation returns them
        # in complex dtype with Im ≈ 0: plotting them as-is triggered a
        # ComplexWarning at the silent cast — the warning class that already
        # hid a real bug (block 5, debt D2). EXPLICIT conversion, with a
        # check on what it discards.
        B_e_vals = np.asarray(B_e_vals)
        C_e_vals = np.asarray(C_e_vals)
        residu_im = float(max(np.abs(B_e_vals.imag).max(),
                              np.abs(C_e_vals.imag).max()))
        if residu_im > 1e-10:
            run.note(f"Non-negligible imaginary part discarded on B^e/C^e: "
                     f"{residu_im:.2e}")
        B_e_vals, C_e_vals = B_e_vals.real, C_e_vals.real

        run.table("coefficients_vs_a", {
            "a_over_e": a_ratios, "B_e": B_e_vals, "C_e": C_e_vals,
            "Re_B": ReB, "Re_C": ReC,
        })

        fig, axes = plt.subplots(1, 2, figsize=DOUBLE)
        axes[0].plot(a_ratios, B_e_vals, "b-", lw=2, label=r"$B^e(a)$")
        axes[0].plot(a_ratios, C_e_vals, "r--", lw=2, label=r"$C^e(a)$")
        axes[0].set_ylabel("Elastic coefficient")
        axes[0].set_title(r"Instantaneous coefficients $B^e(a)$, $C^e(a)$", fontsize=12)

        axes[1].plot(a_ratios, ReB, "b-", lw=2,
                     label=rf"$\mathrm{{Re}}\,B(p, a)$, $\omega = {omega_test}$")
        axes[1].plot(a_ratios, ReC, "r--", lw=2,
                     label=rf"$\mathrm{{Re}}\,C(p, a)$, $\omega = {omega_test}$")
        axes[1].set_ylabel("Real part (passivity test)")
        axes[1].set_title(r"Passivity: $\mathrm{Re}\,B(p,a)$, $\mathrm{Re}\,C(p,a)$",
                          fontsize=12)

        for ax in axes:
            ax.axvline(1.0, color="green", ls=":", alpha=0.7, label=r"$a = e$")
            ax.axhline(0.0, color="black", ls="-", alpha=0.3)
            ax.set_xlabel(r"$a/e$")
            ax.legend(fontsize=10)
            ax.grid(True, alpha=0.3)
        fig.tight_layout()
        run.figure(fig, "fig08_energie_passivite.pdf")
        plt.close(fig)

        # === 2. Reflection error vs a/e — and superconvergence ==============
        # Marigo et al. (2017) show that a = e minimizes the error. We
        # recover this, and we give the reason: a = e is the ONLY
        # SUPERCONVERGENT choice. The relative error on R is there O(eta^2),
        # versus O(eta) for any other a — a full order gained.
        #
        # CAUTION: the reference solver has a precision floor of ~1e-4 in
        # relative error. Below it, one measures noise, not the model. This
        # is exactly Marigo's warning. The floor is plotted.
        a_ratios2 = np.linspace(b["a_over_e_min_err"], b["a_over_e_max_err"],
                                b["n_a_over_e_err"])
        theta = b["theta"]
        colonnes = {"a_over_e": a_ratios2}
        plancher = b["plancher_reference"]
        run.value("plancherReference", float(plancher),
                  "precision floor of the reference solver "
                  "(relative error on R, measured by block 10)", fmt="%.1e")

        fig, axes = plt.subplots(1, 2, figsize=DOUBLE)
        for kh, color in zip(b["kh_values"], ["blue", "green", "red"]):
            R_ref, _ = compute_R_T_reference(kh, theta, p_)   # independent of a
            E_R = []
            for r in a_ratios2:
                R_h, _ = compute_R_T_homogenized(kh, theta, p_, a=r * p_.e_over_h)
                E_R.append(abs(R_h - R_ref) / abs(R_ref))
            E_R = np.asarray(E_R)
            colonnes[f"E_R_kh_{kh}"] = E_R
            i = int(np.argmin(E_R))
            print(f"  kh={kh}: min E_R = {E_R.min():.2e} at a/e = {a_ratios2[i]:.3f}")
            run.note(f"kh={kh}: a/e minimizing E_R = {a_ratios2[i]:.3f}")
            axes[0].semilogy(a_ratios2, E_R, color=color, lw=1.8, label=rf"$kh = {kh}$")

        axes[0].axhline(plancher, color="grey", ls="-.", lw=1.2,
                        label="reference-solver floor")
        axes[0].axvline(1.0, color="black", ls=":", lw=1.4, label=r"$a = e$")
        axes[0].set_xlabel(r"$a/e$")
        axes[0].set_ylabel(r"$E_R$ (relative error on $R$)")
        axes[0].set_title(r"The error is minimal at $a = e$", fontsize=12)

        # --- panel 2: the convergence ORDER as a function of a ----------------
        khs = np.asarray(b["kh_ordre"], dtype=float)
        ordres, a_test = [], np.linspace(0.6, 1.4, b["n_a_ordre"])
        for r in a_test:
            e_k = []
            for kh in khs:
                R_ref, _ = compute_R_T_reference(kh, theta, p_)
                R_h, _ = compute_R_T_homogenized(kh, theta, p_, a=r * p_.e_over_h)
                e_k.append(abs(R_h - R_ref) / abs(R_ref))
            e_k = np.asarray(e_k)
            ordres.append(float(np.polyfit(np.log(khs), np.log(e_k), 1)[0]))
        ordres = np.asarray(ordres)
        colonnes_ordre = {"a_over_e": a_test, "ordre": ordres}

        i_sup = int(np.argmax(ordres))
        run.value("ordreMaxEnA", float(ordres.max()),
                  "maximum convergence order over the a sweep", fmt="%.2f")
        run.value("aOptimalOrdre", float(a_test[i_sup]),
                  "a/e achieving this maximum order", fmt="%.2f")
        print(f"  max order = {ordres.max():.2f} at a/e = {a_test[i_sup]:.3f}  "
              f"(order ~1 elsewhere: a = e gains a full order)")

        axes[1].plot(a_test, ordres, "b-o", lw=1.8, ms=4)
        axes[1].axhline(1.0, color="grey", ls="--", lw=1, label=r"$\mathcal{O}(\eta)$")
        axes[1].axhline(2.0, color="green", ls="--", lw=1, label=r"$\mathcal{O}(\eta^2)$")
        axes[1].axvline(1.0, color="black", ls=":", lw=1.4, label=r"$a = e$")
        axes[1].set_xlabel(r"$a/e$")
        axes[1].set_ylabel("convergence order of $E_R$")
        axes[1].set_title(r"$a = e$ is the only superconvergent choice", fontsize=12)

        for ax in axes:
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3, which="both")
        fig.tight_layout()
        run.figure(fig, "fig11_erreur_vs_a_over_e.pdf")
        plt.close(fig)

        run.table("erreur_vs_a", colonnes)
        run.table("ordre_vs_a", colonnes_ordre)


if __name__ == "__main__":
    main()
