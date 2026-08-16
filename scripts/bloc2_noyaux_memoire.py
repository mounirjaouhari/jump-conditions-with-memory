"""Block 2 — Memory kernels K_B(t), K_C(t) and Prony approximation.

Produces: fig07_noyaux_memoire.pdf
Data: results/<config>/bloc2/*.csv + manifest.json

    python run.py bloc2 [--config config/xxx.yaml]
"""

import numpy as np

from fading_memory import RunRecorder, load_config
from fading_memory.physics import (check_passivity, condition_secteur,
                                   decompose_B, decompose_C, inverse_laplace,
                                   noyau_memoire, prony_eval, prony_fit)
from fading_memory.plotting import DOUBLE, apply_style, plt


def main(cfg=None):
    cfg = cfg or load_config()
    apply_style()
    p_ = cfg.params()
    b = cfg.bloc("bloc2")
    lap = cfg.laplace_kwargs()

    with RunRecorder("bloc2", cfg) as run:
        # B^e, B^v, C^e, C^v are published by block 1: do not redeclare them
        # (one value = one LaTeX macro = a single producer).
        B_e, B_v, K_hat_B = decompose_B(p_)
        C_e, C_v, K_hat_C = decompose_C(p_)

        # --- passivity ---------------------------------------------------------
        # NORMAL coefficient B: [[U]] = h·B·⟨∂₁U⟩. B grows with p (the inclusion
        #   becomes relatively soft). Passivity:  Re[p·K̂_B(p)] ≥ 0.
        # TRANSVERSE coefficient C: enters with a MINUS SIGN,
        #   [[Σ₁]] = h·S·⟨divΣ⟩ − h·C·⟨∂₂Σ₂⟩, and DECREASES with p.
        #   Passivity:  Re[p·K̂_C(p)] ≤ 0,  i.e. −K_C of positive type.
        # The two vary in OPPOSITE DIRECTIONS: opposite signs in the passivity
        #   inequalities (paper, passivity conjecture + sign remark).
        passif_B, min_B = check_passivity(K_hat_B)
        _, max_C = check_passivity(lambda p: -K_hat_C(p))
        passif_C = max_C >= -1e-8
        run.value("passiviteMinB", min_B,
                  "min Re(p K̂_B) ≥ 0 — normal coefficient B", fmt="%.2e")
        run.value("passiviteMinC", max_C,
                  "min Re(−p K̂_C) ≥ 0 — transverse coefficient C", fmt="%.2e")
        print(f"  passivity K_B  : {passif_B} (min Re[+p K̂_B] = {min_B:+.3e})")
        print(f"  passivity −K_C : {passif_C} (min Re[−p K̂_C] = {max_C:+.3e})")
        if not (passif_B and passif_C):
            run.note("PASSIVITY VIOLATED — theorem 7.1 is defeated.")

        # --- sector condition (S_p) — error estimate (supplement S9) ----------
        # On the real axis, (S_p) follows from passivity; off the axis, it is
        # an assumption of the error-estimate theorem: we check it on a grid
        # of the right half-plane. By the symmetry p → p̄ (cf.
        # physics.condition_secteur), Im p ≥ 0 suffices.
        sigmas = np.asarray(b["secteur_sigmas"], dtype=float)
        omegas_s = np.asarray(b["secteur_omegas"], dtype=float)
        lignes = {"sigma": [], "omega": [], "secteur_B": [], "secteur_C": []}
        for sig in sigmas:
            for om in omegas_s:
                sB, sC = condition_secteur(sig + 1j * om, p_)
                lignes["sigma"].append(sig)
                lignes["omega"].append(om)
                lignes["secteur_B"].append(sB)
                lignes["secteur_C"].append(sC)
        sec_B = np.asarray(lignes["secteur_B"])
        sec_C = np.asarray(lignes["secteur_C"])
        run.table("condition_secteur", lignes)
        run.value("secteurMinB", float(sec_B.min()),
                  "min of Re[e^{-i arg p} M_m/B] over the half-plane grid",
                  fmt="%.3g")
        run.value("secteurMinC", float(sec_C.min()),
                  "min of Re[e^{-i arg p} M_m C] over the half-plane grid",
                  fmt="%.3g")
        run.value("secteurNbPoints", int(sec_B.size),
                  "size of the (σ, ω) grid of the right half-plane", fmt="%d")
        run.value("secteurSigmaMax", float(sigmas.max()),
                  "largest abscissa σ of the grid", fmt="%g")
        run.value("secteurOmegaMax", float(omegas_s.max()),
                  "largest ordinate ω of the grid", fmt="%g")
        print(f"  condition (S_p): min secteur_B = {sec_B.min():+.4f}   "
              f"min secteur_C = {sec_C.min():+.4f}  ({sec_B.size} points)")
        if sec_B.min() <= 0 or sec_C.min() <= 0:
            run.note("SECTOR CONDITION (S_p) VIOLATED on the grid — "
                     "the error estimate of supplement S9 loses its hypothesis.")

        # --- time-domain reconstruction ---------------------------------------
        # K̂_X(p) ~ C∞/p as |p| → ∞: the kernel JUMPS at t = 0. We subtract this
        # tail analytically, otherwise the Bromwich integral oscillates (D3b).
        t = np.linspace(b["t_min"], b["t_max"], b["n_t"])
        K_B_hat, C_B, lam = noyau_memoire(p_, quoi="B")
        K_C_hat, C_C, _ = noyau_memoire(p_, quoi="C")
        run.value("KBzero", C_B, "K_B(0+) = B(+∞) − B^e: jump of the kernel at the origin", fmt="%.3f")
        run.value("KCzero", C_C, "K_C(0+) = C(+∞) − C^e", fmt="%.3f")
        run.value("lambdaRelax", lam, "λ = μ_m/β_m: relaxation rate of the matrix", fmt="%.1f")

        print(f"  K_B(0+) = {C_B:+.4f}   K_C(0+) = {C_C:+.4f}   λ = {lam:.1f}")
        K_B_t = inverse_laplace(K_B_hat, t, tail_C=C_B, tail_lambda=lam, **lap)
        K_C_t = inverse_laplace(K_C_hat, t, tail_C=C_C, tail_lambda=lam, **lap)

        # --- Prony approximation ----------------------------------------------
        # alpha_n > 0 by construction (log parametrization): the approximation
        # is passive. This is possible ONLY if the kernel itself is positive —
        # which follows from passivity. With the legacy code, the kernel was
        # negative and the residual was 100%: the "Prony" curve was flat.
        # K_C being negative (stiffness), we fit −K_C: the α_n stay > 0 and
        # the approximation remains passive, up to the global sign.
        n_exp = b["prony_n_exp"]
        alphas_B, taus_B, res_B = prony_fit(t, K_B_t, n_exp=n_exp)
        alphas_C, taus_C, res_C = prony_fit(t, -K_C_t, n_exp=n_exp)
        res_rel_B = res_B / max(np.sum(K_B_t ** 2), 1e-30)
        res_rel_C = res_C / max(np.sum(K_C_t ** 2), 1e-30)
        run.value("pronyNExp", n_exp, "number of internal variables", fmt="%d")
        run.value("pronyResiduB", res_rel_B, "relative residual of the K_B fit", fmt="%.1e")
        run.value("pronyResiduC", res_rel_C, "relative residual of the K_C fit", fmt="%.1e")
        print(f"  Prony K_B: relative residual = {res_rel_B:.2e}   alphas = {np.round(alphas_B, 4)}")
        print(f"  Prony K_C: relative residual = {res_rel_C:.2e}   alphas = {np.round(alphas_C, 4)}")
        run.note(f"alphas_B > 0: {bool(np.all(alphas_B > 0))}; "
                 f"alphas_C > 0: {bool(np.all(alphas_C > 0))} (passivity by construction)")

        run.table("noyaux_memoire", {
            "t": t,
            "K_B": K_B_t, "K_B_prony": prony_eval(t, alphas_B, taus_B),
            "K_C": K_C_t, "K_C_prony": -prony_eval(t, alphas_C, taus_C),
        })
        run.table("prony_coefficients", {
            "n": np.arange(1, n_exp + 1),
            "alpha_B": alphas_B, "tau_B": taus_B,
            "alpha_C": alphas_C, "tau_C": taus_C,
        })

        # --- fig07 -------------------------------------------------------------
        fig, axes = plt.subplots(1, 2, figsize=DOUBLE)
        for ax, (K, Kp, lab, c) in zip(axes, [
            (K_B_t, prony_eval(t, alphas_B, taus_B), "B", "b"),
            (K_C_t, -prony_eval(t, alphas_C, taus_C), "C", "r"),
        ]):
            ax.plot(t, K, c + "-", lw=2, label=rf"$K_{lab}(t)$ (Laplace inversion)")
            ax.plot(t, Kp, "k--", lw=1.5, label=rf"$K_{lab}(t)$ Prony ($N={n_exp}$)")
            ax.set_xlabel(r"$t \, c_m / h$")
            ax.set_ylabel(rf"$K_{lab}(t)$")
            ax.set_title(rf"Memory kernel $K_{lab}(t)$", fontsize=12)
            ax.legend(fontsize=10)
            ax.grid(True, alpha=0.3)
        fig.tight_layout()
        run.figure(fig, "fig07_noyaux_memoire.pdf")
        plt.close(fig)


if __name__ == "__main__":
    main()
