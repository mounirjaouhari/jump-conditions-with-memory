"""Block 7 — In which regime does the memory matter?

Produces: fig17_regime_memoire.pdf
Data    : results/<config>/bloc7/*.csv + manifest.json

WHY THIS BLOCK EXISTS
---------------------
The test case of the paper places the study in a regime where the memory is
NEGLIGIBLE in the harmonic setting: `visc_ratio = ωβ_m/μ_m = 0.1` is defined
at ω = 1, but the figures sweep kh ≤ 0.5, so ωβ_m/μ_m ≤ 0.05 everywhere. The
gain of the memory model there is ~10 %, which led the first version of the
paper to invent a table of errors (18.3 % → 2.1 %) that the code did not
produce.

This block establishes the exact fact, and plots it: the gain of the memory
is a function of the ratio ω/λ, where λ = μ_m/β_m is the relaxation rate of
the matrix. It is small as long as ω ≪ λ, and becomes important when ω ~ λ.

    python run.py bloc7 [--config config/xxx.yaml]
"""

import numpy as np

from fading_memory import RunRecorder, load_config
from fading_memory.physics import (compute_R_T_homogenized,
                                   compute_R_T_reference, decompose_B,
                                   noyau_memoire)
from fading_memory.plotting import DOUBLE, apply_style, plt


def main(cfg=None):
    cfg = cfg or load_config()
    apply_style()
    b = cfg.bloc("bloc7")

    with RunRecorder("bloc7", cfg) as run:
        visc = np.asarray(b["visc_ratios"], dtype=float)
        kh = b["kh"]
        theta = np.deg2rad(b["theta_deg"])

        gains, amplitudes, omega_sur_lambda = [], [], []
        err_mem, err_ins = [], []
        for vr in visc:
            p_ = cfg.params(visc_ratio=vr)
            R_ref, _ = compute_R_T_reference(kh, theta, p_)
            R_mem, _ = compute_R_T_homogenized(kh, theta, p_, with_memory=True)
            R_ins, _ = compute_R_T_homogenized(kh, theta, p_, with_memory=False)
            e_mem = abs(R_mem - R_ref) / abs(R_ref)
            e_ins = abs(R_ins - R_ref) / abs(R_ref)
            gains.append(e_ins / e_mem)
            err_mem.append(e_mem)
            err_ins.append(e_ins)

            # kernel amplitude: K_B(0+) = B(+inf) - B^e
            _, C_inf, lam = noyau_memoire(p_, quoi="B")
            amplitudes.append(C_inf)
            omega_sur_lambda.append(kh / lam)   # omega = kh; lambda = mu_m/beta_m

            print(f"  ωβ/μ = {vr:<6g}  ω/λ = {kh / lam:8.3g}  "
                  f"K_B(0+) = {C_inf:6.3f}  memory gain = {e_ins / e_mem:6.2f}×")

        # REMARKABLE FACT, and absent from the paper: K_B(0+) = B(+inf) - B^e
        # does NOT depend on beta_m. Indeed B(p) = a/h + G(r(p)) and r(p) goes
        # from mu_i/mu_m (at p=0) to 0 (at p=+inf), whatever the viscosity.
        # The viscosity therefore sets the RELAXATION TIME of the kernel
        # (lambda), not its AMPLITUDE — which depends only on the elastic
        # contrast.
        amp = np.asarray(amplitudes)
        constante = bool(np.ptp(amp) < 1e-6 * max(abs(amp.mean()), 1.0))
        run.note(f"K_B(0+) independent of the viscosity: {constante} "
                 f"(value {amp.mean():.4f}). The viscosity sets lambda, not the amplitude.")
        run.value("KBzeroInvariant", float(amp.mean()),
                  "K_B(0+), invariant with respect to the viscosity", fmt="%.3f")

        run.table("regime_memoire", {
            "visc_ratio": visc, "omega_sur_lambda": omega_sur_lambda,
            "K_B_zero": amplitudes, "gain_memoire": gains,
            "E_R_memoire": err_mem, "E_R_instantane": err_ins,
        })

        i_ref = int(np.argmin(np.abs(visc - cfg.params().visc_ratio)))
        run.value("gainMemoireReference", float(gains[i_ref]),
                  "memory gain, test case of the paper", fmt="%.2f")
        run.value("gainMemoireMax", float(max(gains)),
                  "maximal memory gain over the viscosity sweep",
                  fmt="%.1f")
        run.value("viscRatioGainMax", float(visc[int(np.argmax(gains))]),
                  "viscosity achieving this maximal gain", fmt="%g")
        i_min = int(np.argmin(omega_sur_lambda))
        run.value("gainMemoireMin", float(gains[i_min]),
                  "memory gain at the smallest ω/λ of the sweep", fmt="%.2f")
        run.value("omegaLambdaMin", float(omega_sur_lambda[i_min]),
                  "smallest ratio ω/λ of the sweep", fmt="%.0e")
        run.value("omegaLambdaMax", float(max(omega_sur_lambda)),
                  "largest ratio ω/λ of the sweep", fmt="%g")
        # Regime of the harmonic sweeps of the test case: ωβ_m/μ_m = kh·visc_ratio
        # is bounded there by kh_max (block 3) times the reference viscosity.
        regime = float(cfg.bloc("bloc3")["kh_max"]) * float(cfg.params().visc_ratio)
        run.value("regimeBalayagesMax", regime,
                  "bound on ωβ_m/μ_m over the harmonic sweeps of the test case",
                  fmt="%g")

        # === fig17 ============================================================
        fig, axes = plt.subplots(1, 2, figsize=DOUBLE)

        axes[0].loglog(omega_sur_lambda, gains, "bo-", lw=2, ms=5)
        axes[0].axhline(1.0, color="k", ls=":", lw=1)
        axes[0].axvline(1.0, color="r", ls="--", lw=1, alpha=0.7,
                        label=r"$\omega = \lambda$")
        axes[0].plot(omega_sur_lambda[i_ref], gains[i_ref], "r*", ms=15,
                     label="test case of the paper", zorder=5)
        axes[0].set_xlabel(r"$\omega / \lambda$   ($\lambda = \mu_m/\beta_m$)")
        axes[0].set_ylabel(r"gain $= E_R(\mathrm{instantaneous}) / E_R(\mathrm{memory})$")
        axes[0].set_title("Memory matters when $\\omega$ approaches $\\lambda$",
                          fontsize=11)

        axes[1].loglog(omega_sur_lambda, err_ins, "r--s", lw=2, ms=5,
                       label="instantaneous model")
        axes[1].loglog(omega_sur_lambda, err_mem, "b-o", lw=2, ms=5,
                       label="memory model")
        axes[1].axvline(1.0, color="r", ls="--", lw=1, alpha=0.7)
        axes[1].set_xlabel(r"$\omega / \lambda$")
        axes[1].set_ylabel(r"$E_R$ (relative error on $R$)")
        axes[1].set_title("The error of the instantaneous model blows up;\n"
                          "that of the memory model does not move", fontsize=11)

        for ax in axes:
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3, which="both")
        fig.tight_layout()
        run.figure(fig, "fig17_regime_memoire.pdf")
        plt.close(fig)


if __name__ == "__main__":
    main()
