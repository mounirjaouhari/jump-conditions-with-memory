"""Block 5 — Transient propagation of a pulse through the memory interface.

Produces: fig12_signaux_temporels.pdf, fig13_spectres.pdf,
          fig14_influence_viscosite.pdf
Data: results/<config>/bloc5/*.csv + manifest.json

MODEL — and why we do NOT write it by hand
------------------------------------------
The inherited code posed directly, in the time domain,

    U_refl(t) = −(h/2)·[ B^e U̇ + B^v Ü + (K_B * U̇)(t) ] ,

a formula wrong on three counts: the sign is flipped; the memory is
convolved there with U̇ instead of Ü (since [[U]] = h·B·⟨∂₁U⟩ and B(p) = B^e
+ pB^v + p·K̂_B(p): the kernel thus acts on the DERIVATIVE of ⟨∂₁U⟩); and
the inertial coefficient S is absent, even though R = i k h (S − B)/2 at
first order. On top of that came a time step dt = 0.05 unable to resolve a
kernel that relaxes in τ ≈ 0.013.

We do not rewrite this model by hand. The reflection coefficient R(ω) of the
homogenized model is ALREADY validated against the reference (scattering
section): the problem being linear and time-translation invariant,

    U_refl(t) = FT⁻¹[ R(ω) · Û_inc(ω) ] ,

with no additional time-domain convolution scheme (the identity is exact;
its evaluation remains subject to the truncation/step of the frequency grid
and to the per-frequency solver error). The same formula applies to the
REFERENCE solution, with the R(ω) of the actual problem: we therefore
compare the transient of the model to the transient of the actual problem.

FOUR CURVES
-----------
  reference   : R(ω) of the actual problem (finite elements, per frequency)
  memory     : R(ω) of the homogenized model, exact kernel
  Prony       : same, kernel replaced by Σ αₙ e^{−t/τₙ} — this is the model
                actually implementable, with internal variables
  instantaneous  : same, memory ignored (K_B ≡ 0)

    python run.py bloc5 [--config config/xxx.yaml]
"""

import numpy as np

from fading_memory import RunRecorder, load_config
from fading_memory.physics import (compute_R_T_homogenized,
                                   compute_R_T_reference, decompose_B,
                                   inverse_laplace, noyau_memoire,
                                   prony_fit, prony_laplace)
from fading_memory.plotting import DOUBLE, apply_style, plt


def _spectre_utile(U_hat, omegas, kh_max, seuil):
    """Frequencies where the pulse has content AND where the model is valid."""
    amp = np.abs(U_hat)
    return (amp > seuil * amp.max()) & (omegas > 1e-12) & (omegas <= kh_max)


def _signal(R_omega, U_hat, masque, n):
    """u(t) = FT⁻¹[R(ω)·Û(ω)].

    `irfft` reconstructs with e^{+iωt}, while the model is written in
    e^{−iωt} (p = −iω). The factor to apply to the e^{+iωt} component is
    therefore the CONJUGATE of R — this is what guarantees a real output
    signal.
    """
    H = np.zeros_like(U_hat)
    H[masque] = np.conj(R_omega)
    return np.fft.irfft(H * U_hat, n=n)


def main(cfg=None):
    cfg = cfg or load_config()
    apply_style()
    p_ = cfg.params()
    b = cfg.bloc("bloc5")
    lap = cfg.laplace_kwargs()

    with RunRecorder("bloc5", cfg) as run:
        # --- incident pulse ---------------------------------------------------
        dt, n = b["dt"], None
        t = np.arange(0.0, b["t_end"], dt)
        n = len(t)
        U_inc = np.exp(-(t - b["t0"]) ** 2 / b["tau0"] ** 2) * np.cos(b["omega0"] * t)

        # The three pulse parameters cited in the paper (pulse equation).
        run.value("tauZero", float(b["tau0"]), "width of the Gaussian pulse", fmt="%g")
        run.value("omegaZero", float(b["omega0"]), "carrier angular frequency of the pulse",
                  fmt="%g")
        run.value("khMaxModele", float(b["kh_max_modele"]),
                  "kh bound of the validity domain of the first-order model", fmt="%g")

        U_hat = np.fft.rfft(U_inc)
        omegas = 2.0 * np.pi * np.fft.rfftfreq(n, dt)
        masque = _spectre_utile(U_hat, omegas, b["kh_max_modele"], b["seuil_spectre"])
        w = omegas[masque]
        run.note(f"{masque.sum()} frequencies retained in [0, {b['kh_max_modele']}] "
                 f"(threshold {b['seuil_spectre']:g} of the spectrum maximum)")
        print(f"  {masque.sum()} useful frequencies (kh ≤ {b['kh_max_modele']})")

        # A Gaussian has no compact support (Paley-Wiener): the share of the
        # spectrum OUTSIDE the validity band is QUANTIFIED, not decreed. We
        # cite the out-of-band fraction (small but nonzero) rather than the
        # in-band fraction, which would misleadingly round to 100 %.
        energie = np.abs(U_hat) ** 2
        hors_bande = omegas > b["kh_max_modele"]
        frac_hors = float(energie[hors_bande].sum() / energie.sum())
        run.value("fractionHorsBande", frac_hors,
                  "fraction of the pulse spectral energy outside kh <= kh_max_modele",
                  fmt="%.1e")
        print(f"  out-of-band spectral energy: {frac_hors:.3e} "
              f"(i.e. {100 * (1 - frac_hors):.5f} % in the band)")

        # --- kernel and its Prony approximation -------------------------------
        t_k = np.linspace(0.0, cfg.bloc("bloc2")["t_max"], cfg.bloc("bloc2")["n_t"])
        K_hat, C_B, lam = noyau_memoire(p_, quoi="B")
        K_B = inverse_laplace(K_hat, t_k, tail_C=C_B, tail_lambda=lam, **lap)
        n_exp = cfg.bloc("bloc2")["prony_n_exp"]
        alphas, taus, _ = prony_fit(t_k, K_B, n_exp=n_exp)

        B_e, B_v, _ = decompose_B(p_)

        def B_prony(p):
            """B(p) with the exact kernel replaced by the Prony one."""
            return B_e + p * B_v + p * prony_laplace(p, alphas, taus)

        # --- the four reflection coefficients ----------------------------------
        R_ref, R_mem, R_pro, R_ins = [], [], [], []
        for om in w:
            R_ref.append(compute_R_T_reference(om, 0.0, p_)[0])
            R_mem.append(compute_R_T_homogenized(om, 0.0, p_)[0])
            R_pro.append(compute_R_T_homogenized(
                om, 0.0, p_, B_override=B_prony(-1j * om))[0])
            R_ins.append(compute_R_T_homogenized(om, 0.0, p_, with_memory=False)[0])
        R_ref, R_mem = np.array(R_ref), np.array(R_mem)
        R_pro, R_ins = np.array(R_pro), np.array(R_ins)

        u_ref = _signal(R_ref, U_hat, masque, n)
        u_mem = _signal(R_mem, U_hat, masque, n)
        u_pro = _signal(R_pro, U_hat, masque, n)
        u_ins = _signal(R_ins, U_hat, masque, n)

        def err(x):
            return float(np.linalg.norm(x - u_ref)
                         / max(np.linalg.norm(u_ref), 1e-14) * 100)

        E_mem, E_pro, E_ins = err(u_mem), err(u_pro), err(u_ins)
        run.value("ERtransMemoire", E_mem,
                  "L2 error (%) of the memory model on the reflected signal", fmt="%.2f")
        run.value("ERtransProny", E_pro,
                  f"L2 error (%) of the approximated memory model (Prony N={n_exp})",
                  fmt="%.2f")
        run.value("ERtransInstantane", E_ins,
                  "L2 error (%) of the instantaneous model on the reflected signal",
                  fmt="%.1f")
        run.value("gainTransitoire", E_ins / max(E_mem, 1e-12),
                  "gain of the memory in the transient regime", fmt="%.1f")
        print(f"  error / reference:  memory {E_mem:.2f} %   "
              f"Prony {E_pro:.2f} %   instantaneous {E_ins:.1f} %")
        print(f"  memory gain: {E_ins / max(E_mem, 1e-12):.1f}×")

        run.table("signaux_temporels", {
            "t": t, "U_inc": U_inc, "U_refl_reference": u_ref,
            "U_refl_memoire": u_mem, "U_refl_prony": u_pro,
            "U_refl_instantane": u_ins,
        })
        run.table("spectres_reflexion", {
            "omega": w, "abs_R_reference": np.abs(R_ref),
            "abs_R_memoire": np.abs(R_mem), "abs_R_prony": np.abs(R_pro),
            "abs_R_instantane": np.abs(R_ins),
        })

        # === fig12: signals, at TWO viscosities ==============================
        # Left panel: the test case of the paper (weakly viscous matrix).
        # The four curves overlap there — and it must be said: in this
        # regime, the memory changes almost nothing.
        # Right panel: strongly viscous matrix. The instantaneous model
        # visibly breaks away there, while the memory model follows the
        # reference. This is where the paper's thesis is at stake.
        fenetre = (t > b["t0"] - 4 * b["tau0"]) & (t < b["t0"] + 4 * b["tau0"])
        vr_fort = b["visc_ratio_fort"]
        p_fort = cfg.params(visc_ratio=vr_fort)

        Rr2 = np.array([compute_R_T_reference(om, 0.0, p_fort)[0] for om in w])
        Rm2 = np.array([compute_R_T_homogenized(om, 0.0, p_fort)[0] for om in w])
        Ri2 = np.array([compute_R_T_homogenized(om, 0.0, p_fort,
                                                with_memory=False)[0] for om in w])
        u_ref2 = _signal(Rr2, U_hat, masque, n)
        u_mem2 = _signal(Rm2, U_hat, masque, n)
        u_ins2 = _signal(Ri2, U_hat, masque, n)
        nr2 = max(np.linalg.norm(u_ref2), 1e-14)
        E_mem2 = float(np.linalg.norm(u_mem2 - u_ref2) / nr2 * 100)
        E_ins2 = float(np.linalg.norm(u_ins2 - u_ref2) / nr2 * 100)
        run.value("ERtransMemoireFort", E_mem2,
                  f"error (%) of the memory model, viscous matrix (ωβ/μ = {vr_fort})",
                  fmt="%.1f")
        run.value("ERtransInstantaneFort", E_ins2,
                  f"error (%) of the instantaneous model, viscous matrix (ωβ/μ = {vr_fort})",
                  fmt="%.0f")
        run.value("viscRatioFort", vr_fort, "viscosity of the 'dissipative matrix' case",
                  fmt="%g")

        fig, axes = plt.subplots(1, 2, figsize=DOUBLE, sharex=True)
        cas = [
            (u_ref, u_mem, u_pro, u_ins, E_mem, E_pro, E_ins,
             p_.visc_ratio, "Weakly viscous matrix (test case)"),
            (u_ref2, u_mem2, None, u_ins2, E_mem2, None, E_ins2,
             vr_fort, "Dissipative matrix"),
        ]
        for ax, (ur, um, up, ui, em, ep, ei, vr, titre) in zip(axes, cas):
            ax.plot(t[fenetre], ur[fenetre], "k-", lw=2.6,
                    label="Actual problem (reference)")
            ax.plot(t[fenetre], um[fenetre], "b--", lw=1.9,
                    label=f"Homogenized, memory — {em:.1f} %")
            if up is not None:
                ax.plot(t[fenetre], up[fenetre], "g-.", lw=1.5,
                        label=f"Prony $N={n_exp}$ — {ep:.1f} %")
            ax.plot(t[fenetre], ui[fenetre], "r:", lw=2.0,
                    label=f"Instantaneous ($K_B = 0$) — {ei:.0f} %")
            ax.set_xlabel(r"$t \, c_m / h$")
            ax.set_title(rf"{titre}: $\omega\beta_m/\mu_m = {vr:g}$", fontsize=11)
            ax.legend(fontsize=8, loc="upper left")
            ax.grid(True, alpha=0.3)
        axes[0].set_ylabel(r"$U^{\rm refl}(t)$")
        fig.tight_layout()
        run.figure(fig, "fig12_signaux_temporels.pdf")
        plt.close(fig)

        # === the viscosity sweep (data for panel c) ===========================
        e_ins_v, e_mem_v = [], []
        for vr in b["visc_ratios"]:
            p_vr = cfg.params(visc_ratio=vr)
            Rr = np.array([compute_R_T_reference(om, 0.0, p_vr)[0] for om in w])
            Rm = np.array([compute_R_T_homogenized(om, 0.0, p_vr)[0] for om in w])
            Ri = np.array([compute_R_T_homogenized(om, 0.0, p_vr,
                                                   with_memory=False)[0] for om in w])
            ur = _signal(Rr, U_hat, masque, n)
            nr = max(np.linalg.norm(ur), 1e-14)
            e_mem_v.append(np.linalg.norm(_signal(Rm, U_hat, masque, n) - ur) / nr * 100)
            e_ins_v.append(np.linalg.norm(_signal(Ri, U_hat, masque, n) - ur) / nr * 100)
            print(f"  ωβ/μ = {vr:<6g}: memory {e_mem_v[-1]:6.2f} %   "
                  f"instantaneous {e_ins_v[-1]:6.2f} %")
        run.table("influence_viscosite", {
            "visc_ratio": b["visc_ratios"],
            "E_memoire_pct": e_mem_v, "E_instantane_pct": e_ins_v,
        })

        # === fig13: the memory regime, in three panels of the SAME size ======
        # (a) pulse spectrum against lambda, (b) frequency-by-frequency
        # reflection, (c) transient error as a function of viscosity.
        fig, axes = plt.subplots(1, 3, figsize=(12.4, 3.9))

        axes[0].plot(omegas[masque], np.abs(U_hat[masque]) / np.abs(U_hat).max(),
                     "k-", lw=2)
        axes[0].axvline(lam, color="r", ls="--", lw=1.2,
                        label=rf"$\lambda = \mu_m/\beta_m = {lam:.0f}$")
        axes[0].set_xlabel(r"$\omega h / c_m$")
        axes[0].set_ylabel(r"$|\widehat{U}^{\rm inc}(\omega)|$ (normalized)")
        axes[0].set_title("(a) Spectrum of the incident pulse", fontsize=10.5)
        axes[0].legend(fontsize=8, loc="best")

        axes[1].plot(w, np.abs(R_ref), "k-", lw=2.2, label="Reference")
        axes[1].plot(w, np.abs(R_mem), "b--", lw=1.8, label="Memory")
        axes[1].plot(w, np.abs(R_ins), "r:", lw=1.8, label="Instantaneous")
        axes[1].set_xlabel(r"$\omega h / c_m$")
        axes[1].set_ylabel(r"$|R(\omega)|$")
        axes[1].set_title("(b) Frequency-by-frequency reflection", fontsize=10.5)
        axes[1].legend(fontsize=8, loc="best")

        axes[2].loglog(b["visc_ratios"], e_ins_v, "r--s", lw=2, ms=5,
                       label="instantaneous")
        axes[2].loglog(b["visc_ratios"], e_mem_v, "b-o", lw=2, ms=5,
                       label="memory")
        axes[2].set_xlabel(r"$\omega\beta_m/\mu_m$ (viscosity)")
        axes[2].set_ylabel(r"$L^2$ error on $U^{\rm refl}$ (%)")
        axes[2].set_title("(c) Error vs viscosity", fontsize=10.5)
        axes[2].legend(fontsize=8, loc="best")
        fig.tight_layout()
        run.figure(fig, "fig13_spectres.pdf")
        plt.close(fig)


if __name__ == "__main__":
    main()
