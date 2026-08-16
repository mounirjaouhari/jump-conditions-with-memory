"""Block 9 — Passivity on the half-plane: Stieltjes measure.

Reproducible proof of the passivity theorem (complex half-plane). The
interface coefficient G(r) = B_1(r) is a Stieltjes function of the
contrast,

    G(r) = c + ∫_[0,∞) dμ(z)/(z+r),   μ ≥ 0   (Bergman–Milton representation),

and it is the POSITIVITY of μ which, composed with the homography r(p), makes
Re[p·K̂_B(p)] ≥ 0 on the whole half-plane Re(p) > 0. This block establishes
numerically the two load-bearing facts:

  (i)  μ'(z) = -(1/π) Im G(-z + iη) ≥ 0   (positivity of the measure);
  (ii) G(r0) - G(r1) reconstructed from μ ≈ direct computation  (the
       representation really is a Stieltjes one in the contrast variable).

Output: fig14_mesure_stieltjes.pdf
Data  : results/<config>/bloc9/*.csv + manifest.json

    python run.py bloc9 [--config config/xxx.yaml]
"""

import numpy as np

from fading_memory import RunRecorder, load_config
from fading_memory.physics import (identite_herglotz, mesure_stieltjes,
                                   reconstruction_stieltjes)
from fading_memory.plotting import DOUBLE, apply_style, plt


def main(cfg=None):
    cfg = cfg or load_config()
    apply_style()
    p_ = cfg.params()
    b = cfg.bloc("bloc9")

    with RunRecorder("bloc9", cfg) as run:
        # --- (0) Herglotz identity: Im B = -Im r ∫|∇V¹|², Im C = +Im r ∫|∇V²|²
        # This is the PROOF of μ ≥ 0 (passivity theorem); we verify the identity
        # on complex contrasts from both half-planes.
        errs_h = [identite_herglotz(complex(re, im), p_)
                  for re, im in b["herglotz_r"]]
        err_herglotz = float(np.max(errs_h))
        run.value("herglotzResidu", err_herglotz,
                  "max relative deviation of the Herglotz identity (Im B, Im C)",
                  fmt="%.1e")
        print(f"  Herglotz identity: max relative deviation = {err_herglotz:.2e} "
              f"over {len(errs_h)} complex contrasts")
        if err_herglotz > 1e-8:
            run.note("HERGLOTZ IDENTITY VIOLATED — the proof of μ ≥ 0 "
                     "(passivity theorem) loses its numerical foundation.")
        # --- z grid: dense under the peak, coarse in the tail ------------------
        z_dense = np.linspace(b["z_min"], b["z_dense_max"], b["n_z_dense"])
        z_sparse = np.linspace(b["z_dense_max"] + 1e-3, b["z_max"], b["n_z_sparse"])
        z = np.concatenate([z_dense, z_sparse])
        eta = float(b["eta"])

        # --- (i) positivity of the Stieltjes measure --------------------------
        mu = mesure_stieltjes(z, p_, eta=eta)
        mu_min = float(mu.min())
        positif = mu_min >= -1e-3           # tolerance: FEM noise + O(η) smoothing
        run.value("stieltjesMuMin", mu_min,
                  "min of the Stieltjes density μ'(z) — positivity (passivity)",
                  fmt="%.2e")
        run.value("stieltjesEta", eta,
                  "smoothing η of the Stieltjes–Perron inversion", fmt="%.2g")
        run.value("stieltjesNbZ", int(z.size),
                  "number of points of the z grid", fmt="%d")
        print(f"  Stieltjes measure: min μ' = {mu_min:+.3e}  "
              f"(peak {mu.max():.1f} at z≈{z[int(np.argmax(mu))]:.2f})  → "
              f"{'μ ≥ 0 OK' if positif else 'μ < 0: passivity violated'}")
        if not positif:
            run.note("NEGATIVE STIELTJES MEASURE — the passivity theorem "
                     "on the half-plane is contradicted.")

        # --- (ii) reconstruction: consistency of the representation -----------
        r1s = list(b["r1_values"])
        directs, recons, err_rel = [], [], []
        for r1 in r1s:
            d, rec = reconstruction_stieltjes(p_, r1, z, eta=eta)
            directs.append(d)
            recons.append(rec)
            err_rel.append(abs(d - rec) / max(abs(d), 1e-12))
        err_max = 100.0 * float(np.max(err_rel))
        run.value("stieltjesReconErrMax", err_max,
                  "max relative deviation (%) reconstruction from μ vs direct G",
                  fmt="%.1f")
        print(f"  reconstruction G(r0)-G(r1): max relative deviation = {err_max:.1f} %")

        run.table("mesure_stieltjes", {"z": z, "mu_prime": mu})
        run.table("reconstruction_stieltjes", {
            "r1": np.asarray(r1s, dtype=float),
            "direct": np.asarray(directs),
            "reconstruit": np.asarray(recons),
            "error_relative": np.asarray(err_rel),
        })

        # --- fig14 -------------------------------------------------------------
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=DOUBLE)

        ax1.plot(z, mu, "b-", lw=2)
        ax1.axhline(0.0, color="k", lw=0.8, alpha=0.5)
        ax1.fill_between(z, 0, mu, where=(mu >= 0), color="b", alpha=0.12)
        ax1.set_xlim(0, min(b["z_max"], 10.0))
        ax1.set_xlabel(r"$z$")
        # ϑ (and not η): in the paper, η = kh is the small asymptotic
        # parameter; the smoothing of the inversion is written ϑ (S7).
        ax1.set_ylabel(r"$\mu'(z) = -\frac{1}{\pi}\,\mathrm{Im}\,G(-z+i\vartheta)$")
        ax1.set_title(r"Stieltjes measure $\mu' \geq 0$", fontsize=12)
        ax1.grid(True, alpha=0.3)

        ax2.plot(r1s, directs, "ko", ms=7, label="direct $G(r_0)-G(r_1)$")
        ax2.plot(r1s, recons, "r+", ms=11, mew=2, label=r"reconstructed from $\mu'$")
        ax2.set_xlabel(r"$r_1$")
        ax2.set_ylabel(r"$G(r_0) - G(r_1)$")
        ax2.set_title("Consistency of the representation", fontsize=12)
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3)

        fig.tight_layout()
        run.figure(fig, "fig14_mesure_stieltjes.pdf")
        plt.close(fig)


if __name__ == "__main__":
    main()
