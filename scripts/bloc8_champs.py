"""Block 8 — Fields: the choice of the thickness a, seen on the field itself.

Produces: fig18_champs.pdf       (actual field + model error, a = e, oblique)
          fig19_champ_vs_a.pdf   (actual vs homogenized field for a = 0, e, 2e)
          fig20_erreur_H1_vs_a.pdf (H¹ deviation as a function of a/e, near/far)
Data    : results/<config>/bloc8/*.csv + manifest.json

WHAT THESE FIGURES SHOW
-----------------------
`fig19` reproduces, for our viscoelastic matrix, the demonstration of
figure 5 of Marigo et al. (2017): the ACTUAL field (finite elements, which
resolves the microstructure) and the HOMOGENIZED field are placed side by
side for three interface thicknesses a = 0, e, 2e. One sees — and the H¹
deviation quantifies it — that only a = e reproduces the field; a = 0 and
a = 2e distort it.

`fig20` gives the continuous version (figures 6-7 of Marigo): the H¹
deviation of eq. (64), as a function of a/e, split into NEAR field (close to
the row, where the evanescent field lives that the interface encapsulates
without reproducing it) and FAR field. The minimum is sharp at a = e.

`fig18` (oblique incidence, a = e) shows what Marigo's elastic matrix cannot
show: the near field around the inclusions, and the fact that omitting the
coefficient C — absent at θ = 0 — destroys the field as soon as θ ≠ 0.

MEASURE OF THE DEVIATION (Marigo, eq. 64)
-----------------------------------------
    E_H1 = ‖V − V_num‖_{H¹(region)} / ‖V_num‖_{H¹(region)} ,
with the H¹ norm ∫(|V|² + |∇V|²). The left face of the interface is aligned
with the row (x_L = −e/2) for every thickness a: the homogenized field on the
left (incident + reflected, coefficient R(a)) is therefore defined for all a,
and that is where it is compared with the actual field. Near field =
y₁ < x_L; far field = y₁ < x_L − 1 (one period further left), beyond the
evanescent field.

    python run.py bloc8 [--config config/xxx.yaml]
"""

import numpy as np

from fading_memory import RunRecorder, load_config
from fading_memory.physics import compute_R_T_homogenized, matrix_modulus
from matplotlib.patches import Rectangle

from fading_memory.plotting import apply_style, plt
from fading_memory.scattering_fem import _CACHE, R_T_reference


def _homogenized_field(Y1, Y2, kh, theta, p_, a, with_C=True):
    """Model field: plane waves, interface on [−e/2, −e/2 + a]."""
    M_m = matrix_modulus(-1j * kh, p_)
    k = kh * np.sqrt(1.0 / M_m)
    k2 = k * np.sin(theta)
    k1 = np.sqrt(k ** 2 - k2 ** 2)
    R, T = compute_R_T_homogenized(kh, theta, p_, a=a, with_C=with_C)

    x_L, x_R = -p_.e_over_h / 2.0, -p_.e_over_h / 2.0 + a
    U = np.full(Y1.shape, np.nan, dtype=complex)
    g, d = Y1 < x_L, Y1 > x_R
    U[g] = np.exp(1j * k2 * Y2[g]) * (np.exp(1j * k1 * Y1[g])
                                      + R * np.exp(-1j * k1 * Y1[g]))
    U[d] = np.exp(1j * k2 * Y2[d]) * T * np.exp(1j * k1 * Y1[d])
    return U, (g | d)


def _ecart_H1(U, U_ref, y1, y2, seuil):
    """Relative H¹ deviation (Marigo, eq. 64) on the left region {y₁ < seuil}.

    U and U_ref are given on the (y1, y2) grid; we restrict to the left
    subdomain, where the homogenized field is defined for every thickness a
    (no interface hole), then take the H¹ norm ∫(|·|² + |∇·|²). The derivatives
    are numerical on both sides — as in Marigo, who notes that this
    differentiation adds numerical noise close to the near field.
    """
    gauche = y1 < seuil
    if gauche.sum() < 3:
        return float("nan")
    y1g = y1[gauche]
    diff = U[gauche, :] - U_ref[gauche, :]

    def _norme2(F):
        dF1 = np.gradient(F, y1g, axis=0)
        dF2 = np.gradient(F, y2, axis=1)
        return np.sum(np.abs(F) ** 2 + np.abs(dF1) ** 2 + np.abs(dF2) ** 2)

    num = _norme2(diff)
    den = _norme2(U_ref[gauche, :])
    return float(np.sqrt(num / max(den, 1e-30)))


def main(cfg=None):
    cfg = cfg or load_config()
    apply_style()
    p_ = cfg.params()
    b = cfg.bloc("bloc8")

    with RunRecorder("bloc8", cfg) as run:
        kh = b["kh"]
        theta = np.deg2rad(b["theta_deg"])
        X, raf = b["X"], b["raffinement"]
        e_h = p_.e_over_h
        a_e = e_h                                       # a = e
        x_L = -e_h / 2.0                                # left face of the interface
        seuil_proche = x_L                              # near field: y₁ < x_L
        seuil_lointain = x_L - 1.0                      # far field: one period further

        # --- actual reference field (one FEM solve, reused) -------------------
        R_ref, _ = R_T_reference(kh, p_, theta=theta, X=X, raffinement=raf)
        maillage = _CACHE[(p_.e_over_h, p_.phi, X, raf)]
        Y1, Y2, U_ref = maillage.field(n_periodes=b["n_periodes"])
        y1 = Y1[:, 0]
        y2 = Y2[0, :]

        # =====================================================================
        # fig18 — actual field + error (complete model / C omitted), a = e, oblique
        # =====================================================================
        U_c, masque = _homogenized_field(Y1, Y2, kh, theta, p_, a_e, with_C=True)
        U_0, _ = _homogenized_field(Y1, Y2, kh, theta, p_, a_e, with_C=False)

        err_c = np.abs(U_c - U_ref); err_c[~masque] = np.nan
        err_0 = np.abs(U_0 - U_ref); err_0[~masque] = np.nan
        n_ref = np.sqrt(np.nanmean(np.abs(U_ref[masque]) ** 2))
        e_c = float(np.sqrt(np.nanmean(err_c[masque] ** 2)) / n_ref)
        e_0 = float(np.sqrt(np.nanmean(err_0[masque] ** 2)) / n_ref)
        run.value("champErreurAvecC", e_c, "L2 error on the field, complete model",
                  fmt="%.1e")
        run.value("champErreurSansC", e_0, "L2 error on the field, C omitted", fmt="%.2f")
        print(f"  fig18 : field error  with C = {e_c:.2e}   without C = {e_0:.2e}")

        run.table("coupe_champ", {
            "y1": y1,
            "Re_U_reference": np.real(U_ref[:, U_ref.shape[1] // 2]),
            "Re_U_avec_C": np.real(U_c[:, U_c.shape[1] // 2]),
            "Re_U_sans_C": np.real(U_0[:, U_0.shape[1] // 2]),
        })

        fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.6))
        vmax_e = float(np.nanmax(err_0))
        im0 = axes[0].pcolormesh(Y1, Y2, np.real(U_ref), cmap="RdBu_r",
                                 shading="auto", vmin=-1.6, vmax=1.6,
                                 rasterized=True)
        axes[0].set_title(r"Actual field $\mathrm{Re}\,U$" + "\n(reference, finite elements)",
                          fontsize=10)
        fig.colorbar(im0, ax=axes[0], fraction=0.046)
        for ax, err, titre in [
            (axes[1], err_c, r"Error of the complete model" + f"\n({e_c:.1e})"),
            (axes[2], err_0, r"Error when $C$ is omitted" + f"\n({e_0:.2f})"),
        ]:
            im = ax.pcolormesh(Y1, Y2, err, cmap="inferno", shading="auto",
                               vmin=0.0, vmax=vmax_e, rasterized=True)
            ax.set_title(titre, fontsize=10)
            fig.colorbar(im, ax=ax, fraction=0.046)
        _dessine_inclusions(axes, p_, b["n_periodes"])
        for ax in axes:
            ax.set_xlabel(r"$y_1 = X_1/h$"); ax.set_aspect("auto"); ax.grid(False)
        axes[0].set_ylabel(r"$y_2 = X_2/h$")
        fig.suptitle(rf"$kh = {kh}$, $\theta = {b['theta_deg']}^\circ$; the rectangles "
                     r"are the inclusions, the white band the effective interface",
                     fontsize=10, y=1.03)
        fig.tight_layout()
        run.figure(fig, "fig18_champs.pdf")
        plt.close(fig)

        # =====================================================================
        # fig19 — actual vs homogenized field for a = 0, e, 2e (analogue of Fig. 5)
        # =====================================================================
        a_vals = [av * e_h for av in b["a_over_e_values"]]        # a/e -> a
        fields, ecarts = [], []
        for a in a_vals:
            U_a, _ = _homogenized_field(Y1, Y2, kh, theta, p_, a, with_C=True)
            fields.append(U_a)
            ecarts.append(_ecart_H1(U_a, U_ref, y1, y2, seuil_proche))

        # citable values: the H¹ deviation (near field) for a = 0, e, 2e
        for av, ec in zip(b["a_over_e_values"], ecarts):
            nom = {0.0: "ecartHunAzero", 1.0: "ecartHunAe",
                   2.0: "ecartHunADeuxe"}.get(float(av))
            if nom:
                run.value(nom, ec * 100.0,
                          f"H¹ deviation (%) near field, a = {av:g} e", fmt="%.0f")
        i_e = [float(av) for av in b["a_over_e_values"]].index(1.0)
        print("  fig19 : near H¹ deviation  " +
              "  ".join(f"a={av:g}e:{ec*100:.0f}%"
                        for av, ec in zip(b["a_over_e_values"], ecarts)))

        run.table("champ_vs_a", {
            "y1": y1,
            **{f"Re_U_homog_a_{av:g}e": np.real(U[:, U.shape[1] // 2])
               for av, U in zip(b["a_over_e_values"], fields)},
            "Re_U_reference": np.real(U_ref[:, U_ref.shape[1] // 2]),
        })

        n_pan = len(a_vals) + 1
        fig, axes = plt.subplots(1, n_pan, figsize=(3.15 * n_pan, 3.6), sharey=True)
        vlim = 1.6
        im = axes[0].pcolormesh(Y1, Y2, np.real(U_ref), cmap="RdBu_r",
                                shading="auto", vmin=-vlim, vmax=vlim,
                                rasterized=True)
        axes[0].set_title(r"Actual field $\mathrm{Re}\,U$" + "\n(reference)", fontsize=10)
        for ax, av, U, ec in zip(axes[1:], b["a_over_e_values"], fields, ecarts):
            ax.pcolormesh(Y1, Y2, np.real(U), cmap="RdBu_r", shading="auto",
                          vmin=-vlim, vmax=vlim, rasterized=True)
            marque = r"\;\star" if float(av) == 1.0 else ""
            ax.set_title(rf"$a = {av:g}\,e{marque}$" + f"\n$E_{{H^1}} = {ec*100:.0f}\\,\\%$",
                         fontsize=10)
            # white band of the interface [x_L, x_L + a]
            ax.add_patch(Rectangle((x_L, Y2.min()), av * e_h, Y2.max() - Y2.min(),
                                   facecolor="white", edgecolor="none", zorder=3))
        _dessine_inclusions([axes[0]], p_, b["n_periodes"])
        for ax in axes:
            ax.axvline(x_L, color="green", ls=":", lw=1.0, zorder=4)
            ax.set_xlabel(r"$y_1 = X_1/h$"); ax.set_aspect("auto"); ax.grid(False)
        axes[0].set_ylabel(r"$y_2 = X_2/h$")
        fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.02, pad=0.01)
        fig.suptitle(rf"Actual and homogenized fields, $kh = {kh}$, "
                     rf"$\theta = {b['theta_deg']}^\circ$ ; only $a = e$ reproduces the field",
                     fontsize=10, y=1.02)
        run.figure(fig, "fig19_champ_vs_a.pdf")
        plt.close(fig)

        # =====================================================================
        # fig20 — H¹ deviation as a function of a/e, near / far field (Fig. 6-7)
        # =====================================================================
        a_ratios = np.linspace(b["a_over_e_min"], b["a_over_e_max"], b["n_a_courbe"])
        khs_c = b["kh_courbe"]

        fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2), sharey=True)
        couleurs = plt.cm.viridis(np.linspace(0.15, 0.85, len(khs_c)))
        opt_proche = {}
        for khc, coul in zip(khs_c, couleurs):
            R_c, _ = R_T_reference(khc, p_, theta=theta, X=X, raffinement=raf)
            mesh_c = _CACHE[(p_.e_over_h, p_.phi, X, raf)]
            Y1c, Y2c, Uref_c = mesh_c.field(n_periodes=b["n_periodes"])
            y1c, y2c = Y1c[:, 0], Y2c[0, :]
            e_proche, e_lointain = [], []
            for r in a_ratios:
                Ua, _ = _homogenized_field(Y1c, Y2c, khc, theta, p_, r * e_h)
                e_proche.append(_ecart_H1(Ua, Uref_c, y1c, y2c, seuil_proche))
                e_lointain.append(_ecart_H1(Ua, Uref_c, y1c, y2c, seuil_lointain))
            e_proche = np.asarray(e_proche); e_lointain = np.asarray(e_lointain)
            opt_proche[khc] = float(a_ratios[int(np.argmin(e_proche))])
            axes[0].semilogy(a_ratios, e_proche, "-o", color=coul, ms=3, lw=1.6,
                             label=rf"$kh = {khc:g}$")
            axes[1].semilogy(a_ratios, e_lointain, "-o", color=coul, ms=3, lw=1.6,
                             label=rf"$kh = {khc:g}$")
            run.table(f"ecart_H1_vs_a_kh_{khc:g}", {
                "a_over_e": a_ratios, "E_H1_proche": e_proche,
                "E_H1_lointain": e_lointain,
            })
            print(f"  fig20 : kh={khc:g}  near a* = {opt_proche[khc]:.2f}  "
                  f"E_near(a=e) = {e_proche[np.argmin(np.abs(a_ratios-1))]*100:.0f}%")

        for ax, titre in [(axes[0], "Near field  ($y_1 < x_L$)"),
                          (axes[1], "Far field  ($y_1 < x_L - 1$)")]:
            ax.axvline(1.0, color="black", ls="--", lw=1.2, label=r"$a = e$")
            ax.set_xlabel(r"$a/e$")
            ax.set_title(titre, fontsize=11)
            ax.grid(True, which="both", alpha=0.3)
            ax.legend(fontsize=8)
        axes[0].set_ylabel(r"$E_{H^1}$ (relative deviation in the norm $H^1$)")
        run.value("aOptimalChamp", opt_proche[kh] if kh in opt_proche
                  else opt_proche[khs_c[0]],
                  "a/e minimizing the H¹ deviation (near field)", fmt="%.2f")
        fig.suptitle(r"The deviation from the actual field is minimal at $a = e$, "
                     r"for all frequencies", fontsize=11, y=1.0)
        fig.tight_layout()
        run.figure(fig, "fig20_erreur_H1_vs_a.pdf")
        plt.close(fig)


def _dessine_inclusions(axes, p_, n_periodes):
    """Draw the inclusion rectangles (one per period) on each axis."""
    a1, a2 = p_.e_over_h / 2.0, p_.phi / 2.0
    for ax in axes:
        for k in range(n_periodes):
            ax.add_patch(Rectangle((-a1, k + 0.5 - a2), 2 * a1, 2 * a2,
                                   facecolor="none", edgecolor="k",
                                   lw=1.1, zorder=6))


if __name__ == "__main__":
    main()
