"""Block 3 — Harmonic scattering: coefficients R and T as functions of kh.

Produces: fig09_RT_vs_kh.pdf (|R|, |T|, errors — three panels)
Data: results/<config>/bloc3/*.csv + manifest.json

The reference here is the TRUE solution of the inclusion problem (finite
elements, `scattering_fem.py`), and no longer the effective-layer model of the
first version — which saw no microstructure and therefore validated nothing
(debt D1). Radiation is handled there by a ZERO-MODE impedance condition
(exact on Rayleigh order 0; the evanescent modes are not zero on the truncated
boundary, only exponentially small) — hence the truncation deviation
measured below (`troncatureScattering`).

⚠ Normal incidence: at theta = 0, the C term vanishes from the jump
conditions (it is carried by ∂₂Σ₂). These figures validate B and S, NOT C.

    python run.py bloc3 [--config config/xxx.yaml]
"""

import numpy as np

from fading_memory import RunRecorder, load_config
from fading_memory.physics import (compute_R_T_homogenized,
                                   compute_R_T_reference)
from fading_memory.plotting import apply_style, plt
from fading_memory.scattering_fem import R_T_reference


def _error_relative(approx, ref):
    """Relative error, falling back to absolute when the reference vanishes."""
    ref = np.asarray(ref)
    approx = np.asarray(approx)
    denom = np.where(np.abs(ref) > 1e-8, np.abs(ref), 1.0)
    return np.abs(approx - ref) / denom


def main(cfg=None):
    cfg = cfg or load_config()
    apply_style()
    p_ = cfg.params()
    b = cfg.bloc("bloc3")

    with RunRecorder("bloc3", cfg) as run:
        run.note("Reference = FEM solution of the inclusion problem (D1 closed). "
                 "Normal incidence: C plays no role, only B and S are validated.")

        khs = np.linspace(b["kh_min"], b["kh_max"], b["n_kh"])
        theta = b["theta"]

        R_mem, T_mem, R_inst, T_inst, R_ref, T_ref = ([] for _ in range(6))
        for kh in khs:
            omega = kh  # nondimensionalization: c_m / h = 1
            r_m, t_m = compute_R_T_homogenized(omega, theta, p_, with_memory=True)
            r_i, t_i = compute_R_T_homogenized(omega, theta, p_, with_memory=False)
            r_r, t_r = compute_R_T_reference(omega, theta, p_)
            R_mem.append(abs(r_m)); T_mem.append(abs(t_m))
            R_inst.append(abs(r_i)); T_inst.append(abs(t_i))
            R_ref.append(abs(r_r)); T_ref.append(abs(t_r))

        E_R_mem = _error_relative(R_mem, R_ref)
        E_R_inst = _error_relative(R_inst, R_ref)
        E_T_mem = _error_relative(T_mem, T_ref)
        E_T_inst = _error_relative(T_inst, T_ref)

        run.table("RT_vs_kh", {
            "kh": khs,
            "R_ref": R_ref, "R_memoire": R_mem, "R_instantane": R_inst,
            "T_ref": T_ref, "T_memoire": T_mem, "T_instantane": T_inst,
            "E_R_memoire": E_R_mem, "E_R_instantane": E_R_inst,
            "E_T_memoire": E_T_mem, "E_T_instantane": E_T_inst,
        })

        # The radiation condition is zero-mode: the evanescent modes are not
        # zero on the truncated boundary, only exponentially small. The
        # deviation is MEASURED (the paper cites it), it is not assumed.
        kh_tronc = float(khs[len(khs) // 2])
        R_X2, _ = R_T_reference(kh_tronc, p_, X=2.0)
        R_X3, _ = R_T_reference(kh_tronc, p_, X=3.0)
        run.value("troncatureScattering", abs(abs(R_X2) - abs(R_X3)),
                  f"deviation on |R| between truncations X=2 and X=3 (kh = {kh_tronc:g})",
                  fmt="%.1e")
        print(f"  truncation: | |R|(X=2) - |R|(X=3) | = "
              f"{abs(abs(R_X2) - abs(R_X3)):.2e} at kh = {kh_tronc:g}")

        # Citable values: maximum error over the studied kh range
        run.value("ERmemMax", float(E_R_mem.max()),
                  "max relative error on |R|, memory model", fmt="%.2e")
        run.value("ERinstMax", float(E_R_inst.max()),
                  "max relative error on |R|, instantaneous model", fmt="%.2e")
        run.value("khMax", float(khs.max()), "upper bound of the kh sweep", fmt="%.2f")
        print(f"  E_R max: memory = {E_R_mem.max():.3e}   instantaneous = {E_R_inst.max():.3e}")

        # --- local convergence orders -------------------------------------------
        # Order measured AT kh0: log-log slope of E_R (memory model) between
        # kh0/2 and kh0. The grid being linear and kh0 taken on the grid,
        # both points exist exactly (up to the rounding of the step).
        noms = ["ordreKhPetit", "ordreKhMoyen", "ordreKhGrand"]
        for nom, kh0 in zip(noms, b["kh_ordres"]):
            i0 = int(np.argmin(np.abs(khs - kh0 / 2.0)))
            i1 = int(np.argmin(np.abs(khs - kh0)))
            slope = float(np.log(E_R_mem[i1] / E_R_mem[i0])
                          / np.log(khs[i1] / khs[i0]))
            run.value(nom, slope,
                      f"local order of E_R (memory), log-log slope between "
                      f"kh = {khs[i0]:g} and {khs[i1]:g}", fmt="%.2f")
            print(f"  local order at kh = {khs[i1]:g}: {slope:.3f}")
        run.value("khOrdreMin", float(b["kh_ordres"][0]),
                  "smallest kh where the local order is measured", fmt="%g")
        run.value("khOrdreMax", float(b["kh_ordres"][-1]),
                  "largest kh where the local order is measured", fmt="%g")

        # --- fig09: |R|, |T| and errors — three panels of the SAME size ------
        # (a) |R|, (b) |T|, (c) relative errors. A single figure: the three
        # panels have exactly the same dimensions (presentation requirement).
        fig, axes = plt.subplots(1, 3, figsize=(12.4, 3.9))
        for ax, (ref, mem, inst, lab, titre) in zip(axes, [
            (R_ref, R_mem, R_inst, "R", "(a) Reflection coefficient $|R|$"),
            (T_ref, T_mem, T_inst, "T", "(b) Transmission coefficient $|T|$"),
        ]):
            ax.plot(khs, ref, "k-", lw=2.2, label="Actual problem (FEM ref.)")
            ax.plot(khs, mem, "b--", lw=1.8, label="Homogenized, memory")
            ax.plot(khs, inst, "r:", lw=1.8, label="Homogenized, instantaneous")
            ax.set_xlabel(r"$kh$")
            ax.set_ylabel(rf"$|{lab}|$")
            ax.set_title(titre, fontsize=10.5)
            ax.legend(fontsize=8, loc="best")

        axe = axes[2]
        axe.semilogy(khs, E_R_mem, "b-o", lw=1.8, ms=4, label=r"$E_R$ memory")
        axe.semilogy(khs, E_R_inst, "b--s", lw=1.5, ms=4, label=r"$E_R$ instantaneous")
        axe.semilogy(khs, E_T_mem, "r-^", lw=1.8, ms=4, label=r"$E_T$ memory")
        axe.semilogy(khs, E_T_inst, "r--d", lw=1.5, ms=4, label=r"$E_T$ instantaneous")
        axe.set_xlabel(r"$kh$")
        axe.set_ylabel("Relative error")
        axe.set_title(r"(c) Relative errors $E_R$, $E_T$", fontsize=10.5)
        axe.legend(fontsize=8, loc="best")
        fig.tight_layout()
        run.figure(fig, "fig09_RT_vs_kh.pdf")
        plt.close(fig)


if __name__ == "__main__":
    main()
