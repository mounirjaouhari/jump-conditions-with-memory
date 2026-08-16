"""Block 0 — Geometry schematics (figures 1 to 4).

Produces: fig01_geometrie_reelle.pdf, fig02_cellule_elementaire.pdf,
          fig03_interface_homogeneisee.pdf, fig04_passage_schema.pdf

WHY THIS SCRIPT EXISTS
----------------------
These four figures used to ship with the paper without any code to produce
them: orphaned, non-regenerable, non-editable. `run.py verify` flagged them
on every execution.

They are schematics, not computation results: they depend only on the
geometry (phi, e/h, a), which they read from the configuration. Changing phi
in `config/default.yaml` therefore updates the schematics AND the
computations, together.

    python run.py bloc0 [--config config/xxx.yaml]
"""

import numpy as np
from matplotlib.patches import FancyArrowPatch, Rectangle

from fading_memory import RunRecorder, load_config
from fading_memory.plotting import SIMPLE, apply_style, plt

MATRICE = "#cfe2f3"
INCLUSION = "#e06666"
INTERFACE = "#93c47d"


def _inclusions(ax, e_h, phi, n=3, x0=0.0):
    """Draw n rectangular inclusions centered on x = x0, with period 1."""
    for k in range(-(n // 2), n // 2 + 1):
        ax.add_patch(Rectangle((x0 - e_h / 2, k - phi / 2), e_h, phi,
                               facecolor=INCLUSION, edgecolor="k", lw=1.2, zorder=3))
    return n


def main(cfg=None):
    cfg = cfg or load_config()
    apply_style()
    p_ = cfg.params()
    e_h, phi = p_.e_over_h, p_.phi

    with RunRecorder("bloc0", cfg) as run:
        run.note("Purely geometric schematics: no physical computation.")

        # === fig01: geometry of the actual problem ===========================
        # 3 inclusions (k = -1, 0, 1); dimension lines are placed OUTSIDE the motifs.
        fig, ax = plt.subplots(figsize=(7.6, 4.4))
        # the matrix fills the whole plane: the incident wave propagates in it
        ax.add_patch(Rectangle((-5, -2.8), 10, 5.6, facecolor=MATRICE,
                               edgecolor="none", zorder=0))
        _inclusions(ax, e_h, phi, n=3)

        # incident wave, in a free band above the inclusions
        x = np.linspace(-4.6, -1.6, 300)
        ax.plot(x, 2.15 + 0.22 * np.sin(5.5 * x), "k-", lw=1.4, zorder=4)
        ax.add_patch(FancyArrowPatch((-1.4, 2.15), (-0.4, 2.15),
                                     arrowstyle="-|>", mutation_scale=16, lw=1.4,
                                     color="k", zorder=4))
        ax.text(-4.6, 2.5, r"$U^{\rm inc}$", fontsize=12)

        # dimension e: below the domain
        ax.annotate("", xy=(-e_h / 2, -2.05), xytext=(e_h / 2, -2.05),
                    arrowprops=dict(arrowstyle="<->", lw=1.1))
        ax.text(0, -2.55, r"$e$", ha="center", fontsize=12)
        ax.plot([-e_h / 2, -e_h / 2], [-1.75, -2.05], "k:", lw=0.8)
        ax.plot([e_h / 2, e_h / 2], [-1.75, -2.05], "k:", lw=0.8)
        # dimension h: period, between two inclusions, on the right
        ax.annotate("", xy=(2.4, -1.0), xytext=(2.4, 0.0),
                    arrowprops=dict(arrowstyle="<->", lw=1.1))
        ax.text(2.6, -0.5, r"$h$", va="center", fontsize=12)
        # dimension phi.h: height of an inclusion, on the left
        ax.annotate("", xy=(-2.0, -phi / 2), xytext=(-2.0, phi / 2),
                    arrowprops=dict(arrowstyle="<->", lw=1.1))
        ax.plot([-2.0, -e_h / 2], [phi / 2, phi / 2], "k:", lw=0.8)
        ax.plot([-2.0, -e_h / 2], [-phi / 2, -phi / 2], "k:", lw=0.8)
        ax.text(-2.2, 0, r"$\varphi h$", va="center", ha="right", fontsize=12)

        ax.text(-4.8, -2.6, "viscoelastic matrix", fontsize=10)
        ax.text(1.3, 1.35, "elastic\ninclusions", fontsize=10, color=INCLUSION)
        ax.set_xlim(-5, 5); ax.set_ylim(-2.8, 2.8)
        ax.set_xlabel(r"$X_1$"); ax.set_ylabel(r"$X_2$")
        ax.set_aspect("equal"); ax.grid(False)
        fig.tight_layout()
        run.figure(fig, "fig01_geometrie_reelle.pdf")
        plt.close(fig)

        # === fig02: elementary cell =========================================
        fig, ax = plt.subplots(figsize=SIMPLE)
        ax.add_patch(Rectangle((-3, -0.5), 6, 1.0, facecolor=MATRICE,
                               edgecolor="k", lw=1.2, zorder=0))
        ax.add_patch(Rectangle((-e_h / 2, -phi / 2), e_h, phi,
                               facecolor=INCLUSION, edgecolor="k", lw=1.2, zorder=3))
        for y in (-0.5, 0.5):
            ax.axhline(y, color="k", ls="--", lw=1.0, alpha=0.7)
        ax.text(2.4, 0.56, "periodicity", fontsize=9, style="italic")
        ax.annotate("", xy=(-2.9, 0), xytext=(-2.3, 0),
                    arrowprops=dict(arrowstyle="<-", lw=1.2))
        ax.annotate("", xy=(2.9, 0), xytext=(2.3, 0),
                    arrowprops=dict(arrowstyle="<-", lw=1.2))
        ax.text(-2.85, 0.12, r"$y_1 \to -\infty$", fontsize=9)
        ax.text(1.75, 0.12, r"$y_1 \to +\infty$", fontsize=9)
        ax.text(0, 0, r"$M_i$", ha="center", va="center", fontsize=12, zorder=4)
        ax.text(-1.9, -0.3, r"$M_m$", fontsize=12)
        ax.set_xlim(-3.2, 3.2); ax.set_ylim(-0.75, 0.75)
        ax.set_xlabel(r"$y_1 = X_1/h$"); ax.set_ylabel(r"$y_2$")
        ax.set_title(r"Elementary cell $Y_\infty = \mathbb{R} \times [0,1]$",
                     fontsize=12)
        ax.set_aspect("equal"); ax.grid(False)
        fig.tight_layout()
        run.figure(fig, "fig02_cellule_elementaire.pdf")
        plt.close(fig)

        # === fig03: homogenized interface ===================================
        a = e_h  # choice a = e
        fig, ax = plt.subplots(figsize=SIMPLE)
        ax.add_patch(Rectangle((-3, -1.4), 6, 2.8, facecolor=MATRICE,
                               edgecolor="none", zorder=0))
        ax.add_patch(Rectangle((-a / 2, -1.4), a, 2.8, facecolor=INTERFACE,
                               edgecolor="k", lw=1.2, alpha=0.85, zorder=2))
        ax.annotate("", xy=(-a / 2, -1.15), xytext=(a / 2, -1.15),
                    arrowprops=dict(arrowstyle="<->", lw=1.1))
        ax.text(0, -1.35, r"$a$", ha="center", fontsize=12)
        ax.text(0, 0.75, "effective interface\nwith memory", ha="center", fontsize=10,
                zorder=4)
        # Matplotlib's mathtext does not know \llbracket: we write [[ · ]]
        ax.text(0, -0.35,
                r"$[\![U]\!] = h\,B(p,a)\,\langle \partial_1 U \rangle$",
                ha="center", fontsize=10, zorder=4)
        ax.text(-2.85, 1.15, r"$M_m(p)$", fontsize=11)
        ax.text(2.15, 1.15, r"$M_m(p)$", fontsize=11)
        ax.set_xlim(-3, 3); ax.set_ylim(-1.5, 1.5)
        ax.set_xlabel(r"$X_1$"); ax.set_ylabel(r"$X_2$")
        ax.set_aspect("equal"); ax.grid(False)
        fig.tight_layout()
        run.figure(fig, "fig03_interface_homogeneisee.pdf")
        plt.close(fig)

        # === fig04: actual -> effective transition ==========================
        fig, axes = plt.subplots(1, 2, figsize=(11.0, 3.6))
        axes[0].add_patch(Rectangle((-3, -1.6), 6, 3.2, facecolor=MATRICE,
                                    edgecolor="none"))
        _inclusions(axes[0], e_h, phi, n=3)
        axes[0].set_title("Actual problem: row of inclusions", fontsize=11)

        axes[1].add_patch(Rectangle((-3, -1.6), 6, 3.2, facecolor=MATRICE,
                                    edgecolor="none"))
        axes[1].add_patch(Rectangle((-a / 2, -1.6), a, 3.2, facecolor=INTERFACE,
                                    edgecolor="k", lw=1.2, alpha=0.85))
        axes[1].text(0, 0, "memory\n" + r"$K_B(t),\ K_C(t)$", ha="center",
                     va="center", fontsize=10)
        axes[1].set_title(r"Effective interface, $\eta = kh \to 0$", fontsize=11)

        for ax in axes:
            ax.set_xlim(-3, 3); ax.set_ylim(-1.7, 1.7)
            ax.set_xlabel(r"$X_1$")
            ax.set_aspect("equal"); ax.grid(False)
            ax.set_yticks([])
        fig.tight_layout()
        run.figure(fig, "fig04_passage_schema.pdf")
        plt.close(fig)

        print("  4 schematics regenerated — no orphaned figure left")


if __name__ == "__main__":
    main()
