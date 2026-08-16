"""Single figure style — publication quality.

The style used to be copy-pasted identically at the top of the blocks: as many
places to modify to change a font. It now lives here, and nowhere else.

Quality choices (SIAM / J. Comput. Phys. / Phys. Rev. standards):
- **vector** output (PDF): the curves stay sharp at any magnification;
  only the field maps (pcolormesh) are rasterized, at 400 dpi;
- mathematics in **Computer Modern** (`mathtext.fontset = cm`), to match the
  Latin Modern font of the document;
- closed box, tick marks pointing **inwards** on all four sides, visible
  minor ticks (useful for log axes), discreet grid below the data;
- fonts embedded in the PDF (`pdf.fonttype = 42`), a frequent publisher
  requirement.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # non-display backend: essential in CI / headless runs
import matplotlib.pyplot as plt  # noqa: E402
from cycler import cycler  # noqa: E402

# Sober, distinguishable line palette (blue, brick red, green, orange,
# purple, slate) — recognizable even when printed in grayscale thanks to the
# line styles chosen in the scripts.
_PALETTE = ["#1f4e79", "#b2182b", "#1b7837", "#d9820b", "#5e3c99", "#4d4d4d"]

STYLE = {
    # --- fonts ----------------------------------------------------------------
    "font.family": "serif",
    "mathtext.fontset": "cm",
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.titlesize": 11.5,
    # --- lines and colors -----------------------------------------------------
    "lines.linewidth": 1.6,
    "lines.markersize": 4.5,
    "lines.markeredgewidth": 0.8,
    "axes.linewidth": 0.8,
    "patch.linewidth": 0.8,
    "axes.prop_cycle": cycler(color=_PALETTE),
    # --- box, ticks, grid -----------------------------------------------------
    "axes.edgecolor": "#333333",
    "axes.axisbelow": True,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "ytick.right": True,
    "xtick.minor.visible": True,
    "ytick.minor.visible": True,
    "xtick.major.size": 4.0,
    "ytick.major.size": 4.0,
    "xtick.minor.size": 2.2,
    "ytick.minor.size": 2.2,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "xtick.minor.width": 0.6,
    "ytick.minor.width": 0.6,
    "axes.grid": True,
    "grid.color": "#cccccc",
    "grid.alpha": 0.6,
    "grid.linewidth": 0.5,
    "grid.linestyle": "-",
    # --- legend ---------------------------------------------------------------
    "legend.frameon": True,
    "legend.framealpha": 0.92,
    "legend.edgecolor": "#bbbbbb",
    "legend.fancybox": False,
    "legend.borderpad": 0.45,
    "legend.handlelength": 1.9,
    "legend.columnspacing": 1.2,
    # --- output ---------------------------------------------------------------
    "figure.dpi": 150,
    "savefig.dpi": 400,          # only affects rasterized elements (field maps)
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.03,
    "pdf.fonttype": 42,          # embedded (TrueType) fonts — publisher requirement
    "ps.fonttype": 42,
    "pdf.compression": 6,
}

# Standardized figure formats (inches)
SIMPLE = (6.4, 4.0)
SIMPLE_HAUTE = (6.4, 4.5)
DOUBLE = (11.0, 4.0)


def apply_style() -> None:
    """Call once at the top of every block script."""
    plt.rcParams.update(STYLE)


__all__ = ["apply_style", "plt", "SIMPLE", "SIMPLE_HAUTE", "DOUBLE", "STYLE"]
