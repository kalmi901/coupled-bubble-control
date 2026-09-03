import matplotlib.pyplot as plt

RC_PARAMS = {
    "text.usetex": False,
    # betűk
    "font.family": "serif",
    "font.size": 8,
    "font.serif": ["Times New Roman"],
    "mathtext.fontset": "dejavuserif",

    # vektoros export, font beágyazás
    "pdf.fonttype": 42,   # TrueType / beágyazható
    "ps.fonttype": 42,
    "svg.fonttype": "none",

    # vonalak / tengelyek
    "axes.linewidth": 0.6,
    "lines.linewidth": 1.0,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.minor.width": 0.5,
    "ytick.minor.width": 0.5,

    # tickek mérete
    "xtick.major.size": 3,
    "ytick.major.size": 3,
    "xtick.minor.size": 1.5,
    "ytick.minor.size": 1.5,

    # jelmagyarázat / elrendezés
    "legend.frameon": False,
    "legend.fontsize": 7,
    "figure.dpi": 200,
    "savefig.dpi": 300,
}


# Bubble Colors
COLORS = {
    0 : "#D55E00",   # vermilion
    1 : "#0072B2",   # blue
    2 : "#1CC82D",   # green
    3 : "#C81CA3"    # magenta
}

REF_STYLE = {
    "lw": 3.2,
    "alpha": 0.32,
    "solid_capstyle": "round",
    "zorder": 2,
}

GPU_STYLE = {
    "lw": 1.15,
    "alpha": 1.0,
    "solid_capstyle": "round",
    "zorder": 3,
}


def apply_plot_style() -> None:
    """Apply the common Matplotlib settings used for manuscript figures."""
    plt.rcParams.update(RC_PARAMS)

def two_column_figsize(nrows: int, ncols: int, *, width_in=7.0, panel_height_in=1.55,
                       top_in=0.35, bottom_in=0.55, left_in=0.65, right_in=0.15,
                       hspace=0.25, wspace=0.20):
    """
    width_in: két hasábos szélesség (gyakran 6.8–7.2 inch). Kezdd 7.0-val.
    panel_height_in: egy subplot (panel) ajánlott magassága inch-ben.
    A margók inch-ben vannak, így kiszámítható a teljes magasság.
    """
    height_in = top_in + bottom_in + nrows * panel_height_in + (nrows - 1) * (hspace * panel_height_in)
    return (width_in, height_in), dict(left=left_in/width_in,
                                       right=1 - right_in/width_in,
                                       bottom=bottom_in/height_in,
                                       top=1 - top_in/height_in,
                                       wspace=wspace,
                                       hspace=hspace)