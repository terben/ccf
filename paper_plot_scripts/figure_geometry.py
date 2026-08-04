"""
Figure 1 (\\label{fig:admissible} in CCF.tex) -- geometry of the
admissible region, Sect. 5.1.

Purely closed-form: p_3 and sigma_3^2 at order 3 are computed via the
small local helper `p3_sigma3` below rather than through schurcorr's
general recursion, since a two-line closed form is clearer here than a
general-N library call for this fixed, low-order visualisation (the
package's numerical core is exercised in figure_roundtrip.py and
figure_gaussianization.py instead).
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from pathlib import Path

from schurcorr.plotting import aa_plot


# ---------------------------------------------------------------------
# Optional titles -- set to None to disable
# ---------------------------------------------------------------------
TOP_TITLE = None
MIDDLE_TITLE = None

FIGDIR = Path(__file__).resolve().parent / "../figs"
FIGDIR.mkdir(exist_ok=True)


# ---------------------------------------------------------------------
# A&A style   -- height_ratio 0.80 -> 0.70 for a shorter figure
# ---------------------------------------------------------------------
aa_plot(column="double",
        height_ratio=0.70,
        fontsize=8.0,
        use_tex=True)

plt.rcParams["figure.constrained_layout.w_pad"] = 0.01
plt.rcParams["figure.constrained_layout.h_pad"] = 0.01

SIGMA_COLOR = "#6a3d9a"
BACKBONE_COLOR = "#c0392b"
GUIDE_LOWER_COLOR = "#9ecae1"
GUIDE_UPPER_COLOR = "#a1d99b"
BOUNDARY_COLOR = "black"
ALPHA1_GUIDE_COLOR = "#bbbbbb"
REGION_COLOR = "#e9f1f7"


def add_interval_width_fig1(ax, x, y_lower, y_mid, y_upper, label,
                            text_side=1, color=BACKBONE_COLOR, alpha=0.45):
    ax.annotate(
        "",
        xy=(x, y_upper),
        xytext=(x, y_lower),
        arrowprops=dict(
            arrowstyle="<->",
            color=color,
            alpha=alpha,
            lw=2.2,
            shrinkA=0,
            shrinkB=0,
            mutation_scale=13,
        ),
        zorder=8,
    )

    tick = 0.035
    ax.plot(
        [x - tick, x + tick],
        [y_mid, y_mid],
        color=color,
        alpha=0.75,
        lw=1.8,
        zorder=9,
    )

    x_offset = 0.055 * text_side
    ha = "left" if text_side > 0 else "right"
    ax.text(
        x + x_offset,
        0.5 * (y_lower + y_upper),
        label,
        color=color,
        alpha=0.85,
        ha=ha,
        va="center",
        rotation=90,
        zorder=9,
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.55, pad=1.5),
    )


def add_interval_width_fig2(ax, x, y_lower, y_mid, y_upper, label,
                            text_side=1, color=BACKBONE_COLOR, alpha=0.35):
    cap = 0.04

    ax.vlines(
        x, y_lower, y_upper,
        color=color,
        alpha=alpha,
        linewidth=4.0,
        zorder=7,
    )

    ax.hlines(
        [y_lower, y_mid, y_upper],
        x - cap, x + cap,
        color=color,
        alpha=[alpha, 0.75, alpha],
        linewidth=[2.0, 2.0, 2.0],
        zorder=8,
    )

    x_offset = 0.06 * text_side
    ha = "left" if text_side > 0 else "right"
    ax.text(
        x + x_offset,
        0.5 * (y_lower + y_upper),
        label,
        color=color,
        alpha=0.85,
        ha=ha,
        va="center",
        rotation=90,
        zorder=9,
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.55, pad=1.5),
    )


def p3_sigma3(r1, r2):
    a2 = (r2 - r1**2) / (1 - r1**2)
    p3 = r1 * (1 - a2) * r2 + a2 * r1
    sigma3_sq = (1 - r1**2) * (1 - a2**2)
    return p3, sigma3_sq


# ---------------------------------------------------------------------
# Row layout -- "top" now matches "bottom" (both 0.75)
# ---------------------------------------------------------------------
top_active = bool(TOP_TITLE)
mid_active = bool(MIDDLE_TITLE)

rows = ["top", "legend"]
if mid_active:
    rows.append("middle_title")
rows.append("bottom")

heights = {
    "top": 0.75,          # was 1.00 -- matches bottom row
    "legend": 0.16,
    "formula": 0.00,
    "middle_title": 0.00,
    "bottom": 0.75,
}
height_ratios = [heights[r] for r in rows]
row_idx = {r: i for i, r in enumerate(rows)}


# ---------------------------------------------------------------------
# Figure and grid
# ---------------------------------------------------------------------
fig = plt.figure()
outer = fig.add_gridspec(
    len(rows), 1,
    height_ratios=height_ratios,
    hspace=0.05,
)

# Nested gridspec for the top row.
# Outer: [gutter | content | gutter] with width ratios [1, 4, 1].
# The middle column takes 4/6 = 2/3 of the figure width and is
# subdivided into 2 equal panels, each ~1/3 of the figure width --
# the same width as one bottom-row panel. The gutters (1/6 each)
# push the pair of panels toward the centre of the page.
top_gs_outer = outer[row_idx["top"]].subgridspec(
    1, 3, wspace=0.0, width_ratios=[1, 4, 1],
)
top_gs_inner = top_gs_outer[0, 1].subgridspec(1, 2, wspace=0.20)

bot_gs = outer[row_idx["bottom"]].subgridspec(1, 3, wspace=0.00)

ax1 = fig.add_subplot(top_gs_inner[0, 0])
ax2 = fig.add_subplot(top_gs_inner[0, 1])

ax_leg = fig.add_subplot(outer[row_idx["legend"]])
ax_leg.axis("off")

if mid_active:
    ax_middle = fig.add_subplot(outer[row_idx["middle_title"]])
    ax_middle.axis("off")

axes_bot = [fig.add_subplot(bot_gs[0, 0])]
axes_bot.extend(
    fig.add_subplot(bot_gs[0, i], sharey=axes_bot[0]) for i in range(1, 3)
)
axes_bot[1].tick_params(labelleft=False)
axes_bot[2].tick_params(labelleft=False)


# ---------------------------------------------------------------------
# TOP LEFT: correlation space (curved r-space)
# ---------------------------------------------------------------------
alpha1_vals = [-0.7, -0.35, 0.0, 0.35, 0.7]
r1_dense = np.linspace(-1.0, 1.0, 400)
r2_lower_bnd = 2 * r1_dense**2 - 1
r2_upper_bnd = np.ones_like(r1_dense)

ax1.fill_between(r1_dense, r2_lower_bnd, r2_upper_bnd,
                 color=REGION_COLOR, zorder=0)

for a1 in alpha1_vals:
    y_min = 2 * a1**2 - 1
    ax1.vlines(a1, y_min, 1.0, colors=ALPHA1_GUIDE_COLOR,
               linestyles=":", linewidth=1.0, zorder=1)

for a2, col in [(-0.7, GUIDE_LOWER_COLOR), (0.7, GUIDE_UPPER_COLOR)]:
    r2_curve = r1_dense**2 + a2 * (1 - r1_dense**2)
    ax1.plot(r1_dense, r2_curve, color=col, linewidth=1.3, alpha=0.85)

ax1.plot(r1_dense, r1_dense**2, color=BACKBONE_COLOR,
         linewidth=3.0, zorder=5)
ax1.plot(r1_dense, r2_lower_bnd, color=BOUNDARY_COLOR, linewidth=2.2)
ax1.plot(r1_dense, r2_upper_bnd, color=BOUNDARY_COLOR,
         linestyle="--", linewidth=2.2)

r1_width = 0.4
p2_width = r1_width**2
r2_lower_width = 2 * r1_width**2 - 1
r2_upper_width = 1.0

add_interval_width_fig1(
    ax1,
    x=r1_width,
    y_lower=r2_lower_width,
    y_mid=p2_width,
    y_upper=r2_upper_width,
    label=r"$\Delta_2=2\sigma_2^2$",
    text_side=1,
)

for r1_pinch in (-1.0, 1.0):
    ax1.plot(r1_pinch, 1.0, marker="o", markersize=6,
             markerfacecolor="white", markeredgecolor=BACKBONE_COLOR,
             markeredgewidth=1.6, zorder=6)

ax1.set_xlabel(r"$r_1$")
ax1.set_ylabel(r"$r_2$")
ax1.set_xlim(-1.1, 1.1)
ax1.set_ylim(-1.1, 1.1)
ax1.set_aspect("equal")
ax1.grid(True, linestyle="--", alpha=0.3)
ax1.set_title(r"Correlation space (curved $r$-space)")


# ---------------------------------------------------------------------
# TOP RIGHT: natural coordinates (flat alpha-space)
# ---------------------------------------------------------------------
ax2.fill_between([-1, 1], [-1, -1], [1, 1], color=REGION_COLOR, zorder=0)

for a2, col in [(-0.7, GUIDE_LOWER_COLOR), (0.7, GUIDE_UPPER_COLOR)]:
    ax2.axhline(a2, color=col, linewidth=1.3, alpha=0.85)

ax2.axhline(0.0, color=BACKBONE_COLOR, linewidth=3.0, zorder=5)
ax2.axhline(-1.0, color=BOUNDARY_COLOR, linewidth=2.2)
ax2.axhline(1.0, color=BOUNDARY_COLOR, linestyle="--", linewidth=2.2)

for a1 in alpha1_vals:
    ax2.axvline(a1, color=ALPHA1_GUIDE_COLOR,
                linewidth=1.0, linestyle=":", zorder=1)

ax2.set_xlabel(r"$\alpha_1$")
ax2.set_ylabel(r"$\alpha_2$")
ax2.set_xlim(-1.1, 1.1)
ax2.set_ylim(-1.1, 1.1)
ax2.set_aspect("equal")
ax2.grid(True, linestyle="--", alpha=0.3)
ax2.set_title(r"Natural coordinates (flat $\alpha$-space)")


# ---------------------------------------------------------------------
# Shared legend strip
# ---------------------------------------------------------------------
legend_handles = [
    Patch(facecolor=REGION_COLOR, edgecolor="none",
          label="admissible region"),
    Line2D([0], [0], color=BOUNDARY_COLOR, linewidth=2.2,
           label=r"boundary $\alpha_n=-1$"),
    Line2D([0], [0], color=BOUNDARY_COLOR, linestyle="--", linewidth=2.2,
           label=r"boundary $\alpha_n=+1$"),
    Line2D([0], [0], color=BACKBONE_COLOR, linewidth=3.0,
           label=r"$\alpha_n=0$: $r_n=p_n$ (backbone)"),
    Line2D([0], [0], color=GUIDE_LOWER_COLOR, linewidth=1.3, alpha=0.85,
           label=r"guide $\alpha_n=-0.7$"),
    Line2D([0], [0], color=GUIDE_UPPER_COLOR, linewidth=1.3, alpha=0.85,
           label=r"guide $\alpha_n=+0.7$"),
    Line2D([0], [0], color=ALPHA1_GUIDE_COLOR, linewidth=1.0,
           linestyle=":", label=r"constant-$\alpha_1$ family / domain edge"),
    Line2D([0], [0], marker="o", linestyle="None", markersize=5.5,
           markerfacecolor="white", markeredgecolor=BACKBONE_COLOR,
           markeredgewidth=1.6,
           label=r"degenerate point: $\sigma_n^2\to0$"),
]

ax_leg.legend(
    handles=legend_handles,
    loc="center",
    ncol=4,
    frameon=True,
    framealpha=0.92,
    borderaxespad=0.0,
    handlelength=2.2,
    handletextpad=0.6,
    columnspacing=1.5,
)


# ---------------------------------------------------------------------
# Optional middle title
# ---------------------------------------------------------------------
if mid_active:
    ax_middle.text(
        0.5, 0.5, MIDDLE_TITLE,
        transform=ax_middle.transAxes,
        ha="center", va="center",
    )


# ---------------------------------------------------------------------
# BOTTOM: three fixed-r1 slices
# ---------------------------------------------------------------------
r1_values = [-0.5, 0.0, 0.5]

for index, (ax, r1) in enumerate(zip(axes_bot, r1_values)):
    r2_lower = 2 * r1**2 - 1
    r2 = np.linspace(r2_lower + 1e-4, 1.0 - 1e-4, 400)
    p3, sig3 = p3_sigma3(r1, r2)
    r3_lower = p3 - sig3
    r3_upper = p3 + sig3

    ax.fill_between(r2, r3_lower, r3_upper, color=REGION_COLOR, zorder=0)
    ax.plot(r2, r3_lower, color="black", linewidth=2.0)
    ax.plot(r2, r3_upper, color="black", linestyle="--", linewidth=2.0)

    for a3, col in [(-0.7, GUIDE_LOWER_COLOR), (0.7, GUIDE_UPPER_COLOR)]:
        r3_level = p3 + a3 * sig3
        ax.plot(r2, r3_level, color=col, linewidth=1.1, alpha=0.8)

    ax.plot(r2, p3, color=BACKBONE_COLOR, linewidth=3.0,
            linestyle="-", zorder=5)

    r2_width = 0.5
    if r2_lower < r2_width < 1.0:
        p3_width, sig3_width = p3_sigma3(r1, np.array([r2_width]))
        p3_width = p3_width[0]
        sig3_width = sig3_width[0]

        add_interval_width_fig2(
            ax,
            x=r2_width,
            y_lower=p3_width - sig3_width,
            y_mid=p3_width,
            y_upper=p3_width + sig3_width,
            label=r"$\Delta_3=2\sigma_3^2$",
            text_side=1,
        )

    for r2_pinch in (r2_lower, 1.0):
        p3_pinch, sig3_pinch = p3_sigma3(r1, np.array([r2_pinch]))
        ax.plot(r2_pinch, p3_pinch[0], marker="o", markersize=5.5,
                markerfacecolor="white", markeredgecolor=BACKBONE_COLOR,
                markeredgewidth=1.6, zorder=6)

    ax.axvline(r2_lower, color="grey", linewidth=0.8, linestyle=":")
    ax.axvline(1.0, color="grey", linewidth=0.8, linestyle=":")

    ax.set_xlabel(r"$r_2$")
    ax.set_xlim(-1.05, 1.05)
    ax.set_ylim(-1.05, 1.05)
    ax.set_aspect("equal")
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.set_title(rf"$r_1 = {r1:+.1f}$")

axes_bot[0].set_ylabel(r"$r_3$")


# ---------------------------------------------------------------------
# Optional overall title
# ---------------------------------------------------------------------
if top_active:
    fig.suptitle(TOP_TITLE)


fig.savefig(FIGDIR / "fig_admissible_combined.pdf")
plt.close(fig)
print("done")
