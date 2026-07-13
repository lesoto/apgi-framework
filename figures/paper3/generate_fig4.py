"""Paper 3 — Figure 4: Psychiatric Disorder Profiles as Level-Cascade Dysregulation (§4.1).

Three-column figure (MDD, Schizophrenia, DoC) each with miniature
five-level architecture showing direction/magnitude of parameter dysregulation
and cascade propagation arrows.

Run:
    python figures/paper3/generate_fig4.py
    python figures/paper3/generate_fig4.py --no-show
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent.parent))
from figures.utils import save_figure

OUTPUT_DIR = pathlib.Path(__file__).parent / "output"

LEVEL_BASE_COLORS = ["#a50f15", "#de2d26", "#fc8d59", "#fdcc8a", "#f0f0f0"]
LEVEL_NAMES = ["L4", "L3", "L2", "L1", "L0"]

# Shared left-hand key: IDENTICAL Level 0-4 hierarchy/substrate definitions as
# Paper 3 Figure 1 (generate_fig1.py), paired with the functional/active-
# inference read-out for each level (CRITICAL correction: one hierarchy, two
# descriptions — not a second, differently-defined Level 0-4 scheme).
LEVEL_KEY = [
    ("L4", "Neuroendocrine\n(HPA axis)", "Policies"),
    ("L3", "Association cortex\n(limbic, parietal)", "Perceptual models"),
    ("L2", "Cortical reservoirs\n(~2 s memory)", "Percepts"),
    ("L1", "Local cortical\nmicrocircuits", "Sensory signals"),
    ("L0", "Brainstem,\nspinal reflex", "Environmental states"),
]

# Dysregulation specs per disorder
# dir: +1=elevated, -1=reduced; size: 0–2
DISORDERS = [
    {
        "name": "MDD",
        "color": "#2166ac",
        "levels": [
            {"theta": +2, "pi": 0, "primary": False},
            {"theta": +2, "pi": 0, "primary": True},  # L3: primary HPA-driven
            {"theta": +1, "pi": -1, "primary": False},  # L2: propagated Pi blunting
            {"theta": +1, "pi": -1, "primary": False},  # L1: propagated Pi blunting
            {"theta": 0, "pi": 0, "primary": False},
        ],
        "cascade_from": 1,  # L3 → L1 via nearest-neighbour kappa cascade
        "cascade_to": 3,
        "caption": "Primary: L3/L4 (HPA θ↑)\n→ κ cascade propagation raises θ\n"
        "and blunts Π at L1–L2",
    },
    {
        "name": "Schizophrenia",
        "color": "#d6604d",
        "levels": [
            {"theta": 0, "pi": 0, "primary": False},
            {"theta": +1, "pi": 0, "primary": False},  # L3: θ variance elevated
            {"theta": 0, "pi": 0, "primary": False},
            {"theta": 0, "pi": +2, "primary": True},  # L1: aberrant salience Π↑
            {"theta": 0, "pi": 0, "primary": False},
        ],
        "cascade_from": 3,  # L1–L2 coupling disrupted
        "cascade_to": 2,
        "caption": "L1 Π↑ (aberrant salience)\nL1–L2 α₁₂ coupling disrupted",
    },
    {
        "name": "DoC\n(VS/UWS → MCS)",
        "color": "#4dac26",
        "levels": [
            {"theta": 0, "pi": 0, "primary": False},
            {"theta": 0, "pi": 0, "primary": False},
            {"theta": 0, "pi": 0, "primary": False},
            {"theta": +3, "pi": -2, "primary": True},  # L1: ignition excluded (VS)
            {"theta": 0, "pi": 0, "primary": False},
        ],
        "cascade_from": None,
        "cascade_to": None,
        # Second state shown side-by-side: MCS, with L1+L2 partially restored.
        "mcs_levels": [
            {"theta": 0, "pi": 0, "primary": False},
            {"theta": 0, "pi": 0, "primary": False},
            {"theta": +1, "pi": -1, "primary": True},  # L2 partially restored
            {"theta": +1, "pi": -1, "primary": True},  # L1 partially restored
            {"theta": 0, "pi": 0, "primary": False},
        ],
        "caption": "VS/UWS: L1 θ↑↑↑ (ignition excluded)\n"
        "MCS: L1+L2 partially restored (θ↑, Π↓)",
    },
]


def _draw_compact_stack(
    ax,
    levels,
    primary_color,
    box_x,
    box_w,
    top_y=0.84,
    box_h=0.11,
    spacing=0.155,
    show_names=True,
):
    """Draw a miniature five-level stack; returns per-level (cx, cy, y)."""
    centres = []
    for i, (lv_name, base_color, lv_spec) in enumerate(
        zip(LEVEL_NAMES, LEVEL_BASE_COLORS, levels)
    ):
        y = top_y - i * spacing
        primary = lv_spec["primary"]
        rect = mpatches.FancyBboxPatch(
            (box_x, y),
            box_w,
            box_h,
            boxstyle="round,pad=0.01",
            facecolor=base_color,
            edgecolor=primary_color if primary else "#888888",
            lw=2.5 if primary else 1.0,
            alpha=0.85 if primary else 0.45,
            zorder=3,
        )
        ax.add_patch(rect)
        if show_names:
            ax.text(
                box_x - 0.015,
                y + box_h / 2,
                lv_name,
                ha="right",
                va="center",
                fontsize=7,
                fontweight="bold",
            )
        td = lv_spec["theta"]
        if td != 0:
            arrow = "↑" * abs(td) if td > 0 else "↓" * abs(td)
            ax.text(
                box_x + box_w * 0.28,
                y + box_h * 0.62,
                f"θ{arrow}",
                ha="center",
                fontsize=6.0,
                color="#d6604d",
                fontweight="bold",
            )
        pd = lv_spec["pi"]
        if pd != 0:
            arrow = "↑" * abs(pd) if pd > 0 else "↓" * abs(pd)
            ax.text(
                box_x + box_w * 0.72,
                y + box_h * 0.32,
                f"Π{arrow}",
                ha="center",
                fontsize=6.0,
                color="#2166ac",
                fontweight="bold",
            )
        centres.append((box_x + box_w / 2, y + box_h / 2, y))
    return centres


def draw_doc_dual(ax, disorder):
    """DoC column: side-by-side VS/UWS and MCS states to show the transition
    (recovery trajectory as a function of level restoration)."""
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.1, 1.05)
    ax.axis("off")
    ax.set_title(
        disorder["name"], fontsize=10, fontweight="bold", color=disorder["color"]
    )

    ax.text(
        0.27,
        0.94,
        "VS/UWS",
        ha="center",
        fontsize=8,
        fontweight="bold",
        color=disorder["color"],
    )
    ax.text(
        0.82,
        0.94,
        "MCS",
        ha="center",
        fontsize=8,
        fontweight="bold",
        color=disorder["color"],
    )

    c_vs = _draw_compact_stack(
        ax,
        disorder["levels"],
        disorder["color"],
        box_x=0.13,
        box_w=0.30,
        show_names=True,
    )
    _draw_compact_stack(
        ax,
        disorder["mcs_levels"],
        disorder["color"],
        box_x=0.66,
        box_w=0.30,
        show_names=False,
    )

    # Transition arrow at the L1 row (index 3): VS/UWS → MCS recovery
    y_l1 = c_vs[3][1]
    ax.annotate(
        "",
        xy=(0.64, y_l1),
        xytext=(0.45, y_l1),
        arrowprops=dict(
            arrowstyle="-|>", color="#4dac26", lw=2.2, connectionstyle="arc3,rad=-0.25"
        ),
    )
    ax.text(
        0.545,
        y_l1 + 0.10,
        "recovery\n(L1/L2\nrestored)",
        ha="center",
        va="bottom",
        fontsize=6.0,
        color="#4dac26",
        fontweight="bold",
    )

    ax.text(
        0.50,
        -0.07,
        disorder["caption"],
        ha="center",
        va="top",
        fontsize=7,
        color=disorder["color"],
        style="italic",
        multialignment="center",
    )


def draw_disorder_column(ax, disorder):
    if disorder.get("mcs_levels"):
        draw_doc_dual(ax, disorder)
        return

    ax.set_xlim(0, 1)
    ax.set_ylim(-0.1, 1.05)
    ax.axis("off")
    ax.set_title(
        disorder["name"], fontsize=10, fontweight="bold", color=disorder["color"]
    )

    BOX_X, BOX_W, BOX_H = 0.20, 0.60, 0.13
    SPACING = 0.175
    PRIMARY_EDGE_LW = 3.0

    for i, (lv_name, base_color, lv_spec) in enumerate(
        zip(LEVEL_NAMES, LEVEL_BASE_COLORS, disorder["levels"])
    ):
        y = 0.88 - i * SPACING
        primary = lv_spec["primary"]

        rect = mpatches.FancyBboxPatch(
            (BOX_X, y),
            BOX_W,
            BOX_H,
            boxstyle="round,pad=0.01",
            facecolor=base_color,
            edgecolor=disorder["color"] if primary else "#888888",
            lw=PRIMARY_EDGE_LW if primary else 1.2,
            alpha=0.85 if primary else 0.45,
            zorder=3,
        )
        ax.add_patch(rect)

        # Level name
        ax.text(
            BOX_X - 0.02,
            y + BOX_H / 2,
            lv_name,
            ha="right",
            va="center",
            fontsize=8,
            fontweight="bold",
        )

        # θ annotation
        theta_dir = lv_spec["theta"]
        if theta_dir != 0:
            arrow = "↑" * abs(theta_dir) if theta_dir > 0 else "↓" * abs(theta_dir)
            ax.text(
                BOX_X + 0.12,
                y + BOX_H * 0.7,
                f"θ{arrow}",
                ha="center",
                fontsize=7.5,
                color="#d6604d",
                fontweight="bold",
            )

        # Π annotation
        pi_dir = lv_spec["pi"]
        if pi_dir != 0:
            arrow = "↑" * abs(pi_dir) if pi_dir > 0 else "↓" * abs(pi_dir)
            ax.text(
                BOX_X + 0.45,
                y + BOX_H * 0.3,
                f"Π{arrow}",
                ha="center",
                fontsize=7.5,
                color="#2166ac",
                fontweight="bold",
            )

    # Cascade arrow
    cf, ct = disorder.get("cascade_from"), disorder.get("cascade_to")
    if cf is not None and ct is not None:
        y_from = 0.88 - cf * SPACING
        y_to = 0.88 - ct * SPACING + BOX_H
        ax.annotate(
            "",
            xy=(BOX_X + BOX_W + 0.05, y_to),
            xytext=(BOX_X + BOX_W + 0.05, y_from),
            arrowprops=dict(
                arrowstyle="->",
                color="#e6a817",
                lw=2.2,
                connectionstyle="arc3,rad=0.3",
            ),
        )
        ax.text(
            BOX_X + BOX_W + 0.10,
            (y_from + y_to) / 2,
            "cascade\n(α disrupted)",
            fontsize=6.5,
            color="#e6a817",
            ha="left",
            va="center",
        )

    # Caption
    ax.text(
        0.50,
        -0.07,
        disorder["caption"],
        ha="center",
        va="top",
        fontsize=7,
        color=disorder["color"],
        style="italic",
        multialignment="center",
    )


def draw_shared_key(ax):
    """Shared left-hand key: same row geometry as draw_disorder_column
    (top_y=0.88, spacing=0.175, box_h=0.13) so rows line up with the three
    disorder columns. Each row shows BOTH the Fig-1 substrate name and the
    active-inference functional read-out for that level."""
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.1, 1.05)
    ax.axis("off")
    ax.set_title("Level key\n(Fig. 1 hierarchy)", fontsize=10, fontweight="bold")

    BOX_H = 0.13
    SPACING = 0.175
    for i, ((lv_name, substrate, functional), base_color) in enumerate(
        zip(LEVEL_KEY, LEVEL_BASE_COLORS)
    ):
        y = 0.88 - i * SPACING
        rect = mpatches.FancyBboxPatch(
            (0.05, y),
            0.90,
            BOX_H,
            boxstyle="round,pad=0.01",
            facecolor=base_color,
            edgecolor="#888888",
            lw=1.0,
            alpha=0.55,
            zorder=3,
        )
        ax.add_patch(rect)
        ax.text(
            0.10,
            y + BOX_H * 0.68,
            f"{lv_name} — {substrate}",
            ha="left",
            va="center",
            fontsize=6.3,
            fontweight="bold",
            color="#222222",
        )
        ax.text(
            0.10,
            y + BOX_H * 0.24,
            f"active-inference: {functional}",
            ha="left",
            va="center",
            fontsize=6.0,
            color="#555555",
            style="italic",
        )


def plot(show: bool = True) -> None:
    fig, axes = plt.subplots(
        1, 4, figsize=(17, 7), gridspec_kw={"width_ratios": [0.62, 1, 1, 1]}
    )
    draw_shared_key(axes[0])
    for ax, disorder in zip(axes[1:], DISORDERS):
        draw_disorder_column(ax, disorder)

    # Legend
    handles = [
        mpatches.Patch(color="#d6604d", label="θ↑ (threshold elevated)"),
        mpatches.Patch(color="#2166ac", label="Π↓/Π↑ (precision change)"),
        mpatches.Patch(color="#e6a817", label="Cascade (α disrupted)"),
        mpatches.Patch(
            facecolor="none", edgecolor="#333333", lw=3, label="Primary site"
        ),
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        fontsize=8,
        ncol=4,
        framealpha=0.8,
        bbox_to_anchor=(0.5, -0.04),
    )

    fig.tight_layout()
    save_figure(fig, OUTPUT_DIR / "fig4_psychiatric_disorder_profiles.pdf")
    if show:
        plt.show()
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-show", action="store_true")
    args = parser.parse_args()
    plot(show=not args.no_show)


if __name__ == "__main__":
    main()
