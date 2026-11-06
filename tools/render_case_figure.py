"""Render the public verl no-std fixture with matplotlib 3.11.2."""

import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt


def main():
    plt.switch_backend("Agg")
    root = Path(__file__).resolve().parents[1]
    suite = json.loads((root / "tests/fixtures/public-cases.json").read_text())
    evidence = next(
        case["evidence"] for case in suite["cases"] if case["id"] == "verl-3"
    )
    assert evidence["profile"]["normalize_std"] is False
    rows = sorted(evidence["samples"], key=lambda row: (row["group_id"], row["reward"]))
    groups = defaultdict(list)
    for row in rows:
        groups[row["group_id"]].append(row["reward"])
    group_names = {group: f"g{index + 1}" for index, group in enumerate(sorted(groups))}
    labels, expected, observed = [], [], []
    for row in rows:
        rewards = groups[row["group_id"]]
        expected.append(row["reward"] - math.fsum(rewards) / len(rewards))
        observed.append(
            next(
                value
                for value, mask in zip(row["advantage"], row["mask"], strict=True)
                if mask
            )
        )
        labels.append(f"{group_names[row['group_id']]} / reward {row['reward']:g}")

    matplotlib.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "svg.fonttype": "path",
            "svg.hashsalt": "rlreplay-case-v1",
        }
    )
    for theme, background, foreground, muted, recorded, target in (
        ("light", "#ffffff", "#1f2328", "#8c959f", "#9a6700", "#0969da"),
        ("dark", "#0d1117", "#e6edf3", "#9198a1", "#d29922", "#4493f8"),
    ):
        fig, axes = plt.subplots(
            figsize=(8, 3.8), facecolor=background, layout="constrained"
        )
        axes.set_facecolor(background)
        positions = list(range(len(rows)))
        for y, actual, correct in zip(positions, observed, expected, strict=True):
            axes.plot([actual, correct], [y, y], color=muted, linewidth=1.3, zorder=1)
        axes.scatter(
            observed,
            positions,
            color=recorded,
            marker="o",
            s=55,
            label="Recorded driver output",
            zorder=3,
        )
        axes.scatter(
            expected,
            positions,
            color=target,
            marker="D",
            s=45,
            label="Requested no-std contract",
            zorder=3,
        )
        axes.axvline(0, color=muted, linewidth=0.8, zorder=0)
        axes.set_yticks(positions, labels)
        axes.invert_yaxis()
        axes.set_xlabel("Advantage", color=foreground)
        axes.set_title(
            "verl vectorized driver with normalization disabled",
            color=foreground,
            loc="left",
            pad=42,
        )
        axes.tick_params(colors=foreground)
        axes.set_xlim(min(expected) - 0.3, max(expected) + 0.3)
        axes.set_ylim(len(rows) - 0.5, -0.5)
        for spine in axes.spines.values():
            spine.set_color(muted)
        legend = axes.legend(
            loc="lower left", bbox_to_anchor=(0, 1.02), ncols=2, frameon=False
        )
        for label in legend.get_texts():
            label.set_color(foreground)
        suffix = "" if theme == "light" else "-dark"
        fig.savefig(
            root / f"docs/assets/verl-normalization{suffix}.svg",
            metadata={"Date": None},
        )
        plt.close(fig)


if __name__ == "__main__":
    main()
