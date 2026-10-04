"""Leaderboard chart: each model's skill over a constant prediction, with 95% intervals.

Starter for the Data Viz week 1 task. Reads the table written by scripts/benchmark_skill.py.

    python viz/figures/leaderboard.py artifacts/benchmark/benchmark_table.csv --out viz/figures/out/leaderboard

Writes <out>.png and <out>.svg. Two panels: density skill (% better than guessing the
training average) and hotspot PR-AUC (with the no-skill line). Models are grouped into
"climate-only" (usable for 2040 projections) and "nowcast" (uses recent krill catches).
"""
import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

# Okabe-Ito colours: distinguishable with the common forms of colour blindness.
COLORS = {"climate": "#0072B2", "nowcast": "#E69F00"}
LABELS = {"climate": "Climate-only (usable for 2040)", "nowcast": "Nowcast (uses recent krill catches)"}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("table")
    p.add_argument("--out", default="viz/figures/out/leaderboard")
    p.add_argument("--hide-seeds", action="store_true", help="Show only seed-averaged GP rows")
    a = p.parse_args()

    t = pd.read_csv(a.table)
    t = t[t["track"] != "baseline"].copy()
    if a.hide_seeds:
        t = t[~t["model"].str.contains(r"\[", regex=True)]
    t["model"] = t["model"].str.replace(r" \[.*\]", "", regex=True)
    t = t.sort_values(["track", "skill_vs_constant"], ascending=[False, True]).reset_index(drop=True)
    y = range(len(t))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 0.45 * len(t) + 1.8), sharey=True,
                                   gridspec_kw={"width_ratios": [1.2, 1]})
    for track, color in COLORS.items():
        s = t[t["track"] == track]
        ax1.errorbar(100 * s["skill_vs_constant"], s.index,
                     xerr=[100 * (s["skill_vs_constant"] - s["skill_ci_low"]), 100 * (s["skill_ci_high"] - s["skill_vs_constant"])],
                     fmt="o", color=color, ecolor=color, elinewidth=1.5, capsize=3, label=LABELS[track])
        ok = s["pr_auc"].notna()
        ax2.errorbar(s.loc[ok, "pr_auc"], s.index[ok],
                     xerr=[s.loc[ok, "pr_auc"] - s.loc[ok, "pr_auc_ci_low"], s.loc[ok, "pr_auc_ci_high"] - s.loc[ok, "pr_auc"]],
                     fmt="o", color=color, ecolor=color, elinewidth=1.5, capsize=3)

    ax1.axvline(0, color="0.3", lw=1)
    ax1.text(-0.5, len(t) - 0.4, "no better than guessing the average ", fontsize=8, color="0.3", va="bottom", ha="right")
    ax1.set_xlabel("Density error improvement over constant (%)")
    no_skill = (t["pr_auc"] / t["pr_auc_vs_no_skill"]).dropna()
    if len(no_skill):
        ns = float(no_skill.median())
        ax2.axvline(ns, color="0.3", lw=1, ls="--")
        ax2.axvline(2 * ns, color="0.6", lw=1, ls=":")
        ax2.text(ns, len(t) - 0.4, " random", fontsize=8, color="0.3", va="bottom")
        ax2.text(2 * ns, len(t) - 0.4, " 2x random (skill bar)", fontsize=8, color="0.45", va="bottom")
    ax2.set_xlabel("Hotspot PR-AUC (higher is better)")
    ax1.set_yticks(list(y), t["model"])
    ax1.set_ylim(-0.7, len(t) + 0.3)
    for ax in (ax1, ax2):
        ax.grid(axis="x", color="0.9")
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("FishCast benchmark on held-out years (95% intervals from resampling survey years)", fontsize=11, x=0.01, ha="left")
    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, fontsize=9, frameon=False)
    fig.tight_layout(rect=(0, 0.06, 1, 1))

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "svg"):
        fig.savefig(out.with_suffix(f".{ext}"), dpi=200)
    print(f"saved {out}.png and {out}.svg")


if __name__ == "__main__":
    main()
