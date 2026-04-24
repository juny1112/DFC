#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Supplementary Figure: minimum-SOC distribution between full charges
(Nature Energy half-width style; no panel label)

Updated per user request:
- no gaps between horizontal bars
- y-axis shown as numeric SOC ticks (0, 20, ..., 100) instead of interval labels
- bar color changed to #4DBBD5
- no title
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family": "Arial",
    "font.size": 5,
    "axes.labelsize": 5,
    "axes.titlesize": 5,
    "xtick.labelsize": 5,
    "ytick.labelsize": 5,
    "legend.fontsize": 5,
    "xtick.major.width": 0.4,
    "ytick.major.width": 0.4,
    "xtick.minor.width": 0.4,
    "ytick.minor.width": 0.4,
    "xtick.major.size": 2,
    "ytick.major.size": 2,
    "xtick.minor.size": 1.5,
    "ytick.minor.size": 1.5,
})

IN_CSV = r"G:\공유 드라이브\BSG_DFC_result\combined\DFC_완충후이동주차\불량개입\min_soc_between_fullcharges_cases.csv"
OUT_DIR = r"G:\공유 드라이브\BSG_DFC_result\combined\DFC_완충후이동주차\Suppl"
os.makedirs(OUT_DIR, exist_ok=True)

CASE_MODE = "case123"   # "case23" or "case123"
SOC_BIN_WIDTH = 10
SAVE_BASENAME = f"Suppl_minSOC_{CASE_MODE}"

COLOR_BAR = "#4DBBD5"
ALPHA_MAIN = 0.80

FIG_W = 3.5   # ~89 mm single-column width
FIG_H = 2.9
DPI = 300


def thin_spines(ax, lw: float = 0.4) -> None:
    for sp in ax.spines.values():
        sp.set_linewidth(lw)


def parse_minima_column(series: pd.Series) -> np.ndarray:
    values = []
    for s in series.dropna():
        s = str(s).strip()
        if not s:
            continue
        for p in s.split(","):
            p = p.strip()
            if not p:
                continue
            try:
                values.append(float(p))
            except ValueError:
                continue
    return np.asarray(values, dtype=float) if values else np.array([], dtype=float)


def build_minima(df: pd.DataFrame, case_mode: str) -> np.ndarray:
    mins1 = parse_minima_column(df["case1_mins"]) if "case1_mins" in df.columns else np.array([], dtype=float)
    mins2 = parse_minima_column(df["case2_mins"]) if "case2_mins" in df.columns else np.array([], dtype=float)
    mins3 = parse_minima_column(df["case3_mins"]) if "case3_mins" in df.columns else np.array([], dtype=float)

    if case_mode == "case23":
        arrs = [a for a in (mins2, mins3) if a.size > 0]
    elif case_mode == "case123":
        arrs = [a for a in (mins1, mins2, mins3) if a.size > 0]
    else:
        raise ValueError("CASE_MODE must be 'case23' or 'case123'.")

    if not arrs:
        return np.array([], dtype=float)
    return np.concatenate(arrs)


def plot_minsoc_halfwidth(minima: np.ndarray, out_base: str) -> None:
    data = np.asarray(minima, dtype=float)
    data = data[np.isfinite(data)]
    if data.size == 0:
        raise ValueError("No minimum-SOC data found.")

    data = np.clip(data, 0.0, 100.0)

    # 0-10, 10-20, ..., 90-100
    edges = np.arange(0.0, 100.0 + SOC_BIN_WIDTH, SOC_BIN_WIDTH)
    counts, _ = np.histogram(data, bins=edges)
    y_bottoms = edges[:-1]

    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H), dpi=DPI)

    # height == bin width -> bars touch with no vertical gaps
    ax.barh(
        y_bottoms,
        counts,
        height=SOC_BIN_WIDTH,
        align="edge",
        color=COLOR_BAR,
        alpha=ALPHA_MAIN,
        edgecolor="black",
        linewidth=0.4,
    )

    ax.set_xlabel("Count", labelpad=1.0)
    ax.set_ylabel("Minimum SOC (%)", labelpad=1.0)

    ax.set_ylim(0, 100)
    ax.set_yticks(np.arange(0, 101, 20))
    ax.set_yticklabels([str(v) for v in range(0, 101, 20)])

    xmax = int(np.nanmax(counts)) if len(counts) else 1
    ax.set_xlim(0, xmax * 1.05 if xmax > 0 else 1)

    ax.tick_params(axis="both", width=0.4, length=2.0, pad=1.0)
    ax.grid(False)
    thin_spines(ax, lw=0.4)

    fig.tight_layout(pad=0.4)
    fig.savefig(out_base + ".png", dpi=DPI, bbox_inches="tight")
    fig.savefig(out_base + ".pdf", dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"[SAVE] {out_base}.png / {out_base}.pdf")


if __name__ == "__main__":
    df = pd.read_csv(IN_CSV)
    if "n_fullcharge_Rcharg" not in df.columns:
        raise ValueError("Required column missing: n_fullcharge_Rcharg")

    df_valid = df.loc[pd.to_numeric(df["n_fullcharge_Rcharg"], errors="coerce") > 0].copy()
    minima = build_minima(df_valid, CASE_MODE)
    out_base = os.path.join(OUT_DIR, SAVE_BASENAME)
    plot_minsoc_halfwidth(minima, out_base=out_base)
