#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Figure 2 (Nature Energy style):

- Full-width (2-column, 180 mm) composite figure
- Left: large panel a (cluster scatter: AVG(Δt_100%) vs N)
- Right: 2×2 panels b–e (t95 histograms with insets)
- Text size <= 7 pt, font = Arial
"""

import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.ticker import MultipleLocator

# ─────────────────────────────────────────────────────────────
# Global style (Nature guide: sans-serif, max 7 pt)
# ─────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "Arial",
    "font.size": 5,      # base
    "axes.labelsize": 5,
    "axes.titlesize": 5,
    "xtick.labelsize": 5,
    "ytick.labelsize": 5,
    "legend.fontsize": 5,
    # 축/눈금 두께 & 길이
    "xtick.major.width": 0.4,
    "ytick.major.width": 0.4,
    "xtick.minor.width": 0.4,
    "ytick.minor.width": 0.4,
    "xtick.major.size": 2,
    "ytick.major.size": 2,
    "xtick.minor.size": 1.5,
    "ytick.minor.size": 1.5,
})

# ========= 경로 =========
CSV_T95   = r"G:\공유 드라이브\BSG_DFC_result\combined\DFC_완충후이동주차\t95\t95_before_after_delta_combined.csv"
CSV_CLUST = r"G:\공유 드라이브\BSG_DFC_result\combined\DFC_완충후이동주차\monthly_cluster\dfc_features_with_clusters.csv"
OUT_DIR   = os.path.dirname(CSV_T95)
os.makedirs(OUT_DIR, exist_ok=True)

# ========= 플롯 옵션 =========
BIN_WIDTH_DEFAULT        = 5.0    # main histogram bin width
INSET_BIN_STEP_DEFAULT   = 5.0    # inset histogram bin
INSET_TICK_STEP_DEFAULT  = 50.0   # inset major tick step
LABEL_EVERY              = 10     # x tick label interval (bins)

# 색상 팔레트
COLOR_APPLIED   = "#0073c2"   # DFC
COLOR_NOT       = "#cd534c"   # non-DFC
COLOR_DELTA     = "#efc000"   # Δt
ALPHA_MAIN      = 0.450
ALPHA_INSET     = 0.450

# inset 크기 (축 좌표 기준)
INSET_WIDTH_FRAC  = 0.40
INSET_HEIGHT_FRAC = 0.40
INSET_PAD         = 0.05
EXTRA_DOWN        = 0

# figure별 main-x/y + inset-x/y + bin 설정
PLOT_LIMITS = {
    "ALL": {
        "main_mode": "fixed",
        "main_x": (0, 200),
        "main_y": 200,

        "inset_mode": "fixed",
        "inset_x": (0, 150),
        "inset_y": 150,

        "bin_width":       5.0,
        "inset_bin_step":  5.0,
        "inset_tick_step": 50.0,
    },
    "cluster0": {
        "main_mode": "fixed",
        "main_x": (0, 200),
        "main_y": 200,

        "inset_mode": "auto",
        "inset_x": None,
        "inset_y": None,

        "bin_width":       5.0,
        "inset_bin_step":  5.0,
        "inset_tick_step": 50.0,
    },
    "cluster1": {
        "main_mode": "fixed",
        "main_x": (0, 410),
        "main_y": None,

        "inset_mode": "auto",
        "inset_x": None,
        "inset_y": None,

        "bin_width":       10.0,
        "inset_bin_step":  10.0,
        # d inset: 0, 100, 200, ... (100 단위)
        "inset_tick_step": 100.0,
    },
    "cluster2": {
        "main_mode": "auto",
        "main_x": None,
        "main_y": None,

        "inset_mode": "auto",
        "inset_x": None,
        "inset_y": None,

        "bin_width":       10.0,
        "inset_bin_step":  10.0,
        # e inset: 0, 200, 400 ... (200 단위)
        "inset_tick_step": 200.0,
    }
}

# ========= base_key 생성 =========
RE_BASE = re.compile(r'(bms_(?:altitude_)?\d+_\d{4}-\d{2})', re.IGNORECASE)

def ensure_base_key_col(df: pd.DataFrame) -> pd.DataFrame:
    """각 행에서 파일명 패턴을 찾아 base_key 컬럼을 추가."""
    if "base_key" in df.columns:
        return df
    keys = []
    for _, row in df.iterrows():
        k = None
        for val in row.values:
            if isinstance(val, str):
                m = RE_BASE.search(val)
                if m:
                    k = m.group(1)
                    break
        keys.append(k)
    df = df.copy()
    df["base_key"] = keys
    return df

# ========= 공통: 축 윤곽선 가늘게 =========
def thin_spines(ax, lw=0.4):
    for spine in ax.spines.values():
        spine.set_linewidth(lw)

# ========= cluster feature column 선택 (panel a용) =========
def pick_feature_columns(df: pd.DataFrame):
    """
    panel a용:
    - N축: N_col
    - AVG(Δt_100%): mean_col
    (기본 컬럼 이름을 찾되, 실제 플롯은 N_used / mean_used가 있으면 그걸 우선 사용)
    """
    N_candidates    = ["delta_t95_event_N", "N_events", "N_events_applied", "N_events_total"]
    mean_candidates = ["delta_t95_event_mean_h", "delta_t95_mean_h", "delayed_mean_h"]

    N_col = next((c for c in N_candidates if c in df.columns), None)
    mean_col = next((c for c in mean_candidates if c in df.columns), None)

    if N_col is None or mean_col is None:
        raise ValueError(
            "panel a용 피처 컬럼이 없습니다. "
            "delta_t95_event_N / delta_t95_event_mean_h (또는 후보) 컬럼이 필요합니다."
        )
    return N_col, mean_col

# ========= panel a: cluster scatter (N 가로, AVG 세로, winsor 반영) =========
def plot_cluster_scatter_ax(ax, df_feat: pd.DataFrame):
    """
    panel a:
      - x축: N(DFC)        → N_used   있으면 N_used,   없으면 N_col
      - y축: AVG(Δt_100%)  → mean_used 있으면 mean_used, 없으면 mean_col
      - 색: 클러스터 (0,1,2 → Minimal / Frequent / Long)
    이미 클러스터링된 dfc_features_with_clusters.csv 를 사용.
    """
    df_feat = df_feat.copy()
    if "cluster" not in df_feat.columns:
        ax.axis("off")
        return

    # 기본 feature column 이름
    N_col, mean_col = pick_feature_columns(df_feat)

    # winsorized 값이 있으면 우선 사용
    x_base_col = "N_used"    if "N_used"    in df_feat.columns else N_col
    y_base_col = "mean_used" if "mean_used" in df_feat.columns else mean_col

    # 유효 표본 (cluster, x, y 모두 숫자)
    x_num  = pd.to_numeric(df_feat[x_base_col], errors="coerce")
    y_num  = pd.to_numeric(df_feat[y_base_col], errors="coerce")
    cl_num = pd.to_numeric(df_feat["cluster"], errors="coerce")

    mask = x_num.notna() & y_num.notna() & cl_num.notna()
    used = df_feat.loc[mask].copy()
    if used.empty:
        ax.axis("off")
        return

    used[x_base_col]  = pd.to_numeric(used[x_base_col],  errors="coerce")
    used[y_base_col]  = pd.to_numeric(used[y_base_col],  errors="coerce")
    used["cluster"]   = pd.to_numeric(used["cluster"],   errors="coerce").astype(int)

    # 클러스터 라벨(legend 텍스트) & 팔레트 & 마커
    cluster_labels = {
        0: r"Minimal $R_{\mathrm{FC}}$",
        1: r"Frequent $R_{\mathrm{FC}}$",
        2: r"Long $R_{\mathrm{FC}}$",
    }
    palette = ["#cd534c", "#4dbbd5", "#0073c2"]  # cluster 0,1,2
    #markers = ['o', 's', '^']

    for cid in sorted(used["cluster"].unique()):
        sub = used[used["cluster"] == cid]
        if sub.empty:
            continue
        ax.scatter(
            sub[x_base_col],     # x축: N(DFC)
            sub[y_base_col],     # y축: AVG(Δt_100%)
            s=18,
            marker='o',
            c=palette[cid % len(palette)],
            edgecolor='k',
            linewidth=0.3,
            alpha=0.7,
            label=cluster_labels.get(cid, f"Cluster {cid}"),
        )

    # 축 범위 여유 (x: N, y: mean) — winsorized 값 기준
    x_min, x_max = used[x_base_col].min(), used[x_base_col].max()
    y_min, y_max = used[y_base_col].min(), used[y_base_col].max()
    pad_x = (x_max - x_min) * 0.05 if x_max > x_min else 1.0
    pad_y = (y_max - y_min) * 0.05 if y_max > y_min else 1.0
    ax.set_xlim(x_min - pad_x, x_max + pad_x)
    ax.set_ylim(y_min - pad_y, y_max + pad_y)

    # 축 라벨: x = N, y = AVG
    ax.set_xlabel("N(DFC)", fontsize=6)
    ax.set_ylabel(r"AVG($\Delta t_{100\%}$) (h)", fontsize=6)
    ax.xaxis.labelpad = 1.0  # 라벨-숫자 간격 축소
    ax.yaxis.labelpad = 1.0

    # 눈금 스타일
    ax.tick_params(axis='both', labelsize=5, width=0.4, length=2.5, pad=1.5)
    thin_spines(ax, lw=0.4)

    # legend
    leg = ax.legend(
        fontsize=5,
        loc="upper right",
        frameon=True,
    )
    frame = leg.get_frame()
    frame.set_edgecolor("grey")
    frame.set_linewidth(0.4)


# ========= overlapped histogram + inset (단일 Axes 버전) =========
def plot_overlapped_hist_with_inset_ax(ax, df_t95, cfg_key="ALL"):

    cfg = PLOT_LIMITS[cfg_key]

    main_mode = cfg["main_mode"]
    main_x    = cfg["main_x"]
    main_y    = cfg["main_y"]

    inset_mode       = cfg["inset_mode"]
    inset_x          = cfg["inset_x"]
    inset_y          = cfg["inset_y"]
    bin_w            = cfg.get("bin_width",       BIN_WIDTH_DEFAULT)
    inset_bin_step   = cfg.get("inset_bin_step",  INSET_BIN_STEP_DEFAULT)
    inset_tick_step  = cfg.get("inset_tick_step", INSET_TICK_STEP_DEFAULT)

    usecols = ["t95_before_h", "t95_after_h", "delta_t_h"]
    for c in usecols:
        if c not in df_t95.columns:
            raise ValueError(f"Missing column: {c}")

    after  = pd.to_numeric(df_t95["t95_after_h"], errors="coerce").dropna().to_numpy()
    before = pd.to_numeric(df_t95["t95_before_h"], errors="coerce").dropna().to_numpy()

    if after.size == 0 and before.size == 0:
        ax.axis("off")
        return

    # 전체 x 범위 계산
    all_data = np.concatenate([after, before])
    dmin, dmax = float(all_data.min()), float(all_data.max())

    left  = np.floor(dmin / bin_w) * bin_w
    right = np.ceil(dmax / bin_w)  * bin_w

    if main_mode == "fixed" and main_x is not None:
        left  = min(left,  main_x[0])
        right = max(right, np.ceil(main_x[1] / bin_w) * bin_w)

    edges   = np.arange(left, right + bin_w * 0.999, bin_w)
    centers = (edges[:-1] + edges[1:]) / 2

    cnt_after,  _ = np.histogram(after,  bins=edges)
    cnt_before, _ = np.histogram(before, bins=edges)

    # overlapped bars: bin 꽉 채우게
    bar_width = bin_w

    # (변경) non DFC 먼저, DFC 나중
    h_non = ax.bar(
        centers, cnt_before,
        width=bar_width, color=COLOR_NOT,
        alpha=ALPHA_MAIN, edgecolor="k", linewidth=0.3, label="non DFC"
    )
    h_dfc = ax.bar(
        centers, cnt_after,
        width=bar_width, color=COLOR_APPLIED,
        alpha=ALPHA_MAIN, edgecolor="k", linewidth=0.3, label="DFC"
    )

    ax.set_ylabel("Count")
    ax.set_xlabel(r"Total $t_{100\%}$ (h)")
    ax.xaxis.labelpad = 1.0
    ax.yaxis.labelpad = 1.0

    peak = max(cnt_after.max(), cnt_before.max())

    if main_mode == "fixed" and main_x is not None and main_y is not None:
        ax.set_xlim(*main_x)
        ax.set_ylim(0, main_y)
    else:
        ax.set_xlim(edges[0], edges[-1])
        ax.set_ylim(0, peak * 1.15)

    # x tick 간격 및 라벨
    tick_positions = np.arange(ax.get_xlim()[0], ax.get_xlim()[1] + 1e-9, bin_w)
    tick_idx = np.arange(0, len(tick_positions), LABEL_EVERY)
    ax.set_xticks(tick_positions[tick_idx])
    ax.set_xticklabels([str(int(x)) for x in tick_positions[tick_idx]])

    ax.grid(False)

    leg = ax.legend(
        handles=[h_non[0], h_dfc[0]],
        labels=["non DFC", "DFC"],
        loc="upper right",
        frameon=True
    )
    frame = leg.get_frame()
    frame.set_edgecolor("grey")
    frame.set_linewidth(0.4)

    # 숫자를 축에 조금 더 붙게 pad 줄임
    ax.tick_params(axis="both", width=0.4, pad=1.5)
    thin_spines(ax, lw=0.4)

    # ───── inset: Δt 히스토그램 ─────
    delta = pd.to_numeric(df_t95["delta_t_h"], errors="coerce").dropna().to_numpy()
    if delta.size > 0:

        fig = ax.figure
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        leg_bbox = leg.get_window_extent(renderer=renderer)

        x_disp = ax.bbox.x1
        y_disp = leg_bbox.y0
        _, y0_ax = ax.transAxes.inverted().transform((x_disp, y_disp))

        x0 = 1 - INSET_WIDTH_FRAC - INSET_PAD
        y0 = max(INSET_PAD, y0_ax - INSET_HEIGHT_FRAC - INSET_PAD - EXTRA_DOWN)

        axins = inset_axes(
            ax,
            width="100%", height="100%",
            loc="lower left",
            bbox_to_anchor=(x0, y0, INSET_WIDTH_FRAC, INSET_HEIGHT_FRAC),
            bbox_transform=ax.transAxes,
            borderpad=0.0
        )

        if inset_mode == "fixed" and inset_x is not None:
            lo, hi = inset_x
            lo_edge = np.floor(lo / inset_bin_step) * inset_bin_step
            hi_edge = np.ceil(hi / inset_bin_step) * inset_bin_step
        else:
            dmin2, dmax2 = delta.min(), delta.max()
            lo_edge = np.floor(dmin2 / inset_bin_step) * inset_bin_step
            hi_edge = np.ceil(dmax2  / inset_bin_step) * inset_bin_step
            lo, hi = lo_edge, hi_edge

        bins_dt = np.arange(lo_edge, hi_edge + inset_bin_step * 0.999, inset_bin_step)

        axins.hist(
            delta, bins=bins_dt,
            color=COLOR_DELTA, alpha=ALPHA_INSET,
            edgecolor="k", linewidth=0.3
        )

        axins.set_xlim(lo, hi)

        if inset_mode == "fixed" and inset_y is not None:
            axins.set_ylim(0, inset_y)

        # x 레이블만 사용, y-label "Count"는 생략
        axins.set_xlabel(r'$\Delta t_{100\%}$ (h)')
        # x축 라벨을 축에 더 붙게
        axins.xaxis.labelpad = 0.5
        # 숫자도 축에 더 붙게 pad 줄임
        axins.tick_params(axis="both", width=0.4, labelsize=5, pad=1.0)

        # 모든 inset: major tick만 사용 (minor tick 제거)
        axins.xaxis.set_major_locator(MultipleLocator(inset_tick_step))
        axins.minorticks_off()

        thin_spines(axins, lw=0.4)

# ========= 패널 레이블 유틸 (axes 좌표계에 그림) =========
def add_panel_label(ax, label, fontsize=7):
    """
    패널 레이블 (a,b,c,d,e)를 각 subplot 좌표계 기준으로 그림.
    - x=-0.08: 축 왼쪽 바깥
    - y=1.02:  축 상단 바로 위
    """
    ax.text(
        -0.11, 1.02, label,
        transform=ax.transAxes,
        fontsize=fontsize,
        fontweight="bold",
        ha="right",
        va="bottom",
        clip_on=False,   # 축 밖까지 그려지도록
    )

# ========= Figure 2 생성 =========
def make_figure2(merged, sub0, sub1, sub2, df_feat, out_dir):
    """
    Figure 2 (full-width, multi-panel):

    - a: 클러스터 산점도 (AVG(Δt_100%) vs N(DFC), winsor 반영)
    - b: ALL
    - c: Cluster 0
    - d: Cluster 1
    - e: Cluster 2
    """

    # 2-column width: 180 mm ~ 7.09 inch → 약 7.1 inch 사용
    fig = plt.figure(figsize=(7.2, 2.8), dpi=300)

    # 2행 3열 GridSpec
    gs = GridSpec(
        nrows=2, ncols=3,
        width_ratios=[2, 1, 1], # 컬럼 비율
        height_ratios=[1.0, 1.0],
        wspace=0.18,   # 패널 가로 간격 축소
        hspace=0.22,   # 패널 세로 간격 축소
        figure=fig
    )

    # Panel a: 왼쪽 큰 패널 (두 행 span) — 클러스터 산점도
    axA = fig.add_subplot(gs[:, 0])
    plot_cluster_scatter_ax(axA, df_feat)

    # Panel b–e: 오른쪽 2×2
    axB = fig.add_subplot(gs[0, 1])
    axC = fig.add_subplot(gs[0, 2])
    axD = fig.add_subplot(gs[1, 1])
    axE = fig.add_subplot(gs[1, 2])

    plot_overlapped_hist_with_inset_ax(axB, merged, cfg_key="ALL")

    if len(sub0) > 0:
        plot_overlapped_hist_with_inset_ax(axC, sub0, cfg_key="cluster0")
    else:
        axC.axis("off")

    if len(sub1) > 0:
        plot_overlapped_hist_with_inset_ax(axD, sub1, cfg_key="cluster1")
    else:
        axD.axis("off")

    if len(sub2) > 0:
        plot_overlapped_hist_with_inset_ax(axE, sub2, cfg_key="cluster2")
    else:
        axE.axis("off")

    # 레이아웃 먼저 맞추고
    fig.tight_layout(rect=[0, 0, 1, 0.97])


    # 각 axes 기준으로 패널 레이블 추가
    add_panel_label(axA, "a")
    add_panel_label(axB, "b")
    add_panel_label(axC, "c")
    add_panel_label(axD, "d")
    add_panel_label(axE, "e")

    base = os.path.join(out_dir, "Figure2")

    # 출판용 벡터(PDF) + 리뷰/슬라이드용 PNG 동시 저장
    fig.savefig(base + ".pdf", dpi=300, bbox_inches="tight")
    fig.savefig(base + ".png", dpi=300, bbox_inches="tight")

    plt.close(fig)
    print(f"[SAVE] Figure 2 -> {base}.pdf / {base}.png")


# ========= 실행부 =========
if __name__ == "__main__":

    # 데이터 로드 및 merge
    t95 = pd.read_csv(CSV_T95)
    cl  = pd.read_csv(CSV_CLUST)

    t95 = ensure_base_key_col(t95)
    cl  = ensure_base_key_col(cl)

    cl["cluster"] = pd.to_numeric(cl["cluster"], errors="coerce")
    merged = pd.merge(t95, cl[["base_key", "cluster"]], how="left", on="base_key")

    # 클러스터별 subset (b–e용)
    sub0 = merged[merged["cluster"] == 0]
    sub1 = merged[merged["cluster"] == 1]
    sub2 = merged[merged["cluster"] == 2]

    # panel a 는 cl (feature + cluster 라벨 + mean_used / N_used) 그대로 사용
    make_figure2(
        merged=merged,
        sub0=sub0,
        sub1=sub1,
        sub2=sub2,
        df_feat=cl,
        out_dir=OUT_DIR
    )
