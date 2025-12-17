# ───────────────── Plot: overall + per-cluster (0/1/2) ─────────────────
import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.ticker import MultipleLocator, NullFormatter, FormatStrFormatter

# 폰트: Arial
plt.rcParams["font.family"] = "Arial"

# ========= 경로 =========
CSV_T95   = r"G:\공유 드라이브\BSG_DFC_result\combined\DFC_완충후이동주차\t95\t95_before_after_delta_combined.csv"
CSV_CLUST = r"G:\공유 드라이브\BSG_DFC_result\combined\DFC_완충후이동주차\monthly_cluster\dfc_features_with_clusters.csv"
OUT_DIR   = os.path.dirname(CSV_T95)

# ========= 공통 플롯 옵션 =========
BIN_WIDTH       = 5.0
LABEL_EVERY     = 10

# 🔹 색상 팔레트
COLOR_APPLIED   = "#0073c2"   # DFC
COLOR_NOT       = "#cd534c"   # non-DFC
COLOR_DELTA     = "#efc000"   # Δt
ALPHA_MAIN      = 0.60
ALPHA_INSET     = 0.60

# 🔹 inset 크기 고정 (A 옵션)
INSET_WIDTH_FRAC  = 0.35
INSET_HEIGHT_FRAC = 0.35
INSET_PAD         = 0.01
EXTRA_DOWN        = 0.08

# 🔹 figure별 main-x/y 범위 + inset-x/y 범위 설정
PLOT_LIMITS = {
    "ALL": {
        "main_mode": "fixed",
        "main_x": (0, 200),
        "main_y": 200,

        "inset_mode": "fixed",
        "inset_x": (0, 150),
        "inset_y": 150,
    },
    "cluster0": {
        "main_mode": "auto",
        "main_x": None,
        "main_y": None,

        "inset_mode": "auto",
        "inset_x": None,
        "inset_y": None,
    },
    "cluster1": {
        "main_mode": "auto",
        "main_x": None,
        "main_y": None,

        "inset_mode": "auto",
        "inset_x": None,
        "inset_y": None,
    },
    "cluster2": {
        "main_mode": "auto",
        "main_x": None,
        "main_y": None,

        "inset_mode": "auto",
        "inset_x": None,
        "inset_y": None,
    }
}

# ========= base_key 생성 =========
RE_BASE = re.compile(r'(bms_(?:altitude_)?\d+_\d{4}-\d{2})', re.IGNORECASE)
def ensure_base_key_col(df):
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

# ========= (1) 단일 figure 버전 (기존) =========
def plot_grouped_hist_with_inset(df_t95, out_png, cfg_key="ALL"):

    cfg = PLOT_LIMITS[cfg_key]

    main_mode = cfg["main_mode"]
    main_x    = cfg["main_x"]
    main_y    = cfg["main_y"]

    inset_mode = cfg["inset_mode"]
    inset_x    = cfg["inset_x"]
    inset_y    = cfg["inset_y"]

    usecols = ["t95_before_h", "t95_after_h", "delta_t_h"]
    for c in usecols:
        if c not in df_t95.columns:
            raise ValueError(f"Missing column: {c}")

    after  = pd.to_numeric(df_t95["t95_after_h"], errors="coerce").dropna().to_numpy()
    before = pd.to_numeric(df_t95["t95_before_h"], errors="coerce").dropna().to_numpy()

    if after.size == 0 and before.size == 0:
        print("[SKIP] No data to plot.")
        return

    # 전체 범위
    all_data = np.concatenate([after, before])
    dmin, dmax = float(all_data.min()), float(all_data.max())

    left  = np.floor(dmin / BIN_WIDTH) * BIN_WIDTH
    right = np.ceil(dmax / BIN_WIDTH)  * BIN_WIDTH

    if main_mode == "fixed" and main_x is not None:
        left  = min(left,  main_x[0])
        right = max(right, np.ceil(main_x[1]/BIN_WIDTH)*BIN_WIDTH)

    edges   = np.arange(left, right + BIN_WIDTH*0.999, BIN_WIDTH)
    centers = (edges[:-1] + edges[1:]) / 2

    cnt_after,  _ = np.histogram(after,  bins=edges)
    cnt_before, _ = np.histogram(before, bins=edges)

    bar_width = BIN_WIDTH * 0.45

    fig, ax = plt.subplots(figsize=(7, 6), dpi=150)

    # main bars (윤곽선 포함)
    ax.bar(centers - bar_width/2, cnt_after,
           width=bar_width, color=COLOR_APPLIED,
           alpha=ALPHA_MAIN, edgecolor="k", linewidth=0.5, label="DFC")

    ax.bar(centers + bar_width/2, cnt_before,
           width=bar_width, color=COLOR_NOT,
           alpha=ALPHA_MAIN, edgecolor="k", linewidth=0.5, label="non DFC")

    ax.set_ylabel("Count", fontsize=8)
    ax.set_xlabel(r"Total $t_{100\%}$ (h)", fontsize=8)

    peak = max(cnt_after.max(), cnt_before.max())

    # axis limits
    if main_mode == "fixed" and main_x is not None and main_y is not None:
        ax.set_xlim(*main_x)
        ax.set_ylim(0, main_y)
    else:
        ax.set_xlim(edges[0], edges[-1])
        ax.set_ylim(0, peak * 1.15)

    # tick labels
    tick_positions = np.arange(ax.get_xlim()[0], ax.get_xlim()[1]+1e-9, BIN_WIDTH)
    tick_idx = np.arange(0, len(tick_positions), LABEL_EVERY)
    ax.set_xticks(tick_positions[tick_idx])
    ax.set_xticklabels([str(int(x)) for x in tick_positions[tick_idx]], fontsize=8)

    # grid OFF
    ax.grid(False)

    # legend 윤곽선
    leg = ax.legend(loc="upper right", frameon=True, fontsize=8)
    frame = leg.get_frame()
    frame.set_edgecolor("grey")
    frame.set_linewidth(0.6)

    ax.tick_params(axis="both", labelsize=8)

    # ───── inset ─────
    delta = pd.to_numeric(df_t95["delta_t_h"], errors="coerce").dropna().to_numpy()
    if delta.size > 0:

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

        # inset axis range
        if inset_mode == "fixed" and inset_x is not None:
            lo, hi = inset_x
            lo_edge = np.floor(lo / 5) * 5
            hi_edge = np.ceil(hi / 5) * 5
        else:
            dmin2, dmax2 = delta.min(), delta.max()
            lo_edge = np.floor(dmin2 / 5) * 5
            hi_edge = np.ceil(dmax2  / 5) * 5
            lo, hi = lo_edge, hi_edge

        bins_dt = np.arange(lo_edge, hi_edge + 4.999, 5)

        axins.hist(delta, bins=bins_dt, color=COLOR_DELTA,
                   alpha=ALPHA_INSET, edgecolor="k", linewidth=0.5)

        axins.set_xlim(lo, hi)

        if inset_mode == "fixed" and inset_y is not None:
            axins.set_ylim(0, inset_y)

        axins.set_xlabel(r'$\Delta t_{100\%}$ (h)', fontsize=8)
        axins.set_ylabel("Count", fontsize=8)
        axins.tick_params(labelsize=8)

        axins.xaxis.set_major_locator(MultipleLocator(50))
        axins.xaxis.set_minor_locator(MultipleLocator(5))
        axins.xaxis.set_minor_formatter(NullFormatter())

    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[SAVE] {out_png}")

# ========= (2) 2×2 subplot용: ax에 그리는 버전 =========
def plot_grouped_hist_with_inset_ax(ax, df_t95, cfg_key="ALL"):

    cfg = PLOT_LIMITS[cfg_key]

    main_mode = cfg["main_mode"]
    main_x    = cfg["main_x"]
    main_y    = cfg["main_y"]

    inset_mode = cfg["inset_mode"]
    inset_x    = cfg["inset_x"]
    inset_y    = cfg["inset_y"]

    usecols = ["t95_before_h", "t95_after_h", "delta_t_h"]
    for c in usecols:
        if c not in df_t95.columns:
            raise ValueError(f"Missing column: {c}")

    after  = pd.to_numeric(df_t95["t95_after_h"], errors="coerce").dropna().to_numpy()
    before = pd.to_numeric(df_t95["t95_before_h"], errors="coerce").dropna().to_numpy()

    if after.size == 0 and before.size == 0:
        ax.axis("off")
        return

    # 전체 범위
    all_data = np.concatenate([after, before])
    dmin, dmax = float(all_data.min()), float(all_data.max())

    left  = np.floor(dmin / BIN_WIDTH) * BIN_WIDTH
    right = np.ceil(dmax / BIN_WIDTH)  * BIN_WIDTH

    if main_mode == "fixed" and main_x is not None:
        left  = min(left,  main_x[0])
        right = max(right, np.ceil(main_x[1]/BIN_WIDTH)*BIN_WIDTH)

    edges   = np.arange(left, right + BIN_WIDTH*0.999, BIN_WIDTH)
    centers = (edges[:-1] + edges[1:]) / 2

    cnt_after,  _ = np.histogram(after,  bins=edges)
    cnt_before, _ = np.histogram(before, bins=edges)

    bar_width = BIN_WIDTH * 0.45

    # main bars (윤곽선 포함)
    ax.bar(centers - bar_width/2, cnt_after,
           width=bar_width, color=COLOR_APPLIED,
           alpha=ALPHA_MAIN, edgecolor="k", linewidth=0.5, label="DFC")

    ax.bar(centers + bar_width/2, cnt_before,
           width=bar_width, color=COLOR_NOT,
           alpha=ALPHA_MAIN, edgecolor="k", linewidth=0.5, label="non DFC")

    ax.set_ylabel("Count", fontsize=8)
    ax.set_xlabel(r"Total $t_{100\%}$ (h)", fontsize=8)

    peak = max(cnt_after.max(), cnt_before.max())

    # axis limits
    if main_mode == "fixed" and main_x is not None and main_y is not None:
        ax.set_xlim(*main_x)
        ax.set_ylim(0, main_y)
    else:
        ax.set_xlim(edges[0], edges[-1])
        ax.set_ylim(0, peak * 1.15)

    # tick labels
    tick_positions = np.arange(ax.get_xlim()[0], ax.get_xlim()[1]+1e-9, BIN_WIDTH)
    tick_idx = np.arange(0, len(tick_positions), LABEL_EVERY)
    ax.set_xticks(tick_positions[tick_idx])
    ax.set_xticklabels([str(int(x)) for x in tick_positions[tick_idx]], fontsize=8)

    # grid OFF
    ax.grid(False)

    # legend 윤곽선
    leg = ax.legend(loc="upper right", frameon=True, fontsize=8)
    frame = leg.get_frame()
    frame.set_edgecolor("grey")
    frame.set_linewidth(0.6)

    ax.tick_params(axis="both", labelsize=8)

    # ───── inset ─────
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

        # inset axis range
        if inset_mode == "fixed" and inset_x is not None:
            lo, hi = inset_x
            lo_edge = np.floor(lo / 5) * 5
            hi_edge = np.ceil(hi / 5) * 5
        else:
            dmin2, dmax2 = delta.min(), delta.max()
            lo_edge = np.floor(dmin2 / 5) * 5
            hi_edge = np.ceil(dmax2  / 5) * 5
            lo, hi = lo_edge, hi_edge

        bins_dt = np.arange(lo_edge, hi_edge + 4.999, 5)

        axins.hist(delta, bins=bins_dt, color=COLOR_DELTA,
                   alpha=ALPHA_INSET, edgecolor="k", linewidth=0.5)

        axins.set_xlim(lo, hi)

        if inset_mode == "fixed" and inset_y is not None:
            axins.set_ylim(0, inset_y)

        axins.set_xlabel(r'$\Delta t_{100\%}$ (h)', fontsize=8)
        axins.set_ylabel("Count", fontsize=8)
        axins.tick_params(labelsize=8)

        axins.xaxis.set_major_locator(MultipleLocator(50))
        axins.xaxis.set_minor_locator(MultipleLocator(5))
        axins.xaxis.set_minor_formatter(NullFormatter())

from PIL import Image

def combine_2x2_images(img_all, img_c0, img_c1, img_c2, out_path):
    # 이미지 로드
    im_all = Image.open(img_all)
    im_c0  = Image.open(img_c0)
    im_c1  = Image.open(img_c1)
    im_c2  = Image.open(img_c2)

    # 이미지 크기 확인 (모두 동일해야 함)
    w, h = im_all.size

    # 2×2 캔버스 생성
    canvas = Image.new("RGB", (w*2, h*2), (255,255,255))

    # 배치
    canvas.paste(im_all, (0,0))
    canvas.paste(im_c0,  (w,0))
    canvas.paste(im_c1,  (0,h))
    canvas.paste(im_c2,  (w,h))

    # 저장
    canvas.save(out_path, quality=95)
    print("[SAVE 2×2]", out_path)


# ========= 실행부 =========
if __name__ == "__main__":

    t95 = pd.read_csv(CSV_T95)
    cl  = pd.read_csv(CSV_CLUST)

    t95 = ensure_base_key_col(t95)
    cl  = ensure_base_key_col(cl)

    cl["cluster"] = pd.to_numeric(cl["cluster"], errors="coerce")
    merged = pd.merge(t95, cl[["base_key", "cluster"]], how="left", on="base_key")

    # (1) 기존처럼 각각 개별 figure 저장
    plot_grouped_hist_with_inset(
        merged,
        os.path.join(OUT_DIR, "t95_hist_grouped_all.png"),
        cfg_key="ALL"
    )

    for cid in [0, 1, 2]:
        sub = merged[merged["cluster"] == cid]
        if len(sub) == 0:
            continue
        plot_grouped_hist_with_inset(
            sub,
            os.path.join(OUT_DIR, f"t95_hist_grouped_cluster{cid}.png"),
            cfg_key=f"cluster{cid}"
        )

    # --- 4개 이미지를 2x2로 합치기 ---
    img_all = os.path.join(OUT_DIR, "t95_hist_grouped_all.png")
    img_c0  = os.path.join(OUT_DIR, "t95_hist_grouped_cluster0.png")
    img_c1  = os.path.join(OUT_DIR, "t95_hist_grouped_cluster1.png")
    img_c2  = os.path.join(OUT_DIR, "t95_hist_grouped_cluster2.png")

    out_2x2 = os.path.join(OUT_DIR, "t95_hist_grouped_2x2.png")
    combine_2x2_images(img_all, img_c0, img_c1, img_c2, out_2x2)


    # (2) 2×2 subplot 한 장으로 저장 (All, C0, C1, C2)
    fig, axes = plt.subplots(2, 2, figsize=(10, 10), dpi=150)

    # (0,0): ALL
    plot_grouped_hist_with_inset_ax(
        axes[0, 0],
        merged,
        cfg_key="ALL"
    )

    # (0,1): cluster 0
    sub0 = merged[merged["cluster"] == 0]
    if len(sub0) > 0:
        plot_grouped_hist_with_inset_ax(
            axes[0, 1],
            sub0,
            cfg_key="cluster0"
        )
    else:
        axes[0, 1].axis("off")

    # (1,0): cluster 1
    sub1 = merged[merged["cluster"] == 1]
    if len(sub1) > 0:
        plot_grouped_hist_with_inset_ax(
            axes[1, 0],
            sub1,
            cfg_key="cluster1"
        )
    else:
        axes[1, 0].axis("off")

    # (1,1): cluster 2
    sub2 = merged[merged["cluster"] == 2]
    if len(sub2) > 0:
        plot_grouped_hist_with_inset_ax(
            axes[1, 1],
            sub2,
            cfg_key="cluster2"
        )
    else:
        axes[1, 1].axis("off")

    plt.tight_layout()
    out_2x2 = os.path.join(OUT_DIR, "t95_hist_grouped_2x2.png")
    fig.savefig(out_2x2, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[SAVE] {out_2x2}")
