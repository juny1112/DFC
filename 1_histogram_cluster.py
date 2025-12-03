# ───────────────── Plot: overall + per-cluster (0/1/2) ─────────────────
import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.ticker import MultipleLocator, NullFormatter, FormatStrFormatter

# ========= 경로 =========
CSV_T95   = r'G:\공유 드라이브\BSG_DFC_result\combined\DFC_완충후이동주차\t95_before_after_delta_combined.csv' # t95 계산
CSV_CLUST = r'G:\공유 드라이브\BSG_DFC_result\combined\DFC_완충후이동주차\dfc_features_with_clusters.csv'   # 클러스터 결과

OUT_DIR   = os.path.dirname(CSV_T95)

# ========= 플롯 옵션 =========
MAIN_LIMIT_MODE = 'fixed'     # 'auto' or 'fixed'
MAIN_X_RANGE    = (0, 550)
MAIN_Y_MAX      = 425
BIN_WIDTH       = 5.0
LABEL_EVERY     = 10

INSET_LIMIT_MODE = 'fixed'          # 'auto' or 'fixed'
INSET_X_RANGE    = (0.0, 450)
INSET_Y_MAX      = 500
INSET_BIN_STEP   = 5.0
INSET_TICK_STEP  = 50.0
EXTRA_DOWN       = 0.08

COLOR_APPLIED   = "#9ECAE1"   # DFC applied
COLOR_NOT       = "#FDAE6B"   # DFC not applied
COLOR_DELTA     = "#E15759"   # Δt
ALPHA_MAIN      = 0.90
ALPHA_INSET     = 0.90

# === 파일명 접미사(메인 모드 기준) ===
SUFFIX = "_fixed" if MAIN_LIMIT_MODE.lower() == "fixed" else "_auto"

# ========= 유틸: base_key 추출 =========
RE_BASE = re.compile(r'(bms_(?:altitude_)?\d+_\d{4}-\d{2})', re.IGNORECASE)

def ensure_base_key_col(df: pd.DataFrame) -> pd.DataFrame:
    if 'base_key' in df.columns:
        return df
    base_keys = []
    for _, row in df.iterrows():
        key = None
        for _, v in row.items():
            if isinstance(v, str):
                m = RE_BASE.search(v)
                if m:
                    key = m.group(1)
                    break
        base_keys.append(key)
    df = df.copy()
    df['base_key'] = base_keys
    return df

# ========= 공통 플로팅 함수 =========
def plot_grouped_hist_with_inset(df_t95: pd.DataFrame, out_png: str, title_suffix: str = ''):
    usecols = ['t95_before_h', 't95_after_h', 'delta_t_h']
    for c in usecols:
        if c not in df_t95.columns:
            raise ValueError(f"required column missing: {c}")

    after  = pd.to_numeric(df_t95['t95_after_h'],  errors='coerce').dropna().to_numpy()
    before = pd.to_numeric(df_t95['t95_before_h'], errors='coerce').dropna().to_numpy()

    all_data = np.concatenate([after, before]) if (after.size or before.size) else np.array([])
    if all_data.size == 0:
        print(f"[SKIP] No data to plot for {title_suffix}.")
        return

    dmin, dmax = float(np.min(all_data)), float(np.max(all_data))
    left  = np.floor(dmin / BIN_WIDTH) * BIN_WIDTH
    right = np.ceil(dmax / BIN_WIDTH)  * BIN_WIDTH
    if MAIN_LIMIT_MODE == 'fixed':
        left  = min(left, MAIN_X_RANGE[0])
        right = max(right, np.ceil(MAIN_X_RANGE[1] / BIN_WIDTH) * BIN_WIDTH)

    edges   = np.arange(left, right + BIN_WIDTH*0.999, BIN_WIDTH)
    centers = (edges[:-1] + edges[1:]) / 2

    cnt_after,  _ = np.histogram(after,  bins=edges)
    cnt_before, _ = np.histogram(before, bins=edges)

    bar_total_width = BIN_WIDTH * 0.90
    bar_width_each  = bar_total_width / 2.0
    offset          = bar_width_each / 2.0

    fig, ax = plt.subplots(figsize=(12, 5.5), dpi=150)
    ax.bar(centers - offset, cnt_after,  width=bar_width_each, label='DFC applied',
           color=COLOR_APPLIED, alpha=ALPHA_MAIN, edgecolor='none', align='center')
    ax.bar(centers + offset, cnt_before, width=bar_width_each, label='DFC not applied',
           color=COLOR_NOT,   alpha=ALPHA_MAIN, edgecolor='none', align='center')

    ttl = 't_95% comparison' + (f' — {title_suffix}' if title_suffix else '')
    ax.set_title(ttl)
    ax.set_ylabel('Count')
    ax.set_xlabel('Total t_95% (hours)')

    peak = int(max(cnt_after.max() if cnt_after.size else 0,
                   cnt_before.max() if cnt_before.size else 0))

    if MAIN_LIMIT_MODE == 'fixed':
        ax.set_xlim(*MAIN_X_RANGE)
        ax.set_ylim(0, MAIN_Y_MAX)
    else:
        ax.set_xlim(edges[0], edges[-1])
        ax.set_ylim(0, peak * 1.15 if peak > 0 else 1)

    if MAIN_LIMIT_MODE == 'fixed':
        tick_positions = np.arange(MAIN_X_RANGE[0], MAIN_X_RANGE[1] + 1e-9, BIN_WIDTH)
    else:
        tick_positions = edges[:-1]
    tick_idx = np.arange(0, len(tick_positions), LABEL_EVERY)
    ax.set_xticks(tick_positions[tick_idx])
    ax.set_xticklabels([f"{int(x)}" for x in tick_positions[tick_idx]], rotation=0)

    ax.xaxis.set_minor_locator(MultipleLocator(BIN_WIDTH))
    ax.grid(axis='y', alpha=0.25, linewidth=0.7)
    legend = ax.legend(loc='upper right', frameon=False)

    # ── Δt inset ──
    delta = pd.to_numeric(df_t95['delta_t_h'], errors='coerce').dropna().to_numpy()
    if delta.size > 0:
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        leg_bbox = legend.get_window_extent(renderer=renderer)
        x_disp = ax.bbox.x1
        y_disp = leg_bbox.y0
        _, y0_ax = ax.transAxes.inverted().transform((x_disp, y_disp))

        width_frac, height_frac, pad = 0.35, 0.35, 0.01
        x0 = 1 - width_frac - pad
        y0 = max(pad, y0_ax - height_frac - pad - EXTRA_DOWN)

        axins = inset_axes(ax, width="100%", height="100%",
                           loc="lower left",
                           bbox_to_anchor=(x0, y0, width_frac, height_frac),
                           bbox_transform=ax.transAxes, borderpad=0.0)

        if INSET_LIMIT_MODE == 'fixed':
            lo, hi = INSET_X_RANGE
            lo_edge = np.floor(lo / INSET_BIN_STEP) * INSET_BIN_STEP
            hi_edge = np.ceil(hi / INSET_BIN_STEP) * INSET_BIN_STEP
        else:
            dmin_i, dmax_i = float(np.min(delta)), float(np.max(delta))
            lo_edge = np.floor(dmin_i / INSET_BIN_STEP) * INSET_BIN_STEP
            hi_edge = np.ceil(dmax_i  / INSET_BIN_STEP) * INSET_BIN_STEP
            lo, hi = lo_edge, hi_edge

        bins_dt = np.arange(lo_edge, hi_edge + INSET_BIN_STEP*0.999, INSET_BIN_STEP)
        axins.hist(delta, bins=bins_dt, alpha=ALPHA_INSET, color=COLOR_DELTA, edgecolor='none')
        axins.axvline(0, color='k', linestyle='--', linewidth=0.8)
        axins.set_title('Δt_>95% distribution', fontsize=9)
        axins.set_xlabel('Δt_>95% (h)', fontsize=8)
        axins.set_ylabel('Count', fontsize=8)
        axins.tick_params(labelsize=8)

        axins.set_xlim(lo, hi)
        axins.xaxis.set_major_locator(MultipleLocator(INSET_TICK_STEP))
        axins.xaxis.set_minor_locator(MultipleLocator(INSET_BIN_STEP))
        axins.xaxis.set_minor_formatter(NullFormatter())
        axins.xaxis.set_major_formatter(FormatStrFormatter('%.0f'))
        if INSET_LIMIT_MODE == 'fixed' and INSET_Y_MAX is not None:
            axins.set_ylim(0, float(INSET_Y_MAX))

    fig.savefig(out_png, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"[SAVE] {out_png}")

# ========= 메인 =========
if __name__ == "__main__":
    # 1) 데이터 로드
    t95 = pd.read_csv(CSV_T95)
    cl  = pd.read_csv(CSV_CLUST)

    # 2) base_key 정규화 후 병합
    t95  = ensure_base_key_col(t95)
    cl   = ensure_base_key_col(cl)

    if 'cluster' not in cl.columns:
        raise SystemExit("[ERR] cluster 컬럼이 없습니다. 먼저 클러스터링을 수행하세요.")

    cl['cluster'] = pd.to_numeric(cl['cluster'], errors='coerce')

    merged = pd.merge(t95, cl[['base_key', 'cluster']], how='left', on='base_key')

    # 3) 전체 히스토그램
    plot_grouped_hist_with_inset(
        merged,
        os.path.join(OUT_DIR, f't95_hist_grouped_all{SUFFIX}.png'),
        'ALL'
    )

    # 4) 클러스터별 히스토그램 (0,1,2)
    for cid in [0, 1, 2]:
        sub = merged[merged['cluster'] == cid].copy()
        if len(sub) == 0:
            print(f"[INFO] cluster {cid}: rows=0 → 스킵")
            continue
        out_png = os.path.join(OUT_DIR, f't95_hist_grouped_cluster{cid}{SUFFIX}.png')
        plot_grouped_hist_with_inset(sub, out_png, f'Cluster {cid}')
