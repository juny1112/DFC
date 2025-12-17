#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os, re
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.dates import AutoDateLocator, ConciseDateFormatter
from pandas.tseries.offsets import MonthEnd
from concurrent.futures import ProcessPoolExecutor, as_completed   # 멀티프로세싱

# ─────────────────────────────────────────────────────
# 경로/설정
# ─────────────────────────────────────────────────────
CLUSTER_CSV = r"G:\공유 드라이브\BSG_DFC_result\combined\DFC_완충후이동주차\monthly_cluster\dfc_features_with_clusters.csv"

DIR_CR_MAP = {
    "EV6":    r"Z:\SamsungSTF\Processed_Data\DFC\EV6\CR_parsing",
    "Ioniq5": r"Z:\SamsungSTF\Processed_Data\DFC\Ioniq5\CR_parsing",
}
DIR_DFC_MAP = {
    "EV6":    r"Z:\SamsungSTF\Processed_Data\DFC\EV6\DFC_완충후이동주차",
    "Ioniq5": r"Z:\SamsungSTF\Processed_Data\DFC\Ioniq5\DFC_완충후이동주차",
}

BASE_OUT = r"G:\공유 드라이브\BSG_DFC_result\combined\DFC_완충후이동주차\Fig3"

CLUSTER_TO_DIR = {0: "Minimal", 1: "Frequent", 2: "Long"}

N_PER_CLUSTER = {
    0: 10,   # Minimal R_FC
    1: 10,   # Frequent R_FC (N 큰 순)
    2: 20,   # Long R_FC (AVG(t100%) 큰 순)
}
DEFAULT_N_PER_CLUSTER = 0

APPEND_MODE = True
N_WORKERS   = 8

# ───────────────── 스타일 ─────────────────
CLR_CR   = '#cd534c'; LS_CR = '-'
CLR_APPL = '#0073c2'; LS_APPL = '--'
LW_CR = 2.2; LW_APPL = 2.2

ALPHA_CR   = 0.9
ALPHA_APPL = 0.9

LAB_APPL = 'DFC'
LAB_CR   = 'non DFC'

# ───────────────── 파일/키 유틸 ─────────────────
BASE_KEY = r'(bms_(?:altitude_)?\d+_\d{4}-\d{2})'
RE_KEY_CR  = re.compile(rf'^{BASE_KEY}_CR(?:_.*)?\.csv$',  re.IGNORECASE)
RE_KEY_DFC = re.compile(rf'^{BASE_KEY}_DFC(?:_.*)?\.csv$', re.IGNORECASE)
RE_BASE_FLEX = re.compile(BASE_KEY, re.IGNORECASE)

def scan_dir_cr(root: str, model: str):
    mp = {}
    if not os.path.isdir(root):
        print(f"[WARN] not a dir (CR): {root}")
        return mp
    for fn in os.listdir(root):
        if not fn.lower().endswith('.csv'):
            continue
        m = RE_KEY_CR.match(fn)
        if m:
            key = m.group(1)
            mp[(model, key)] = os.path.join(root, fn)
    return mp

def scan_dir_dfc(root: str, model: str):
    mp = {}
    if not os.path.isdir(root):
        print(f"[WARN] not a dir (DFC): {root}")
        return mp
    for fn in os.listdir(root):
        if not fn.lower().endswith('.csv'):
            continue
        m = RE_KEY_DFC.match(fn)
        if m:
            key = m.group(1)
            mp[(model, key)] = os.path.join(root, fn)
    return mp

def extract_base_key_from_row(row: pd.Series):
    for _, val in row.items():
        if isinstance(val, str):
            m = RE_BASE_FLEX.search(val)
            if m:
                return m.group(1)
    return None

# ───────────────── 로딩/플롯 ─────────────────
def load_bms_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if 'time' not in df.columns or 'soc' not in df.columns:
        raise ValueError(f"[ERR] required columns (time, soc) not found: {path}")
    df['time'] = pd.to_datetime(df['time'], errors='coerce')
    df['soc']  = pd.to_numeric(df['soc'],  errors='coerce')
    return df.dropna(subset=['time','soc']).sort_values('time')

def month_windows(year: int, month: int):
    start = pd.Timestamp(year=year, month=month, day=1)
    end   = (start + MonthEnd(1)).replace(
        hour=23, minute=59, second=59, microsecond=999999
    )
    return [
        (start,                        start + pd.Timedelta(days=7)  - pd.Timedelta(seconds=1)),
        (start + pd.Timedelta(days=7), start + pd.Timedelta(days=14) - pd.Timedelta(seconds=1)),
        (start + pd.Timedelta(days=14),start + pd.Timedelta(days=21) - pd.Timedelta(seconds=1)),
        (start + pd.Timedelta(days=21),end)
    ]

def slice_range(df: pd.DataFrame, s, e):
    return df.loc[(df['time'] >= s) & (df['time'] <= e)]

def style_axes(ax):
    for sp in ax.spines.values():
        sp.set_linewidth(1.3)
    ax.tick_params(axis='both', width=1.1, length=4, labelsize=9)
    ax.axhline(95, color='0.5', linestyle=':', linewidth=1.2)
    ax.set_ylim(0, 100)
    ax.set_yticks([0,20,40,60,80,100])
    loc = AutoDateLocator()
    ax.xaxis.set_major_locator(loc)
    ax.xaxis.set_major_formatter(ConciseDateFormatter(loc))
    for lab in ax.get_xticklabels():
        lab.set_rotation(0)
        lab.set_horizontalalignment('center')

def draw_pair(df_cr: pd.DataFrame,
              df_ap: pd.DataFrame,
              ym: str,
              base_key: str,
              vehicle_model: str,
              out_path: str):
    year, month = map(int, ym.split('-'))
    windows = month_windows(year, month)

    fig, axs = plt.subplots(2, 2, figsize=(12, 8), dpi=300)
    axs = axs.flatten()

    for i, (s, e) in enumerate(windows):
        d1 = slice_range(df_cr, s, e)
        d2 = slice_range(df_ap, s, e)

        axs[i].plot(
            d1['time'], d1['soc'],
            color=CLR_CR, linestyle=LS_CR, linewidth=LW_CR,
            alpha=ALPHA_CR,
            label=LAB_CR
        )
        axs[i].plot(
            d2['time'], d2['soc'],
            color=CLR_APPL, linestyle=LS_APPL, linewidth=LW_APPL,
            alpha=ALPHA_APPL,
            label=LAB_APPL
        )

        style_axes(axs[i])
        axs[i].set_ylabel('SOC (%)')
        axs[i].set_xlabel('Time (day)')
        axs[i].legend(
            [LAB_APPL, LAB_CR],
            loc='lower left', bbox_to_anchor=(0.02, 0.02),
            frameon=True, framealpha=0.9, fontsize=9, borderaxespad=0.0
        )
        axs[i].set_title(f'{s:%b %d} – {e:%b %d}', fontsize=10, pad=6)

    fig.suptitle(
        f'{vehicle_model}: SOC vs Time by week — {ym}  (KEY: {base_key})',
        fontsize=13, y=0.98
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, bbox_inches='tight', dpi=300)
    plt.close(fig)
    print(f"[SAVE] {out_path}")

# ───────────────── worker ─────────────────
def worker_generate_plot(model: str,
                         key: str,
                         ym: str,
                         cr_path: str,
                         dfc_path: str,
                         out_path: str):
    try:
        # 이미 파일이 있으면 worker 쪽에서도 한 번 더 방어
        if os.path.exists(out_path):
            return (model, key, "exists")
        df_cr = load_bms_csv(cr_path)
        df_ap = load_bms_csv(dfc_path)
        draw_pair(df_cr, df_ap, ym, key, model, out_path)
        return (model, key, None)
    except Exception as e:
        return (model, key, str(e))

# ───────────────── 메인 ─────────────────
if __name__ == "__main__":
    dfc = pd.read_csv(CLUSTER_CSV)
    if "cluster" not in dfc.columns:
        raise SystemExit("[ERR] 'cluster' 컬럼이 없습니다.")
    if "vehicle model" not in dfc.columns:
        raise SystemExit("[ERR] 'vehicle model' 컬럼이 없습니다.")

    if "base_key" not in dfc.columns:
        dfc["base_key"] = dfc.apply(extract_base_key_from_row, axis=1)

    dfc["cluster"] = pd.to_numeric(dfc["cluster"], errors="coerce")
    valid = dfc["base_key"].notna() & dfc["cluster"].isin([0, 1, 2])
    dfc = dfc.loc[valid].copy()
    if dfc.empty:
        raise SystemExit("[INFO] 유효한 base_key/cluster 데이터를 찾지 못했습니다.")

    def pick_feature_columns(df):
        N_candidates    = ["N_used", "delta_t95_event_N", "N_events", "N_events_applied", "N_events_total"]
        mean_candidates = ["mean_used", "delta_t95_event_mean_h", "delta_t95_mean_h", "delayed_mean_h"]
        N_col = next((c for c in N_candidates if c in df.columns), None)
        mean_col = next((c for c in mean_candidates if c in df.columns), None)
        if N_col is None or mean_col is None:
            raise ValueError(
                "정렬에 필요한 컬럼이 없습니다. "
                "N_used / delta_t95_event_N, mean_used / delta_t95_event_mean_h 중 하나 이상 필요."
            )
        return N_col, mean_col

    N_col, mean_col = pick_feature_columns(dfc)

    map_cr   = {}
    map_appl = {}
    for model, root in DIR_CR_MAP.items():
        map_cr.update(scan_dir_cr(root, model))
    for model, root in DIR_DFC_MAP.items():
        map_appl.update(scan_dir_dfc(root, model))

    for cid in [0, 1, 2]:
        cluster_name = CLUSTER_TO_DIR.get(cid, f"cluster_{cid}")
        out_dir = Path(BASE_OUT) / cluster_name
        out_dir.mkdir(parents=True, exist_ok=True)

        sub = dfc[dfc["cluster"] == cid].copy()
        if sub.empty:
            print(f"[INFO] cluster {cid} ({cluster_name}): 후보 없음")
            continue

        if cid == 2:
            sort_col = mean_col
            sort_desc = True
        elif cid == 1:
            sort_col = N_col
            sort_desc = True
        else:
            sort_col = None
            sort_desc = True

        def has_both_files(row):
            model = str(row["vehicle model"])
            key   = row["base_key"]
            return (model, key) in map_cr and (model, key) in map_appl

        sub["has_files"] = sub.apply(has_both_files, axis=1)
        sub = sub[sub["has_files"]].copy()
        if sub.empty:
            print(f"[INFO] cluster {cid} ({cluster_name}): 유효 CR/DFC 쌍 없음")
            continue

        if sort_col is not None and sort_col in sub.columns:
            sub = sub.sort_values(sort_col, ascending=not sort_desc)
        else:
            sub = sub.sample(frac=1.0, random_state=42).copy()

        # 후보 만들면서, 이미 파일 있으면 바로 스킵 (append 모드일 때)
        all_candidates = []
        seen_pairs = set()
        for _, row in sub.iterrows():
            model = str(row["vehicle model"])
            key   = row["base_key"]
            pair  = (model, key)
            if pair in seen_pairs:
                continue
            seen_pairs.add(pair)

            ym = key.split('_')[-1]
            out_path = str(out_dir / f"{model}_{key}_CR_vs_DFC_APPLIED.png")

            if APPEND_MODE and os.path.exists(out_path):
                # 이미 그림이 있으니 후보로 추가하지 않음
                continue

            all_candidates.append((model, key, ym, out_path))

        if not all_candidates:
            print(f"[INFO] cluster {cid} ({cluster_name}): 새로 생성할 후보 없음")
            continue

        need = int(N_PER_CLUSTER.get(cid, DEFAULT_N_PER_CLUSTER))
        if need <= 0:
            print(f"[INFO] cluster {cid} ({cluster_name}): 생성 개수 0 → 건너뜀")
            continue

        picked = all_candidates[:need]

        jobs = []
        for model, key, ym, out_path in picked:
            cr_path  = map_cr[(model, key)]
            dfc_path = map_appl[(model, key)]
            jobs.append((model, key, ym, cr_path, dfc_path, out_path))

        gen = 0
        if jobs:
            with ProcessPoolExecutor(max_workers=N_WORKERS) as ex:
                futures = [ex.submit(worker_generate_plot, *job) for job in jobs]
                for fut in as_completed(futures):
                    model, key, err = fut.result()
                    if err is None:
                        gen += 1
                    elif err == "exists":
                        # 방어 로직에 걸린 경우 (이미 존재)
                        pass
                    else:
                        print(f"[SKIP] {model}, {key} - {err}")

        print(
            f"[INFO] cluster {cid} ({cluster_name}): "
            f"requested={need}, generated={gen}, folder={out_dir}"
        )
