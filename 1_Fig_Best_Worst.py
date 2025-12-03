#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os, re
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.dates import AutoDateLocator, ConciseDateFormatter
from pandas.tseries.offsets import MonthEnd

# ─────────────────────────────────────────────────────
# 경로/설정
# ─────────────────────────────────────────────────────
# 클러스터 결과 CSV (앞 단계에서 생성된 dfc_features_with_clusters.csv)
CLUSTER_CSV = r"G:\공유 드라이브\BSG_DFC_result\EV6\DFC_완충후이동주차\dfc_features_with_clusters.csv"

# 원천 데이터 폴더들
DIR_CR       = r"Z:\SamsungSTF\Processed_Data\DFC\EV6\CR_parsing"             # ..._CR.csv 만 허용 (_r 불가)
DIR_DFC_APPL = r"Z:\SamsungSTF\Processed_Data\DFC\EV6\DFC_완충후이동주차"      # ..._DFC.csv (완충후이동주차 적용)

# 출력 폴더(클러스터별 하위폴더에 저장)
BASE_OUT = r"G:\공유 드라이브\BSG_DFC_result\EV6\DFC_완충후이동주차"

# 클러스터별 저장 폴더명 매핑
CLUSTER_TO_DIR = {0: "Worst", 1: "Best_1", 2: "Best_2"}

# [초기 생성 모드] 고정 개수로 생성하고 싶을 때 사용
CLUSTER_QUOTAS = {0: 10, 1: 10, 2: 10}

# [추가 생성 모드] 이미 저장된 것을 건너뛰고, 새로 N개씩만 더 생성
APPEND_MODE = True         # ← True면 append 모드로 동작
ADD_PER_CLUSTER = 5       # ← 클러스터당 추가 생성 개수

# 스타일 고정: CR(파랑 실선), DFC applied(빨강 점선)
CLR_CR   = '#1f77b4'; LS_CR = '-'
CLR_APPL = '#d62728'; LS_APPL = '--'
LW_CR = 2.2; LW_APPL = 2.0
LAB_CR   = 'DFC not applied'
LAB_APPL = 'DFC applied'

# ─────────────────────────────────────────────────────
# 파일 키/스캐너 유틸
# ─────────────────────────────────────────────────────
# BASE_KEY: bms_(altitude_)?<digits>_<YYYY-MM>
BASE_KEY = r'(bms_(?:altitude_)?\d+_\d{4}-\d{2})'
RE_KEY_CR   = re.compile(rf'^{BASE_KEY}_CR(?:_.*)?\.csv$',  re.IGNORECASE)
RE_KEY_DFC  = re.compile(rf'^{BASE_KEY}_DFC(?:_.*)?\.csv$', re.IGNORECASE)
RE_BASE_FLEX = re.compile(BASE_KEY, re.IGNORECASE)

# 이미 저장된 그림 파일에서 base_key 추출 (중복 생성 방지용)
RE_SAVED = re.compile(rf'^{BASE_KEY}_CR_vs_DFC_APPLIED\.png$', re.IGNORECASE)

def already_saved_keys(out_dir: Path):
    """해당 출력 폴더에서 이미 생성된 base_key 집합 반환"""
    keys = set()
    if not out_dir.exists():
        return keys
    for fn in os.listdir(out_dir):
        m = RE_SAVED.match(fn)
        if m:
            keys.add(m.group(1))
    return keys

def scan_dir_cr(root: str):
    """CR 폴더: ..._CR.csv 만 허용 ( _r 금지 ) → {base_key: fullpath}"""
    mp = {}
    if not os.path.isdir(root):
        print(f"[WARN] not a dir: {root}"); return mp
    for fn in os.listdir(root):
        if not fn.lower().endswith('.csv'):
            continue
        m = RE_KEY_CR.match(fn)
        if m:
            key = m.group(1)  # base_key
            mp[key] = os.path.join(root, fn)
    return mp

def scan_dir_dfc(root: str):
    """DFC 폴더: ..._DFC.csv 만 허용 → {base_key: fullpath}"""
    mp = {}
    if not os.path.isdir(root):
        print(f"[WARN] not a dir: {root}"); return mp
    for fn in os.listdir(root):
        if not fn.lower().endswith('.csv'):
            continue
        m = RE_KEY_DFC.match(fn)
        if m:
            key = m.group(1)
            mp[key] = os.path.join(root, fn)
    return mp

def extract_base_key_from_row(row: pd.Series):
    """여러 텍스트 컬럼에서 base_key 패턴을 찾아 반환"""
    for _, val in row.items():
        if isinstance(val, str):
            m = RE_BASE_FLEX.search(val)
            if m:
                return m.group(1)
    # 보조: id/month 형태로 있을 수도 있음 → 힌트형 조합 시도(안 맞으면 무시)
    id_cols = [c for c in row.index if 'id' in c.lower()]
    ym_cols = [c for c in row.index if re.search(r'(ym|month|yyyy|mm)', c.lower())]
    for ic in id_cols:
        for mc in ym_cols:
            try:
                iid = str(row[ic]); ym = str(row[mc])
                cand = f"bms_{iid}_{ym}"
                if RE_BASE_FLEX.fullmatch(cand):
                    return cand
            except Exception:
                pass
    return None

# ─────────────────────────────────────────────────────
# 로딩/슬라이싱/플롯
# ─────────────────────────────────────────────────────
def load_bms_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if 'time' not in df.columns or 'soc' not in df.columns:
        raise ValueError(f"[ERR] required columns (time, soc) not found: {path}")
    df['time'] = pd.to_datetime(df['time'], errors='coerce')
    df['soc']  = pd.to_numeric(df['soc'],  errors='coerce')
    return df.dropna(subset=['time','soc']).sort_values('time')

def month_windows(year: int, month: int):
    start = pd.Timestamp(year=year, month=month, day=1)
    end   = (start + MonthEnd(1)).replace(hour=23, minute=59, second=59, microsecond=999999)
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
        lab.set_rotation(45)
        lab.set_horizontalalignment('right')

def draw_pair(df_cr: pd.DataFrame, df_ap: pd.DataFrame, ym: str, base_key: str, out_path: str):
    """CR(파랑 실선) vs DFC applied(빨강 점선) — 모든 서브플롯에 x/y 라벨 및 범례"""
    year, month = map(int, ym.split('-'))
    windows = month_windows(year, month)
    fig, axs = plt.subplots(2,2, figsize=(12,8), dpi=300); axs = axs.flatten()
    for i,(s,e) in enumerate(windows):
        d1 = slice_range(df_cr, s, e)
        d2 = slice_range(df_ap, s, e)
        axs[i].plot(d1['time'], d1['soc'], color=CLR_CR,   linestyle=LS_CR,   linewidth=LW_CR,   label=LAB_CR)
        axs[i].plot(d2['time'], d2['soc'], color=CLR_APPL, linestyle=LS_APPL, linewidth=LW_APPL, label=LAB_APPL)
        style_axes(axs[i])
        axs[i].set_ylabel('SOC (%)')
        axs[i].set_xlabel('Time (day)')
        axs[i].legend([LAB_CR, LAB_APPL], loc='lower left', bbox_to_anchor=(0.02, 0.02),
                      frameon=True, framealpha=0.9, fontsize=9, borderaxespad=0.0)
        axs[i].set_title(f'{s:%b %d} – {e:%b %d}', fontsize=10, pad=6)
    fig.suptitle(f'SOC vs Time by week — {ym}  (KEY: {base_key})', fontsize=13, y=0.98)
    fig.tight_layout(rect=[0,0,1,0.96])
    fig.savefig(out_path, bbox_inches='tight', dpi=300)
    plt.close(fig)
    print(f"[SAVE] {out_path}")

# ─────────────────────────────────────────────────────
# 메인 로직
# ─────────────────────────────────────────────────────
if __name__ == "__main__":
    # 1) 클러스터 CSV 로드
    dfc = pd.read_csv(CLUSTER_CSV)
    if "cluster" not in dfc.columns:
        raise SystemExit("[ERR] 'cluster' 컬럼이 없습니다. 먼저 클러스터링을 수행해 주세요.")

    # 2) base_key 컬럼 생성(유연 추출)
    if "base_key" not in dfc.columns:
        dfc["base_key"] = dfc.apply(extract_base_key_from_row, axis=1)

    # 유효 base_key + cluster 정수만 사용
    dfc["cluster"] = pd.to_numeric(dfc["cluster"], errors="coerce")
    valid = dfc["base_key"].notna() & dfc["cluster"].isin([0,1,2])
    dfc = dfc.loc[valid].copy()

    if dfc.empty:
        raise SystemExit("[INFO] 유효한 base_key/cluster 데이터를 찾지 못했습니다.")

    # 3) 원천 파일 매핑 스캔
    map_cr   = scan_dir_cr(DIR_CR)
    map_appl = scan_dir_dfc(DIR_DFC_APPL)

    # 4) 클러스터별 후보 수집(두 폴더 모두 존재하는 키만)
    for cid in [0,1,2]:
        # 출력 폴더
        out_dir = Path(BASE_OUT) / CLUSTER_TO_DIR.get(cid, f"cluster_{cid}")
        out_dir.mkdir(parents=True, exist_ok=True)

        # 이 클러스터의 전체 후보
        all_candidates = []
        for _, row in dfc[dfc["cluster"] == cid].iterrows():
            k = row["base_key"]
            if k in map_cr and k in map_appl:
                all_candidates.append(k)

        if not all_candidates:
            print(f"[INFO] cluster {cid}: 후보 없음")
            continue

        # 이미 저장된 키 제외 (append 모드)
        if APPEND_MODE:
            done = already_saved_keys(out_dir)
            candidates = [k for k in all_candidates if k not in done]
            need = int(ADD_PER_CLUSTER)
        else:
            candidates = all_candidates
            need = int(CLUSTER_QUOTAS.get(cid, 0))

        if need <= 0:
            print(f"[INFO] cluster {cid}: 생성 개수 0 → 건너뜀")
            continue

        # 중복 제거 후 앞에서 need개
        picked, seen = [], set()
        for k in candidates:
            if k in seen:
                continue
            seen.add(k)
            picked.append(k)
            if len(picked) >= need:
                break

        # 그림 생성
        gen = 0
        for k in picked:
            ym = k.split('_')[-1]  # YYYY-MM
            try:
                df_cr  = load_bms_csv(map_cr[k])
                df_ap  = load_bms_csv(map_appl[k])
                out_path = str(out_dir / f"{k}_CR_vs_DFC_APPLIED.png")
                draw_pair(df_cr, df_ap, ym, k, out_path)
                gen += 1
            except Exception as e:
                print(f"[SKIP] {k} - {e}")

        print(f"[INFO] cluster {cid}: requested={need}, generated={gen}, folder={out_dir}")
