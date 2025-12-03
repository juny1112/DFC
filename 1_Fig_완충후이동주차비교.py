#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os, re
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.dates import AutoDateLocator, ConciseDateFormatter
from pandas.tseries.offsets import MonthEnd

# ─────────────────────────────────────────────────────────
# 옵션
# ─────────────────────────────────────────────────────────
# current 같이 그릴지 여부 (True/False)
PLOT_CURRENT = True

# 선택적으로 그릴 base_key 목록 (비워두면 전체)
# 예시: ['bms_01241225206_2023-01', 'bms_01241225206_2023-02']
SELECT_BASE_KEYS = []

# ─────────────────────────────────────────────────────────
# 경로 설정
# ─────────────────────────────────────────────────────────
DIR_CR        = r'Z:\SamsungSTF\Processed_Data\DFC\EV6\CR_parsing'            # ..._CR.csv 만 ( _r 불가 )
DIR_DFC_ORIG  = r'Z:\SamsungSTF\Processed_Data\DFC\EV6\DFC_원본'               # ..._DFC.csv (완충후이동주차 미적용)
DIR_DFC_APPL  = r'Z:\SamsungSTF\Processed_Data\DFC\EV6\DFC_완충후이동주차'     # ..._DFC.csv (완충후이동주차 적용)
APPLIED_SUMMARY_CSV = r'G:\공유 드라이브\BSG_DFC_result\EV6\DFC_완충후이동주차\FULLCHARGE_PARKING_SUMMARY_FROM_OFF.csv'

OUT_DIR = r'G:\공유 드라이브\BSG_DFC_result\EV6\DFC_완충후이동주차\완충후이동주차_적용_비교'
Path(OUT_DIR).mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────────────────
# 스타일 (고정)
# ─────────────────────────────────────────────────────────
CLR_CR            = '#1f77b4'  # 파랑
LS_CR             = '-'        # 실선
CLR_DFC_APPLIED   = '#d62728'  # 빨강(완충후이동주차 적용)
LS_DFC_APPLIED    = '--'       # 점선
CLR_DFC_ORIGINAL  = '#ff7f0e'  # 주황(DFC 원본)
LS_DFC_ORIGINAL   = '-.'       # dash-dot
LW_CR   = 2.2
LW_DFC  = 2.0

# current 라인 스타일 (SOC와 구분 위해 점선/얇게)
LS_CURR           = ':'        # current 점선
LW_CURR           = 1.0

# 범례 라벨
LAB_CR           = 'DFC not applied'
LAB_DFC_ORIG     = 'DFC applied'
LAB_DFC_APPLIED  = 'DFC applied with auto parking'

# current는 CR만 표시
LAB_CR_CURR      = 'Current (DFC not applied)'

# ─────────────────────────────────────────────────────────
# 정규식 (altitude 유무 포함)
# ─────────────────────────────────────────────────────────
BASE_KEY = r'(bms_(?:altitude_)?\d+_\d{4}-\d{2})'
RE_KEY_CR   = re.compile(rf'^{BASE_KEY}_CR(?:_.*)?\.csv$',  re.IGNORECASE)
RE_KEY_DFC  = re.compile(rf'^{BASE_KEY}_DFC(?:_.*)?\.csv$', re.IGNORECASE)
# 요약 CSV에서 base_key 추출 (_r/경로/확장자/altitude 유무 모두 허용)
RE_SUM_FLEX = re.compile(rf'{BASE_KEY}(?:_r)?(?:_DFC)?(?:\.csv)?', re.IGNORECASE)
RE_SORT     = re.compile(r'^bms_(?:altitude_)?(?P<id>\d+?)_(?P<ym>\d{4}-\d{2})$', re.IGNORECASE)

def sort_key(base_key: str):
    m = RE_SORT.match(base_key)
    if not m:
        return (base_key, '9999-99')
    return (int(m.group('id')), m.group('ym'))

# ─────────────────────────────────────────────────────────
# 스캐너
# ─────────────────────────────────────────────────────────
def scan_dir_cr_only(root: str):
    """_CR.csv만 허용 → {base_key: fullpath}"""
    mp = {}
    if not os.path.isdir(root):
        print(f"[WARN] not a dir: {root}")
        return mp
    for fn in os.listdir(root):
        if not fn.lower().endswith('.csv'):
            continue
        m = RE_KEY_CR.match(fn)
        if m:
            mp[m.group(1)] = os.path.join(root, fn)
    return mp

def scan_dir_dfc(root: str):
    """_DFC.csv만 허용 → {base_key: fullpath}"""
    mp = {}
    if not os.path.isdir(root):
        print(f"[WARN] not a dir: {root}")
        return mp
    for fn in os.listdir(root):
        if not fn.lower().endswith('.csv'):
            continue
        m = RE_KEY_DFC.match(fn)
        if m:
            mp[m.group(1)] = os.path.join(root, fn)
    return mp

# ─────────────────────────────────────────────────────────
# 요약 CSV → 적용 대상 base_key 수집
# ─────────────────────────────────────────────────────────
def collect_applied_keys(summary_csv: str):
    df = pd.read_csv(summary_csv)
    keys = set()

    def _scan_series(series: pd.Series):
        for raw in series.dropna().astype(str):
            base = os.path.basename(raw)
            for tok in re.split(r'[\s,;|]+', base):
                if not tok:
                    continue
                m = RE_SUM_FLEX.search(tok)
                if m:
                    keys.add(m.group(1))

    for c in df.columns:
        if str(df[c].dtype) == 'object':
            _scan_series(df[c])

    for c in ("file", "filename", "file_stem", "path"):
        if c in df.columns:
            _scan_series(df[c])

    print(f"[DEBUG] collected {len(keys)} applied base_keys (first 10): {sorted(keys, key=sort_key)[:10]}")
    return keys

# ─────────────────────────────────────────────────────────
# 로딩/유틸
# ─────────────────────────────────────────────────────────
def load_bms_csv(path):
    df = pd.read_csv(path)
    if 'time' not in df.columns or 'soc' not in df.columns:
        raise ValueError(f"[ERR] required columns (time, soc) not found: {path}")
    df['time'] = pd.to_datetime(df['time'], errors='coerce')
    df['soc']  = pd.to_numeric(df['soc'],  errors='coerce')

    # pack_current 있으면 숫자로 변환해서 같이 사용, 없으면 NaN 컬럼 생성
    if 'pack_current' in df.columns:
        df['pack_current'] = pd.to_numeric(df['pack_current'], errors='coerce')
    else:
        df['pack_current'] = np.nan

    return df.dropna(subset=['time','soc']).sort_values('time')

def month_windows(year, month):
    start = pd.Timestamp(year=year, month=month, day=1)
    end   = (start + MonthEnd(1)).replace(hour=23, minute=59, second=59, microsecond=999999)
    return [
        (start,                        start + pd.Timedelta(days=7)  - pd.Timedelta(seconds=1)),
        (start + pd.Timedelta(days=7), start + pd.Timedelta(days=14) - pd.Timedelta(seconds=1)),
        (start + pd.Timedelta(days=14),start + pd.Timedelta(days=21) - pd.Timedelta(seconds=1)),
        (start + pd.Timedelta(days=21),end)
    ]

def slice_range(df, s, e):
    return df.loc[(df['time'] >= s) & (df['time'] <= e)]

def style_axes(ax):
    """공통 축 스타일 (범례 제외)"""
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

# ─────────────────────────────────────────────────────────
# 트리플 오버레이 (SOC + 옵션 current(CR만))
# ─────────────────────────────────────────────────────────
def draw_triple(df_cr, df_or, df_ap, ym, key, out_path):
    """CR(파랑), DFC 원본(주황 dash-dot), DFC 적용(빨강 점선) + (옵션) current (CR만)"""
    year, month = map(int, ym.split('-'))
    windows = month_windows(year, month)
    fig, axs = plt.subplots(2, 2, figsize=(12, 8), dpi=300)
    axs = axs.flatten()

    for i, (s, e) in enumerate(windows):
        d1 = slice_range(df_cr, s, e)
        d2 = slice_range(df_or, s, e)
        d3 = slice_range(df_ap, s, e)

        ax_soc = axs[i]

        # SOC 라인
        line_cr_soc, = ax_soc.plot(
            d1['time'], d1['soc'],
            color=CLR_CR, linestyle=LS_CR,
            linewidth=LW_CR, label=LAB_CR
        )
        line_or_soc, = ax_soc.plot(
            d2['time'], d2['soc'],
            color=CLR_DFC_ORIGINAL, linestyle=LS_DFC_ORIGINAL,
            linewidth=LW_DFC, label=LAB_DFC_ORIG
        )
        line_ap_soc, = ax_soc.plot(
            d3['time'], d3['soc'],
            color=CLR_DFC_APPLIED, linestyle=LS_DFC_APPLIED,
            linewidth=LW_DFC, label=LAB_DFC_APPLIED
        )

        style_axes(ax_soc)
        ax_soc.set_ylabel('SOC (%)')
        ax_soc.set_xlabel('Time (day)')
        ax_soc.set_title(f'{s:%b %d} – {e:%b %d}', fontsize=10, pad=6)

        # 기본 범례 (SOC 3개)
        lines = [line_cr_soc, line_or_soc, line_ap_soc]
        labels = [LAB_CR, LAB_DFC_ORIG, LAB_DFC_APPLIED]

        # ── current: CR 만 오른쪽 축에 표시 ──
        if PLOT_CURRENT and not d1['pack_current'].isna().all():
            ax_cur = ax_soc.twinx()
            ax_cur.set_ylabel('Current (A)')
            ax_cur.tick_params(axis='y', width=1.0, length=3, labelsize=8)

            line_cr_cur, = ax_cur.plot(
                d1['time'], d1['pack_current'],
                color=CLR_CR, linestyle=LS_CURR, linewidth=LW_CURR,
                label=LAB_CR_CURR
            )

            lines.append(line_cr_cur)
            labels.append(LAB_CR_CURR)

        # SOC + (옵션) current 합쳐서 범례
        leg = ax_soc.legend(
            lines, labels,
            loc='lower right', bbox_to_anchor=(0.98, 0.02),
            frameon=True, framealpha=0.9, fontsize=8, borderaxespad=0.0
        )
        for line in leg.get_lines():
            line.set_linewidth(LW_CR)

    fig.suptitle(
        f'SOC vs Time by week — {ym}  (KEY: {key})  [TRIPLE OVERLAY]',
        fontsize=13, y=0.98
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, bbox_inches='tight', dpi=300)
    plt.close(fig)
    print(f"[SAVE] {out_path}")

# ─────────────────────────────────────────────────────────
# 실행 (트리플 오버레이 전용)
# ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    applied_keys = collect_applied_keys(APPLIED_SUMMARY_CSV)
    print(f"[INFO] applied keys from summary (normalized): {len(applied_keys)}")

    map_cr   = scan_dir_cr_only(DIR_CR)      # _CR만
    map_orig = scan_dir_dfc(DIR_DFC_ORIG)    # _DFC (원본)
    map_appl = scan_dir_dfc(DIR_DFC_APPL)    # _DFC (완충후이동주차 적용)

    # 세 경로 모두 존재하는 키만 처리
    keys_to_process = sorted(
        [k for k in applied_keys if (k in map_cr) and (k in map_orig) and (k in map_appl)],
        key=sort_key
    )

    # 선택된 key만 사용 (SELECT_BASE_KEYS가 비어있으면 전체)
    if SELECT_BASE_KEYS:
        selected_set = set(SELECT_BASE_KEYS)
        before = len(keys_to_process)
        keys_to_process = [k for k in keys_to_process if k in selected_set]
        print(f"[INFO] SELECT_BASE_KEYS 사용: {before}개 → {len(keys_to_process)}개로 필터링")
    else:
        print("[INFO] SELECT_BASE_KEYS 비어있음 → 모든 triple-available key 사용")

    print(f"[INFO] triple-available keys 최종 개수: {len(keys_to_process)}")
    if not keys_to_process:
        miss_cr   = [k for k in applied_keys if k not in map_cr][:10]
        miss_orig = [k for k in applied_keys if k not in map_orig][:10]
        miss_appl = [k for k in applied_keys if k not in map_appl][:10]
        print(f"[DEBUG] sample missing CR   : {miss_cr}")
        print(f"[DEBUG] sample missing ORIG : {miss_orig}")
        print(f"[DEBUG] sample missing APPL : {miss_appl}")
        raise SystemExit("[INFO] No triple-overlay targets. Check folders & summary CSV.]")

    for key in keys_to_process:
        ym = key.split('_')[-1]  # YYYY-MM
        df_cr = load_bms_csv(map_cr[key])
        df_or = load_bms_csv(map_orig[key])
        df_ap = load_bms_csv(map_appl[key])

        out_path = os.path.join(OUT_DIR, f"{key}_TRIPLE_OVERLAY.png")
        draw_triple(df_cr, df_or, df_ap, ym, key, out_path)
