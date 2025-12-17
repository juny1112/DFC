#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
from pathlib import Path
import pandas as pd
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.dates import AutoDateLocator, ConciseDateFormatter
from pandas.tseries.offsets import MonthEnd
from concurrent.futures import ProcessPoolExecutor, as_completed

# =========================================================
# (A) 설정
# =========================================================

# 1) SOC0_count 결과 CSV
SOC0_CSV = r"G:\공유 드라이브\BSG_DFC_result\combined\DFC_완충후이동주차\불량개입\bad_intervention_soc0_count.csv"

# 2) 원본(비조작) 데이터 루트: R_parsing_완충후이동주차
DIR_ORIG_MAP = {
    "EV6":    r"Z:\SamsungSTF\Processed_Data\DFC\EV6\R_parsing_완충후이동주차",
    "Ioniq5": r"Z:\SamsungSTF\Processed_Data\DFC\Ioniq5\R_parsing_완충후이동주차",
}

# 3) 조작(DFC) 데이터 루트: 불량개입 생성 폴더(당신 코드 결과물)
DIR_DFC_MAP = {
    "EV6":    r"Z:\SamsungSTF\Processed_Data\DFC\EV6\불량개입",
    "Ioniq5": r"Z:\SamsungSTF\Processed_Data\DFC\Ioniq5\불량개입",
}

# 4) 그림 저장 폴더
OUT_DIR = r"G:\공유 드라이브\BSG_DFC_result\combined\DFC_완충후이동주차\불량개입\불량개입_fig"

# 5) 생성 개수(“새로 생성되는” 것 기준)
N_TO_PLOT = 80

# 6) APPEND 모드: True면 이미 생성된 그림은 스킵
APPEND_MODE = True

# 7) 병렬 워커 수
N_WORKERS = 8

# 8) 스타일(주신 예시 기반)
CLR_CR   = '#cd534c'; LS_CR = '-'
CLR_APPL = '#0073c2'; LS_APPL = '--'
LW_CR = 2.2; LW_APPL = 2.2
ALPHA_CR   = 0.9
ALPHA_APPL = 0.9
LAB_APPL = 'DFC'
LAB_CR   = 'non DFC'

# =========================================================
# (B) 유틸
# =========================================================

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def to_datetime_soc(df: pd.DataFrame) -> pd.DataFrame:
    if 'time' not in df.columns or 'soc' not in df.columns:
        raise ValueError("required columns (time, soc) not found")
    df = df.copy()
    df['time'] = pd.to_datetime(df['time'], errors='coerce')
    df['soc']  = pd.to_numeric(df['soc'], errors='coerce')
    df = df.dropna(subset=['time', 'soc']).sort_values('time')
    return df

def load_csv_min(path: str) -> pd.DataFrame:
    # 원본/조작본 모두 time,soc만 있으면 됨. (원본은 컬럼 많아도 OK)
    df = pd.read_csv(path, usecols=lambda c: c in {'time', 'soc'})
    return to_datetime_soc(df)

def infer_ym_from_filename(file_name: str):
    """
    file_name 예: bms_01241228049_2023-07_r.csv
    여기서 '2023-07' 추출
    """
    # 가장 보수적으로 "_YYYY-MM" 패턴을 찾음
    import re
    m = re.search(r'_(\d{4}-\d{2})', file_name)
    if not m:
        return None
    return m.group(1)

def month_windows(year: int, month: int):
    start = pd.Timestamp(year=year, month=month, day=1)
    end   = (start + MonthEnd(1)).replace(
        hour=23, minute=59, second=59, microsecond=999999
    )
    return [
        (start,                         start + pd.Timedelta(days=7)  - pd.Timedelta(seconds=1)),
        (start + pd.Timedelta(days=7),  start + pd.Timedelta(days=14) - pd.Timedelta(seconds=1)),
        (start + pd.Timedelta(days=14), start + pd.Timedelta(days=21) - pd.Timedelta(seconds=1)),
        (start + pd.Timedelta(days=21), end),
    ]

def slice_range(df: pd.DataFrame, s, e):
    return df.loc[(df['time'] >= s) & (df['time'] <= e)]

def style_axes(ax):
    for sp in ax.spines.values():
        sp.set_linewidth(1.3)
    ax.tick_params(axis='both', width=1.1, length=4, labelsize=9)
    ax.axhline(95, color='0.5', linestyle=':', linewidth=1.2)
    ax.set_ylim(0, 100)
    ax.set_yticks([0, 20, 40, 60, 80, 100])

    loc = AutoDateLocator()
    ax.xaxis.set_major_locator(loc)
    ax.xaxis.set_major_formatter(ConciseDateFormatter(loc))
    for lab in ax.get_xticklabels():
        lab.set_rotation(0)
        lab.set_horizontalalignment('center')

def draw_pair(df_cr: pd.DataFrame,
              df_ap: pd.DataFrame,
              ym: str,
              car_type: str,
              file_name: str,
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
            alpha=ALPHA_CR, label=LAB_CR
        )
        axs[i].plot(
            d2['time'], d2['soc'],
            color=CLR_APPL, linestyle=LS_APPL, linewidth=LW_APPL,
            alpha=ALPHA_APPL, label=LAB_APPL
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
        f'{car_type}: SOC vs Time by week — {ym}  (FILE: {file_name})',
        fontsize=13, y=0.98
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, bbox_inches='tight', dpi=300)
    plt.close(fig)

def find_file_recursive(root: str, target_name: str):
    """
    root 하위에서 정확히 파일명(target_name) 일치하는 첫 경로 반환
    """
    rootp = Path(root)
    for p in rootp.rglob(target_name):
        if p.is_file() and p.name.lower() == target_name.lower():
            return str(p)
    return None

def derive_dfc_filename_from_input(file_name: str):
    """
    입력 file_name이 *_r.csv이면 -> *_DFC.csv 로 변환
    아니면 -> stem + _DFC.csv
    """
    p = Path(file_name)
    stem = p.stem  # 확장자 제외
    if stem.endswith("_r"):
        stem = stem[:-2]
    return f"{stem}_DFC.csv"

# =========================================================
# (C) 워커
# =========================================================

def worker(job):
    car_type, file_name, orig_root, dfc_root, out_path = job
    try:
        if APPEND_MODE and os.path.exists(out_path):
            return (car_type, file_name, "exists")

        ym = infer_ym_from_filename(file_name)
        if ym is None:
            return (car_type, file_name, "ym_not_found")

        # 원본 파일 찾기
        orig_path = find_file_recursive(orig_root, file_name)
        if orig_path is None:
            return (car_type, file_name, "orig_not_found")

        # 조작(DFC) 파일 찾기: file_name -> *_DFC.csv
        dfc_name = derive_dfc_filename_from_input(file_name)
        dfc_path = find_file_recursive(dfc_root, dfc_name)
        if dfc_path is None:
            return (car_type, file_name, f"dfc_not_found({dfc_name})")

        df_cr = load_csv_min(orig_path)
        df_ap = load_csv_min(dfc_path)

        draw_pair(df_cr, df_ap, ym, car_type, file_name, out_path)
        return (car_type, file_name, None)

    except Exception as e:
        return (car_type, file_name, str(e))

# =========================================================
# (D) 메인
# =========================================================

if __name__ == "__main__":
    ensure_dir(OUT_DIR)

    df = pd.read_csv(SOC0_CSV)
    # 필수 컬럼 체크
    for c in ["car_type", "file_name", "SOC0_count"]:
        if c not in df.columns:
            raise SystemExit(f"[ERR] '{c}' 컬럼이 없습니다: {SOC0_CSV}")

    df["SOC0_count"] = pd.to_numeric(df["SOC0_count"], errors="coerce").fillna(0).astype(int)

    # 1) SOC0_count >= 1 필터
    sub = df[df["SOC0_count"] >= 1].copy()
    if sub.empty:
        raise SystemExit("[INFO] SOC0_count >= 1 인 파일이 없습니다.")

    # 2) file_name 순 정렬
    sub = sub.sort_values("file_name", ascending=True).reset_index(drop=True)

    # 3) 후보 목록 생성 (append 모드면 이미 있는 그림은 제외)
    candidates = []
    for _, r in sub.iterrows():
        car_type  = str(r["car_type"])
        file_name = str(r["file_name"])

        if car_type not in DIR_ORIG_MAP or car_type not in DIR_DFC_MAP:
            continue

        # file_name에서 .csv 제거 (대소문자 무시)
        base = Path(file_name).stem  # bms_..._2023-07_r  (확장자 제거)
        fig_name = f"{car_type}_{base}.png"
        out_path = str(Path(OUT_DIR) / fig_name)

        if APPEND_MODE and os.path.exists(out_path):
            continue

        candidates.append((car_type, file_name, DIR_ORIG_MAP[car_type], DIR_DFC_MAP[car_type], out_path))

    if not candidates:
        raise SystemExit("[INFO] 새로 생성할 후보가 없습니다. (이미 모두 생성됨)")

    # 4) 사용자가 지정한 개수만큼만 뽑기
    picked = candidates[:max(0, int(N_TO_PLOT))]
    if not picked:
        raise SystemExit("[INFO] N_TO_PLOT이 0이거나 후보가 없습니다.")

    print(f"[INFO] total SOC0>=1 files={len(sub)} | new candidates={len(candidates)} | will_generate={len(picked)}")
    print(f"[INFO] out_dir={OUT_DIR} | workers={N_WORKERS} | append={APPEND_MODE}")

    gen = 0
    with ProcessPoolExecutor(max_workers=N_WORKERS) as ex:
        futures = [ex.submit(worker, j) for j in picked]
        for fut in tqdm(as_completed(futures), total=len(futures), unit="fig"):
            car_type, file_name, err = fut.result()
            if err is None:
                gen += 1
            elif err == "exists":
                pass
            else:
                print(f"[SKIP] {car_type} | {file_name} -> {err}")

    print(f"[DONE] generated={gen} / requested={len(picked)}")
