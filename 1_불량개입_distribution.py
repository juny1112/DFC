#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ========= 경로 =========
IN_CSV  = r"G:\공유 드라이브\BSG_DFC_result\combined\DFC_완충후이동주차\불량개입\min_soc_between_fullcharges_cases.csv"
OUT_DIR = r"G:\공유 드라이브\BSG_DFC_result\combined\DFC_완충후이동주차\불량개입"

os.makedirs(OUT_DIR, exist_ok=True)

# ========= 옵션 =========
MAX_RATE   = 1.0     # rate 히스토그램 x축 상한 (넘어가면 클리핑)
NBINS_RATE = 10      # 0~MAX_RATE 구간을 NBINS_RATE개로 등분

# SOC 최저점 분포용 옵션
SOC_BIN_WIDTH = 10  # SOC 히스토그램 bin 폭(%). 예: 0-5, 5-10, ... 95-100

# ========= 컬러 팔레트 =========
COLOR_SOC10_COUNT = "#A1D99B"  # 연한 그린 (10% counts)
COLOR_SOC20_COUNT = "#FC9272"  # 연한 레드/코랄 (20% counts)
COLOR_SOC10_RATE  = "#6BAED6"  # 블루 톤 (10% rate)
COLOR_SOC20_RATE  = "#FDBB84"  # 오렌지 톤 (20% rate)

# SOC 최저점 히스토그램 색 (원하는 걸로 골라 써도 됨)
COLOR_MINS_CASE23  = "#C7E9C0"  # 연한 그린
COLOR_MINS_CASE123 = "#C7E9C0"  # 연한 그린

ALPHA_MAIN = 0.9


# ========= 데이터 로드 & 파생 변수 계산 =========
def load_and_prepare(in_csv: str) -> pd.DataFrame:
    df = pd.read_csv(in_csv)

    req = [
        "car_type", "file_name", "n_fullcharge_Rcharg",
        "case1_total", "case1_10_count", "case1_20_count",
        "case2_total", "case2_10_count", "case2_20_count",
        "case3_total", "case3_10_count", "case3_20_count",
    ]
    for c in req:
        if c not in df.columns:
            raise ValueError(f"필수 컬럼 없음: {c}")

    # ----- per-file bad counts -----
    # Case 2+3만
    df["bad10_case23"] = df["case2_10_count"] + df["case3_10_count"]
    df["bad20_case23"] = df["case2_20_count"] + df["case3_20_count"]

    # Case 1+2+3 모두
    df["bad10_case123"] = df["case1_10_count"] + df["bad10_case23"]
    df["bad20_case123"] = df["case1_20_count"] + df["bad20_case23"]

    # ----- n_fullcharge_Rcharg == 0 인 행은 bad count도 NaN 처리 -----
    mask_full = df["n_fullcharge_Rcharg"] > 0
    for col in ["bad10_case23", "bad20_case23", "bad10_case123", "bad20_case123"]:
        df.loc[~mask_full, col] = np.nan

    # ----- per-file denominator (몇 번의 "기회"가 있었는지) -----
    denom23  = df["n_fullcharge_Rcharg"].astype(float)
    denom123 = (df["n_fullcharge_Rcharg"] + df["case1_total"]).astype(float)

    df["denom_case23"]  = denom23
    df["denom_case123"] = denom123

    # ----- per-file rate (probability처럼 해석할 값) -----
    # Case1 미포함: (case2+case3 bad counts) / fullcharge 횟수
    df["rate10_case23"] = np.where(
        denom23 > 0,
        df["bad10_case23"] / denom23,
        np.nan,
    )
    df["rate20_case23"] = np.where(
        denom23 > 0,
        df["bad20_case23"] / denom23,
        np.nan,
    )

    # Case1 포함: (case1+2+3 bad counts) / (fullcharge 횟수 + case1_total)
    df["rate10_case123"] = np.where(
        denom123 > 0,
        df["bad10_case123"] / denom123,
        np.nan,
    )
    df["rate20_case123"] = np.where(
        denom123 > 0,
        df["bad20_case123"] / denom123,
        np.nan,
    )

    return df


# ========= 유틸: SOC 최저점 문자열 파싱 =========
def parse_minima_column(series: pd.Series) -> np.ndarray:
    """
    '40,35,8,3' 이런 문자열이 들어있는 caseX_mins 컬럼에서
    모든 값을 flatten 해서 float array로 반환.
    """
    values = []
    for s in series.dropna():
        # 혹시 숫자가 이미 들어있어도 str(...)로 통일
        s = str(s).strip()
        if not s:
            continue
        parts = s.split(",")
        for p in parts:
            p = p.strip()
            if not p:
                continue
            try:
                values.append(float(p))
            except ValueError:
                # 이상한 값 있으면 무시
                continue
    if not values:
        return np.array([], dtype=float)
    return np.array(values, dtype=float)


# ========= 히스토그램 유틸 =========
def plot_hist_counts_int(series: pd.Series, out_png: str, title: str, color: str, xlabel: str):
    """
    정수 count에 대한 히스토그램
    - bin: [0,1), [1,2), [2,3), ...
    - x tick: 0,1,2,...
    """
    data = series.to_numpy()
    data = data[~np.isnan(data)]   # 혹시 모를 NaN 제거 (df_valid 쓰지만 안전빵)
    data = data.astype(int)
    data = data[data >= 0]
    if data.size == 0:
        print(f"[SKIP] no data for {title}")
        return

    max_count = int(data.max())
    xs = np.arange(0, max_count + 1)
    counts = np.bincount(data, minlength=max_count + 1)

    fig, ax = plt.subplots(figsize=(6, 4), dpi=150)
    ax.bar(
        xs,
        counts,
        width=1.0,
        align="edge",      # [k, k+1) 구간
        color=color,
        alpha=ALPHA_MAIN,
        edgecolor="black",
    )

    ax.set_xlim(0, max_count + 1)
    ax.set_xticks(xs)

    ax.set_xlabel(xlabel)
    ax.set_ylabel("Number of files")
    ax.set_title(title)
    ax.grid(False)

    plt.tight_layout()
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[SAVE] {out_png}")


def plot_hist_rate(series: pd.Series, out_png: str, title: str, color: str, xlabel: str):
    """
    rate(확률 비슷한 값) 히스토그램
    """
    data = series.to_numpy().astype(float)
    data = data[~np.isnan(data)]
    if data.size == 0:
        print(f"[SKIP] no data for {title}")
        return

    data_clip = np.clip(data, 0, MAX_RATE)
    bins = np.linspace(0, MAX_RATE, NBINS_RATE + 1)

    fig, ax = plt.subplots(figsize=(6, 4), dpi=150)
    ax.hist(
        data_clip,
        bins=bins,
        color=color,
        alpha=ALPHA_MAIN,
        edgecolor="black",
    )

    ax.set_xlabel(xlabel)
    ax.set_ylabel("Number of files")
    ax.set_title(title)
    ax.grid(False)

    plt.tight_layout()
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[SAVE] {out_png}")


# ========= SOC 최저점 "거꾸로 히스토그램" =========
def plot_minsoc_horizontal(minima: np.ndarray,
                           out_png: str,
                           title: str,
                           color: str):
    """
    minima: SOC 최저점 값들의 배열 (0~100%)

    bin 정의:
      - 첫 bin: [0, 5]
      - 이후 bin: (5,10], (10,15], (15,20], ... 이런 식으로
        → 즉, "15-20"은 15 초과 20 이하 (15,20]에 해당
    y축: SOC (%), x축: count  → 가로 막대(histogram 느낌)
    """
    data = np.asarray(minima, dtype=float)
    data = data[np.isfinite(data)]
    if data.size == 0:
        print(f"[SKIP] no minima data for {title}")
        return

    # SOC 범위 0~100%로 클리핑
    data = np.clip(data, 0.0, 100.0)

    # bin 경계: 0,5,10,...,100
    bins = np.arange(0.0, 100.0 + SOC_BIN_WIDTH, SOC_BIN_WIDTH)

    cats = pd.cut(
        data,
        bins=bins,
        right=True,         # (a, b]
        include_lowest=True # 첫 bin만 [0,5]
    )

    # 구간별 개수 (bin 순서 유지: 카테고리 인덱스 기준 정렬)
    counts = cats.value_counts().sort_index()

    # y 위치는 각 Interval의 중앙값
    y_centers = np.array([(iv.left + iv.right) / 2.0 for iv in counts.index])
    height = SOC_BIN_WIDTH  # 막대 높이(= bin 폭)

    fig, ax = plt.subplots(figsize=(6, 6), dpi=150)
    ax.barh(
        y_centers,
        counts.to_numpy(),
        height=height,
        color=color,
        alpha=ALPHA_MAIN,
        edgecolor="black",
    )

    ax.set_xlabel("Count")
    ax.set_ylabel("Minimum SOC bin")
    ax.set_title(title)

    ax.set_ylim(0, 100)
    ax.grid(False)

    plt.tight_layout()
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[SAVE] {out_png}")



# ========= 메인 =========
if __name__ == "__main__":
    df = load_and_prepare(IN_CSV)

    # per-file 파생값까지 포함한 CSV (전체 파일 기준) 저장
    out_with_rates = os.path.join(
        OUT_DIR, "min_soc_between_fullcharges_cases_with_rates.csv"
    )
    df.to_csv(out_with_rates, index=False, encoding="utf-8-sig")
    print(f"[SAVE] {out_with_rates}")

    # ===== 분석/플롯 대상: 완전충전 1회 이상 있는 파일만 =====
    df_valid = df[df["n_fullcharge_Rcharg"] > 0].copy()
    print(f"[INFO] n_fullcharge_Rcharg > 0 인 파일 수: {len(df_valid)}")

    # --------- 1) "횟수" 분포 (counts) ---------
    # Case1 미포함 (Case2+3)
    out_cnt_10_c23 = os.path.join(OUT_DIR, "hist_badcount_soc10_case23.png")
    out_cnt_20_c23 = os.path.join(OUT_DIR, "hist_badcount_soc20_case23.png")

    plot_hist_counts_int(
        df_valid["bad10_case23"],
        out_cnt_10_c23,
        title="Distribution of SOC < 10% bad counts (Case2+3 only)",
        color=COLOR_SOC10_COUNT,
        xlabel="Number of SOC < 10% bad events per file (Case2+3)",
    )

    plot_hist_counts_int(
        df_valid["bad20_case23"],
        out_cnt_20_c23,
        title="Distribution of SOC < 20% bad counts (Case2+3 only)",
        color=COLOR_SOC20_COUNT,
        xlabel="Number of SOC < 20% bad events per file (Case2+3)",
    )

    # Case1 포함 (Case1+2+3)
    out_cnt_10_c123 = os.path.join(OUT_DIR, "hist_badcount_soc10_case123.png")
    out_cnt_20_c123 = os.path.join(OUT_DIR, "hist_badcount_soc20_case123.png")

    plot_hist_counts_int(
        df_valid["bad10_case123"],
        out_cnt_10_c123,
        title="Distribution of SOC < 10% bad counts (Case1+2+3)",
        color=COLOR_SOC10_COUNT,
        xlabel="Number of SOC < 10% bad events per file (Case1+2+3)",
    )

    plot_hist_counts_int(
        df_valid["bad20_case123"],
        out_cnt_20_c123,
        title="Distribution of SOC < 20% bad counts (Case1+2+3)",
        color=COLOR_SOC20_COUNT,
        xlabel="Number of SOC < 20% bad events per file (Case1+2+3)",
    )

    # --------- 2) "확률 / rate" 분포 ---------
    # Case1 미포함 (Case2+3)
    out_rate_10_c23 = os.path.join(OUT_DIR, "hist_badrate_soc10_case23.png")
    out_rate_20_c23 = os.path.join(OUT_DIR, "hist_badrate_soc20_case23.png")

    plot_hist_rate(
        df_valid["rate10_case23"],
        out_rate_10_c23,
        title="Distribution of SOC < 10% bad rate per full charge (Case2+3 only)",
        color=COLOR_SOC10_RATE,
        xlabel="Bad rate of SOC < 10% per full charge (Case2+3)",
    )

    plot_hist_rate(
        df_valid["rate20_case23"],
        out_rate_20_c23,
        title="Distribution of SOC < 20% bad rate per full charge (Case2+3 only)",
        color=COLOR_SOC20_RATE,
        xlabel="Bad rate of SOC < 20% per full charge (Case2+3)",
    )

    # Case1 포함 (Case1+2+3)
    out_rate_10_c123 = os.path.join(OUT_DIR, "hist_badrate_soc10_case123.png")
    out_rate_20_c123 = os.path.join(OUT_DIR, "hist_badrate_soc20_case123.png")

    plot_hist_rate(
        df_valid["rate10_case123"],
        out_rate_10_c123,
        title="Distribution of SOC < 10% bad rate per opportunity (Case1+2+3)",
        color=COLOR_SOC10_RATE,
        xlabel="Bad rate of SOC < 10% per (full charge + Case1) (Case1+2+3)",
    )

    plot_hist_rate(
        df_valid["rate20_case123"],
        out_rate_20_c123,
        title="Distribution of SOC < 20% bad rate per opportunity (Case1+2+3)",
        color=COLOR_SOC20_RATE,
        xlabel="Bad rate of SOC < 20% per (full charge + Case1) (Case1+2+3)",
    )

    # --------- 3) SOC 최저점 분포 (거꾸로 히스토그램) ---------
    # caseX_mins 컬럼이 있는 경우에만 수행 (없으면 빈 array로 끝)
    mins1  = parse_minima_column(df_valid["case1_mins"]) if "case1_mins" in df_valid.columns else np.array([], dtype=float)
    mins2  = parse_minima_column(df_valid["case2_mins"]) if "case2_mins" in df_valid.columns else np.array([], dtype=float)
    mins3  = parse_minima_column(df_valid["case3_mins"]) if "case3_mins" in df_valid.columns else np.array([], dtype=float)

    # Case1 미포함 (Case2+3)
    if mins2.size + mins3.size > 0:
        mins_case23 = np.concatenate([mins2, mins3])
    else:
        mins_case23 = np.array([], dtype=float)

    # Case1 포함 (Case1+2+3)
    if mins1.size + mins2.size + mins3.size > 0:
        mins_case123 = np.concatenate([mins1, mins2, mins3])
    else:
        mins_case123 = np.array([], dtype=float)

    out_mins_c23 = os.path.join(OUT_DIR, "hist_minsoc_case23_horizontal.png")
    out_mins_c123 = os.path.join(OUT_DIR, "hist_minsoc_case123_horizontal.png")

    plot_minsoc_horizontal(
        mins_case23,
        out_mins_c23,
        title="Distribution of minimum SOC between full charges (Case2+3 only)",
        color=COLOR_MINS_CASE23,
    )

    plot_minsoc_horizontal(
        mins_case123,
        out_mins_c123,
        title="Distribution of minimum SOC between full charges (Case1+2+3)",
        color=COLOR_MINS_CASE123,
    )
