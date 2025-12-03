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
NBINS_RATE = 20      # 0~MAX_RATE 구간을 NBINS_RATE개로 등분

# ========= 컬러 팔레트 =========
COLOR_SOC10_COUNT = "#A1D99B"  # 연한 그린 (10% counts)
COLOR_SOC20_COUNT = "#FC9272"  # 연한 레드/코랄 (20% counts)
COLOR_SOC10_RATE  = "#6BAED6"  # 블루 톤 (10% rate)
COLOR_SOC20_RATE  = "#FDBB84"  # 오렌지 톤 (20% rate)

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
