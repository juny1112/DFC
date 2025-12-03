#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import pandas as pd
from pathlib import Path

# ① 각 차종별 입력 CSV (이미 너의 계산 코드로 생성된 결과)
CSV_EV6    = r"G:\공유 드라이브\BSG_DFC_result\EV6\DFC_완충후이동주차\t95_before_after_delta.csv"
CSV_IONIQ5 = r"G:\공유 드라이브\BSG_DFC_result\Ioniq5\DFC_완충후이동주차\t95_before_after_delta.csv"

# ② 합친 파일 저장 위치
OUT_DIR = r"G:\공유 드라이브\BSG_DFC_result\combined\DFC_완충후이동주차"
OUT_PATH = Path(OUT_DIR) / "t95_before_after_delta_combined.csv"

# ③ 읽을 컬럼(히스토그램 코드가 사용하는 최소 컬럼)
USECOLS = ["file", "t95_before_h", "t95_after_h", "delta_t_h"]

def load_with_model(path: str, model_name: str) -> pd.DataFrame:
    df = pd.read_csv(path, usecols=[c for c in USECOLS if c in pd.read_csv(path, nrows=0).columns])
    # 없는 컬럼이 있으면 만들어두기(히스토그램은 t95_before/after/delta만 써도 됨)
    for c in USECOLS:
        if c not in df.columns:
            df[c] = pd.NA
    df["model"] = model_name
    return df[["model"] + USECOLS]

def main():
    Path(OUT_DIR).mkdir(parents=True, exist_ok=True)

    df_ev6    = load_with_model(CSV_EV6, "EV6")
    df_ioniq5 = load_with_model(CSV_IONIQ5, "Ioniq5")

    # 세로로 붙이기 (row-wise concat)
    df_all = pd.concat([df_ev6, df_ioniq5], axis=0, ignore_index=True)

    # before/after가 모두 NaN인 행은 버림(그래프에 영향 없음)
    mask_keep = df_all[["t95_before_h", "t95_after_h", "delta_t_h"]].notna().any(axis=1)
    df_all = df_all.loc[mask_keep].copy()

    # 저장
    df_all.to_csv(OUT_PATH, index=False, encoding="utf-8-sig")
    print(f"[SAVE] combined CSV -> {OUT_PATH}")
    print(df_all.head())

if __name__ == "__main__":
    main()
