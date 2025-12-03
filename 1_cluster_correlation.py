#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
히트맵 위주 리포트 (숫자 주석 포함)

입력 CSV(필수 컬럼): user_id, cluster(월 라벨), user_cluster(사용자 라벨)
출력:
  - alignment_crosstab_counts.csv
  - alignment_crosstab_rowratio.csv
  - fig_heatmap_rowratio_annot.png
  - fig_heatmap_counts_annot.png
  - alignment_overall_match.txt
"""

import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ===== 여기만 채워도 됨 =====
INPUT_CSV = r"G:\공유 드라이브\BSG_DFC_result\combined\DFC_완충후이동주차\user_dfc_features_with_user_clusters.csv"
OUT_DIR   = r"G:\공유 드라이브\BSG_DFC_result\combined\DFC_완충후이동주차"
# ==========================

def _ensure_paths():
    global INPUT_CSV, OUT_DIR
    if not INPUT_CSV:
        INPUT_CSV = input("입력 CSV 경로(user_dfc_features_with_user_clusters.csv): ").strip('"').strip()
    if not OUT_DIR:
        OUT_DIR = input("출력 폴더 경로: ").strip('"').strip()
    inp = Path(INPUT_CSV)
    out = Path(OUT_DIR); out.mkdir(parents=True, exist_ok=True)
    if not inp.exists():
        raise FileNotFoundError(f"입력 CSV가 없습니다: {inp}")
    return inp, out

def safe_write_csv(df: pd.DataFrame, path: Path, **kwargs):
    kwargs.setdefault("index", False)
    kwargs.setdefault("encoding", "utf-8-sig")
    df.to_csv(path, **kwargs)
    print(f"[SAVE] {path}")

def _annotated_heatmap(matrix: pd.DataFrame, title: str, out_path: Path,
                       fmt="percent", cmap="Blues"):
    """
    fmt: "percent" → 0~1 비율을 %로 표기
         "int"     → 정수 개수로 표기
    """
    # 축 눈금을 항상 "정수 문자열"로 만들기
    def _as_int_str(seq):
        out = []
        for v in seq:
            try:
                if pd.isna(v):
                    out.append("")
                else:
                    out.append(str(int(round(float(v)))))
            except Exception:
                out.append(str(v))
        return out

    plt.figure(figsize=(7.5, 5.5))
    im = plt.imshow(matrix.values, aspect="auto", interpolation="nearest", cmap=cmap)
    cb_label = "row-wise ratio" if fmt == "percent" else "count"
    plt.colorbar(im, fraction=0.046, pad=0.04, label=cb_label)

    # ▶ 정수 눈금 라벨 적용
    plt.xticks(range(matrix.shape[1]), _as_int_str(matrix.columns))
    plt.yticks(range(matrix.shape[0]), _as_int_str(matrix.index))

    plt.xlabel("monthly cluster")
    plt.ylabel("user cluster")  # ▶ 요청: user_cluster → user cluster
    plt.title(title)

    # 값 주석
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            val = matrix.values[i, j]
            if pd.isna(val):
                txt = ""
            else:
                txt = f"{val*100:.1f}%" if fmt == "percent" else f"{int(round(val))}"
            plt.text(j, i, txt, ha="center", va="center", fontsize=9)

    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"[SAVE] {out_path}")

def run(inp: Path, out: Path):
    df = pd.read_csv(inp)
    print(f"[INFO] 로드: {inp} (rows={len(df)})")

    req = ["user_id", "cluster", "user_cluster"]
    missing = [c for c in req if c not in df.columns]
    if missing:
        raise ValueError(f"필수 컬럼 누락: {missing}")

    # 숫자화(가능하면)
    for c in ["cluster", "user_cluster"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # 크로스탭 (행: user_cluster, 열: cluster)
    ct = pd.crosstab(df["user_cluster"], df["cluster"]).sort_index().sort_index(axis=1)
    ct_ratio = ct.div(ct.sum(axis=1).replace(0, np.nan), axis=0)

    safe_write_csv(ct, out / "alignment_crosstab_counts.csv")
    safe_write_csv(ct_ratio, out / "alignment_crosstab_rowratio.csv")

    # 히트맵 (숫자 주석 포함)
    _annotated_heatmap(ct_ratio.fillna(0.0),
                       "User vs. Monthly Cluster (row-wise ratio)",
                       out / "fig_heatmap_rowratio_annot.png",
                       fmt="percent", cmap="Blues")

    _annotated_heatmap(ct.astype(float),
                       "User vs. Monthly Cluster (counts)",
                       out / "fig_heatmap_counts_annot.png",
                       fmt="int", cmap="Greys")

    # 전체 일치율(참고)
    overall_match = float((df["user_cluster"] == df["cluster"]).mean())
    (out / "alignment_overall_match.txt").write_text(f"{overall_match:.6f}", encoding="utf-8")
    print(f"[SAVE] {out / 'alignment_overall_match.txt'} (overall_match={overall_match:.4f})")

    # 콘솔 해석 힌트
    print("\n[읽는 법]")
    print("- 각 행(user_cluster)은 100%가 되도록 정규화되어 있습니다.")
    print("- 한 행에서 가장 진한 칸이, 그 사용자군에서 가장 많이 나온 월 라벨입니다.")
    print("- 대각선이 진할수록 사용자 라벨과 월 라벨의 일치도가 높습니다.")
    print("- off-diagonal이 진하면 혼합/변동이 있다는 뜻입니다.")

if __name__ == "__main__":
    inp, out = _ensure_paths()
    run(inp, out)
