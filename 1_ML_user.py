#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import re
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler

# ─────────────────────────────────────────────────────────
# (선택) Windows 스레드 설정
# ─────────────────────────────────────────────────────────
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "8")

# ==========================================================
# 사용자 입력
# ==========================================================
COMBINED_CSV = r"G:\공유 드라이브\BSG_DFC_result\combined\DFC_완충후이동주차\dfc_features_with_clusters.csv"
OUT_DIR      = r"G:\공유 드라이브\BSG_DFC_result\combined\DFC_완충후이동주차"

# ==========================================================
# 저장 유틸 (권한 문제 대비)
# ==========================================================
def _ts(): return datetime.now().strftime("%Y%m%d_%H%M%S")

def safe_write_csv(df: pd.DataFrame, path: Path, **kwargs):
    path = Path(path); kwargs.setdefault("index", False); kwargs.setdefault("encoding", "utf-8-sig")
    try:
        df.to_csv(path, **kwargs); print(f"[SAVE] {path}"); return str(path)
    except PermissionError as e:
        alt = path.with_name(f"{path.stem}_{_ts()}_{os.getpid()}{path.suffix}")
        df.to_csv(alt, **kwargs)
        print(f"[WARN] PermissionError → 이름 변경 저장: {alt}\n       원인: {e}")
        return str(alt)

def safe_savefig(fig, path: Path, **kwargs):
    path = Path(path)
    try:
        fig.savefig(path, **kwargs); print(f"[SAVE] {path}"); return str(path)
    except PermissionError as e:
        alt = path.with_name(f"{path.stem}_{_ts()}_{os.getpid()}{path.suffix}")
        fig.savefig(alt, **kwargs)
        print(f"[WARN] PermissionError → 이름 변경 저장: {alt}\n       원인: {e}")
        return str(alt)

# ==========================================================
# file_stem → (user_id, ym) 파싱
#  규칙: "사용자_YYYY-MM_..." → 사용자 = YYYY-MM 앞까지 전부
# ==========================================================
RE_USER_YM = re.compile(r"^(?P<user>.+?)_(?P<ym>\d{4}-\d{2})(?:[_\.].*)?$", re.IGNORECASE)

def parse_user_ym_from_stem(stem: str):
    m = RE_USER_YM.match(str(stem))
    if m:
        return m.group("user"), m.group("ym")
    return None, None

# ==========================================================
# 클러스터 학습 + 경계 시각화 + 요약출력
#   입력 df_for_fit: [user_id, delta_t95_event_N, delta_t95_event_mean_h]
#   - 여기서 delta_t95_event_N 은 "N_per_month" (월 평균 건수)
#   - delta_t95_event_mean_h 은 "delta_t95_event_mean_month_h" (산술평균)
# ==========================================================
def fit_and_plot_userlevel(df_for_fit: pd.DataFrame, out_dir: Path,
                           plot_name="user_dfc_clusters_boundary_users_FROM_COMBINED.png",
                           winsorize_mean=True, cap_pct=99.9,
                           user_feat_full: pd.DataFrame | None = None):
    out_dir.mkdir(parents=True, exist_ok=True)

    # 내부 연산 컬럼(매핑된 이름)
    N_col = "delta_t95_event_N"          # 실제로는 N_per_month
    M_col = "delta_t95_event_mean_h"     # 실제로는 delta_t95_event_mean_month_h

    # 플롯 축 라벨(요청명)
    DISP_X = "N_per_month"
    DISP_Y = "delta_t95_event_mean_month_h"

    df = df_for_fit.copy()

    # 전체 사용자 수(입력 기준)
    total_users_input = int(df["user_id"].nunique()) if "user_id" in df.columns else len(df)

    # 사용 마스크
    N_num = pd.to_numeric(df[N_col], errors="coerce")
    M_num = pd.to_numeric(df[M_col], errors="coerce")
    mask_used = (N_num > 0) & np.isfinite(N_num) & np.isfinite(M_num)
    used = df.loc[mask_used, [N_col, M_col]].astype(float)
    used_user_ids = df.loc[mask_used, "user_id"] if "user_id" in df.columns else None
    total_users_used = int(used_user_ids.nunique()) if used_user_ids is not None else int(mask_used.sum())
    total_users_skipped = total_users_input - total_users_used  # 요약용

    # 표본 부족 처리
    if len(used) < 3:
        print("[WARN] 유효 유저 표본 < 3 → 클러스터링 생략")
        out_csv = out_dir / "user_user_level_fit_input.csv"
        safe_write_csv(df, out_csv)
        return str(out_csv), None

    # 윈저라이즈(평균만)
    cap_val = np.nan
    n_clipped = 0
    used_w = used.copy()
    if winsorize_mean:
        cap_val = np.nanpercentile(used_w[M_col], cap_pct)
        clipped_mask = used_w[M_col] > cap_val
        n_clipped = int(clipped_mask.sum())
        used_w[M_col] = np.minimum(used_w[M_col], cap_val)
        print(f"[INFO] winsorize(mean only) @{cap_pct}pct → {M_col} cap={cap_val:.3f} (clip {n_clipped})")

    # 표준화 → KMeans & Logistic 동일 스케일
    scaler = StandardScaler()
    Xs = scaler.fit_transform(used_w.values)

    # KMeans (표준화 공간)
    km = KMeans(n_clusters=3, n_init=10, random_state=42).fit(Xs)
    labels_raw = km.labels_

    # === 크기 기준 재라벨링: 가장 큰 군집→0, 그다음→1, 마지막→2 ===
    counts = np.bincount(labels_raw, minlength=km.n_clusters)
    order = np.argsort(-counts)
    relabel = {old: new for new, old in enumerate(order)}
    labels = np.array([relabel[l] for l in labels_raw])

    # 센터도 동일한 순서로 재정렬 (표준화 공간 기준)
    centers_std = km.cluster_centers_[order]
    print(f"[INFO] relabel by size desc: old→new {relabel} (counts={counts.tolist()})")

    # silhouette / 분류 성능
    try:
        sil = silhouette_score(Xs, labels) if len(set(labels)) >= 2 else float("nan")
    except Exception:
        sil = float("nan")

    Xtr, Xte, ytr, yte = train_test_split(Xs, labels, test_size=0.25, random_state=42, stratify=labels)
    clf = LogisticRegression(max_iter=1000, random_state=42).fit(Xtr, ytr)
    try:
        acc = accuracy_score(yte, clf.predict(Xte))
    except Exception:
        acc = float("nan")
    try:
        n_splits = int(min(5, np.bincount(labels).min()))
        cv_acc = cross_val_score(
            clf, Xs, labels,
            cv=StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        ).mean() if n_splits >= 2 else np.nan
    except Exception:
        cv_acc = np.nan

    # 결정경계: 원 단위 그리드 생성 → 표준화 후 예측
    x_min, x_max = used_w[N_col].min(), used_w[N_col].max()
    y_min, y_max = used_w[M_col].min(), used_w[M_col].max()
    pad_x = (x_max - x_min) * 0.05 if x_max > x_min else 1.0
    pad_y = (y_max - y_min) * 0.05 if y_max > y_min else 1.0
    gx = np.linspace(x_min - pad_x, x_max + pad_x, 300)
    gy = np.linspace(y_min - pad_y, y_max + pad_y, 300)
    Xo, Yo = np.meshgrid(gx, gy)
    Z = clf.predict(scaler.transform(np.c_[Xo.ravel(), Yo.ravel()])).reshape(Xo.shape)

    # 클러스터 센터: 원 단위로 역변환 (재정렬된 순서 기반)
    centers_orig = scaler.inverse_transform(centers_std)
    centers_df = pd.DataFrame(centers_orig, columns=[DISP_X, DISP_Y]).round(3)

    # 플롯
    fig = plt.figure(figsize=(8, 7))
    plt.contourf(Xo, Yo, Z, levels=np.arange(Z.max()+2)-0.5, alpha=0.25, cmap="coolwarm")
    plt.contour(Xo, Yo, Z, colors="k", linewidths=1)
    cs = plt.contour(Xo, Yo, Xo*Yo, levels=[50,100,150,300,450,600,750,900], alpha=0.65, cmap="viridis")
    plt.clabel(cs, inline=True, fontsize=8, fmt='%d')

    plt.scatter(used_w[N_col], used_w[M_col], s=18, c=labels, cmap="tab10", edgecolor="k")
    #plt.scatter(centers_orig[:,0], centers_orig[:,1], marker="X", s=150, color="red", label="Cluster centers")

    plt.xlabel(DISP_X); plt.ylabel(DISP_Y)
    plt.title(
        f"User-level Clustering (k=3, winz={'on' if winsorize_mean else 'off'}, "
        f"sil={None if np.isnan(sil) else round(sil,3)}, acc={None if np.isnan(acc) else round(acc,3)})"
    )
    plt.legend(); plt.tight_layout()
    plot_path = out_dir / plot_name
    safe_savefig(fig, plot_path, dpi=200)
    plt.close(fig)

    # 라벨 저장 (user_id 유지)
    df_out = df.copy()
    df_out["cluster"] = np.nan
    df_out.loc[mask_used, "cluster"] = labels
    out_csv = out_dir / "user_user_level_fit_with_labels.csv"
    safe_write_csv(df_out, out_csv)

    # ===== 콘솔 요약 =====
    num_users_12mo_total = None
    num_users_12mo_used  = None
    cluster_file_counts  = None
    if user_feat_full is not None:
        num_users_12mo_total = int((user_feat_full["months_covered"] >= 12).sum())

        if "user_id" in user_feat_full.columns and "user_id" in df_out.columns:
            merged_tmp = user_feat_full.merge(df_out[["user_id","cluster"]], on="user_id", how="left")
            users_per_cluster = merged_tmp.groupby("cluster", dropna=False)["user_id"].nunique()
            files_per_cluster = merged_tmp.groupby("cluster", dropna=False)["months_covered"].sum()
            cluster_file_counts = (users_per_cluster, files_per_cluster)

            if used_user_ids is not None:
                used_12 = merged_tmp.loc[merged_tmp["user_id"].isin(used_user_ids)]
                num_users_12mo_used = int((used_12["months_covered"] >= 12).sum())

    print("\n=== 결과 요약 ===")
    if winsorize_mean:
        print(f"- winsor cap(mean only) @{cap_pct}pct: {M_col}={None if np.isnan(cap_val) else round(cap_val,3)}")
        print(f"- clipped: {M_col} {n_clipped}개")
    print(f"- 출력 폴더                         : {out_dir}")
    print(f"- 전체 사용자 수(입력)              : {total_users_input}명")
    print(f"- 클러스터 적용 사용자 수(라벨 대상): {total_users_used}명")
    print(f"- 제외된 사용자 수(스케일/유효성)   : {total_users_skipped}명")
    if num_users_12mo_total is not None:
        print(f"- 12개월(파일 12개 이상) 사용자(전체) : {num_users_12mo_total}명")
    if num_users_12mo_used is not None:
        print(f"- 12개월(파일 12개 이상) 사용자(클러스터): {num_users_12mo_used}명")
    print(f"- silhouette                       : {None if np.isnan(sil) else round(sil,3)}")
    print(f"- test acc                         : {None if np.isnan(acc) else round(acc,3)}")
    print(f"- cv acc                           : {None if np.isnan(cv_acc) else round(cv_acc,3)}")
    print(f"- 산점도                           : {plot_path}")
    print(f"- 라벨 CSV                         : {out_csv}")

    try:
        w = clf.coef_[0]; b = clf.intercept_[0]
        slope = -w[0]/w[1]; intercept = -b/w[1]
        print("\n[1] 로지스틱 회귀 경계 기준(표준화 공간)")
        print(f" - 결정 경계식: {w[0]:.3f}*x + {w[1]:.3f}*y + {b:.3f} = 0")
        print(f" - slope={slope:.3f}, intercept={intercept:.3f}")
    except Exception:
        pass

    print("\n[2] 클러스터 중심좌표 (원 단위)")
    print(pd.DataFrame(centers_orig, columns=[DISP_X, DISP_Y]).round(3).to_string(index=True))

    print("\n[3] 클러스터별 크기")
    vc_users = pd.Series(labels).value_counts().sort_index()
    for cid, cnt in vc_users.items():
        extra = ""
        if cluster_file_counts is not None:
            users_per_cluster, files_per_cluster = cluster_file_counts
            try:
                files = int(files_per_cluster.loc[cid])
                extra = f", 파일 합계={files}개"
            except Exception:
                pass
        print(f" - Cluster {cid}: 사용자 {cnt}명{extra}")

    return str(out_csv), str(plot_path), pd.DataFrame(centers_orig, columns=[DISP_X, DISP_Y]).round(3)

# ==========================================================
# 파이프라인
# ==========================================================
def run_user_level_from_combined(combined_csv: str, out_dir: str,
                                 winsorize_mean=True, cap_pct=99.9):
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(combined_csv)
    print(f"[INFO] 로드: {combined_csv} (rows={len(df)})")

    # 3개 컬럼만 사용
    needed = ["file_stem", "delta_t95_event_N", "delta_t95_event_mean_h"]
    for c in needed:
        if c not in df.columns:
            raise ValueError(f"필요 컬럼이 없습니다: {c}")

    # file_stem → user_id, ym 파싱
    users, yms = [], []
    for stem in df["file_stem"].astype(str).fillna(""):
        u, ym = parse_user_ym_from_stem(stem)
        users.append(u); yms.append(ym)
    df["user_id"] = users
    df["ym"] = yms

    bad_u = int(df["user_id"].isna().sum()); bad_m = int(df["ym"].isna().sum())
    if bad_u or bad_m:
        print(f"[WARN] 파싱 실패: user_id {bad_u}행, ym {bad_m}행 → 제외")
    df = df.dropna(subset=["user_id","ym"]).copy()

    # 숫자화 (원본 df에 컬럼 추가하지 않음)
    N_series = pd.to_numeric(df["delta_t95_event_N"], errors="coerce")
    M_series = pd.to_numeric(df["delta_t95_event_mean_h"], errors="coerce")

    # 사용자 집계용 임시 DF (원본 변형 없음)
    tmp = pd.DataFrame({
        "user_id": df["user_id"].values,
        "ym": df["ym"].values,
        "_N": np.clip(N_series.values, a_min=0, a_max=None),
        "_M": M_series.values
    })

    user_feat = (
        tmp.groupby("user_id", as_index=False)
           .agg(
               months_covered=("ym","nunique"),
               N_total=("_N", "sum"),
               delta_t95_event_mean_month_h=("_M", "mean")
           )
    )
    user_feat["N_per_month"] = user_feat["N_total"] / user_feat["months_covered"]
    user_feat.replace([np.inf, -np.inf], np.nan, inplace=True)

    # 학습 입력 (이름 매핑)
    df_for_fit = user_feat[["user_id","N_per_month","delta_t95_event_mean_month_h"]].rename(columns={
        "N_per_month": "delta_t95_event_N",
        "delta_t95_event_mean_month_h": "delta_t95_event_mean_h"
    })

    # 학습/플롯/요약
    labeled_csv, plot_path, centers_df = fit_and_plot_userlevel(
        df_for_fit, out,
        plot_name="user_dfc_clusters_boundary_users_FROM_COMBINED.png",
        winsorize_mean=winsorize_mean, cap_pct=cap_pct,
        user_feat_full=user_feat
    )

    # 라벨 병합 저장 (ID 기준)
    df_labels = pd.read_csv(labeled_csv)  # columns: user_id, delta_t95_event_N, delta_t95_event_mean_h, cluster

    merged = user_feat.merge(df_labels[["user_id","cluster"]], on="user_id", how="left")

    out_all = Path(out) / f"user_user_level_clusters_FROM_COMBINED_winz_{'on' if winsorize_mean else 'off'}.csv"
    safe_write_csv(
        merged[["user_id","months_covered","N_total","N_per_month","delta_t95_event_mean_month_h","cluster"]],
        out_all
    )

    # 12개월 사용자만
    only12 = merged[merged["months_covered"] >= 12].copy()
    out_12 = Path(out) / f"user_user_level_clusters_12MO_FROM_COMBINED_winz_{'on' if winsorize_mean else 'off'}.csv"
    safe_write_csv(
        only12[["user_id","months_covered","N_total","N_per_month","delta_t95_event_mean_month_h","cluster"]],
        out_12
    )

    # (추가) 원본 월별 파일에 user_cluster 컬럼 병합 후 저장
    df_with_user_cluster = df.merge(
        df_labels[["user_id","cluster"]].rename(columns={"cluster":"user_cluster"}),
        on="user_id", how="left"
    )
    out_monthly = Path(out) / "user_dfc_features_with_user_clusters.csv"
    safe_write_csv(df_with_user_cluster, out_monthly)

    print(f"\n[DONE] 전체 사용자 결과 CSV : {out_all}")
    print(f"[DONE] 12개월 사용자 결과 CSV: {out_12}")
    print(f"[DONE] 월별 원본 + user_cluster CSV: {out_monthly}")
    if plot_path:
        print(f"[PLOT] {plot_path}")
    return str(out_all), str(out_12), str(out_monthly), plot_path

# ==========================================================
# 실행부
# ==========================================================
if __name__ == "__main__":
    run_user_level_from_combined(
        COMBINED_CSV,
        OUT_DIR,
        winsorize_mean=False,   # 평균 피처만 윈저라이즈(토글)
        #cap_pct=99.9
    )
