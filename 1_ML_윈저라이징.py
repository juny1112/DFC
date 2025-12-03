#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler

# (선택) Windows 경고 완화
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "8")

# ----------------------------------------------------------
# 사용자 입력 (단일/다중 둘 다 예시 제공)
# ----------------------------------------------------------
single_input_path = r"G:\공유 드라이브\BSG_DFC_result\EV6\DFC_원본\dfc_features_summary.csv"
single_save_path  = r"G:\공유 드라이브\BSG_DFC_result\EV6\DFC_원본"

multi_inputs = [
    ("EV6",   r"G:\공유 드라이브\BSG_DFC_result\EV6\DFC_완충후이동주차\dfc_features_summary.csv"),
    ("Ioniq5",r"G:\공유 드라이브\BSG_DFC_result\Ioniq5\DFC_완충후이동주차\dfc_features_summary.csv"),
]
multi_save_path = r"G:\공유 드라이브\BSG_DFC_result\combined\DFC_완충후이동주차"

# ----------------------------------------------------------
# 유틸 함수
# ----------------------------------------------------------
def pick_feature_columns(df):
    """
    x축: N, y축: mean 자동 선택.
    먼저 delta_t95_event_* (이번 파이프라인 표준)를 우선 사용.
    """
    N_candidates    = ["delta_t95_event_N", "N_events", "N_events_applied", "N_events_total"]
    mean_candidates = ["delta_t95_event_mean_h", "delta_t95_mean_h", "delayed_mean_h"]

    N_col = next((c for c in N_candidates if c in df.columns), None)
    mean_col = next((c for c in mean_candidates if c in df.columns), None)

    if N_col is None or mean_col is None:
        raise ValueError(
            "필요 컬럼이 없습니다. delta_t95_event_N, delta_t95_event_mean_h (또는 대체 후보) 컬럼을 포함해 주세요."
        )
    return N_col, mean_col

def winsorize_mean_only(work: pd.DataFrame, mean_col: str, cap_pct: float = 99.0):
    """
    mean_col만 상위 cap_pct 퍼센타일로 캡(윈저라이즈)한다.
    N_col은 손대지 않는다.
    반환: work_w(윈저 적용 데이터프레임), cap_value(상한), clipped_mask(캡 적용 여부 시리즈)
    """
    if len(work) == 0:
        return work.copy(), np.nan, pd.Series([], dtype=bool, index=work.index)

    cap_value = np.nanpercentile(work[mean_col], cap_pct)
    clipped_mask = work[mean_col] > cap_value
    work_w = work.copy()
    work_w[mean_col] = np.minimum(work_w[mean_col], cap_value)
    return work_w, cap_value, clipped_mask

def _fit_cluster_and_plot(df, out_dir: Path, plot_name: str = "dfc_clusters_boundary.png"):
    """
    단일 DataFrame을 받아:
      - N>0 & 유효 표본만으로 KMeans + Logistic 회귀 경계 학습
      - 전체 행에는 cluster를 채우되(미사용/결측은 NaN), 사용된 표본만 라벨 부여
      - 산점도/결정경계 플롯 저장
      - 라벨 CSV 저장
    반환: (라벨 CSV 경로, 플롯 경로)
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    df = df.copy()

    # 피처 선택
    N_col, mean_col = pick_feature_columns(df)

    # === 마스크 정의 ===
    N_num = pd.to_numeric(df[N_col], errors="coerce")
    mean_num = pd.to_numeric(df[mean_col], errors="coerce")

    mask_npos   = N_num > 0
    mask_finite = np.isfinite(N_num) & np.isfinite(mean_num)
    mask_used   = mask_npos & mask_finite

    total_rows = len(df)
    n_removed_N0 = int((~mask_npos).sum())
    n_removed_nonfinite = int((~mask_finite & mask_npos).sum())
    used_idx = df.index[mask_used]

    # 작업용 데이터 (모델 입력)
    work = df.loc[used_idx, [N_col, mean_col]].astype(float)

    # === mean만 윈저라이즈 ===
    CAP_PCT = 99.9  # 상위 (1-CAP_PCT)% 이상은 상한선으로 잘라냄 (상위0.1%). 상한선 = 데이터가 적어서 0.1%의 데이터가 정확히 없으므로 linear interpolation 한 값으로 됨 (예: 8.981 번째값 = 8번째 값 - 9번째 값을 선형보간해서 얻음)
    work_w, pM_hi, clipped_M_mask = winsorize_mean_only(work, mean_col, CAP_PCT)

    # 최소 표본 체크 (k>=3 위해 3개 이상 권장)
    if len(work_w) < 3:
        df_out = df.copy()
        df_out["cluster"] = np.nan
        df_out["N_used"] = np.nan
        df_out["mean_used"] = np.nan
        out_csv = out_dir / "dfc_features_with_clusters.csv"
        df_out.to_csv(out_csv, index=False, encoding="utf-8-sig")
        print("\n[WARN] 유효 표본 3개 미만 → k≥3 클러스터링 불가, 라벨은 NaN으로 저장합니다.")
        print(f"- 전체 행 수           : {total_rows}")
        print(f"- N==0 (출력에는 포함) : {n_removed_N0}개")
        print(f"- 유효성 미충족(N>0)   : {n_removed_nonfinite}개 (출력에는 포함, cluster=NaN)")
        print(f"- 라벨 CSV(전부 NaN)   : {out_csv}")
        return str(out_csv), None

    X = work_w.values

    # 진단 출력
    n_clipM = int(clipped_M_mask.sum())
    if n_clipM:
        print(f"[INFO] winsorize(mean only) @{CAP_PCT}pct → {mean_col} cap={pM_hi:.3f} (clip {n_clipM})")

    # -------------------------------
    # KMeans (k=3 고정)
    # -------------------------------
    best_k = 3
    km = KMeans(n_clusters=3, n_init=10, random_state=42)
    best_model = km.fit(X)
    labels_raw = best_model.labels_

    # === [중요] 클러스터 크기 내림차순 재라벨링: 가장 큰 군집→0, 그다음→1, 마지막→2 ===
    counts = np.bincount(labels_raw, minlength=best_k)
    order = np.argsort(-counts)  # 내림차순
    relabel = {old: new for new, old in enumerate(order)}
    labels = np.array([relabel[l] for l in labels_raw], dtype=int)
    centers_ordered = best_model.cluster_centers_[order]
    print(f"[INFO] relabel by size desc: old→new {relabel} (counts={counts.tolist()})")

    # -------------------------------
    # Logistic Regression 경계 학습
    # -------------------------------
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, labels, test_size=0.25, random_state=42, stratify=None
    )
    clf = LogisticRegression(max_iter=1000, random_state=42)
    clf.fit(X_train, y_train)
    try:
        acc = accuracy_score(y_test, clf.predict(X_test))
    except Exception:
        acc = float("nan")

    # silhouette (재라벨링된 labels 기준)
    try:
        if len(set(labels)) >= 2:
            best_s = silhouette_score(X, labels)
        else:
            best_s = float("nan")
    except Exception:
        best_s = float("nan")

    # 소표본 안전 처리한 CV
    try:
        n_splits = int(min(5, np.bincount(labels).min()))
        if n_splits >= 2:
            cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
            cv_acc = cross_val_score(clf, X_scaled, labels, cv=cv).mean()
        else:
            cv_acc = np.nan
    except Exception:
        cv_acc = np.nan

    # -------------------------------
    # 시각화 (결정 경계 + 등고선) — 원 단위 2D 격자에서 예측
    # -------------------------------
    x_min_o, x_max_o = work_w[N_col].min(), work_w[N_col].max()
    y_min_o, y_max_o = work_w[mean_col].min(), work_w[mean_col].max()
    pad_x = (x_max_o - x_min_o) * 0.05 if x_max_o > x_min_o else 1.0
    pad_y = (y_max_o - y_min_o) * 0.05 if y_max_o > y_min_o else 1.0
    x_lin = np.linspace(x_min_o - pad_x, x_max_o + pad_x, 300)
    y_lin = np.linspace(y_min_o - pad_y, y_max_o + pad_y, 300)
    Xo, Yo = np.meshgrid(x_lin, y_lin)

    # 격자 표준화 후 예측
    grid_orig = np.c_[Xo.ravel(), Yo.ravel()]
    grid_scaled = scaler.transform(grid_orig)
    Z = clf.predict(grid_scaled).reshape(Xo.shape)

    # 플롯
    plt.figure(figsize=(8, 7))
    # 결정 영역
    plt.contourf(Xo, Yo, Z, levels=np.arange(Z.max() + 2) - 0.5, alpha=0.25, cmap="coolwarm")
    plt.contour(Xo, Yo, Z, colors="k", linewidths=1)

    # N*mean 등고선 (원 단위)
    total_effect = Xo * Yo
    cs = plt.contour(Xo, Yo, total_effect,
                     levels=[50,100,150,300,450,600,750,900],
                     alpha=0.65, cmap="viridis")
    plt.clabel(cs, inline=True, fontsize=8, fmt='%d')

    # 산점도(윈저라이즈된 원 단위)
    plt.scatter(work_w[N_col], work_w[mean_col], s=18, c=labels, cmap="tab10", edgecolor="k")

    # 클러스터 센터(윈저라이즈된 원 단위, 재정렬 반영)
    # plt.scatter(centers_ordered[:, 0], centers_ordered[:, 1],marker="X", s=150, color="red", label="Cluster centers")

    plt.xlabel(N_col)
    plt.ylabel(mean_col)
    plt.title(f"Monthly data Clustering (k={best_k}, silhouette={None if np.isnan(best_s) else round(best_s,3)}, "
              f"acc={None if np.isnan(acc) else round(acc,3)})")
    plt.legend()
    plt.tight_layout()
    plot_path = out_dir / plot_name
    plt.savefig(plot_path, dpi=200)
    plt.close()

    # -------------------------------
    # 결과 저장
    # -------------------------------
    df_out = df.copy()
    df_out["cluster"] = np.nan
    df_out["N_used"] = np.nan
    df_out["mean_used"] = np.nan
    df_out.loc[used_idx, "cluster"] = labels
    df_out.loc[used_idx, "N_used"] = work[N_col].values          # N은 원래 값 저장
    df_out.loc[used_idx, "mean_used"] = work_w[mean_col].values  # mean만 캡 반영
    df_out.loc[used_idx, f"{mean_col}_clipped"] = clipped_M_mask.values  # 캡 여부

    out_csv = out_dir / "dfc_features_with_clusters.csv"
    df_out.to_csv(out_csv, index=False, encoding="utf-8-sig")

    # 콘솔 요약
    print("\n=== 결과 요약 ===")
    if len(work_w) > 0:
        print(f"- winsor cap(mean only) @{CAP_PCT}pct: {mean_col}={pM_hi:.3f}")
        print(f"- clipped: {mean_col} {n_clipM}개")
    print(f"- 출력 폴더            : {out_dir}")
    print(f"- 전체 행 수           : {total_rows}")
    print(f"- N==0 (출력에는 포함) : {n_removed_N0}개")
    print(f"- 유효성 미충족(N>0)   : {n_removed_nonfinite}개 (출력에는 포함, cluster=NaN)")
    print(f"- 학습/클러스터링 사용 : {len(work_w)}개 (N>0 & 유효, winsorized-mean-only)")
    print(f"- k(고정)              : {best_k}")
    print(f"- silhouette           : {None if np.isnan(best_s) else round(best_s, 3)}")
    print(f"- test acc             : {None if np.isnan(acc) else round(acc, 3)}")
    print(f"- cv acc               : {None if np.isnan(cv_acc) else round(cv_acc, 3)}")
    print(f"- 산점도               : {plot_path}")
    print(f"- 라벨 CSV             : {out_csv}")

    # 추가 정보(가능하면 출력)
    try:
        w = clf.coef_[0]; b = clf.intercept_[0]
        slope = -w[0] / w[1]; intercept = -b / w[1]
        print("\n[1] 로지스틱 회귀 경계 기준")
        print(f" - 결정 경계식 (표준화 공간): {w[0]:.3f}*x + {w[1]:.3f}*y + {b:.3f} = 0")
        print(f" - 경계선 기울기(slope): {slope:.3f}, 절편(intercept): {intercept:.3f}")
    except Exception:
        pass

    print("\n[2] 클러스터 중심좌표 (윈저라이즈-mean-only 적용 후, 원 단위, 재정렬 반영)")
    centers_df = pd.DataFrame(centers_ordered, columns=[N_col, mean_col])
    print(centers_df.to_string(index=True, float_format=lambda x: f'{x:.3f}'))

    print("\n[3] 클러스터별 데이터 개수 (사용된 표본만, 재라벨링 후)")
    cluster_counts = pd.Series(labels).value_counts().sort_index()
    for cid, cnt in cluster_counts.items():
        print(f" - Cluster {cid}: {cnt}개")

    return str(out_csv), str(plot_path)

# ----------------------------------------------------------
# 단일 CSV 파이프라인
# ----------------------------------------------------------
def run_pipeline(input_path, save_path):
    out_dir = Path(save_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(input_path)
    print(f"[INFO] 단일 CSV 로드: {input_path} (rows={len(df)})")
    _fit_cluster_and_plot(df, out_dir, plot_name="dfc_clusters_boundary.png")

# ----------------------------------------------------------
# 다중 CSV 파이프라인 (합쳐서 분석)
# ----------------------------------------------------------
def run_pipeline_multi(inputs, save_path):
    """
    inputs: [("EV6", path1), ("Ioniq5", path2), ...]
    - 각 CSV를 읽어 'vehicle model' 컬럼을 추가한 뒤 세로로 concat
    - 공통/비공통 컬럼은 outer join으로 합침(비공통은 NaN)
    - 합쳐진 df에 대해 단일 파이프라인과 동일 작업 수행
    - 산점도는 합쳐서 그리되, cluster 색상은 같고(학습 결과),
      산점도 모양(marker)을 vehicle model별로 달리 표시
    """
    out_dir = Path(save_path)
    out_dir.mkdir(parents=True, exist_ok=True)

    dfs = []
    for label, path in inputs:
        df_i = pd.read_csv(path)
        df_i = df_i.copy()
        df_i["vehicle model"] = label
        dfs.append(df_i)
        print(f"[INFO] 로드: {label} ({path}) rows={len(df_i)})")

    # 열 기준 통합(outer join)
    df_all = pd.concat(dfs, axis=0, ignore_index=True, sort=False)

    # ---- 학습/출력(합본) ----
    labeled_csv, plot_path = _fit_cluster_and_plot(df_all, out_dir, plot_name="dfc_clusters_boundary_combined.png")

    # ---- vehicle model별 모양으로 산점도 추가 ----
    N_col, mean_col = pick_feature_columns(df_all)
    N_num = pd.to_numeric(df_all[N_col], errors="coerce")
    mean_num = pd.to_numeric(df_all[mean_col], errors="coerce")
    mask_used = (N_num > 0) & np.isfinite(N_num) & np.isfinite(mean_num)

    # 라벨 결과 로드
    df_labeled = pd.read_csv(labeled_csv)

    cols_need = [N_col, mean_col, "cluster", "vehicle model"]
    used = df_labeled.loc[mask_used, cols_need].dropna(subset=[N_col, mean_col, "cluster"], how="any") \
                     if all(c in df_labeled.columns for c in cols_need) else pd.DataFrame(columns=cols_need)
    if len(used):
        markers = ['o', 's', '^', 'D', 'P', 'X', '*', 'v', '<', '>']
        vm_unique = used["vehicle model"].dropna().unique().tolist()
        marker_map = {vm: markers[i % len(markers)] for i, vm in enumerate(vm_unique)}

        plt.figure(figsize=(8, 7))
        for vm in vm_unique:
            sub = used[used["vehicle model"] == vm]
            plt.scatter(sub[N_col], sub[mean_col],
                        s=22, c=sub["cluster"], cmap="tab10",
                        edgecolor="k", marker=marker_map[vm], label=vm)
        plt.xlabel(N_col); plt.ylabel(mean_col)
        plt.title("Combined scatter by vehicle model (marker) & cluster (color)")
        plt.legend(title="vehicle model")
        plt.tight_layout()
        scatter_by_vm = out_dir / "dfc_scatter_combined_by_vehicle_model.png"
        plt.savefig(scatter_by_vm, dpi=200)
        plt.close()
        print(f"[SAVE] 합본 산점도(차종별 마커): {scatter_by_vm}")
    else:
        print("[WARN] 시각화할 유효 표본이 없어 합본 산점도(차종별) 생략")

    # ---- 차종별 라벨 CSV도 분리 저장 ----
    try:
        if "vehicle model" in df_labeled.columns:
            for vm in df_labeled["vehicle model"].dropna().unique():
                sub = df_labeled[df_labeled["vehicle model"] == vm].copy()
                out_csv_vm = out_dir / f"dfc_features_with_clusters_{vm}.csv"
                sub.to_csv(out_csv_vm, index=False, encoding="utf-8-sig")
                print(f"[SAVE] 차종별 라벨 CSV: {out_csv_vm}")
    except Exception as e:
        print(f"[WARN] 차종별 라벨 CSV 저장 중 문제: {e}")

    return labeled_csv, plot_path

# ----------------------------------------------------------
# 실행
# ----------------------------------------------------------
if __name__ == "__main__":
    # 1) 단일 CSV (예: EV6만) — 필요 시 주석 해제
    # run_pipeline(single_input_path, single_save_path)

    # 2) 다중 CSV (예: EV6 + Ioniq5) — 합쳐서 분석/시각화
    run_pipeline_multi(multi_inputs, multi_save_path)
