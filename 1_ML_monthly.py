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

plt.rcParams["font.family"] = "Arial"

# (선택) Windows 경고 완화
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "8")

# ----------------------------------------------------------
# 사용자 입력 (단일/다중 둘 다 예시 제공)
# ----------------------------------------------------------
single_input_path = r"G:\공유 드라이브\BSG_DFC_result\EV6\DFC_원본\dfc_features_summary.csv"
single_save_path  = r"G:\공유 드라이브\BSG_DFC_result\EV6\DFC_원본"

multi_inputs = [
    ("EV6",    r"G:\공유 드라이브\BSG_DFC_result\EV6\DFC_완충후이동주차\dfc_features_summary.csv"),
    ("Ioniq5", r"G:\공유 드라이브\BSG_DFC_result\Ioniq5\DFC_완충후이동주차\dfc_features_summary.csv"),
]
multi_save_path = r"G:\공유 드라이브\BSG_DFC_result\combined\DFC_완충후이동주차\monthly_cluster"

# ----------------------------------------------------------
# 등고선(iso N*AVG) 토글/설정  ✅
# ----------------------------------------------------------
SHOW_ISO_T100_CONTOUR = True
ISO_T100_LEVELS       = [35, 50, 100, 300, 1000, 3000, 10000]  # N*AVG (hour-event)
ISO_CONTOUR_LABELS    = True
ISO_CONTOUR_COLOR     = "k"
ISO_CONTOUR_LS        = "--"
ISO_CONTOUR_LW        = 0.8
ISO_CONTOUR_ALPHA     = 0.6
ISO_CONTOUR_FMT       = r"$N \times AVG=%.0f$"

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

def add_iso_product_contours_linear(ax, x_min, x_max, y_min, y_max,
                                    levels, show_labels=True, fmt=None,
                                    color="k", ls="--", lw=0.8, alpha=0.6):
    """
    linear 축에서 iso (x*y = const) 곡선을 그림: y = C/x.
    """
    if x_max <= 0:
        return

    # x는 양수만 의미가 있으므로 안전하게 시작점 보정
    x0 = max(1e-6, x_min)
    xs = np.linspace(x0, x_max, 600)

    for C in levels:
        ys = C / xs
        m = np.isfinite(ys) & (ys >= y_min) & (ys <= y_max)
        if not np.any(m):
            continue

        ax.plot(xs[m], ys[m], color=color, linestyle=ls, linewidth=lw, alpha=alpha)

        if show_labels:
            idx = np.flatnonzero(m)
            mid_idx = idx[len(idx) // 2]
            x_lab = xs[mid_idx]
            y_lab = ys[mid_idx]
            text = (fmt % C) if (fmt is not None) else f"{C:g}"
            ax.text(
                x_lab, y_lab, text,
                fontsize=8, color=color, alpha=min(1.0, alpha + 0.2),
                ha="left", va="bottom"
            )

# ----------------------------------------------------------
# 메인: clustering + plot + save
# ----------------------------------------------------------
def _fit_cluster_and_plot(df, out_dir: Path, plot_name: str = "dfc_clusters_boundary.png"):
    """
    단일 DataFrame을 받아:
      - N>0 & 유효 표본만으로 KMeans + Logistic 회귀 학습
      - 전체 행에는 cluster를 채우되(미사용/결측은 NaN), 사용된 표본만 라벨 부여
      - 산점도 플롯 저장 (+ 등고선 on/off)
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

    # 작업용 데이터 (모델 입력, 원 단위)
    work = df.loc[used_idx, [N_col, mean_col]].astype(float)

    # === mean만 윈저라이즈 ===
    CAP_PCT = 99.9  # 상위 0.1% 캡
    work_w, pM_hi, clipped_M_mask = winsorize_mean_only(work, mean_col, CAP_PCT)

    # 최소 표본 체크
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

    # --------------------------------------------------
    # N축 가중치 적용
    # --------------------------------------------------
    WEIGHT_N = 1.2   # N 축 가중치

    X_orig = work_w[[N_col, mean_col]].to_numpy().astype(float)   # 원 단위
    X_for_model = X_orig.copy()
    X_for_model[:, 0] *= WEIGHT_N   # N 축 확대

    # 진단 출력
    n_clipM = int(clipped_M_mask.sum())
    if n_clipM:
        print(f"[INFO] winsorize(mean only) @{CAP_PCT}pct → {mean_col} cap={pM_hi:.3f} (clip {n_clipM})")
    print(f"[INFO] N axis weight: WEIGHT_N={WEIGHT_N}")

    # ------------------------------- #
    # KMeans (k=3) + 크기 기준 재라벨 #
    # ------------------------------- #
    best_k = 3
    km = KMeans(n_clusters=best_k, n_init=10, random_state=42)
    best_model = km.fit(X_for_model)
    labels_raw = best_model.labels_

    counts = np.bincount(labels_raw, minlength=best_k)
    order = np.argsort(-counts)  # 내림차순
    relabel = {old: new for new, old in enumerate(order)}
    labels = np.array([relabel[l] for l in labels_raw], dtype=int)

    centers_weighted = best_model.cluster_centers_[order]
    centers_ordered_orig = centers_weighted.copy()
    centers_ordered_orig[:, 0] /= WEIGHT_N

    print(f"[INFO] relabel by size desc: old→new {relabel} (counts={counts.tolist()})")

    # ------------------------------- #
    # Logistic Regression (진단용)    #
    # ------------------------------- #
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_for_model)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, labels, test_size=0.25, random_state=42, stratify=None
    )
    clf = LogisticRegression(max_iter=1000, random_state=42)
    clf.fit(X_train, y_train)
    try:
        acc = accuracy_score(y_test, clf.predict(X_test))
    except Exception:
        acc = float("nan")

    try:
        if len(set(labels)) >= 2:
            best_s = silhouette_score(X_for_model, labels)
        else:
            best_s = float("nan")
    except Exception:
        best_s = float("nan")

    try:
        n_splits = int(min(5, np.bincount(labels).min()))
        if n_splits >= 2:
            cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
            cv_acc = cross_val_score(clf, X_scaled, labels, cv=cv).mean()
        else:
            cv_acc = np.nan
    except Exception:
        cv_acc = np.nan

    # ==================================================
    # ==== Plot : 산점도 + (옵션) iso N*AVG 등고선 ====
    # ==================================================
    fig, ax = plt.subplots(figsize=(7.5, 6))

    cluster_labels = {
        0: r"Minimal $R_{\mathrm{FC}}$",
        1: r"Frequent $R_{\mathrm{FC}}$",
        2: r"Long $R_{\mathrm{FC}}$",
    }
    markers = ["o", "s", "^"]
    palette = ["#cd534c", "#20854e", "#0073c2"]

    for cid in range(best_k):
        m = (labels == cid)
        if not np.any(m):
            continue
        ax.scatter(
            work_w.loc[m, N_col],
            work_w.loc[m, mean_col],
            s=30,
            marker=markers[cid % len(markers)],
            c=palette[cid % len(palette)],
            edgecolor="k",
            linewidth=0.5,
            alpha=0.9,
            label=cluster_labels.get(cid, f"Cluster {cid}"),
        )

    # 축 범위 약간 여유
    x_min, x_max = work_w[N_col].min(), work_w[N_col].max()
    y_min, y_max = work_w[mean_col].min(), work_w[mean_col].max()
    pad_x = (x_max - x_min) * 0.05 if x_max > x_min else 1.0
    pad_y = (y_max - y_min) * 0.05 if y_max > y_min else 1.0
    ax.set_xlim(x_min - pad_x, x_max + pad_x)
    ax.set_ylim(y_min - pad_y, y_max + pad_y)

    # ✅ 등고선 on/off
    if SHOW_ISO_T100_CONTOUR:
        xlo, xhi = ax.get_xlim()
        ylo, yhi = ax.get_ylim()
        add_iso_product_contours_linear(
            ax, xlo, xhi, ylo, yhi,
            levels=ISO_T100_LEVELS,
            show_labels=ISO_CONTOUR_LABELS,
            fmt=ISO_CONTOUR_FMT,
            color=ISO_CONTOUR_COLOR,
            ls=ISO_CONTOUR_LS,
            lw=ISO_CONTOUR_LW,
            alpha=ISO_CONTOUR_ALPHA,
        )

    ax.set_xlabel("N(DFC)", fontsize=8)
    ax.set_ylabel(r"AVG($\Delta t_{100\%}$)", fontsize=8)
    ax.tick_params(axis="both", labelsize=8, width=1.2, length=5)

    leg = ax.legend(fontsize=8, loc="upper right", frameon=True)
    frame = leg.get_frame()
    frame.set_edgecolor("Grey")
    frame.set_linewidth(0.8)

    fig.tight_layout()
    plot_path = out_dir / plot_name
    fig.savefig(plot_path, dpi=200)
    plt.close(fig)

    # ------------------------------- #
    # 결과 저장                       #
    # ------------------------------- #
    df_out = df.copy()
    df_out["cluster"] = np.nan
    df_out["N_used"] = np.nan
    df_out["mean_used"] = np.nan
    df_out.loc[used_idx, "cluster"] = labels
    df_out.loc[used_idx, "N_used"] = work[N_col].values
    df_out.loc[used_idx, "mean_used"] = work_w[mean_col].values
    df_out.loc[used_idx, f"{mean_col}_clipped"] = clipped_M_mask.values

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
    print(f"- iso contour (N*AVG)  : {'ON' if SHOW_ISO_T100_CONTOUR else 'OFF'}")
    print(f"- 산점도               : {plot_path}")
    print(f"- 라벨 CSV             : {out_csv}")

    print("\n[2] 클러스터 중심좌표 (윈저라이즈-mean-only, 원 단위, 재정렬 반영)")
    centers_df = pd.DataFrame(centers_ordered_orig, columns=[N_col, mean_col])
    print(centers_df.to_string(index=True, float_format=lambda x: f"{x:.3f}"))

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
    out_dir = Path(save_path)
    out_dir.mkdir(parents=True, exist_ok=True)

    dfs = []
    for label, path in inputs:
        df_i = pd.read_csv(path)
        df_i = df_i.copy()
        df_i["vehicle model"] = label
        dfs.append(df_i)
        print(f"[INFO] 로드: {label} ({path}) rows={len(df_i)})")

    df_all = pd.concat(dfs, axis=0, ignore_index=True, sort=False)

    # ① 전체 차량 합쳐서 클러스터링 + 메인 figure
    labeled_csv, plot_path = _fit_cluster_and_plot(
        df_all, out_dir, plot_name="dfc_clusters_boundary_combined.png"
    )

    # ② 차종별로 marker를 달리한 산점도 figure 추가
    try:
        N_col, mean_col = pick_feature_columns(df_all)

        df_labeled = pd.read_csv(labeled_csv)

        needed_cols = [N_col, mean_col, "cluster", "vehicle model"]
        if not all(c in df_labeled.columns for c in needed_cols):
            print("[WARN] 차종별 figure를 그리기에 필요한 컬럼이 부족합니다.")
            return labeled_csv, plot_path

        used = df_labeled.dropna(subset=needed_cols)
        if len(used) == 0:
            print("[WARN] 차종별 figure에서 사용할 유효 표본이 없습니다.")
            return labeled_csv, plot_path

        cluster_labels = {
            0: r"Minimal $R_{\mathrm{FC}}$",
            1: r"Frequent $R_{\mathrm{FC}}$",
            2: r"Long $R_{\mathrm{FC}}$",
        }
        palette = ["#cd534c", "#20854e", "#0073c2"]

        vm_unique = used["vehicle model"].unique().tolist()
        marker_list = ["o", "s", "^", "D", "P", "X"]
        marker_map = {vm: marker_list[i % len(marker_list)] for i, vm in enumerate(vm_unique)}

        fig2, ax2 = plt.subplots(figsize=(7.5, 6))

        best_k = 3
        for cid in range(best_k):
            for i, vm in enumerate(vm_unique):
                sub = used[(used["vehicle model"] == vm) & (used["cluster"] == cid)]
                if len(sub) == 0:
                    continue
                label = cluster_labels.get(cid, f"Cluster {cid}") if i == 0 else None
                ax2.scatter(
                    sub[N_col],
                    sub[mean_col],
                    s=30,
                    marker=marker_map[vm],
                    c=palette[cid % len(palette)],
                    edgecolor="k",
                    linewidth=0.5,
                    alpha=0.9,
                    label=label,
                )

        # 축 범위
        x_min, x_max = used[N_col].min(), used[N_col].max()
        y_min, y_max = used[mean_col].min(), used[mean_col].max()
        pad_x = (x_max - x_min) * 0.05 if x_max > x_min else 1.0
        pad_y = (y_max - y_min) * 0.05 if y_max > y_min else 1.0
        ax2.set_xlim(x_min - pad_x, x_max + pad_x)
        ax2.set_ylim(y_min - pad_y, y_max + pad_y)

        # ✅ 등고선 on/off
        if SHOW_ISO_T100_CONTOUR:
            xlo, xhi = ax2.get_xlim()
            ylo, yhi = ax2.get_ylim()
            add_iso_product_contours_linear(
                ax2, xlo, xhi, ylo, yhi,
                levels=ISO_T100_LEVELS,
                show_labels=ISO_CONTOUR_LABELS,
                fmt=ISO_CONTOUR_FMT,
                color=ISO_CONTOUR_COLOR,
                ls=ISO_CONTOUR_LS,
                lw=ISO_CONTOUR_LW,
                alpha=ISO_CONTOUR_ALPHA,
            )

        ax2.set_xlabel("N(DFC)", fontsize=8)
        ax2.set_ylabel(r"AVG($\Delta t_{100\%}$)", fontsize=8)
        ax2.tick_params(axis="both", labelsize=8, width=1.2, length=5)

        leg2 = ax2.legend(fontsize=8, loc="upper right", frameon=True)
        frame2 = leg2.get_frame()
        frame2.set_edgecolor("Grey")
        frame2.set_linewidth(0.6)

        fig2.tight_layout()
        scatter_by_vm_path = out_dir / "dfc_clusters_boundary_by_vehicle_model.png"
        fig2.savefig(scatter_by_vm_path, dpi=200)
        plt.close(fig2)

        print(f"[SAVE] 차종별 산점도 figure: {scatter_by_vm_path}")

    except Exception as e:
        print(f"[WARN] 차종별 figure 생성 중 문제 발생: {e}")

    return labeled_csv, plot_path

# ----------------------------------------------------------
# 실행
# ----------------------------------------------------------
if __name__ == "__main__":
    # run_pipeline(single_input_path, single_save_path)
    run_pipeline_multi(multi_inputs, multi_save_path)
