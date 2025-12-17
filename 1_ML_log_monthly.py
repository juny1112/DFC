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
single_input_path = r"G:\공유 드라이브\BSG_DFC_result\EV6\DFC_완충후이동주차\dfc_features_summary.csv"
single_save_path  = r"G:\공유 드라이브\BSG_DFC_result\EV6\DFC_완충후이동주차"

multi_inputs = [
    ("EV6",   r"G:\공유 드라이브\BSG_DFC_result\EV6\DFC_완충후이동주차\dfc_features_summary.csv"),
    ("Ioniq5",r"G:\공유 드라이브\BSG_DFC_result\Ioniq5\DFC_완충후이동주차\dfc_features_summary.csv"),
]
multi_save_path = r"G:\공유 드라이브\BSG_DFC_result\combined\DFC_완충후이동주차\monthly_cluster"

# ----------------------------------------------------------
# 토글/하이퍼파라미터
# ----------------------------------------------------------
USE_WINSORIZE_MEAN = False    # winsorize on/off (상단 캡; 필요 시 ON)
WINSOR_CAP_PCT     = 99.9     # 상위 퍼센타일 캡

WEIGHT_N = 1.0               # log-space에서 log10(N) 축 가중치
K_FIXED  = 3                 # k 고정

# ----------------------------------------------------------
# ✅ 해결책 1: log-space에서 AVG(=mean) 하단 clip (저AVG 과밀 영향 완화)
# ----------------------------------------------------------
CLIP_LOG_MEAN_LO = True
LOG_MEAN_LO_PCT  = 5.0   # 하단 2%를 바닥으로 올림 (1~5 추천)

# ----------------------------------------------------------
# 등고선(iso N*AVG) 토글/설정
# ----------------------------------------------------------
SHOW_ISO_T100_CONTOUR = True
ISO_T100_LEVELS       = [35, 50, 100, 300, 1000]
ISO_CONTOUR_LABELS    = True
ISO_CONTOUR_FMT       = r"$N \times \mathrm{AVG}=%.0f$"

# ----------------------------------------------------------
# 출력 파일 prefix (요청: 맨 앞에 "log" 붙이기)
# ----------------------------------------------------------
OUT_PREFIX = "log_"

# ----------------------------------------------------------
# 유틸 함수
# ----------------------------------------------------------
def pick_feature_columns(df):
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
    if len(work) == 0:
        return work.copy(), np.nan, pd.Series([], dtype=bool, index=work.index)

    cap_value = np.nanpercentile(work[mean_col], cap_pct)
    clipped_mask = work[mean_col] > cap_value
    work_w = work.copy()
    work_w[mean_col] = np.minimum(work_w[mean_col], cap_value)
    return work_w, cap_value, clipped_mask

def relabel_by_centers(centers_orig: np.ndarray):
    """
    centers_orig: shape (3,2) in original units [N, mean]
    Rule:
      - max(mean) => Long (new_id=2)
      - among remaining, max(N) => Frequent (new_id=1)
      - remaining => Minimal (new_id=0)
    """
    if centers_orig.shape[0] != 3:
        raise ValueError("This relabel rule is defined for k=3 only.")

    Ns = centers_orig[:, 0]
    Ms = centers_orig[:, 1]

    long_old = int(np.nanargmax(Ms))
    remaining = [i for i in range(3) if i != long_old]
    frequent_old = int(remaining[int(np.nanargmax(Ns[remaining]))])
    minimal_old = int([i for i in range(3) if i not in (long_old, frequent_old)][0])

    map_old_to_new = {minimal_old: 0, frequent_old: 1, long_old: 2}
    name_by_new = {
        0: r"Minimal $R_{\mathrm{FC}}$",
        1: r"Frequent $R_{\mathrm{FC}}$",
        2: r"Long $R_{\mathrm{FC}}$",
    }
    return map_old_to_new, name_by_new

def add_iso_product_contours(ax, x_min, x_max, y_min, y_max, levels, show_labels=True, fmt=None):
    """
    log-log axes에서 iso (x*y = const) 등고선 추가.
    """
    if x_min <= 0 or y_min <= 0:
        return

    xs = np.logspace(np.log10(x_min), np.log10(x_max), 300)
    ys = np.logspace(np.log10(y_min), np.log10(y_max), 300)
    X, Y = np.meshgrid(xs, ys)
    Z = X * Y  # N * AVG

    cs = ax.contour(
        X, Y, Z,
        levels=levels,
        colors="k",
        linewidths=0.6,
        linestyles="dashed",
        alpha=0.6
    )

    if show_labels:
        ax.clabel(
            cs,
            inline=True,
            fontsize=7,
            fmt=fmt if fmt is not None else "%.0f"
        )

# ----------------------------------------------------------
# 핵심: clustering + plot + save
# ----------------------------------------------------------
def _fit_cluster_and_plot(df, out_dir: Path, plot_name: str = "dfc_clusters_boundary.png"):
    """
    - (log space) N>0 & mean>0 & 유효 표본으로 KMeans 학습
    - 산점도는 x,y log scale
    - iso (N*AVG=const) 등고선 on/off
    - cluster 라벨은 '센터 위치 기준'으로 재정의 (0=Minimal,1=Frequent,2=Long)
    - 출력물 파일명은 OUT_PREFIX("log_") prefix
    - ✅ 해결책1: log10(mean) 하단을 퍼센타일 기준으로 clip
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    df = df.copy()

    # 파일명 prefix 강제
    plot_name = OUT_PREFIX + Path(plot_name).name

    N_col, mean_col = pick_feature_columns(df)

    # === 마스크 (log 위해 양수 필요) ===
    N_num = pd.to_numeric(df[N_col], errors="coerce")
    mean_num = pd.to_numeric(df[mean_col], errors="coerce")

    mask_npos   = N_num > 0
    mask_mpos   = mean_num > 0
    mask_finite = np.isfinite(N_num) & np.isfinite(mean_num)

    mask_used = mask_npos & mask_mpos & mask_finite
    used_idx = df.index[mask_used]

    total_rows = len(df)
    n_removed_N0        = int((~mask_npos).sum())
    n_removed_mean0     = int((mask_npos & ~mask_mpos).sum())
    n_removed_nonfinite = int((mask_npos & mask_mpos & ~mask_finite).sum())

    work = df.loc[used_idx, [N_col, mean_col]].astype(float)

    # === winsorize 토글 (상단 캡) ===
    if USE_WINSORIZE_MEAN:
        work_w, pM_hi, clipped_M_mask = winsorize_mean_only(work, mean_col, WINSOR_CAP_PCT)
    else:
        work_w = work.copy()
        pM_hi = np.nan
        clipped_M_mask = pd.Series(False, index=work.index)

    if len(work_w) < K_FIXED:
        df_out = df.copy()
        df_out["cluster"] = np.nan
        df_out["N_used"] = np.nan
        df_out["mean_used"] = np.nan
        df_out[f"{mean_col}_clipped"] = np.nan

        out_csv = out_dir / f"{OUT_PREFIX}dfc_features_with_clusters.csv"
        df_out.to_csv(out_csv, index=False, encoding="utf-8-sig")

        print("\n[WARN] 유효 표본이 충분하지 않습니다 → 라벨은 NaN으로 저장합니다.")
        print(f"- 전체 행 수              : {total_rows}")
        print(f"- N==0 (출력에는 포함)     : {n_removed_N0}개")
        print(f"- mean<=0 (출력에는 포함)  : {n_removed_mean0}개")
        print(f"- 유효성 미충족(N>0,mean>0): {n_removed_nonfinite}개 (출력에는 포함, cluster=NaN)")
        print(f"- 학습/클러스터링 사용     : {len(work_w)}개")
        print(f"- 라벨 CSV                 : {out_csv}")
        return str(out_csv), None

    # --- log-space clustering ---
    X_orig = work_w[[N_col, mean_col]].to_numpy().astype(float)
    X_log = np.column_stack([np.log10(X_orig[:, 0]), np.log10(X_orig[:, 1])])

    # ✅ 해결책 1: log10(mean) 하단 clip
    if CLIP_LOG_MEAN_LO:
        lo = float(np.nanpercentile(X_log[:, 1], LOG_MEAN_LO_PCT))
        X_log[:, 1] = np.maximum(X_log[:, 1], lo)
        print(f"[INFO] clip log10(AVG) lower @{LOG_MEAN_LO_PCT}pct: lo={lo:.4f}")

    # N 축 가중치 (log-space)
    X_for_model = X_log.copy()
    X_for_model[:, 0] *= WEIGHT_N

    n_clipM = int(clipped_M_mask.sum())
    if USE_WINSORIZE_MEAN:
        print(f"[INFO] winsorize(mean only) ON @{WINSOR_CAP_PCT}pct → cap={pM_hi:.3f} (clip {n_clipM})")
    else:
        print("[INFO] winsorize(mean only) OFF")
    print("[INFO] clustering space: log10(N), log10(mean)")
    print(f"[INFO] N axis weight (in log space): WEIGHT_N={WEIGHT_N}")

    # --- KMeans ---
    best_k = K_FIXED
    km = KMeans(n_clusters=best_k, n_init=10, random_state=42)
    best_model = km.fit(X_for_model)
    labels_raw = best_model.labels_.astype(int)

    # centers: weighted log -> unweight -> orig units
    centers_log_weighted = best_model.cluster_centers_.copy()
    centers_log = centers_log_weighted.copy()
    centers_log[:, 0] /= WEIGHT_N
    centers_orig = np.column_stack([10 ** centers_log[:, 0], 10 ** centers_log[:, 1]])

    # center 위치 기준 재라벨 (0 Minimal, 1 Frequent, 2 Long)
    map_old_to_new, name_by_new = relabel_by_centers(centers_orig)
    labels = np.array([map_old_to_new[int(l)] for l in labels_raw], dtype=int)

    centers_reordered = np.zeros_like(centers_orig)
    for old_id, new_id in map_old_to_new.items():
        centers_reordered[new_id, :] = centers_orig[old_id, :]

    # diagnostics
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
        best_s = silhouette_score(X_for_model, labels) if len(set(labels)) >= 2 else float("nan")
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

    # --- plot ---
    fig, ax = plt.subplots(figsize=(7.5, 6))

    markers = ['o', 's', '^']  # 0 Minimal, 1 Frequent, 2 Long
    palette = ["#cd534c", "#20854e", "#0073c2"]

    for cid in [0, 1, 2]:
        m = (labels == cid)
        if not np.any(m):
            continue
        ax.scatter(
            work_w.loc[m, N_col],
            work_w.loc[m, mean_col],
            s=30,
            marker=markers[cid],
            c=palette[cid],
            edgecolor="k",
            linewidth=0.5,
            alpha=0.9,
            label=name_by_new.get(cid, f"Cluster {cid}")
        )

    ax.set_xscale("log")
    ax.set_yscale("log")

    x_min, x_max = work_w[N_col].min(), work_w[N_col].max()
    y_min, y_max = work_w[mean_col].min(), work_w[mean_col].max()
    pad = 0.08
    x_lo = x_min / (1 + pad)
    x_hi = x_max * (1 + pad)
    y_lo = y_min / (1 + pad)
    y_hi = y_max * (1 + pad)

    ax.set_xlim(x_lo, x_hi)
    ax.set_ylim(y_lo, y_hi)

    if SHOW_ISO_T100_CONTOUR:
        add_iso_product_contours(
            ax, x_lo, x_hi, y_lo, y_hi,
            levels=ISO_T100_LEVELS,
            show_labels=ISO_CONTOUR_LABELS,
            fmt=ISO_CONTOUR_FMT
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

    # --- save labeled csv ---
    df_out = df.copy()
    df_out["cluster"] = np.nan
    df_out["N_used"] = np.nan
    df_out["mean_used"] = np.nan
    df_out[f"{mean_col}_clipped"] = np.nan

    df_out.loc[used_idx, "cluster"] = labels
    df_out.loc[used_idx, "N_used"] = work[N_col].values
    df_out.loc[used_idx, "mean_used"] = work_w[mean_col].values
    df_out.loc[used_idx, f"{mean_col}_clipped"] = clipped_M_mask.values.astype(int)

    out_csv = out_dir / f"{OUT_PREFIX}dfc_features_with_clusters.csv"
    df_out.to_csv(out_csv, index=False, encoding="utf-8-sig")

    # summary
    print("\n=== 결과 요약 ===")
    if USE_WINSORIZE_MEAN:
        print(f"- winsor cap(mean only) @{WINSOR_CAP_PCT}pct: {mean_col}={pM_hi:.3f}")
        print(f"- clipped: {mean_col} {int(clipped_M_mask.sum())}개")
    else:
        print("- winsorize(mean only): OFF")
    print(f"- clip log10(AVG) low     : {'ON' if CLIP_LOG_MEAN_LO else 'OFF'} (pct={LOG_MEAN_LO_PCT})")
    print(f"- 출력 폴더              : {out_dir}")
    print(f"- 전체 행 수              : {total_rows}")
    print(f"- N==0 (출력에는 포함)     : {n_removed_N0}개")
    print(f"- mean<=0 (출력에는 포함)  : {n_removed_mean0}개")
    print(f"- 유효성 미충족(N>0,mean>0): {n_removed_nonfinite}개 (출력에는 포함, cluster=NaN)")
    print(f"- 학습/클러스터링 사용     : {len(work_w)}개 (N>0 & mean>0 & 유효)")
    print(f"- k(고정)                  : {best_k}")
    print(f"- silhouette (log space)   : {None if np.isnan(best_s) else round(best_s, 3)}")
    print(f"- test acc                 : {None if np.isnan(acc) else round(acc, 3)}")
    print(f"- cv acc                   : {None if np.isnan(cv_acc) else round(cv_acc, 3)}")
    print(f"- iso contour (N*AVG)       : {'ON' if SHOW_ISO_T100_CONTOUR else 'OFF'}")
    print(f"- 산점도                   : {plot_path}")
    print(f"- 라벨 CSV                 : {out_csv}")

    print("\n[2] 클러스터 중심좌표 (센터 위치 기반 재라벨 후: 0 Minimal, 1 Frequent, 2 Long)")
    centers_df = pd.DataFrame(centers_reordered, columns=[N_col, mean_col], index=[0, 1, 2])
    print(centers_df.to_string(index=True, float_format=lambda x: f"{x:.3f}"))

    print("\n[3] 클러스터별 데이터 개수 (사용된 표본만, 재라벨링 후)")
    cluster_counts = pd.Series(labels).value_counts().sort_index()
    for cid, cnt in cluster_counts.items():
        print(f" - Cluster {cid} ({name_by_new[cid]}): {cnt}개")

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

    # ② 차종별 산점도 figure 추가 (파일명 log_ prefix + iso contour 토글 반영)
    try:
        N_col, mean_col = pick_feature_columns(df_all)
        df_labeled = pd.read_csv(labeled_csv)

        needed_cols = [N_col, mean_col, "cluster", "vehicle model"]
        if not all(c in df_labeled.columns for c in needed_cols):
            print("[WARN] 차종별 figure를 그리기에 필요한 컬럼이 부족합니다.")
            return labeled_csv, plot_path

        used = df_labeled.dropna(subset=needed_cols).copy()
        if len(used) == 0:
            print("[WARN] 차종별 figure에서 사용할 유효 표본이 없습니다.")
            return labeled_csv, plot_path

        used[N_col] = pd.to_numeric(used[N_col], errors="coerce")
        used[mean_col] = pd.to_numeric(used[mean_col], errors="coerce")
        used = used[(used[N_col] > 0) & (used[mean_col] > 0)]
        if len(used) == 0:
            print("[WARN] 차종별 figure에서 (N>0 & mean>0) 유효 표본이 없습니다.")
            return labeled_csv, plot_path

        name_by_new = {
            0: r"Minimal $R_{\mathrm{FC}}$",
            1: r"Frequent $R_{\mathrm{FC}}$",
            2: r"Long $R_{\mathrm{FC}}$",
        }
        palette = ["#cd534c", "#20854e", "#0073c2"]

        vm_unique = used["vehicle model"].unique().tolist()
        marker_list = ["o", "s", "^", "D", "P", "X"]
        marker_map = {vm: marker_list[i % len(marker_list)] for i, vm in enumerate(vm_unique)}

        fig2, ax2 = plt.subplots(figsize=(7.5, 6))

        used["cluster"] = pd.to_numeric(used["cluster"], errors="coerce").astype(int)

        for cid in [0, 1, 2]:
            for i, vm in enumerate(vm_unique):
                sub = used[(used["vehicle model"] == vm) & (used["cluster"] == cid)]
                if len(sub) == 0:
                    continue
                label = name_by_new.get(cid, f"Cluster {cid}") if i == 0 else None
                ax2.scatter(
                    sub[N_col],
                    sub[mean_col],
                    s=30,
                    marker=marker_map[vm],
                    c=palette[cid],
                    edgecolor="k",
                    linewidth=0.5,
                    alpha=0.9,
                    label=label,
                )

        ax2.set_xscale("log")
        ax2.set_yscale("log")

        x_min, x_max = used[N_col].min(), used[N_col].max()
        y_min, y_max = used[mean_col].min(), used[mean_col].max()
        pad = 0.08
        x_lo = x_min / (1 + pad)
        x_hi = x_max * (1 + pad)
        y_lo = y_min / (1 + pad)
        y_hi = y_max * (1 + pad)

        ax2.set_xlim(x_lo, x_hi)
        ax2.set_ylim(y_lo, y_hi)

        if SHOW_ISO_T100_CONTOUR:
            add_iso_product_contours(
                ax2, x_lo, x_hi, y_lo, y_hi,
                levels=ISO_T100_LEVELS,
                show_labels=ISO_CONTOUR_LABELS,
                fmt=ISO_CONTOUR_FMT
            )

        ax2.set_xlabel("N(DFC)", fontsize=8)
        ax2.set_ylabel(r"AVG($\Delta t_{100\%}$)", fontsize=8)
        ax2.tick_params(axis="both", labelsize=8, width=1.2, length=5)

        leg2 = ax2.legend(fontsize=8, loc="upper right", frameon=True)
        frame2 = leg2.get_frame()
        frame2.set_edgecolor("Grey")
        frame2.set_linewidth(0.6)

        fig2.tight_layout()
        scatter_by_vm_path = out_dir / f"{OUT_PREFIX}dfc_clusters_boundary_by_vehicle_model.png"
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
