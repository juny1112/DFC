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

plt.rcParams["font.family"] = "Arial"

# ─────────────────────────────────────────────────────────
# (선택) Windows 스레드 설정
# ─────────────────────────────────────────────────────────
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "8")

# ==========================================================
# 사용자 입력
# ==========================================================
COMBINED_CSV = r"G:\공유 드라이브\BSG_DFC_result\combined\DFC_완충후이동주차\monthly_cluster\dfc_features_with_clusters.csv"
OUT_DIR      = r"G:\공유 드라이브\BSG_DFC_result\combined\DFC_완충후이동주차\user_cluster"

# ==========================================================
# ✅ LOG SCALE 옵션
# ==========================================================
USE_LOG_SPACE_FOR_CLUSTERING = True   # True: log10(N), log10(M)에서 클러스터링
PLOT_LOG_AXES                = True   # True: 산점도 축 log-log
WEIGHT_LOG_N                 = 1.0    # log-space에서 log10(N)축 가중치 (1이면 없음)
K_FIXED                      = 3

# (선택) log-space 하단 clip (과밀/저AVG 영향 완화)
CLIP_LOG_MEAN_LO = False
LOG_MEAN_LO_PCT  = 5.0

# (선택) iso-line (x*y=const) 토글 (N_per_month * mean_month)
SHOW_ISO_CONTOUR   = False
ISO_LEVELS         = [35, 50, 100, 300]
ISO_LABELS         = True
ISO_FMT            = r"$N \times \mathrm{AVG}=%.0f$"

# ==========================================================
# ✅ 클러스터 이름(라벨) 재정의 규칙: "센터 위치" 기준
#   - mean(AVG) 가장 큰 cluster => Long (cluster=2)
#   - 나머지 중 N 가장 큰 cluster => Frequent (cluster=1)
#   - 나머지 => Minimal (cluster=0)
# ==========================================================
USE_CENTER_BASED_NAMING = True

# ==========================================================
# 저장 유틸 (권한 문제 대비)
# ==========================================================
def _ts():
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def safe_write_csv(df: pd.DataFrame, path: Path, **kwargs):
    path = Path(path)
    kwargs.setdefault("index", False)
    kwargs.setdefault("encoding", "utf-8-sig")
    try:
        df.to_csv(path, **kwargs)
        print(f"[SAVE] {path}")
        return str(path)
    except PermissionError as e:
        alt = path.with_name(f"{path.stem}_{_ts()}_{os.getpid()}{path.suffix}")
        df.to_csv(alt, **kwargs)
        print(f"[WARN] PermissionError → 이름 변경 저장: {alt}\n       원인: {e}")
        return str(alt)

def safe_savefig(fig, path: Path, **kwargs):
    path = Path(path)
    try:
        fig.savefig(path, **kwargs)
        print(f"[SAVE] {path}")
        return str(path)
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
# 유틸: 윈저라이즈(평균만)
# ==========================================================
def winsorize_mean_only(work: pd.DataFrame, mean_col: str, cap_pct: float = 99.0):
    if len(work) == 0:
        return work.copy(), np.nan, pd.Series([], dtype=bool, index=work.index)

    cap_value = np.nanpercentile(work[mean_col], cap_pct)
    clipped_mask = work[mean_col] > cap_value
    work_w = work.copy()
    work_w[mean_col] = np.minimum(work_w[mean_col], cap_value)
    return work_w, cap_value, clipped_mask

# ==========================================================
# 유틸: iso-line (x*y=const) 등고선 (log-log)
# ==========================================================
def add_iso_product_contours(ax, x_min, x_max, y_min, y_max,
                             levels, show_labels=True, fmt=None):
    if x_min <= 0 or y_min <= 0:
        return

    xs = np.logspace(np.log10(x_min), np.log10(x_max), 300)
    ys = np.logspace(np.log10(y_min), np.log10(y_max), 300)
    X, Y = np.meshgrid(xs, ys)
    Z = X * Y

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

# ==========================================================
# ✅ 센터 위치 기반 재라벨링
#   입력: centers_orig (shape (3,2)) where columns are [N, M] in ORIGINAL units
#   출력: map_old_to_new, name_by_new, new_centers_orig
# ==========================================================
def relabel_by_centers_orig(centers_orig: np.ndarray):
    if centers_orig.shape[0] != 3:
        raise ValueError("relabel_by_centers_orig is defined for k=3 only.")

    Ns = centers_orig[:, 0]
    Ms = centers_orig[:, 1]

    long_old = int(np.nanargmax(Ms))  # mean 최대
    remaining = [i for i in range(3) if i != long_old]
    frequent_old = int(remaining[int(np.nanargmax(Ns[remaining]))])  # 남은 것 중 N 최대
    minimal_old = int([i for i in range(3) if i not in (long_old, frequent_old)][0])

    map_old_to_new = {minimal_old: 0, frequent_old: 1, long_old: 2}
    name_by_new = {
        0: r"Minimal $R_{\mathrm{FC}}$",
        1: r"Frequent $R_{\mathrm{FC}}$",
        2: r"Long $R_{\mathrm{FC}}$",
    }

    new_centers = np.zeros_like(centers_orig)
    for old_id, new_id in map_old_to_new.items():
        new_centers[new_id, :] = centers_orig[old_id, :]

    return map_old_to_new, name_by_new, new_centers

# ==========================================================
# 클러스터 학습 + 시각화 + 요약출력
#   입력 df_for_fit: [user_id, delta_t95_event_N, delta_t95_event_mean_h]
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

    total_users_input = int(df["user_id"].nunique()) if "user_id" in df.columns else len(df)

    # 사용 마스크 (log 가능하도록 양수 제약 포함)
    N_num = pd.to_numeric(df[N_col], errors="coerce")
    M_num = pd.to_numeric(df[M_col], errors="coerce")

    if PLOT_LOG_AXES or USE_LOG_SPACE_FOR_CLUSTERING:
        mask_used = (N_num > 0) & (M_num > 0) & np.isfinite(N_num) & np.isfinite(M_num)
    else:
        mask_used = (N_num > 0) & np.isfinite(N_num) & np.isfinite(M_num)

    used = df.loc[mask_used, [N_col, M_col]].astype(float)
    used_user_ids = df.loc[mask_used, "user_id"] if "user_id" in df.columns else None
    total_users_used = int(used_user_ids.nunique()) if used_user_ids is not None else int(mask_used.sum())
    total_users_skipped = total_users_input - total_users_used

    if len(used) < K_FIXED:
        print("[WARN] 유효 유저 표본 < 3 → 클러스터링 생략")
        out_csv = out_dir / "user_user_level_fit_input.csv"
        safe_write_csv(df, out_csv)
        return str(out_csv), None

    # 윈저라이즈(평균만, 원 단위)
    cap_val = np.nan
    n_clipped = 0
    used_w = used.copy()
    clipped_mask = pd.Series(False, index=used_w.index)
    if winsorize_mean:
        used_w, cap_val, clipped_mask = winsorize_mean_only(used_w, M_col, cap_pct)
        n_clipped = int(clipped_mask.sum())
        print(f"[INFO] winsorize(mean only) @{cap_pct}pct → {M_col} cap={cap_val:.3f} (clip {n_clipped})")

    # ===== 클러스터링 입력 공간 구성 =====
    X_orig = used_w[[N_col, M_col]].to_numpy().astype(float)

    if USE_LOG_SPACE_FOR_CLUSTERING:
        X_log = np.column_stack([np.log10(X_orig[:, 0]), np.log10(X_orig[:, 1])])

        if CLIP_LOG_MEAN_LO:
            lo = float(np.nanpercentile(X_log[:, 1], LOG_MEAN_LO_PCT))
            X_log[:, 1] = np.maximum(X_log[:, 1], lo)
            print(f"[INFO] clip log10(mean) lower @{LOG_MEAN_LO_PCT}pct: lo={lo:.4f}")

        X_for_model = X_log.copy()
        X_for_model[:, 0] *= WEIGHT_LOG_N

        print("[INFO] clustering space: log10(N_per_month), log10(mean_month)")
        print(f"[INFO] N axis weight (log space): WEIGHT_LOG_N={WEIGHT_LOG_N}")
    else:
        X_for_model = X_orig.copy()
        print("[INFO] clustering space: original units (N_per_month, mean_month)")

    # 표준화 후 KMeans
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X_for_model)

    km = KMeans(n_clusters=K_FIXED, n_init=10, random_state=42).fit(Xs)
    labels_raw = km.labels_.astype(int)  # 0..2

    # centers: 표준화 역변환 -> 모델공간 -> 원단위 centers_orig 계산
    centers_std = km.cluster_centers_.copy()          # std space
    centers_model = scaler.inverse_transform(centers_std)  # model space (log or orig)

    if USE_LOG_SPACE_FOR_CLUSTERING:
        centers_log = centers_model.copy()
        centers_log[:, 0] /= WEIGHT_LOG_N
        centers_orig = np.column_stack([10 ** centers_log[:, 0], 10 ** centers_log[:, 1]])
    else:
        centers_orig = centers_model.copy()

    # ======================================================
    # ✅ 여기서 "개수 기준"이 아니라 "센터 기준" 재라벨링 적용
    # ======================================================
    if USE_CENTER_BASED_NAMING:
        map_old_to_new, name_by_new, centers_reordered_orig = relabel_by_centers_orig(centers_orig)
        labels = np.array([map_old_to_new[int(l)] for l in labels_raw], dtype=int)
        print(f"[INFO] relabel by centers: old→new {map_old_to_new}")
    else:
        # (옵션) 이전 방식(개수 기준)도 남겨둠
        counts = np.bincount(labels_raw, minlength=km.n_clusters)
        order = np.argsort(-counts)
        relabel = {old: new for new, old in enumerate(order)}
        labels = np.array([relabel[int(l)] for l in labels_raw], dtype=int)
        centers_reordered_orig = centers_orig[order]
        name_by_new = {
            0: r"Minimal $R_{\mathrm{FC}}$",
            1: r"Frequent $R_{\mathrm{FC}}$",
            2: r"Long $R_{\mathrm{FC}}$",
        }
        print(f"[INFO] relabel by size desc: old→new {relabel} (counts={counts.tolist()})")

    # silhouette / 분류 성능 (라벨 변경 후 기준)
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

    # ─────────────────────────────────────────────
    # 플롯 (원 단위 scatter + log axes)
    # ─────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7.5, 6))

    palette = ["#cd534c", "#20854e", "#0073c2"]  # 0 Minimal,1 Frequent,2 Long
    markers = ["o", "s", "^"]

    for cid in [0, 1, 2]:
        m = (labels == cid)
        if not np.any(m):
            continue
        ax.scatter(
            used_w.loc[m, N_col],
            used_w.loc[m, M_col],
            s=30,
            marker=markers[cid],
            c=palette[cid],
            edgecolor="k",
            linewidth=0.5,
            alpha=0.9,
            label=name_by_new.get(cid, f"Cluster {cid}")
        )

    if PLOT_LOG_AXES:
        ax.set_xscale("log")
        ax.set_yscale("log")

    x_min, x_max = float(used_w[N_col].min()), float(used_w[N_col].max())
    y_min, y_max = float(used_w[M_col].min()), float(used_w[M_col].max())

    if PLOT_LOG_AXES:
        pad = 0.08
        x_lo = x_min / (1 + pad)
        x_hi = x_max * (1 + pad)
        y_lo = y_min / (1 + pad)
        y_hi = y_max * (1 + pad)
        ax.set_xlim(x_lo, x_hi)
        ax.set_ylim(y_lo, y_hi)

        if SHOW_ISO_CONTOUR:
            add_iso_product_contours(
                ax, x_lo, x_hi, y_lo, y_hi,
                levels=ISO_LEVELS,
                show_labels=ISO_LABELS,
                fmt=ISO_FMT
            )
    else:
        pad_x = (x_max - x_min) * 0.05 if x_max > x_min else 1.0
        pad_y = (y_max - y_min) * 0.05 if y_max > y_min else 1.0
        ax.set_xlim(x_min - pad_x, x_max + pad_x)
        ax.set_ylim(y_min - pad_y, y_max + pad_y)

    ax.set_xlabel(r"N(DFC)/month", fontsize=8)
    ax.set_ylabel(r"AVG($\Delta t_{100\%}$)/month", fontsize=8)
    ax.tick_params(axis="both", labelsize=8, width=1.2, length=5)

    leg = ax.legend(fontsize=8, loc="upper right", frameon=True)
    frame = leg.get_frame()
    frame.set_edgecolor("Grey")
    frame.set_linewidth(0.6)

    fig.tight_layout()
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
    print("\n=== 결과 요약 ===")
    print(f"- log clustering                    : {'ON' if USE_LOG_SPACE_FOR_CLUSTERING else 'OFF'}")
    if USE_LOG_SPACE_FOR_CLUSTERING:
        print(f"- clip log10(mean) low              : {'ON' if CLIP_LOG_MEAN_LO else 'OFF'} (pct={LOG_MEAN_LO_PCT})")
        print(f"- WEIGHT_LOG_N                      : {WEIGHT_LOG_N}")
    print(f"- plot log-log axes                 : {'ON' if PLOT_LOG_AXES else 'OFF'}")
    print(f"- relabel rule                      : {'center-based' if USE_CENTER_BASED_NAMING else 'size-based'}")
    if winsorize_mean:
        print(f"- winsor cap(mean only) @{cap_pct}pct: {M_col}={None if np.isnan(cap_val) else round(cap_val, 3)}")
        print(f"- clipped: {M_col} {n_clipped}개")
    print(f"- 출력 폴더                         : {out_dir}")
    print(f"- 전체 사용자 수(입력)              : {total_users_input}명")
    print(f"- 클러스터 적용 사용자 수(라벨 대상): {total_users_used}명")
    print(f"- 제외된 사용자 수(스케일/유효성)   : {total_users_skipped}명")
    print(f"- silhouette                        : {None if np.isnan(sil) else round(sil, 3)}")
    print(f"- test acc                          : {None if np.isnan(acc) else round(acc, 3)}")
    print(f"- cv acc                            : {None if np.isnan(cv_acc) else round(cv_acc, 3)}")
    print(f"- iso contour (N*AVG)               : {'ON' if SHOW_ISO_CONTOUR else 'OFF'}")
    print(f"- 산점도                            : {plot_path}")
    print(f"- 라벨 CSV                          : {out_csv}")

    print("\n[센터(원 단위) - 재라벨 후: 0 Minimal, 1 Frequent, 2 Long]")
    print(pd.DataFrame(centers_reordered_orig, columns=[DISP_X, DISP_Y], index=[0, 1, 2]).round(3).to_string())

    print("\n[클러스터별 크기(유저 수)]")
    vc_users = pd.Series(labels).value_counts().sort_index()
    for cid, cnt in vc_users.items():
        print(f" - Cluster {cid} ({name_by_new.get(cid,'')}): 사용자 {cnt}명")

    return str(out_csv), str(plot_path), pd.DataFrame(centers_reordered_orig, columns=[DISP_X, DISP_Y]).round(3)

# ==========================================================
# 파이프라인
# ==========================================================
def run_user_level_from_combined(combined_csv: str, out_dir: str,
                                 winsorize_mean=True, cap_pct=99.9):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(combined_csv)
    print(f"[INFO] 로드: {combined_csv} (rows={len(df)})")

    needed = ["file_stem", "delta_t95_event_N", "delta_t95_event_mean_h"]
    for c in needed:
        if c not in df.columns:
            raise ValueError(f"필요 컬럼이 없습니다: {c}")

    # file_stem → user_id, ym 파싱
    users, yms = [], []
    for stem in df["file_stem"].astype(str).fillna(""):
        u, ym = parse_user_ym_from_stem(stem)
        users.append(u)
        yms.append(ym)
    df["user_id"] = users
    df["ym"] = yms

    bad_u = int(df["user_id"].isna().sum())
    bad_m = int(df["ym"].isna().sum())
    if bad_u or bad_m:
        print(f"[WARN] 파싱 실패: user_id {bad_u}행, ym {bad_m}행 → 제외")
    df = df.dropna(subset=["user_id", "ym"]).copy()

    # 숫자화
    N_series = pd.to_numeric(df["delta_t95_event_N"], errors="coerce")
    M_series = pd.to_numeric(df["delta_t95_event_mean_h"], errors="coerce")

    tmp = pd.DataFrame({
        "user_id": df["user_id"].values,
        "ym": df["ym"].values,
        "_N": np.clip(N_series.values, a_min=0, a_max=None),
        "_M": M_series.values
    })

    user_feat = (
        tmp.groupby("user_id", as_index=False)
           .agg(
               months_covered=("ym", "nunique"),
               N_total=("_N", "sum"),
               delta_t95_event_mean_month_h=("_M", "mean")
           )
    )
    user_feat["N_per_month"] = user_feat["N_total"] / user_feat["months_covered"]
    user_feat.replace([np.inf, -np.inf], np.nan, inplace=True)

    df_for_fit = user_feat[["user_id", "N_per_month", "delta_t95_event_mean_month_h"]].rename(columns={
        "N_per_month": "delta_t95_event_N",
        "delta_t95_event_mean_month_h": "delta_t95_event_mean_h"
    })

    labeled_csv, plot_path, centers_df = fit_and_plot_userlevel(
        df_for_fit, out,
        plot_name="user_dfc_clusters_boundary_users_FROM_COMBINED_log.png" if PLOT_LOG_AXES else
                  "user_dfc_clusters_boundary_users_FROM_COMBINED.png",
        winsorize_mean=winsorize_mean, cap_pct=cap_pct,
        user_feat_full=user_feat
    )

    df_labels = pd.read_csv(labeled_csv)
    merged = user_feat.merge(df_labels[["user_id", "cluster"]], on="user_id", how="left")

    out_all = Path(out) / f"user_user_level_clusters_FROM_COMBINED_winz_{'on' if winsorize_mean else 'off'}_log_{'on' if PLOT_LOG_AXES else 'off'}_centerlabel_{'on' if USE_CENTER_BASED_NAMING else 'off'}.csv"
    safe_write_csv(
        merged[["user_id", "months_covered", "N_total", "N_per_month", "delta_t95_event_mean_month_h", "cluster"]],
        out_all
    )

    only12 = merged[merged["months_covered"] >= 12].copy()
    out_12 = Path(out) / f"user_user_level_clusters_12MO_FROM_COMBINED_winz_{'on' if winsorize_mean else 'off'}_log_{'on' if PLOT_LOG_AXES else 'off'}_centerlabel_{'on' if USE_CENTER_BASED_NAMING else 'off'}.csv"
    safe_write_csv(
        only12[["user_id", "months_covered", "N_total", "N_per_month", "delta_t95_event_mean_month_h", "cluster"]],
        out_12
    )

    df_with_user_cluster = df.merge(
        df_labels[["user_id", "cluster"]].rename(columns={"cluster": "user_cluster"}),
        on="user_id", how="left"
    )
    out_monthly = Path(out) / f"user_dfc_features_with_user_clusters_log_{'on' if PLOT_LOG_AXES else 'off'}_centerlabel_{'on' if USE_CENTER_BASED_NAMING else 'off'}.csv"
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
        winsorize_mean=False,
        # cap_pct=99.9
    )
