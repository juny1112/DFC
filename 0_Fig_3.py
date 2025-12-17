#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import re
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ─────────────────────────────────────────────────────
# Global style (Nature Energy-like)
# ─────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "Arial",
    "font.size": 5,          # base
    "axes.labelsize": 5,
    "axes.titlesize": 5,
    "xtick.labelsize": 5,
    "ytick.labelsize": 5,
    "legend.fontsize": 5,
    "xtick.major.width": 0.4,
    "ytick.major.width": 0.4,
    "xtick.minor.width": 0.4,
    "ytick.minor.width": 0.4,
    "xtick.major.size": 2,
    "ytick.major.size": 2,
    "xtick.minor.size": 1.5,
    "ytick.minor.size": 1.5,
})

# ─────────────────────────────────────────────────────
# 경로/설정
# ─────────────────────────────────────────────────────
CLUSTER_CSV = r"G:\공유 드라이브\BSG_DFC_result\combined\DFC_완충후이동주차\monthly_cluster\dfc_features_with_clusters.csv"

DIR_CR_MAP = {
    "EV6":    r"Z:\SamsungSTF\Processed_Data\DFC\EV6\CR_parsing",
    "Ioniq5": r"Z:\SamsungSTF\Processed_Data\DFC\Ioniq5\CR_parsing",
}

# (기존) 정상 DFC 폴더
DIR_DFC_MAP = {
    "EV6":    r"Z:\SamsungSTF\Processed_Data\DFC\EV6\DFC_완충후이동주차",
    "Ioniq5": r"Z:\SamsungSTF\Processed_Data\DFC\Ioniq5\DFC_완충후이동주차",
}

# (추가) 불량개입 DFC 폴더
DIR_BAD_DFC_MAP = {
    "EV6":    r"Z:\SamsungSTF\Processed_Data\DFC\EV6\불량개입",
    "Ioniq5": r"Z:\SamsungSTF\Processed_Data\DFC\Ioniq5\불량개입",
}

BASE_OUT = r"G:\공유 드라이브\BSG_DFC_result\combined\DFC_완충후이동주차\Fig3"
os.makedirs(BASE_OUT, exist_ok=True)

# ─────────────────────────────────────────────────────
# 패널별 설정
# - d 패널은 cluster 없이 vehicle_model + base_key + 기간만 지정하면 됨
# - 시간까지 지정하려면:
#     (1) start_ts, end_ts 를 "YYYY-MM-DD HH:MM:SS"로 직접 주거나
#     (2) start_day + start_time / end_day + end_time 조합으로 준다.
# ─────────────────────────────────────────────────────
PANEL_SPECS = {
    "a": {
        "cluster": 1,
        "base_key": "bms_01241248817_2023-04",
        "start_day": 8,
        "end_day": 14,
    },
    "b": {
        "cluster": 2,
        "base_key": "bms_01241228090_2023-04",
        "start_day": 9,
        "end_day": 15,
    },
    "c": {
        "cluster": 0,
        "base_key": "bms_altitude_01241248932_2024-05",
        "start_day": 24,
        "end_day": 30,
    },
    "d": {
        "dfc_variant": "bad",
        "base_key": "bms_01241228037_2023-04",
        "vehicle_model": "Ioniq5",  # ★ 불량개입은 cluster 없으니 명시
        "start_day": 4,
        "end_day": 10,

        # 시간까지 지정하려면:
        # "start_ts": "2023-07-08 06:00:00",
        # "end_ts":   "2023-07-15 06:00:00",
        # 또는
        # "start_time": "06:00:00",
        # "end_time":   "06:00:00",
    },
}

# 색상/스타일 (non DFC vs DFC)
CLR_CR      = "#cd534c"
LS_CR       = "-"
CLR_APPL    = "#0073c2"
LS_APPL     = "--"
LW_CR       = 1.3
LW_APPL     = 1.3
ALPHA_CR    = 0.9
ALPHA_APPL  = 0.9
LAB_CR      = "non DFC"
LAB_APPL    = "DFC"

RE_BASE = re.compile(r"(bms_(?:altitude_)?\d+_\d{4}-\d{2})", re.IGNORECASE)

# ─────────────────────────────────────────────────────
# 유틸 함수
# ─────────────────────────────────────────────────────
def ensure_base_key_col(df: pd.DataFrame) -> pd.DataFrame:
    if "base_key" in df.columns:
        return df
    keys = []
    for _, row in df.iterrows():
        k = None
        for val in row.values:
            if isinstance(val, str):
                m = RE_BASE.search(val)
                if m:
                    k = m.group(1)
                    break
        keys.append(k)
    df = df.copy()
    df["base_key"] = keys
    return df


def load_bms_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "time" not in df.columns or "soc" not in df.columns:
        raise ValueError(f"[ERR] required columns (time, soc) not found: {path}")
    df["time"] = pd.to_datetime(df["time"], errors="coerce")
    df["soc"] = pd.to_numeric(df["soc"], errors="coerce")
    df = df.dropna(subset=["time", "soc"]).sort_values("time")
    return df


def thin_spines(ax, lw: float = 0.4):
    for sp in ax.spines.values():
        sp.set_linewidth(lw)


def find_vehicle_model(
    df_clusters: pd.DataFrame,
    base_key: str,
    cluster: int,
) -> str:
    if "vehicle model" not in df_clusters.columns:
        raise ValueError("[ERR] 'vehicle model' 컬럼이 없습니다.")

    sub = df_clusters[
        (df_clusters["base_key"] == base_key) &
        (df_clusters["cluster"] == cluster)
    ]

    if sub.empty:
        sub2 = df_clusters[df_clusters["base_key"] == base_key]
        if sub2.empty:
            raise ValueError(f"[ERR] base_key={base_key} 에 해당하는 vehicle model을 찾지 못했습니다.")
        vm = str(sub2["vehicle model"].iloc[0])
    else:
        vm = str(sub["vehicle model"].iloc[0])

    return vm


def build_month_paths(vehicle_model: str, base_key: str, dfc_variant: str = "normal") -> Tuple[str, str]:
    """
    파일명: {base_key}_CR.csv / {base_key}_DFC.csv
    dfc_variant:
      - "normal": DIR_DFC_MAP
      - "bad":    DIR_BAD_DFC_MAP (불량개입)
    """
    if vehicle_model not in DIR_CR_MAP:
        raise ValueError(f"[ERR] 지원하지 않는 vehicle model: {vehicle_model}")

    cr_dir = DIR_CR_MAP[vehicle_model]

    if dfc_variant == "bad":
        if vehicle_model not in DIR_BAD_DFC_MAP:
            raise ValueError(f"[ERR] DIR_BAD_DFC_MAP에 vehicle model 없음: {vehicle_model}")
        dfc_dir = DIR_BAD_DFC_MAP[vehicle_model]
    else:
        if vehicle_model not in DIR_DFC_MAP:
            raise ValueError(f"[ERR] DIR_DFC_MAP에 vehicle model 없음: {vehicle_model}")
        dfc_dir = DIR_DFC_MAP[vehicle_model]

    cr_path = os.path.join(cr_dir, f"{base_key}_CR.csv")
    dfc_path = os.path.join(dfc_dir, f"{base_key}_DFC.csv")

    if not os.path.isfile(cr_path):
        raise FileNotFoundError(f"[ERR] CR file not found: {cr_path}")
    if not os.path.isfile(dfc_path):
        raise FileNotFoundError(f"[ERR] DFC file not found: {dfc_path}")

    return cr_path, dfc_path


def parse_start_end_from_spec(spec: dict, base_key: str) -> Tuple[pd.Timestamp, pd.Timestamp]:
    """
    우선순위:
      1) start_ts/end_ts 직접 지정 (시간 포함 가능)
      2) base_key의 YYYY-MM + start_day(+start_time), end_day(+end_time)
      3) end 미지정이면 start + 7일
    """
    if spec.get("start_ts") is not None:
        start_ts = pd.to_datetime(spec["start_ts"], errors="raise")
        if spec.get("end_ts") is None:
            end_ts = start_ts + pd.Timedelta(days=7)
        else:
            end_ts = pd.to_datetime(spec["end_ts"], errors="raise")
        return start_ts, end_ts

    try:
        ym = base_key.split("_")[-1]
        year, month = map(int, ym.split("-"))
    except Exception:
        raise ValueError(f"[ERR] base_key에서 year-month 파싱 실패: {base_key}")

    if "start_day" not in spec:
        raise ValueError("[ERR] start_day 또는 start_ts가 필요합니다.")

    start_day = int(spec["start_day"])
    start_time = str(spec.get("start_time", "00:00:00"))
    start_ts = pd.to_datetime(f"{year:04d}-{month:02d}-{start_day:02d} {start_time}", errors="raise")

    if spec.get("end_ts") is not None:
        end_ts = pd.to_datetime(spec["end_ts"], errors="raise")
        return start_ts, end_ts

    if "end_day" in spec:
        end_day = int(spec["end_day"])
        end_time = str(spec.get("end_time", "23:59:59"))
        end_ts = pd.to_datetime(f"{year:04d}-{month:02d}-{end_day:02d} {end_time}", errors="raise")
        return start_ts, end_ts

    end_ts = start_ts + pd.Timedelta(days=7)
    return start_ts, end_ts


def slice_and_pad_range(df: pd.DataFrame, start_ts: pd.Timestamp, end_ts: pd.Timestamp) -> pd.DataFrame:
    if df.empty:
        return df
    if end_ts <= start_ts:
        return pd.DataFrame(columns=df.columns)

    win = df[(df["time"] >= start_ts) & (df["time"] <= end_ts)].copy()

    # start padding
    prior_start = df[df["time"] <= start_ts].tail(1)
    if not prior_start.empty:
        soc_start = prior_start["soc"].iloc[0]
    else:
        after_start = df[df["time"] >= start_ts].head(1)
        if after_start.empty:
            return pd.DataFrame(columns=df.columns)
        soc_start = after_start["soc"].iloc[0]

    if win.empty or win["time"].min() > start_ts:
        win = pd.concat([pd.DataFrame({"time": [start_ts], "soc": [soc_start]}), win], ignore_index=True)

    # end padding
    prior_end = df[df["time"] <= end_ts].tail(1)
    if not prior_end.empty:
        soc_end = prior_end["soc"].iloc[0]
    else:
        after_end = df[df["time"] >= end_ts].head(1)
        soc_end = after_end["soc"].iloc[0] if not after_end.empty else win["soc"].iloc[-1]

    if win["time"].max() < end_ts:
        win = pd.concat([win, pd.DataFrame({"time": [end_ts], "soc": [soc_end]})], ignore_index=True)

    win = win.sort_values("time").reset_index(drop=True)
    return win


def convert_to_day_axis(df: pd.DataFrame, start_ts: pd.Timestamp, end_ts: pd.Timestamp) -> pd.DataFrame:
    df = df.copy()
    span = (end_ts - start_ts)
    if span.total_seconds() <= 0:
        df["day_idx"] = 1.0
        return df
    frac = (df["time"] - start_ts) / span
    df["day_idx"] = frac * 7.0
    return df


def style_time_soc_axes(ax):
    ax.set_xlim(0, 7)
    ax.set_xticks(np.arange(0, 8))
    ax.set_xticklabels([str(i) for i in range(0, 8)])

    ax.set_ylim(0, 100)
    ax.set_yticks([0, 20, 40, 60, 80, 100])

    ax.set_xlabel("Time (day)", labelpad=1.0)
    ax.set_ylabel("SOC (%)", labelpad=1.0)

    ax.tick_params(axis="both", width=0.4, length=2.0, pad=1.0)
    thin_spines(ax, lw=0.4)


def draw_panel(ax, df_cr: pd.DataFrame, df_ap: pd.DataFrame, start_ts: pd.Timestamp, end_ts: pd.Timestamp):
    if df_cr.empty and df_ap.empty:
        ax.axis("off")
        return

    cr = convert_to_day_axis(df_cr, start_ts, end_ts)
    ap = convert_to_day_axis(df_ap, start_ts, end_ts)

    ax.plot(cr["day_idx"], cr["soc"], color=CLR_CR, linestyle=LS_CR, linewidth=LW_CR,
            alpha=ALPHA_CR, label=LAB_CR)
    ax.plot(ap["day_idx"], ap["soc"], color=CLR_APPL, linestyle=LS_APPL, linewidth=LW_APPL,
            alpha=ALPHA_APPL, label=LAB_APPL)

    style_time_soc_axes(ax)
    ax.tick_params(axis="x", labelbottom=True)

    leg = ax.legend(loc="lower left", frameon=True, framealpha=0.9, borderaxespad=0.2)
    frame = leg.get_frame()
    frame.set_edgecolor("grey")
    frame.set_linewidth(0.4)


def add_panel_label(ax, label, fontsize=7):
    ax.text(-0.08, 1.02, label, transform=ax.transAxes,
            fontsize=fontsize, fontweight="bold",
            ha="right", va="bottom", clip_on=False)


# ─────────────────────────────────────────────────────
# 메인 Figure 생성
# ─────────────────────────────────────────────────────
def make_figure_weekly_examples():
    cl = pd.read_csv(CLUSTER_CSV)
    cl = ensure_base_key_col(cl)
    cl["cluster"] = pd.to_numeric(cl["cluster"], errors="coerce")

    fig, axes = plt.subplots(
        nrows=4, ncols=1,
        figsize=(3.5, 6.0),
        dpi=300,
        sharex=True, sharey=True,
    )

    panel_labels = ["a", "b", "c", "d"]

    for i, label in enumerate(panel_labels):
        spec = PANEL_SPECS.get(label, None)
        ax = axes[i]

        if spec is None:
            ax.axis("off")
            add_panel_label(ax, label)
            continue

        base_key = spec["base_key"]
        dfc_variant = str(spec.get("dfc_variant", "normal")).lower()

        # 기간(날짜/시간) 파싱
        try:
            start_ts, end_ts = parse_start_end_from_spec(spec, base_key)
        except Exception as e:
            print(f"[SKIP] panel {label}: time range parse error - {e}")
            ax.axis("off")
            add_panel_label(ax, label)
            continue

        # ★ vehicle model 결정 로직
        # - spec에 vehicle_model이 있으면(=불량개입 d 패널) 그대로 사용
        # - 없으면 cluster 기반으로 찾아옴(a/b/c)
        try:
            if spec.get("vehicle_model") is not None:
                vm = str(spec["vehicle_model"])
            else:
                if "cluster" not in spec:
                    raise ValueError("[ERR] cluster 또는 vehicle_model 중 하나는 필요합니다.")
                cluster = int(spec["cluster"])
                vm = find_vehicle_model(cl, base_key, cluster)
        except Exception as e:
            print(f"[SKIP] panel {label}: {e}")
            ax.axis("off")
            add_panel_label(ax, label)
            continue

        # 원천 파일 경로
        try:
            cr_path, dfc_path = build_month_paths(vm, base_key, dfc_variant=dfc_variant)
        except Exception as e:
            print(f"[SKIP] panel {label}: {e}")
            ax.axis("off")
            add_panel_label(ax, label)
            continue

        # 데이터 로딩
        try:
            df_cr_full = load_bms_csv(cr_path)
            df_ap_full = load_bms_csv(dfc_path)
        except Exception as e:
            print(f"[SKIP] panel {label}: {e}")
            ax.axis("off")
            add_panel_label(ax, label)
            continue

        # slice + padding
        df_cr_win = slice_and_pad_range(df_cr_full, start_ts, end_ts)
        df_ap_win = slice_and_pad_range(df_ap_full, start_ts, end_ts)

        if df_cr_win.empty and df_ap_win.empty:
            print(f"[SKIP] panel {label}: no data in selected range.")
            ax.axis("off")
            add_panel_label(ax, label)
            continue

        draw_panel(ax, df_cr_win, df_ap_win, start_ts, end_ts)
        add_panel_label(ax, label)

    fig.tight_layout(rect=[0.08, 0.03, 0.99, 0.98])

    base = os.path.join(BASE_OUT, "Figure3_weekly_SOC_clusters")
    fig.savefig(base + ".png", dpi=300, bbox_inches="tight")
    fig.savefig(base + ".pdf", dpi=300, bbox_inches="tight")

    plt.close(fig)
    print(f"[SAVE] {base}.png / {base}.pdf")


if __name__ == "__main__":
    make_figure_weekly_examples()
