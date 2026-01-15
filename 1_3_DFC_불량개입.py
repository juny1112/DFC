#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm
import multiprocessing as mp

# =========================================================
# (A) 사용자 설정
# =========================================================

# 1) 불량개입 파일 목록 소스 (1_불량개입_count 결과물)
BAD_LIST_CSV = r"G:\공유 드라이브\BSG_DFC_result\combined\DFC_완충후이동주차\불량개입\min_soc_between_fullcharges_cases.csv"

# 2) 토글: case1_20_count도 불량개입 판정에 포함할지
INCLUDE_CASE1 = False  # True면 case1도 포함, False면 case2+case3만

# 3) 입력(원본) 폴더: 차종별 R_parsing_완충후이동주차
INPUT_DIRS = {
    "EV6":    r"Z:\SamsungSTF\Processed_Data\DFC\EV6\R_parsing_완충후이동주차",
    "Ioniq5": r"Z:\SamsungSTF\Processed_Data\DFC\Ioniq5\R_parsing_완충후이동주차",
}

# 4) 출력(불량개입 manipulation 결과) 폴더
OUTPUT_DIRS = {
    "EV6":    r"Z:\SamsungSTF\Processed_Data\DFC\EV6\불량개입",
    "Ioniq5": r"Z:\SamsungSTF\Processed_Data\DFC\Ioniq5\불량개입",
}

# 5) SOC0_count 결과 저장 위치
OUT_SOC0_COUNT_CSV = r"G:\공유 드라이브\BSG_DFC_result\combined\DFC_완충후이동주차\불량개입\bad_intervention_soc0_count.csv"
OUT_SOC0_COUNT_CSV_TEST = r"G:\공유 드라이브\BSG_DFC_result\combined\DFC_완충후이동주차\불량개입\bad_intervention_soc0_count_TEST.csv"

# 6) SOC 클리핑(주행실패 표시)
CLIP_AT_ZERO = True

# 7) 멀티프로세싱 워커 수
N_WORKERS = 8

# 8) 시험 모드: True면 일부 파일만 단일 프로세스로 실행
TEST_MODE = False
TEST_N_FILES = 10  # car_type 합쳐서 앞에서 N개

# 9) SOC0_count 정의 옵션
#    - "samples": 조작 후 soc==0인 "행(샘플) 수" (요청사항 그대로)
#    - "segments": 조작 후 soc==0이 된 "연속 구간 개수" (원하면 바꿔 쓸 수 있게만 제공)
SOC0_COUNT_MODE = "segments"  # 요청사항: "samples"


# =========================================================
# (B) DFC 이벤트 탐지에 필요한 유틸 (원본 로직 최소 이식)
# =========================================================

def remove_consecutive_ones(data: pd.DataFrame) -> pd.DataFrame:
    if 'R_aftercharg' not in data.columns:
        return data

    s = data['R_aftercharg'].fillna(0).astype(int)
    grp = (s != s.shift(fill_value=s.iloc[0])).cumsum()

    group_sizes = grp.map(grp.value_counts())
    pos_from_start = data.groupby(grp).cumcount()
    pos_from_end = data.iloc[::-1].groupby(grp.iloc[::-1]).cumcount()[::-1]

    keep = (
        (s == 0) |
        ((s == 1) & ((group_sizes < 3) | (pos_from_start == 0) | (pos_from_end == 0)))
    )
    return data.loc[keep].reset_index(drop=True)


def detect_dfc_events_only(data: pd.DataFrame):
    """
    반환: list of dict
      - delay_start_idx (dstart)
      - charge_end_idx  (cend)
      - after_end_idx   (aend)
    """
    data = remove_consecutive_ones(data)

    if 'time' in data.columns and not pd.api.types.is_datetime64_any_dtype(data['time']):
        data['time'] = pd.to_datetime(data['time'], format='%Y-%m-%d %H:%M:%S', errors='coerce')

    if not {'R_charg', 'R_aftercharg', 'soc'}.issubset(set(data.columns)):
        return []

    # 충전 구간 경계
    charg = []
    if int(data.loc[0, 'R_charg']) == 1:
        charg.append(0)
    for i in range(len(data) - 1):
        if int(data.loc[i, 'R_charg']) != int(data.loc[i + 1, 'R_charg']):
            charg.append(i + 1)
    if int(data.loc[len(data) - 1, 'R_charg']) == 1:
        charg.append(len(data) - 1)

    def any_after_near(idx):
        lo = max(0, idx - 1)
        hi = min(len(data) - 1, idx + 1)
        return (data.loc[lo:hi, 'R_aftercharg'].fillna(0).astype(int) == 1).any()

    # DFC 적용 충전구간 후보
    dfc_charg = []
    for i in range(len(charg) - 1):
        if (int(data.loc[charg[i], 'R_charg']) == 1) and any_after_near(charg[i + 1] - 1):
            dfc_charg.append(charg[i])
            dfc_charg.append(charg[i + 1] - 1)

    # SOC 80 delay_start 찾기
    charg_2_pairs = []   # (delay_start, charge_end)
    for j in range(0, len(dfc_charg) - 1, 2):
        start_j, end_j = dfc_charg[j], dfc_charg[j + 1]
        found = False

        for i in range(start_j, max(start_j, end_j)):
            soc_i  = pd.to_numeric(data.loc[i, 'soc'], errors='coerce')
            soc_i1 = pd.to_numeric(data.loc[i + 1, 'soc'], errors='coerce')
            if pd.notna(soc_i) and pd.notna(soc_i1) and (soc_i < 80) and (soc_i1 == 80):
                charg_2_pairs.append((i + 1, end_j))
                found = True
                break

        if (not found):
            soc_s = pd.to_numeric(data.loc[start_j, 'soc'], errors='coerce')
            if pd.notna(soc_s) and (soc_s >= 80):
                charg_2_pairs.append((start_j, end_j))

    # aftercharge segment 매칭
    dfc_events = []
    ch = data['R_charg'].fillna(0).astype(int).to_numpy()
    ac = data['R_aftercharg'].fillna(0).astype(int).to_numpy()

    transitions_ac = np.diff(np.r_[0, ac, 0])
    astarts = np.where(transitions_ac == +1)[0]
    aends   = np.where(transitions_ac == -1)[0] - 1

    transitions_ch = np.diff(np.r_[0, ch])
    cstarts = np.where(transitions_ch == +1)[0]

    astarts.sort(); aends.sort(); cstarts.sort()

    for dstart, cend in charg_2_pairs:
        soc_d = pd.to_numeric(data.loc[dstart, 'soc'], errors='coerce')
        # 원본 정책: delay_start SOC>=95면 DFC 적용 안 함
        if pd.notna(soc_d) and soc_d >= 95:
            continue

        pos_c = np.searchsorted(cstarts, cend + 1, side='left')
        next_charge_start = cstarts[pos_c] if pos_c < len(cstarts) else None

        pos_a = np.searchsorted(astarts, cend, side='left')
        if pos_a >= len(astarts):
            continue

        astart = astarts[pos_a]
        if (next_charge_start is not None) and (astart >= next_charge_start):
            continue

        aend = aends[pos_a]

        dfc_events.append({
            "delay_start_idx": int(dstart),
            "charge_end_idx":  int(cend),
            "after_end_idx":   int(aend),
        })

    return dfc_events


# =========================================================
# (C) SOC0_count helper
# =========================================================

def count_soc0(soc_arr: np.ndarray, mode: str = "samples") -> int:
    """
    mode:
      - "samples": soc==0인 행(샘플) 수
      - "segments": soc==0 연속 구간 수 (0-run count)
    """
    valid = np.isfinite(soc_arr)
    z = valid & (soc_arr == 0.0)

    if mode == "samples":
        return int(np.sum(z))

    if mode == "segments":
        # z가 False->True로 바뀌는 순간(구간 시작) 개수
        return int(np.sum(np.diff(np.r_[False, z]) == 1))

    raise ValueError(f"Unknown SOC0 count mode: {mode}")


# =========================================================
# (D) SOC manipulation (요청한 규칙 그대로)
# =========================================================

def apply_bad_intervention_soc80(data: pd.DataFrame, dfc_events: list):
    if data.empty or (not dfc_events):
        out_cols = ["time", "speed", "soc", "chrg_cable_conn", "pack_current", "charging", "rest"]
        out_cols = [c for c in out_cols if c in data.columns]
        # 조작이 없으니 soc==0도 그냥 원본 기준
        soc0_count = count_soc0(pd.to_numeric(data["soc"], errors="coerce").to_numpy(dtype=float), SOC0_COUNT_MODE) \
                     if "soc" in data.columns else 0
        return data[out_cols].copy(), soc0_count

    if "time" in data.columns:
        data = data.sort_values("time").reset_index(drop=True)
    else:
        data = data.reset_index(drop=True)

    soc_orig = pd.to_numeric(data["soc"], errors="coerce").to_numpy(dtype=float)
    soc_new  = soc_orig.copy()

    rest = data["rest"].fillna(0).astype(int).to_numpy()
    rch  = data["R_charg"].fillna(0).astype(int).to_numpy()

    # 다음 충전 시작(0->1 전이) 인덱스
    cstarts = np.where(np.diff(np.r_[0, rch]) == 1)[0]
    cstarts.sort()

    dfc_events_sorted = sorted(dfc_events, key=lambda x: x["delay_start_idx"])

    for ev in dfc_events_sorted:
        dstart = int(ev["delay_start_idx"])
        if dstart < 0 or dstart >= len(data):
            continue

        # dstart가 rest=1 구간에 속하도록 보정(필요 시 직전 rest=1로 이동)
        if rest[dstart] != 1:
            k = dstart
            while k > 0 and rest[k] == 0:
                k -= 1
            if rest[k] == 1:
                dstart = k
            else:
                continue

        # rest_end: dstart부터 연속된 rest=1의 마지막 인덱스
        rest_end = dstart
        while (rest_end + 1 < len(data)) and (rest[rest_end + 1] == 1):
            rest_end += 1

        drive_start = rest_end + 1  # rest가 0이 되며 주행 시작(i)
        if drive_start >= len(data):
            continue

        # 1) dstart~rest_end: SOC=80 평탄화
        soc_new[dstart:rest_end + 1] = 80.0

        # 2) drive_start에서 SOC를 80으로 만들기 위한 delta
        if not np.isfinite(soc_orig[drive_start]):
            continue
        delta = soc_orig[drive_start] - 80.0

        # 3) delta 적용 범위 = drive_start ~ 다음 충전 시작 직전
        pos_next = np.searchsorted(cstarts, drive_start, side="right")
        next_charge_start = int(cstarts[pos_next]) if pos_next < len(cstarts) else len(data)

        seg_lo = drive_start
        seg_hi = next_charge_start  # exclusive

        seg = soc_orig[seg_lo:seg_hi] - delta
        if CLIP_AT_ZERO:
            seg = np.maximum(seg, 0.0)

        soc_new[seg_lo:seg_hi] = seg

        # 4) 다음 충전 시작부터는 원상복구(오프셋 제거)
        if next_charge_start < len(data):
            soc_new[next_charge_start:] = soc_orig[next_charge_start:]

    # ★ SOC0_count: 조작 후 soc==0 카운트 (요청사항)
    soc0_count = count_soc0(soc_new, SOC0_COUNT_MODE)

    data_out = data.copy()
    data_out["soc"] = soc_new

    out_cols = ["time", "speed", "soc", "chrg_cable_conn", "pack_current", "charging", "rest"]
    out_cols = [c for c in out_cols if c in data_out.columns]

    return data_out[out_cols].copy(), soc0_count


# =========================================================
# (E) 불량개입 파일 리스트 생성
# =========================================================

def load_bad_file_list(bad_list_csv: str, include_case1: bool):
    df = pd.read_csv(bad_list_csv)

    req = ["car_type", "file_name", "case1_20_count", "case2_20_count", "case3_20_count"]
    for c in req:
        if c not in df.columns:
            raise ValueError(f"[BAD_LIST_CSV] 필수 컬럼 없음: {c}")

    c1 = pd.to_numeric(df["case1_20_count"], errors="coerce").fillna(0).astype(int)
    c2 = pd.to_numeric(df["case2_20_count"], errors="coerce").fillna(0).astype(int)
    c3 = pd.to_numeric(df["case3_20_count"], errors="coerce").fillna(0).astype(int)

    if include_case1:
        mask = (c1 >= 1) | (c2 >= 1) | (c3 >= 1)
    else:
        mask = (c2 >= 1) | (c3 >= 1)

    df_bad = df.loc[mask, ["car_type", "file_name"]].copy()
    df_bad["file_name_lower"] = df_bad["file_name"].astype(str).str.lower()
    df_bad["car_type_lower"]  = df_bad["car_type"].astype(str).str.lower()
    return df_bad


# =========================================================
# (F) 파일 1개 처리 워커
# =========================================================

def process_one(args):
    car_type, in_path, out_path = args
    try:
        header = pd.read_csv(in_path, nrows=0)
        cols_available = set(header.columns)

        usecols = [
            "time", "speed", "soc", "chrg_cable_conn", "pack_current", "charging", "rest",
            "R_charg", "R_aftercharg",
        ]
        usecols = [c for c in usecols if c in cols_available]

        must = {"soc", "rest", "R_charg", "R_aftercharg"}
        if not must.issubset(set(usecols)):
            return None

        df = pd.read_csv(in_path, usecols=usecols)
        if df.empty:
            return None

        dfc_events = detect_dfc_events_only(df)

        df_out, soc0_count = apply_bad_intervention_soc80(df, dfc_events)

        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        df_out.to_csv(out_path, index=False, encoding="utf-8-sig")

        return {
            "car_type": car_type,
            "file_name": os.path.basename(in_path),
            "SOC0_count": int(soc0_count),
            "out_path": out_path,
        }

    except Exception as e:
        print(f"[WARN] {car_type} | {os.path.basename(in_path)} 실패: {e}")
        return {
            "car_type": car_type,
            "file_name": os.path.basename(in_path),
            "SOC0_count": 0,
            "out_path": out_path,
            "error": str(e),
        }


# =========================================================
# (G) 메인
# =========================================================

if __name__ == "__main__":
    bad_df = load_bad_file_list(BAD_LIST_CSV, INCLUDE_CASE1)

    jobs = []

    for car_type in ["EV6", "Ioniq5"]:
        in_root  = Path(INPUT_DIRS[car_type])
        out_root = Path(OUTPUT_DIRS[car_type])
        out_root.mkdir(parents=True, exist_ok=True)

        targets = set(
            bad_df.loc[bad_df["car_type_lower"] == car_type.lower(), "file_name_lower"].tolist()
        )
        if not targets:
            print(f"[INFO] {car_type}: 불량개입 대상 파일 0개")
            continue

        all_csv = list(in_root.rglob("*.csv"))
        print(f"[INFO] {car_type}: 입력 CSV 발견 {len(all_csv)}개")

        for p in all_csv:
            if p.name.lower() in targets:
                out_name = p.name.replace("_r.csv", "_DFC.csv")
                out_path = str(out_root / out_name)
                jobs.append((car_type, str(p), out_path))

    print(f"[INFO] 총 처리 대상: {len(jobs)} files | workers={N_WORKERS} | test={TEST_MODE} | soc0_mode={SOC0_COUNT_MODE}")

    records = []

    if not jobs:
        print("[INFO] 처리 대상 파일이 없습니다.")
        raise SystemExit(0)

    if TEST_MODE:
        test_jobs = jobs[:TEST_N_FILES]
        print(f"[TEST] 시험 모드 ON: {len(test_jobs)}개 파일만 단일 실행")

        for j in tqdm(test_jobs, total=len(test_jobs), unit="file"):
            rec = process_one(j)
            if rec is not None:
                records.append(rec)

        if records:
            out_df = pd.DataFrame(records)
            os.makedirs(os.path.dirname(OUT_SOC0_COUNT_CSV_TEST), exist_ok=True)
            out_df[["car_type", "file_name", "SOC0_count"]].to_csv(
                OUT_SOC0_COUNT_CSV_TEST, index=False, encoding="utf-8-sig"
            )
            print(f"[SAVE][TEST] SOC0_count: {OUT_SOC0_COUNT_CSV_TEST} (rows={len(out_df)})")
        else:
            print("[TEST] 기록할 시험 결과가 없습니다.")

    else:
        print("[RUN] 전체 모드: 멀티프로세싱 실행")
        with mp.Pool(processes=N_WORKERS) as pool:
            for rec in tqdm(pool.imap_unordered(process_one, jobs), total=len(jobs), unit="file"):
                if rec is not None:
                    records.append(rec)

        if records:
            out_df = pd.DataFrame(records)
            os.makedirs(os.path.dirname(OUT_SOC0_COUNT_CSV), exist_ok=True)
            out_df[["car_type", "file_name", "SOC0_count"]].to_csv(
                OUT_SOC0_COUNT_CSV, index=False, encoding="utf-8-sig"
            )
            print(f"[SAVE] SOC0_count: {OUT_SOC0_COUNT_CSV} (rows={len(out_df)})")
        else:
            print("[INFO] 기록할 전체 결과가 없습니다.")
