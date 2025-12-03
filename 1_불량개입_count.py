#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm
import multiprocessing as mp

# ───────────── 경로 설정 ─────────────
BASE_DIRS = [
    r"Z:\SamsungSTF\Processed_Data\DFC\Ioniq5\R_parsing_완충후이동주차",
    r"Z:\SamsungSTF\Processed_Data\DFC\EV6\R_parsing_완충후이동주차",
]

OUT_CSV = r"G:\공유 드라이브\BSG_DFC_result\combined\DFC_완충후이동주차\불량개입\min_soc_between_fullcharges_cases.csv"

# 원본 CSV 컬럼명
COL_SOC    = "soc"
COL_RCHARG = "R_charg"

# ───────────── 선택 파일만 돌리고 싶을 때 ─────────────
# 빈 리스트면 → 모든 파일 대상
# 예시:
# TARGET_FILENAMES = [
#     "bms_01241227999_2023-09_r.csv",
#     "bms_altitude_01241364592_2023-11_r.csv",
# ]
TARGET_FILENAMES = []
TARGET_SET = {name.lower() for name in TARGET_FILENAMES} if TARGET_FILENAMES else None


# ───────────── 유틸: 각 case별 최저 SOC 계산 ─────────────
def compute_minima_by_cases(soc: np.ndarray, rch: np.ndarray):
    """
    soc: 1D array (float)
    rch: 1D array (int, 0/1) ─ R_charg 플래그

    정의:
      - 완충구간 = R_charg == 1 인 연속 block
      - block j 의 시작 인덱스 = start_j (0→1 전이)
      - block j 의 끝   인덱스 = end_j   (1→0 전이 직전, 또는 파일 끝)

    Case 1:
      파일 시작(0) ~ 첫 번째 완충 시작 전(start_0) 구간의 SOC 최저점 1개
      (파일이 R_charg=1 상태에서 시작하면 start_0 == 0 이라 case1 없음)

    Case 2:
      각 j=0..(k-2) 에 대해,
        구간: end_j+1 ~ start_{j+1}-1
        (이전 완충 block 끝 이후 ~ 다음 완충 block 시작 전,
         즉 완충구간 사이의 SOC만 사용)
        각 구간의 SOC 최저점 1개씩 → 리스트

      완전충전 block 이 1개뿐이면 (k=1) case2 없음

    Case 3:
      마지막 완충 block 의 **끝(end_last) 이후** ~ 파일 끝 구간의 SOC 최저점 1개
      - 마지막이 충전 중으로 끝나는 경우(end_last == n-1) → case3 없음

    반환:
      minima1, minima2, minima3, n_full
    """
    n = len(soc)
    if n == 0:
        return [], [], [], 0

    is_full = (rch == 1)

    # block 시작/끝 인덱스 찾기
    prev = np.concatenate(([False], is_full[:-1]))
    next_ = np.concatenate((is_full[1:], [False]))

    start_idx = np.flatnonzero(is_full & ~prev)   # 0→1 (block 시작)
    end_idx   = np.flatnonzero(is_full & ~next_)  # 1→0 (block 끝)

    n_full = int(start_idx.size)
    minima1, minima2, minima3 = [], [], []

    if n_full == 0:
        # 완충 block 이 없으면 세 case 모두 없음
        return minima1, minima2, minima3, 0

    # ─ Case 1: 파일 시작 ~ 첫 번째 완충 시작 전
    first_start = int(start_idx[0])
    if first_start > 0:
        seg1 = soc[:first_start]
        seg1 = seg1[np.isfinite(seg1)]
        if seg1.size > 0:
            minima1.append(float(np.min(seg1)))
    # first_start == 0 이면 파일이 완충 상태에서 시작 → case1 없음

    # ─ Case 2: 각 완충 block 끝 이후 ~ 다음 완충 block 시작 전
    if n_full >= 2:
        for j in range(n_full - 1):
            end_j      = int(end_idx[j])
            next_start = int(start_idx[j + 1])

            # 사이에 샘플이 하나라도 있어야 의미 있는 구간
            if next_start > end_j + 1:
                seg2 = soc[end_j + 1: next_start]
                seg2 = seg2[np.isfinite(seg2)]
                if seg2.size > 0:
                    minima2.append(float(np.min(seg2)))
            # next_start <= end_j+1 이면 완충 블럭이 겹치거나 붙어있어서 사이 구간 없음

    # ─ Case 3: 마지막 완충 block 끝 이후 ~ 파일 끝
    last_end = int(end_idx[-1])
    if last_end < n - 1:
        seg3 = soc[last_end + 1:]
        seg3 = seg3[np.isfinite(seg3)]
        if seg3.size > 0:
            minima3.append(float(np.min(seg3)))
    # last_end == n-1 이면 마지막이 충전 중으로 끝 → case3 없음

    return minima1, minima2, minima3, n_full


def count_below_threshold(minima, thr):
    """최저점 리스트 중 SOC ≤ thr 인 구간 개수"""
    return int(sum(1 for v in minima if v <= thr))


def format_minima(minima):
    """최저점 리스트를 '40,35,8,3,20' 이런 문자열로 변환 (소수점은 반올림)"""
    if not minima:
        return ""
    return ",".join(str(int(round(v))) for v in minima)


# ───────────── 워커용 함수 ─────────────
def process_one(args):
    csv_path, car_type = args
    fname = csv_path.name

    try:
        df = pd.read_csv(
            csv_path,
            usecols=[COL_SOC, COL_RCHARG],
            dtype={COL_SOC: "float32", COL_RCHARG: "int8"},
            engine="pyarrow",
        )
    except Exception as e:
        print(f"[WARN] 파일 읽기 실패: {csv_path} ({e})")
        return None

    if df.empty:
        return None

    soc = df[COL_SOC].to_numpy(dtype=float)
    rch = df[COL_RCHARG].to_numpy(dtype=int)

    # SOC NaN 제거 (R_charg도 같이 제거)
    mask_valid = np.isfinite(soc)
    if not mask_valid.any():
        return None
    soc = soc[mask_valid]
    rch = rch[mask_valid]

    minima1, minima2, minima3, n_full = compute_minima_by_cases(soc, rch)

    rec = {
        "car_type": car_type,
        "file_name": fname,
        "n_fullcharge_Rcharg": n_full,

        # Case 1
        "case1_total": len(minima1),
        "case1_10_count": count_below_threshold(minima1, 10),
        "case1_20_count": count_below_threshold(minima1, 20),
        "case1_mins": format_minima(minima1),

        # Case 2
        "case2_total": len(minima2),
        "case2_10_count": count_below_threshold(minima2, 10),
        "case2_20_count": count_below_threshold(minima2, 20),
        "case2_mins": format_minima(minima2),

        # Case 3
        "case3_total": len(minima3),
        "case3_10_count": count_below_threshold(minima3, 10),
        "case3_20_count": count_below_threshold(minima3, 20),
        "case3_mins": format_minima(minima3),
    }

    return rec


# ───────────── 메인 ─────────────
if __name__ == "__main__":
    tasks = []
    found_target_names = set()

    for base in BASE_DIRS:
        base_path = Path(base)
        # car_type 은 상위 폴더 이름(Ioniq5 / EV6)
        car_type = base_path.parent.name

        csv_files = list(base_path.rglob("*.csv"))
        print(f"[INFO] {car_type}: CSV 파일 {len(csv_files)}개 발견")

        for csv_path in csv_files:
            fname_lower = csv_path.name.lower()

            # 특정 파일만 돌리기 옵션
            if TARGET_SET is not None:
                if fname_lower in TARGET_SET:
                    tasks.append((csv_path, car_type))
                    found_target_names.add(fname_lower)
            else:
                tasks.append((csv_path, car_type))

    if not tasks:
        print("[INFO] 처리할 파일이 없습니다 (tasks=0).")
        if TARGET_SET is not None:
            missing = TARGET_SET - found_target_names
            if missing:
                print("[WARN] 아래 파일은 BASE_DIRS에서 찾지 못했습니다:")
                for name in sorted(missing):
                    print("  -", name)
        raise SystemExit()

    if TARGET_SET is not None:
        missing = TARGET_SET - found_target_names
        if missing:
            print("[WARN] 아래 파일은 BASE_DIRS에서 찾지 못했습니다:")
            for name in sorted(missing):
                print("  -", name)

    print(f"[INFO] 전체 처리 대상 파일 수: {len(tasks)}")

    # 병렬 처리
    n_workers = min(8, max(1, mp.cpu_count() - 1))
    print(f"[INFO] 병렬 워커 수: {n_workers}")

    records = []
    with mp.Pool(processes=n_workers) as pool:
        for rec in tqdm(pool.imap_unordered(process_one, tasks),
                        total=len(tasks), unit="file"):
            if rec is not None:
                records.append(rec)

    if records:
        out_df = pd.DataFrame(records)
        os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
        out_df.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
        print(f"[SAVE] {OUT_CSV} (rows={len(out_df)})")
    else:
        print("[INFO] 기록할 결과가 없습니다 (records=0).")
