#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor  # 멀티프로세싱

# ─────────────────────────────────────────────────────────────
# 1) SOC≥95% 시간 계산 (벡터화)  ─ 기존 그대로 사용
# ─────────────────────────────────────────────────────────────
def soc95_time(data: pd.DataFrame) -> float:
    """
    입력: time(문자열 날짜), soc(숫자)가 있는 DataFrame
    반환: SOC>=95% 상태로 있었던 총 시간(시간 단위, float)
    """
    if 'time' not in data.columns or 'soc' not in data.columns or len(data) < 2:
        return 0.0

    # 안전 파싱
    t = pd.to_datetime(data['time'], format='%Y-%m-%d %H:%M:%S', errors='coerce')
    soc = pd.to_numeric(data['soc'], errors='coerce')

    # 끝 행은 dt가 없으므로 한 칸 쉬프트
    t_next = t.shift(-1)
    dt = (t_next - t)

    # 유효 구간 (dt 존재)
    valid = dt.notna()

    # ① 기본 시간: 현재 행의 SOC가 95% 이상인 구간들의 dt 합
    soc_ge95 = soc >= 95
    base_time = dt[valid & soc_ge95].sum()

    # ② 측정오류 보정:
    #    "시동 꺼짐"으로 간주되는 긴 간격(>5min)이며,
    #    직전은 <95, 다음은 >=95 로 '상향 교차'인 경우 그 dt를 추가
    limit = pd.Timedelta(minutes=5)
    long_gap = dt > limit
    next_ge95 = soc.shift(-1) >= 95
    prev_lt95 = soc < 95

    correction_time = dt[valid & long_gap & prev_lt95 & next_ge95].sum()

    total = pd.to_timedelta(0) + (base_time if pd.notna(base_time) else pd.Timedelta(0)) \
            + (correction_time if pd.notna(correction_time) else pd.Timedelta(0))

    # 초→시간
    return total.total_seconds() / 3600.0


# ─────────────────────────────────────────────────────────────
# 1-1) 한 파일만 t95 계산하는 헬퍼
# ─────────────────────────────────────────────────────────────
def compute_t95_for_file(file_path: str) -> float:
    """
    단일 CSV 파일에 대해 SOC>=95% 시간(t95, h)을 계산해서 반환.
    """
    data = pd.read_csv(file_path)
    t95_h = soc95_time(data)
    return t95_h


# ─────────────────────────────────────────────────────────────
# 2) 파일명 정규화 유틸 (예전 코드 그대로)
# ─────────────────────────────────────────────────────────────
def normalize_stem(name: str) -> str:
    """
    파일명/스템 문자열에서 확장자 제거 후,
    '_DFC', '_r', '_CR' 같은 접미사를 떼고 공통 키를 만든다.

    예)
      bms_..._2023-04_r.csv      → bms_..._2023-04
      bms_..._2023-04_DFC.csv    → bms_..._2023-04
      bms_..._2023-04_CR.csv     → bms_..._2023-04
      bms_..._2023-04_r          → bms_..._2023-04
    """
    stem = os.path.splitext(os.path.basename(str(name)))[0]
    for suf in ['_DFC', '_dfc', '_R', '_r', '_CR', '_cr']:
        if stem.endswith(suf):
            stem = stem[:-len(suf)]
    return stem


# ─────────────────────────────────────────────────────────────
# 3) 폴더별 t95(h) 계산 (멀티/싱글 공통)
# ─────────────────────────────────────────────────────────────
def _t95_worker(args):
    """
    멀티프로세싱용 워커
    입력: (key, file_path)
    출력: (key, t95_h, error_msg or None)
    """
    key, file_path = args
    try:
        data = pd.read_csv(file_path)
        t95_h = soc95_time(data)
        return key, t95_h, None
    except Exception as e:
        return key, np.nan, str(e)


def compute_t95_by_file(folder_path: str, use_mp: bool = True, max_workers=None) -> dict:
    """
    주어진 폴더 내 모든 CSV에 대해 t95(h)를 계산해서
    { 공통키(normalize_stem): t95_h } 딕셔너리로 반환.

    use_mp=False: 단일 프로세스로 순차 처리 (네트워크 드라이브에 안전)
    use_mp=True : 멀티프로세싱으로 병렬 처리
    """
    results = {}
    if not os.path.isdir(folder_path):
        print(f"[WARN] 폴더 미존재: {folder_path}")
        return results

    files = [f for f in os.listdir(folder_path) if f.lower().endswith('.csv')]
    files.sort()

    if not files:
        print(f"[WARN] CSV 파일이 없습니다: {folder_path}")
        return results

    tasks = []
    for filename in files:
        file_path = os.path.join(folder_path, filename)
        key = normalize_stem(filename)
        tasks.append((key, file_path))

    # ── 싱글 프로세스 버전 ───────────────────────────────────────
    if not use_mp:
        for key, file_path in tqdm(tasks, desc=f"Scanning {os.path.basename(folder_path)} (single)"):
            try:
                data = pd.read_csv(file_path)
                t95_h = soc95_time(data)
                if key in results:
                    print(f"[WARN] 중복 키 처리: {key} (기존 값 덮어씀)")
                results[key] = t95_h
            except Exception as e:
                print(f"[SKIP] {key} - error(single): {repr(e)}")
        return results

    # ── 멀티프로세스 버전 ───────────────────────────────────────
    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        for key, t95_h, err in tqdm(
            ex.map(_t95_worker, tasks),
            total=len(tasks),
            desc=f"Scanning {os.path.basename(folder_path)} (mp)",
        ):
            if err is not None:
                print(f"[SKIP] {key} - error(mp): {err}")
                continue
            if key in results:
                print(f"[WARN] 중복 키 처리: {key} (기존 값 덮어씀)")
            results[key] = t95_h

    return results


# ─────────────────────────────────────────────────────────────
# 4) 메인: 원하는 경로들에 대해 t95만 계산해서 CSV 저장
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # ── 여기만 네가 수정해서 쓰면 됨 ───────────────────────────
    # (car_model, folder_path, use_mp) 튜플 리스트
    MODELS = [
        ("EV6",   r"Z:\SamsungSTF\Processed_Data\DFC\EV6\DFC_원본",   True),
        ("Ioniq5",r"Z:\SamsungSTF\Processed_Data\DFC\Ioniq5\DFC_원본", True),
    ]

    # 결과 저장 경로
    OUT_DIR = r"G:\공유 드라이브\BSG_DFC_result\combined\DFC_원본\t95"
    os.makedirs(OUT_DIR, exist_ok=True)
    OUT_CSV = os.path.join(OUT_DIR, "t95_by_file_simple.csv")

    all_rows = []

    for car_model, folder_before, use_mp in MODELS:
        print(f"\n==============================")
        print(f"[INFO] car_model = {car_model}")
        print(f"[INFO] 폴더 = {folder_before}")
        print(f"[INFO] use_mp = {use_mp}")

        before_dict = compute_t95_by_file(
            folder_before,
            use_mp=use_mp,
            max_workers=8 if use_mp else None  # 여기에서 워커 개수 조절
        )
        print(f"[INFO] {car_model}: t95 계산된 파일 수 = {len(before_dict)}")

        for key, t_h in before_dict.items():
            all_rows.append({
                "car_model": car_model,
                "file":      key,   # bms_..._YYYY-MM
                "t95_h":     t_h,   # SOC≥95% 총 시간(h)
            })

    if not all_rows:
        raise SystemExit("[INFO] 생성된 row가 없습니다. 경로/입력 파일을 확인하세요.")

    df_out = pd.DataFrame(
        all_rows,
        columns=["car_model", "file", "t95_h"],
    )

    df_out.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    print("\n===================================")
    print(f"[SAVE] t95 결과 CSV → {OUT_CSV}")
    print(f"[INFO] 총 행 수 = {len(df_out)}")
    print("미리보기:")
    print(df_out.head())
