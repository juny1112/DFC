#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor  # 멀티프로세싱

# ──────────────────────────────────────────────────────────────────────
# 1) SOC≥95% 시간 계산 (벡터화)
# ──────────────────────────────────────────────────────────────────────
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


# ──────────────────────────────────────────────────────────────────────
# 1-1) 한 파일만 t95 계산하는 헬퍼
# ──────────────────────────────────────────────────────────────────────
def compute_t95_for_file(file_path: str) -> float:
    """
    단일 CSV 파일에 대해 SOC>=95% 시간(t95, h)을 계산해서 반환.
    """
    data = pd.read_csv(file_path)
    t95_h = soc95_time(data)
    return t95_h


# ──────────────────────────────────────────────────────────────────────
# 2) 파일명 정규화 유틸
#    - 확장자 제거 + 접미사(_DFC, _r, _CR 등) 제거 → 공통 key
# ──────────────────────────────────────────────────────────────────────
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


# ──────────────────────────────────────────────────────────────────────
# 2-1) 멀티프로세싱용 워커
# ──────────────────────────────────────────────────────────────────────
def _t95_worker(args):
    """
    멀티프로F세싱용 워커
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


# ──────────────────────────────────────────────────────────────────────
# 3) 폴더별 before t95(h) 계산 (멀티/싱글 선택 가능)
# ──────────────────────────────────────────────────────────────────────
def compute_t95_by_file(folder_path: str, use_mp: bool = True, max_workers=None) -> dict:
    """
    주어진 폴더 내 모든 CSV에 대해 t95(h)를 계산해서
    { 공통키(normalize_stem): t95_before_h } 딕셔너리로 반환.

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

    # (key, file_path) 리스트 준비
    tasks = []
    for filename in files:
        file_path = os.path.join(folder_path, filename)
        key = normalize_stem(filename)
        tasks.append((key, file_path))

    # ── 싱글 프로세스 버전 ──────────────────────────────────────────
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

    # ── 멀티프로세스 버전 ───────────────────────────────────────────
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


# ──────────────────────────────────────────────────────────────────────
# 4) DFC event summary 에서 delta_t95 불러오기
# ──────────────────────────────────────────────────────────────────────
def load_delta_t95_map(summary_csv: str) -> dict:
    """
    dfc_features_summary.csv에서
      - file_stem → normalize_stem(file_stem) 를 key로,
      - delta_t95_event_sum_h 를 value로
    하는 딕셔너리를 만든다.
    """
    if not os.path.isfile(summary_csv):
        print(f"[WARN] summary CSV 미존재: {summary_csv}")
        return {}

    df = pd.read_csv(summary_csv)

    if 'file_stem' not in df.columns or 'delta_t95_event_sum_h' not in df.columns:
        print(f"[WARN] summary CSV에 필요한 컬럼이 없습니다: {summary_csv}")
        return {}

    df = df.copy()
    df['key'] = df['file_stem'].astype(str).map(normalize_stem)
    df['delta'] = pd.to_numeric(df['delta_t95_event_sum_h'], errors='coerce')

    # 같은 key가 여러 번 있으면 합산(보통은 1:1일 것)
    grouped = df.groupby('key')['delta'].sum(min_count=1)
    delta_map = grouped.to_dict()

    print(f"[INFO] delta_t95 map loaded: {len(delta_map)} keys from {os.path.basename(summary_csv)}")
    return delta_map


# ──────────────────────────────────────────────────────────────────────
# 5) 메인: EV6 + Ioniq5 합쳐서 한 번에 CSV 생성
# ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # ── 차종별 입력 경로 + 멀티 on/off 설정 ────────────────────────
    MODELS = [
        (
            "EV6",
            r"Z:\SamsungSTF\Processed_Data\DFC\EV6\R_parsing_완충후이동주차",
            r"G:\공유 드라이브\BSG_DFC_result\EV6\DFC_완충후이동주차\dfc_features_summary.csv",
            True,   # use_mp: EV6는 멀티프로세싱 사용
        ),
        (
            "Ioniq5",
            r"Z:\SamsungSTF\Processed_Data\DFC\Ioniq5\R_parsing_완충후이동주차",
            r"G:\공유 드라이브\BSG_DFC_result\Ioniq5\DFC_완충후이동주차\dfc_features_summary.csv",
            False,  # use_mp: Ioniq5는 싱글 프로세스로 안전하게
        ),
    ]

    # ── 출력 경로 ─────────────────────────────────────────────────────
    OUT_DIR = r"G:\공유 드라이브\BSG_DFC_result\combined\t95"
    os.makedirs(OUT_DIR, exist_ok=True)
    OUT_CSV = os.path.join(OUT_DIR, "t95_before_after_delta_combined.csv")

    all_rows = []

    for car_model, folder_before, summary_csv, use_mp in MODELS:
        print(f"\n==============================")
        print(f"[INFO] car_model = {car_model}")
        print(f"[INFO] before 폴더 = {folder_before}")
        print(f"[INFO] delta summary = {summary_csv}")
        print(f"[INFO] use_mp = {use_mp}")

        # ① before t95 계산 (모델별로 멀티 on/off)
        before_dict = compute_t95_by_file(folder_before, use_mp=use_mp, max_workers=8 if use_mp else None)
        print(f"[INFO] {car_model}: before t95 파일 수 = {len(before_dict)}")

        # ② delta_t95_event_sum_h 맵 로딩
        delta_map = load_delta_t95_map(summary_csv)

        # ③ 매칭 & after 계산
        keys_before = set(before_dict.keys())
        keys_delta  = set(delta_map.keys())

        missing_delta = sorted(keys_before - keys_delta)
        if missing_delta:
            print(f"[WARN] {car_model}: delta_t95_event_sum_h 없는 키 {len(missing_delta)}개 (예시 5개) → after NaN 처리")
            for k in missing_delta[:5]:
                print("   -", k)

        extra_delta = sorted(keys_delta - keys_before)
        if extra_delta:
            print(f"[INFO] {car_model}: before t95에는 없고 delta만 있는 키 {len(extra_delta)}개 (예시 5개)")
            for k in extra_delta[:5]:
                print("   -", k)

        for key, t_before in before_dict.items():
            delta = delta_map.get(key, np.nan)

            if pd.isna(delta):
                t_after = np.nan
                delta_t = np.nan
            else:
                # delta_t95 는 'before - after' 로 정의되므로
                # after = before - delta_t95
                t_after = t_before - float(delta)
                delta_t = float(delta)

            all_rows.append({
                "car_model":    car_model,
                "file":         key,           # 접미사 제거된 파일명 (bms_..._YYYY-MM)
                "t95_before_h": t_before,
                "t95_after_h":  t_after,
                "delta_t_h":    delta_t,       # == delta_t95_event_sum_h
            })

    # ── DataFrame 생성 & 저장 ─────────────────────────────────────────
    if not all_rows:
        raise SystemExit("[INFO] 생성된 row가 없습니다. 경로/입력 파일을 확인하세요.")

    df_out = pd.DataFrame(
        all_rows,
        columns=["car_model", "file", "t95_before_h", "t95_after_h", "delta_t_h"],
    )

    df_out.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    print("\n===================================")
    print(f"[SAVE] t95 결과 CSV → {OUT_CSV}")
    print(f"[INFO] 총 행 수 = {len(df_out)}")
    print("미리보기:")
    print(df_out.head())
