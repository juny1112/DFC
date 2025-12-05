#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor

# ─────────────────────────────────────────────────────────────
# 전역 설정(토글)
# ─────────────────────────────────────────────────────────────
FULLCHARGE_PARKING_MODE = True     # 완충 후 이동주차 모드 on/off
MERGE_GAP_MINUTES       = 10       # 흡수 기준 간격(분)

# ─────────────────────────────────────────────────────────────
# 0) 유틸: 연속된 1 블록(start, end) 구하기
# ─────────────────────────────────────────────────────────────
def get_blocks(series: pd.Series):
    """
    series: 0/1 또는 bool 시리즈
    return: [(start_idx, end_idx), ...]  (end 포함)
    """
    arr = (pd.to_numeric(series, errors="coerce").fillna(0) != 0).to_numpy()
    n = len(arr)
    blocks = []
    in_block = False
    start = None

    for i, v in enumerate(arr):
        if v and not in_block:
            in_block = True
            start = i
        elif not v and in_block:
            blocks.append((start, i - 1))
            in_block = False
    if in_block:
        blocks.append((start, n - 1))
    return blocks


# ─────────────────────────────────────────────────────────────
# 1) 완전충전/부분충전 라벨링 (rest_charging)
# ─────────────────────────────────────────────────────────────
def rest_charging(data: pd.DataFrame):
    """
    data: time, soc, charging, rest ... 포함된 원본 DataFrame
    return: (라벨 추가된 DataFrame, full 구간 인덱스 리스트)
    """
    # 충전구간 경계 인덱스
    charg = []
    if data.loc[0, 'charging'] == 1:
        charg.append(0)
    for i in range(len(data) - 1):
        if data.loc[i, 'charging'] != data.loc[i + 1, 'charging']:
            charg.append(i + 1)
    if data.loc[len(data) - 1, 'charging'] == 1:
        charg.append(len(data) - 1)

    # 완전충전구간 (full)
    full = []
    for i in range(len(charg) - 1):
        if (
            data.loc[charg[i], 'charging'] == 1
            and any(data.loc[charg[i + 1]-3:charg[i + 1]+6, 'soc'] >= 95)
        ):
            full.append(charg[i])
            full.append(charg[i + 1])

    full = sorted(set(full))

    # 완전충전 / 부분충전 라벨
    data['R_charg'] = 0
    data['R_partial_charg'] = 0

    # 전체 충전구간을 partial로
    for i in range(0, len(charg) - 1, 2):
        if data.loc[charg[i], 'charging'] == 1:
            data.loc[charg[i]:charg[i + 1] - 1, 'R_partial_charg'] = 1

    # full 구간을 full로 바꾸고 partial 해제
    for i in range(0, len(full) - 1, 2):
        if data.loc[full[i], 'charging'] == 1:
            data.loc[full[i]:full[i + 1] - 1, 'R_partial_charg'] = 0
            data.loc[full[i]:full[i + 1] - 1, 'R_charg'] = 1

    return data, full


# ─────────────────────────────────────────────────────────────
# 2) 완충 후 rest → R_aftercharg (히터 세분화 포함)
# ─────────────────────────────────────────────────────────────
def rest_aftercharg(data: pd.DataFrame, full):
    """
    R_charg 라벨이 있는 data 에서
    R_aftercharg (R_fc) 라벨을 생성 + 히터 세분화 적용
    """
    data = data.copy()
    data['R_aftercharg'] = 0
    data['time'] = pd.to_datetime(
        data['time'],
        format='%Y-%m-%d %H:%M:%S',
        errors='coerce'
    )

    # 1단계: 완충 후 rest → R_aftercharg
    for i in range(len(data) - 1):
        if (
            data.loc[i, 'R_charg'] == 1 and
            data.loc[i + 1, 'R_charg'] == 0 and
            data.loc[i + 1, 'rest'] == 1
        ):
            j = i
            while j < len(data) and data.loc[j, 'rest'] == 1:
                data.at[j, 'R_aftercharg'] = 1
                j += 1

    # 2단계: 출발 직전 히터 구간 세분화
    limit_time = pd.Timedelta(minutes=5)
    limit_rest_gap = pd.Timedelta(minutes=30)
    has_curr = 'pack_current' in data.columns

    for i in range(len(data) - 1):
        if (
            data.loc[i, 'R_charg'] == 1 and
            data.loc[i + 1, 'R_charg'] == 1 and
            data.loc[i + 1, 'time'] - data.loc[i, 'time'] > limit_time
        ):
            j = i
            while j < len(data) and data.loc[j, 'R_charg'] == 1:
                j += 1

            if j <= i + 1:
                continue

            rest_gap = data.loc[j - 1, 'time'] - data.loc[i + 1, 'time']

            heater_start_idx = i + 1
            soc_start = data.loc[heater_start_idx, 'soc']
            if soc_start < 95:
                continue

            if has_curr:
                heater_slice = data.loc[i + 1:j - 1]
                pc = heater_slice['pack_current']
                if (pc <= 0).all():
                    continue

            if rest_gap <= limit_rest_gap:
                data.loc[i:j - 1, 'R_aftercharg'] = 1
                k = i + 1
                while k < len(data) and data.loc[k, 'R_charg'] == 1:
                    data.at[k, 'R_charg'] = 0
                    data.at[k, 'R_aftercharg'] = 1
                    k += 1

    return data


# ─────────────────────────────────────────────────────────────
# 3) R_uncharg 라벨링 + gap_fc / gap_cf 계산
# ─────────────────────────────────────────────────────────────
def rest_uncharg_and_gaps(data: pd.DataFrame,
                          enable_merge: bool | None = None):
    """
    - R_uncharg 기본 라벨링 (벡터화)
    - (옵션) 완충 후 이동주차 모드 로직 적용
    - 동시에 gap_fc, gap_cf 리스트 수집:
        gap_fc:  각 R_fc(R_aftercharg) 블록 end ~ 다음 R_uncharg_start 시간차
        gap_cf:  각 C_f(R_charg)       블록 end ~ 다음 R_uncharg_start 시간차
      (단위: 시간[h], 10분 조건은 gap 기록에는 적용하지 않음)
    """
    data = data.copy()

    # 1) 기본 R_uncharg 라벨 (벡터화)
    cond_uncharg = (
        (data['rest'] == 1) &
        (data['R_charg'] == 0) &
        (data['R_partial_charg'] == 0) &
        (data['R_aftercharg'] == 0)
    )
    data['R_uncharg'] = 0
    data.loc[cond_uncharg, 'R_uncharg'] = 1

    # 2) 모드 on/off
    if enable_merge is None:
        enable = FULLCHARGE_PARKING_MODE
    else:
        enable = enable_merge

    if not pd.api.types.is_datetime64_any_dtype(data['time']):
        data['time'] = pd.to_datetime(
            data['time'],
            format='%Y-%m-%d %H:%M:%S',
            errors='coerce'
        )

    t = data['time']
    time_limit = pd.Timedelta(minutes=MERGE_GAP_MINUTES)
    n = len(data)

    after_blocks   = get_blocks(data['R_aftercharg'])
    uncharg_blocks = get_blocks(data['R_uncharg'])
    charg_blocks   = get_blocks(data['R_charg'])

    gap_fc_list = []
    gap_cf_list = []

    # (a) R_fc_end 기준 gap 계산 + (옵션) 흡수 로직
    for (_, aft_end) in after_blocks:
        next_unc = next(
            ((u_start, u_end) for (u_start, u_end) in uncharg_blocks
             if u_start > aft_end),
            None
        )
        if not next_unc:
            continue

        unc_start, unc_end = next_unc
        if aft_end >= n or unc_start >= n:
            continue

        dt = t.iloc[unc_start] - t.iloc[aft_end]
        if pd.isna(dt):
            continue

        gap_h = dt / np.timedelta64(1, 'h')
        if gap_h >= 0:
            gap_fc_list.append(float(gap_h))

        if enable and dt <= time_limit:
            data.loc[aft_end + 1:unc_end, 'R_aftercharg'] = 1
            data.loc[aft_end + 1:unc_end, 'R_uncharg']    = 0

    # (b) C_f_end 기준 gap 계산 + (옵션) 흡수 로직
    for (_, ch_end) in charg_blocks:
        next_unc = next(
            ((u_start, u_end) for (u_start, u_end) in uncharg_blocks
             if u_start > ch_end),
            None
        )
        if not next_unc:
            continue

        unc_start, unc_end = next_unc
        if ch_end >= n or unc_start >= n:
            continue

        dt = t.iloc[unc_start] - t.iloc[ch_end]
        if pd.isna(dt):
            continue

        gap_h = dt / np.timedelta64(1, 'h')
        if gap_h >= 0:
            gap_cf_list.append(float(gap_h))

        if enable and dt <= time_limit:
            data.loc[ch_end:unc_end, 'R_aftercharg'] = 1
            data.loc[ch_end:unc_end, 'R_uncharg']    = 0

    return data, gap_fc_list, gap_cf_list


# ─────────────────────────────────────────────────────────────
# 4) gap 리스트 → 통계
# ─────────────────────────────────────────────────────────────
def summarize_gaps(gaps):
    """
    gaps: [float, ...] (단위: h)
    return: mean, std, sum, count
    """
    if not gaps:
        return np.nan, np.nan, np.nan, 0

    arr = np.array(gaps, dtype='float64')
    mean_h = float(np.nanmean(arr))
    std_h  = float(np.nanstd(arr, ddof=0))
    sum_h  = float(np.nansum(arr))
    cnt    = int(np.sum(~np.isnan(arr)))
    return mean_h, std_h, sum_h, cnt


# ─────────────────────────────────────────────────────────────
# 5) 파일 하나에서 gap_fc / gap_cf 통계 계산 (pyarrow 사용)
# ─────────────────────────────────────────────────────────────
def compute_gap_stats_for_file(file_path: str,
                               enable_merge: bool | None = None):
    """
    입력: CR_parsing CSV (time, soc, charging, rest, pack_current 등)
    출력: (파일명 기준) gap_fc / gap_cf 통계 dict (에러 시 None)
    """
    # 헤더만 읽어서 실제 있는 컬럼 확인
    try:
        header_df = pd.read_csv(file_path, nrows=0)
        cols_available = set(header_df.columns)
    except Exception as e:
        print(f"[SKIP] {os.path.basename(file_path)} - header read error: {e}")
        return None

    usecols = ['time', 'soc', 'charging', 'rest', 'pack_current']
    usecols = [c for c in usecols if c in cols_available]

    # 필수 컬럼 체크
    if not {'time', 'soc', 'charging', 'rest'}.issubset(set(usecols)):
        return None

    try:
        df = pd.read_csv(file_path, usecols=usecols)
    except Exception as e:
        print(f"[SKIP] {os.path.basename(file_path)} - read error: {e}")
        return None

    df = df.sort_values('time').reset_index(drop=True)

    # 1) 완전충전/부분충전 라벨링
    df_labeled, full = rest_charging(df)

    # 2) 완충 후 + 히터 세분화
    df_labeled = rest_aftercharg(df_labeled, full)

    # 3) R_uncharg + gap_fc / gap_cf
    df_labeled, gap_fc_list, gap_cf_list = rest_uncharg_and_gaps(
        df_labeled,
        enable_merge=enable_merge
    )

    fc_mean, fc_std, fc_sum, fc_cnt = summarize_gaps(gap_fc_list)
    cf_mean, cf_std, cf_sum, cf_cnt = summarize_gaps(gap_cf_list)

    return {
        'file_name': os.path.basename(file_path),
        'file_path': file_path,
        'R_fc_gap_mean_h': fc_mean,
        'R_fc_gap_std_h':  fc_std,
        'R_fc_gap_sum_h':  fc_sum,
        'R_fc_gap_count':  fc_cnt,
        'C_f_gap_mean_h':  cf_mean,
        'C_f_gap_std_h':   cf_std,
        'C_f_gap_sum_h':   cf_sum,
        'C_f_gap_count':   cf_cnt,
    }


# ─────────────────────────────────────────────────────────────
# 6) 멀티프로세싱 워커 (top-level이어야 Windows에서 pickle 가능)
# ─────────────────────────────────────────────────────────────
def worker_gap_stats(args):
    """
    멀티프로세싱용 래퍼.
    args: (file_path, enable_merge)
    """
    file_path, enable_merge = args
    return compute_gap_stats_for_file(file_path, enable_merge=enable_merge)


# ─────────────────────────────────────────────────────────────
# 7) 차종별 CR_parsing 폴더 스캔 + 멀티프로세싱
# ─────────────────────────────────────────────────────────────
def scan_root_for_gap_stats(root_dir: str, car_type: str,
                            enable_merge: bool | None = None,
                            use_multiprocessing: bool = True,
                            n_workers: int | None = None):
    records = []
    if not os.path.isdir(root_dir):
        print(f"[WARN] not a directory: {root_dir}")
        return records

    # 전체 파일 수집
    file_paths = []
    for cur_dir, _, files in os.walk(root_dir):
        csv_files = [f for f in files if f.lower().endswith('.csv')]
        for fn in csv_files:
            file_paths.append(os.path.join(cur_dir, fn))

    if not file_paths:
        return records

    # 멀티 or 싱글
    if use_multiprocessing:
        # 코어 수 제한 (네트워크 드라이브 고려해서 상한 8개)
        if n_workers is None:
            cpu_cnt = os.cpu_count() or 4
            n_workers = min(cpu_cnt, 8)

        args_list = [(fp, enable_merge) for fp in file_paths]

        with ProcessPoolExecutor(max_workers=n_workers) as ex:
            results = list(
                tqdm(
                    ex.map(worker_gap_stats, args_list),
                    total=len(args_list),
                    desc=f"{car_type} (mp, {n_workers} workers)"
                )
            )
    else:
        results = []
        for fp in tqdm(file_paths, desc=f"{car_type} (single)"):
            results.append(compute_gap_stats_for_file(fp, enable_merge=enable_merge))

    # 결과 정리
    for rec in results:
        if rec is None:
            continue
        rec['car_type'] = car_type
        records.append(rec)

    return records


# ─────────────────────────────────────────────────────────────
# 메인 실행부
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    ROOTS = [
        ("EV6",    r"Z:\SamsungSTF\Processed_Data\DFC\EV6\CR_parsing"),
        ("Ioniq5", r"Z:\SamsungSTF\Processed_Data\DFC\Ioniq5\CR_parsing"),
    ]

    all_recs = []
    for car, root in ROOTS:
        recs = scan_root_for_gap_stats(
            root_dir=root,
            car_type=car,
            enable_merge=None,          # None → FULLCHARGE_PARKING_MODE / MERGE_GAP_MINUTES 사용
            use_multiprocessing=True,   # 병렬 처리 on
            n_workers=4             # None → min(os.cpu_count(), 8)
        )
        all_recs.extend(recs)

    if not all_recs:
        print("[INFO] 유효한 gap 데이터를 계산하지 못했습니다.")
    else:
        df_out = pd.DataFrame(all_recs)
        df_out = df_out.sort_values(['car_type', 'file_name']).reset_index(drop=True)

        out_dir = r"G:\공유 드라이브\BSG_DFC_result\combined\DFC_완충후이동주차"
        os.makedirs(out_dir, exist_ok=True)
        out_csv = os.path.join(out_dir, "R_fc_C_f_gap_stats_from_CR_parsing.csv")
        df_out.to_csv(out_csv, index=False, encoding='utf-8-sig')
        print(f"[SAVE] gap stats CSV -> {out_csv}")
