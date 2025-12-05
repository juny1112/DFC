import os
import pandas as pd
import numpy as np
from tqdm import tqdm

# ─────────────────────────────────────────────────────────────
# R_aftercharg (R_fc) 구간 시간 계산
#   - R_aftercharg == 1 인 구간에서만 dt를 합산
#   - 연속 블록별 duration을 구해서 (h 단위) 배열로 반환
# ─────────────────────────────────────────────────────────────
def rfc_segment_durations_hours(df: pd.DataFrame, col_time='time', col_flag='R_aftercharg'):
    """
    입력: time, R_aftercharg 컬럼이 포함된 DataFrame
    반환: 각 R_aftercharg=1 연속 구간의 duration[h] 리스트 (없으면 빈 리스트)
    """
    if col_time not in df.columns or col_flag not in df.columns or len(df) < 2:
        return []

    # time 파싱 & 정렬
    t = pd.to_datetime(df[col_time], errors='coerce')
    flag = pd.to_numeric(df[col_flag], errors='coerce').fillna(0)

    # 시간 정렬(혹시라도 뒤섞여 있을 수 있으므로)
    tmp = pd.DataFrame({'time': t, 'flag': flag})
    tmp = tmp.dropna(subset=['time']).sort_values('time').reset_index(drop=True)

    t = tmp['time']
    flag = tmp['flag'] != 0   # 0이 아니면 True (1, 1.0 등)

    # 다음 시점과의 시간 차이(dt)
    t_next = t.shift(-1)
    dt = t_next - t

    # dt가 존재하고, 해당 구간의 시작이 R_aftercharg=1인 경우만 유효
    valid = flag & dt.notna()
    if not valid.any():
        return []

    # 연속 블록 id 생성
    #  - valid가 True인 구간만 대상으로,
    #  - (valid가 True이고, 이전이 False인 지점)을 블록 시작으로 보고 누적 카운트
    start_flag = valid & (~valid.shift(fill_value=False))
    seg_id = start_flag.cumsum()  # 0,1,1,1,2,2,...

    # valid 위치에 대해서만 그룹핑
    dt_valid = dt[valid]
    seg_valid = seg_id[valid]

    # 블록별 dt 합산 (Timedelta 형식)
    seg_sum = dt_valid.groupby(seg_valid).sum()

    # Timedelta → 시간(h)
    durations_h = seg_sum / np.timedelta64(1, 'h')
    return durations_h.to_numpy().tolist()


# ─────────────────────────────────────────────────────────────
# 파일 하나에 대해 R_fc mean/std/sum 계산
# ─────────────────────────────────────────────────────────────
def compute_rfc_stats_for_file(file_path: str):
    """
    단일 CSV 파일에 대해,
    R_aftercharg 구간별 duration[h]을 구하고
    mean / std / sum(h)를 반환.
    (구간이 없으면 모두 np.nan)
    """
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        print(f"[SKIP] {os.path.basename(file_path)} - read error: {e}")
        return np.nan, np.nan, np.nan

    if 'R_aftercharg' not in df.columns or 'time' not in df.columns:
        # 해당 파일엔 라벨이 없거나 time이 없으면 스킵
        return np.nan, np.nan, np.nan

    durations = rfc_segment_durations_hours(df, col_time='time', col_flag='R_aftercharg')
    if len(durations) == 0:
        return np.nan, np.nan, np.nan

    durations_arr = np.array(durations, dtype='float64')
    mean_h = float(np.nanmean(durations_arr))
    std_h  = float(np.nanstd(durations_arr, ddof=0))  # 모집단 표준편차
    sum_h  = float(np.nansum(durations_arr))

    return mean_h, std_h, sum_h


# ─────────────────────────────────────────────────────────────
# 루트 폴더(차종별) 아래의 CSV들을 전부 훑어서 통계 계산
#   - 재귀적으로 *.csv 탐색
#   - 각 파일에 대해 R_fc_mean, R_fc_std, R_fc_sum 계산
# ─────────────────────────────────────────────────────────────
def scan_root_for_rfc(root_dir: str, car_type: str):
    """
    root_dir 아래의 모든 .csv를 재귀 탐색하면서
    R_aftercharg 통계를 계산.
    반환: records 리스트 (사전들)
    """
    records = []

    if not os.path.isdir(root_dir):
        print(f"[WARN] not a directory: {root_dir}")
        return records

    for cur_dir, _, files in os.walk(root_dir):
        csv_files = [f for f in files if f.lower().endswith('.csv')]
        if not csv_files:
            continue

        # tqdm으로 현재 디렉토리 진행 상황 표시
        for fn in tqdm(csv_files, desc=f"{car_type} - {os.path.relpath(cur_dir, root_dir)}"):
            file_path = os.path.join(cur_dir, fn)

            mean_h, std_h, sum_h = compute_rfc_stats_for_file(file_path)

            # 모두 NaN이면 의미 없는 파일 → 그래도 한 줄 남겨두고 싶으면 조건 제거
            if np.isnan(mean_h) and np.isnan(std_h) and np.isnan(sum_h):
                continue

            records.append({
                'car_type': car_type,
                'file_name': fn,
                'R_fc_mean_h': mean_h,
                'R_fc_std_h': std_h,
                'R_fc_sum_h': sum_h,
            })

    return records


# ─────────────────────────────────────────────────────────────
# 메인 실행부
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # 차종별 루트 경로
    ROOTS = [
        ("Ioniq5", r"Z:\SamsungSTF\Processed_Data\DFC\Ioniq5\R_parsing_완충후이동주차"),
        ("EV6",    r"Z:\SamsungSTF\Processed_Data\DFC\EV6\R_parsing_완충후이동주차"),
    ]

    all_records = []

    for car, root in ROOTS:
        recs = scan_root_for_rfc(root, car_type=car)
        all_records.extend(recs)

    if not all_records:
        print("[INFO] 유효한 R_aftercharg 데이터를 가진 파일을 찾지 못했습니다.")
    else:
        df_out = pd.DataFrame(all_records)

        # 필요하면 파일명 기준으로 정렬
        df_out = df_out.sort_values(['car_type', 'file_name']).reset_index(drop=True)

        # 저장 위치는 원하는 대로 바꿔도 됨
        out_dir = r"G:\공유 드라이브\BSG_DFC_result\combined\DFC_완충후이동주차"
        os.makedirs(out_dir, exist_ok=True)
        out_csv = os.path.join(out_dir, "R_fc_time_by_file.csv")

        df_out.to_csv(out_csv, index=False, encoding='utf-8-sig')
        print(f"[SAVE] R_fc stats CSV -> {out_csv}")
