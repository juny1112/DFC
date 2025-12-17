import pandas as pd
import numpy as np
import os
from tqdm import tqdm
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

# ─────────────────────────────────────────────────────────────
# 충전후 구간 불필요한 데이터 삭제 (벡터화, 인덱스 안전)
# ─────────────────────────────────────────────────────────────
def remove_consecutive_ones(data):
    if 'R_aftercharg' not in data.columns:
        return data

    s = data['R_aftercharg'].fillna(0).astype(int)
    grp = (s != s.shift(fill_value=s.iloc[0])).cumsum()          # 연속 구간 라벨
    group_sizes = grp.map(grp.value_counts())                    # 각 행이 속한 구간 길이
    pos_from_start = data.groupby(grp).cumcount()
    pos_from_end = data.iloc[::-1].groupby(grp.iloc[::-1]).cumcount()[::-1]

    # 1이 3개 이상인 구간은 첫/마지막만 보존, 나머지 제거
    keep = (
        (s == 0) |
        ((s == 1) & (group_sizes < 3)) |
        ((s == 1) & (group_sizes >= 3) & ((pos_from_start == 0) | (pos_from_end == 0)))
    )
    return data.loc[keep].reset_index(drop=True)

# ─────────────────────────────────────────────────────────────
# DFC 알고리즘 적용 + 이벤트 통계(선택)
#   - 실제 time shift가 발생한 이벤트들의 delayed_time을 수집
#   - collect_stats=True일 때 (events 리스트, 요약 dict)도 함께 반환
# ─────────────────────────────────────────────────────────────
def DFC(data, collect_stats=False):
    # R_aftercharg 연속 1 구간 정리
    data = remove_consecutive_ones(data)

    # time / soc 파싱 안전화
    if 'time' in data.columns and not pd.api.types.is_datetime64_any_dtype(data['time']):
        data['time'] = pd.to_datetime(data['time'], format='%Y-%m-%d %H:%M:%S', errors='coerce')
    if 'soc' in data.columns and not pd.api.types.is_numeric_dtype(data['soc']):
        data['soc'] = pd.to_numeric(data['soc'], errors='coerce')

    # 넘파이 배열 미리 뽑기 (속도용)
    soc_arr = data['soc'].to_numpy()
    r_charg_arr = data['R_charg'].fillna(0).astype(int).to_numpy()
    r_after_arr = data['R_aftercharg'].fillna(0).astype(int).to_numpy()

    n = len(data)

    # ─ 충전 구간(start, end) 한 번에 찾기 ──────────────────────────────
    # transitions_ch: 0→1 이면 start, 1→0 이면 end+1
    transitions_ch = np.diff(np.r_[0, r_charg_arr, 0])
    cstarts = np.where(transitions_ch == 1)[0]
    cends   = np.where(transitions_ch == -1)[0] - 1

    # aftercharge 구간(start, end) 한 번에 찾기
    transitions_ac = np.diff(np.r_[0, r_after_arr, 0])
    astarts = np.where(transitions_ac == 1)[0]
    aends   = np.where(transitions_ac == -1)[0] - 1

    astarts.sort(); aends.sort(); cstarts.sort(); cends.sort()

    # cend±1 안에 aftercharg=1 있는지 빠르게 체크
    def any_after_near(idx: int) -> bool:
        lo = max(0, idx - 1)
        hi = min(n - 1, idx + 1)
        return (r_after_arr[lo:hi + 1] == 1).any()

    # DFC 적용 충전구간 (start, end) 쌍
    dfc_charg = []
    for cs, ce in zip(cstarts, cends):
        if any_after_near(ce):
            dfc_charg.append(cs)
            dfc_charg.append(ce)

    # ─ 79→80% cross 지점 전체를 한 번에 계산 ──────────────────────────
    # soc[i] < 80 and soc[i+1] == 80 인 지점의 "다음 인덱스(i+1)"들
    cross_mask = (soc_arr[:-1] < 80) & (soc_arr[1:] == 80)
    cross_idxs = np.where(cross_mask)[0] + 1      # delay 시작 후보 인덱스들

    # ─ 충전단계2: 80 기준으로 delay_start, charge_end 쌍 만들기 ────────
    charg_2_pairs = []   # (delay_start, charge_end)
    for j in range(0, len(dfc_charg) - 1, 2):
        start_j, end_j = dfc_charg[j], dfc_charg[j + 1]

        # 이 충전구간 안에 있는 cross_idxs 중 첫 번째를 사용
        in_seg = (cross_idxs >= start_j) & (cross_idxs <= end_j)
        candidates = cross_idxs[in_seg]

        if len(candidates) > 0:
            dstart = int(candidates[0])
            charg_2_pairs.append((dstart, end_j))
        else:
            # 79→80이 없고 시작 SOC가 80 이상이면 시작 시점부터 지연
            if soc_arr[start_j] >= 80:
                charg_2_pairs.append((start_j, end_j))
            # 그 외에는 DFC 적용하지 않음

    dfc_events = []
    rows_to_drop = []   # 나중에 한 번에 drop 할 (lo, hi) 범위들
    t_margin = pd.Timedelta(hours=1)

    if len(charg_2_pairs) and ('R_charg' in data.columns) and ('R_aftercharg' in data.columns):
        # 위에서 이미 cstarts, astarts, aends 계산해둠

        for dstart, cend in charg_2_pairs:
            # 다음 충전 시작 시점
            pos_c = np.searchsorted(cstarts, cend + 1, side='left')
            next_charge_start = cstarts[pos_c] if pos_c < len(cstarts) else None

            # cend 이후(또는 같음)에서 시작하는 aftercharge 구간 찾기
            pos_a = np.searchsorted(astarts, cend, side='left')
            if pos_a >= len(astarts):
                continue
            astart = int(astarts[pos_a])

            # 다음 충전 시작 전에만 유효
            if (next_charge_start is not None) and (astart >= next_charge_start):
                continue

            aend = int(aends[pos_a])  # 동일 aftercharge 세그먼트 끝

            # 시간 계산
            t0 = data.loc[cend, 'time']
            t1 = data.loc[aend, 'time']
            if pd.isna(t0) or pd.isna(t1):
                continue

            delayed_time = (t1 - t0 - t_margin)
            if (delayed_time <= pd.Timedelta(0)) or (dstart + 1 > cend):
                continue

            # 이벤트 정보 기록
            dfc_events.append({
                'charge_end_idx': int(cend),
                'after_end_idx': int(aend),
                'charge_end_time': t0,
                'after_end_time': t1,
                'delay_hours': delayed_time.total_seconds() / 3600.0
            })

            # ─ SOC(cend) > SOC(aend) 이면 부분 shift + tail 삭제 ──────
            soc_cend = soc_arr[cend]
            soc_aend = soc_arr[aend]

            if pd.notna(soc_cend) and pd.notna(soc_aend) and (soc_cend > soc_aend):
                # dstart~cend 구간에서 soc == soc_aend 인 지점들을 넘파이로 탐색
                seg_soc = soc_arr[dstart:cend + 1]  # [dstart, cend]
                rel_idx = np.where(seg_soc == soc_aend)[0]

                if len(rel_idx) > 0:
                    # 마지막으로 soc == soc_aend 가 되는 절대 인덱스
                    cut_idx = int(dstart + rel_idx[-1])

                    # 1) dstart+1 ~ cut_idx 까지만 time shift
                    data.loc[dstart + 1:cut_idx, 'time'] = (
                        data.loc[dstart + 1:cut_idx, 'time'] + delayed_time
                    )

                    # 2) cut_idx+1 ~ cend 는 나중에 drop
                    if cut_idx + 1 <= cend:
                        rows_to_drop.append((cut_idx + 1, cend))
                else:
                    # 매칭되는 SOC가 없으면 기존 로직 유지
                    data.loc[dstart + 1:cend, 'time'] = (
                        data.loc[dstart + 1:cend, 'time'] + delayed_time
                    )
            else:
                # SOC(cend) <= SOC(aend) 이면 기존 로직 그대로
                data.loc[dstart + 1:cend, 'time'] = (
                    data.loc[dstart + 1:cend, 'time'] + delayed_time
                )

    # ─ rows_to_drop에 모인 구간들을 한 번에 삭제 + 인덱스 리셋 ─
    if rows_to_drop:
        all_drop_idx = []
        for lo, hi in rows_to_drop:
            all_drop_idx.extend(range(lo, hi + 1))
        if all_drop_idx:
            data = data.drop(index=sorted(set(all_drop_idx))).reset_index(drop=True)

    # 세분화 컬럼 삭제(존재할 때만)
    columns_to_delete = ['R_charg', 'R_partial_charg', 'R_aftercharg', 'R_uncharg']
    data = data.drop(columns=[c for c in columns_to_delete if c in data.columns], errors='ignore')

    if not collect_stats:
        return data

    # ─ 파일 단위 요약 통계 (delta_t95_event 네이밍 고정) ───────────────
    delays = pd.to_numeric(
        pd.Series([e['delay_hours'] for e in dfc_events], dtype='float64'),
        errors='coerce'
    ).dropna()
    N = int(len(delays))
    mean = float(delays.mean()) if N > 0 else 0.0
    std  = float(delays.std(ddof=1)) if N > 1 else 0.0
    summ = float(delays.sum()) if N > 0 else 0.0

    stats = {
        'delta_t95_event_N': N,
        'delta_t95_event_mean_h': mean,
        'delta_t95_event_std_h': std,
        'delta_t95_event_sum_h': summ
    }
    return data, dfc_events, stats


# ─────────────────────────────────────────────────────────────
# 파일 하나 돌리기 (저장 on/off + 요약 리턴)
# ─────────────────────────────────────────────────────────────
def process_DFC_file(file_path, save_path=None, collect_stats=True, write_output=True):
    """
    write_output=False 이면 변환된 DFC CSV를 저장하지 않고 통계만 반환.
    """
    data = pd.read_csv(file_path)
    if collect_stats:
        result = DFC(data, collect_stats=True)
        data = result[0]
        _events = result[1]
        stats = result[2]
    else:
        data = DFC(data, collect_stats=False)
        _events, stats = [], None

    if write_output:
        if save_path is None:
            base, ext = os.path.splitext(file_path)
            save_path = f"{base.rstrip('_r')}_DFC{ext}"
        data.to_csv(save_path, index=False)

    return data, stats

# ─────────────────────────────────────────────────────────────
# 요약 CSV (delta_t95_event 네이밍 고정)
# ─────────────────────────────────────────────────────────────
SUMMARY_COLUMNS = [
    'file_stem', 'id_token', 'ym',
    'delta_t95_event_N',
    'delta_t95_event_mean_h',
    'delta_t95_event_std_h',
    'delta_t95_event_sum_h',
]

def parse_id_token_and_ym(p: Path):
    # 파일명 예: bms_01241228021_2023-02_r.csv → id_token=01241228021, ym=2023-02
    id_token, ym = "unknown", "0000-00"
    parts = p.stem.split("_")
    if len(parts) >= 3:
        id_token = parts[1]
        ym = parts[2]
    return id_token, ym

def _collect_input_files(input_folder, pattern="*.csv"):
    input_dir = Path(input_folder)
    files = [p for p in sorted(input_dir.glob(pattern))]
    return files

def process_DFC_folder(input_folder, output_folder, summary_csv_path=None,
                       pattern="*.csv", write_outputs=True, skip_existing=True):
    files = _collect_input_files(input_folder, pattern=pattern)
    return _process_files_and_summary(files, output_folder, summary_csv_path,
                                      write_outputs=write_outputs, skip_existing=skip_existing)


def process_DFC_folder_slice(input_folder, output_folder, start_idx=0, end_idx=None,
                             summary_csv_path=None, pattern="*.csv", write_outputs=True,
                             skip_existing=True):
    files = _collect_input_files(input_folder, pattern=pattern)
    if end_idx is None:
        sel = files[start_idx:]
    else:
        sel = files[start_idx:end_idx+1]  # inclusive
    return _process_files_and_summary(sel, output_folder, summary_csv_path,
                                      write_outputs=write_outputs, skip_existing=skip_existing)


def _process_files_and_summary(files, output_folder, summary_csv_path=None,
                               write_outputs=True, skip_existing=True):
    output_dir = Path(output_folder)
    if write_outputs:
        output_dir.mkdir(parents=True, exist_ok=True)
    summary_rows = []

    for p in tqdm(files, desc='Processing Files'):
        try:
            save_path = None
            if write_outputs:
                out_name = p.name.replace('_r.csv', '_DFC.csv')
                save_path = output_dir / out_name

                # ── 여기! 결과가 이미 있으면 스킵 ─────────────────────────
                if skip_existing and save_path.exists():
                    # 필요하면 로그만 남기고 요약에서는 제외(빠르게 돌리기 목적)
                    # tqdm.write(f"[skip] {out_name}")
                    continue
                # ─────────────────────────────────────────────────────────

            _, stats = process_DFC_file(
                str(p),
                save_path=str(save_path) if save_path else None,
                collect_stats=True,
                write_output=write_outputs
            )

            # 요약 수집
            id_token, ym = parse_id_token_and_ym(p)
            if stats is None:
                stats = {
                    'delta_t95_event_N': 0,
                    'delta_t95_event_mean_h': 0.0,
                    'delta_t95_event_std_h': 0.0,
                    'delta_t95_event_sum_h': 0.0
                }

            summary_rows.append({
                'file_stem': p.stem,
                'id_token': id_token,
                'ym': ym,
                'delta_t95_event_N': stats['delta_t95_event_N'],
                'delta_t95_event_mean_h': stats['delta_t95_event_mean_h'],
                'delta_t95_event_std_h': stats['delta_t95_event_std_h'],
                'delta_t95_event_sum_h': stats['delta_t95_event_sum_h'],
            })

        except Exception as e:
            print(f"Error processing {p.name}: {str(e)}")
            id_token, ym = parse_id_token_and_ym(p)
            summary_rows.append({
                'file_stem': p.stem,
                'id_token': id_token,
                'ym': ym,
                'delta_t95_event_N': 0,
                'delta_t95_event_mean_h': 0.0,
                'delta_t95_event_std_h': 0.0,
                'delta_t95_event_sum_h': 0.0,
            })
            continue

    # 요약 CSV 저장
    if summary_csv_path is None:
        summary_csv_path = Path(output_folder) / "dfc_summary.csv"
    summary_df = pd.DataFrame(summary_rows, columns=SUMMARY_COLUMNS)
    summary_df.to_csv(summary_csv_path, index=False, encoding='utf-8-sig')
    print(f"\n✅ 요약 저장: {summary_csv_path} (rows={len(summary_df)})")

    return summary_df


def _dfc_one_file_job(args):
    """
    (in_path, out_path_or_None, write_outputs, skip_existing) 받아서
    - DFC 처리(옵션으로 파일 저장)
    - stats dict 반환 (요약용)
    """
    in_path, out_path, write_outputs, skip_existing = args
    p = Path(in_path)

    # 출력 파일이 있고 스킵이면: 통계도 스킵할지 정책 선택 필요
    # 지금은 "스킵되면 요약에서도 제외" = 기존 단일프로세스와 동일 동작
    if write_outputs and out_path and skip_existing and Path(out_path).exists():
        return ("skip", p.stem, None)

    try:
        _, stats = process_DFC_file(
            str(p),
            save_path=str(out_path) if (write_outputs and out_path) else None,
            collect_stats=True,
            write_output=write_outputs
        )

        if stats is None:
            stats = {
                'delta_t95_event_N': 0,
                'delta_t95_event_mean_h': 0.0,
                'delta_t95_event_std_h': 0.0,
                'delta_t95_event_sum_h': 0.0
            }

        # 요약 row 생성
        id_token, ym = parse_id_token_and_ym(p)
        row = {
            'file_stem': p.stem,
            'id_token': id_token,
            'ym': ym,
            'delta_t95_event_N': int(stats['delta_t95_event_N']),
            'delta_t95_event_mean_h': float(stats['delta_t95_event_mean_h']),
            'delta_t95_event_std_h': float(stats['delta_t95_event_std_h']),
            'delta_t95_event_sum_h': float(stats['delta_t95_event_sum_h']),
        }
        return ("ok", p.stem, row)

    except Exception as e:
        # 에러도 요약에 0으로 남김(기존 로직 유지)
        id_token, ym = parse_id_token_and_ym(p)
        row = {
            'file_stem': p.stem,
            'id_token': id_token,
            'ym': ym,
            'delta_t95_event_N': 0,
            'delta_t95_event_mean_h': 0.0,
            'delta_t95_event_std_h': 0.0,
            'delta_t95_event_sum_h': 0.0,
        }
        return ("error", f"{p.name}: {e}", row)


def process_DFC_folder_mp(
    input_folder,
    output_folder,
    summary_csv_path=None,
    pattern="*.csv",
    write_outputs=True,
    skip_existing=True,
    workers=None,
):
    """
    DFC 멀티프로세스 폴더 처리 + summary 생성
    - write_outputs=True : *_DFC.csv 저장
    - skip_existing=True : *_DFC.csv 있으면 스킵(요약에서도 제외: 기존과 동일)
    """
    files = _collect_input_files(input_folder, pattern=pattern)
    if not files:
        print("[info] 처리할 CSV가 없습니다.")
        return pd.DataFrame(columns=SUMMARY_COLUMNS)

    output_dir = Path(output_folder)
    if write_outputs:
        output_dir.mkdir(parents=True, exist_ok=True)

    # 작업 리스트
    jobs = []
    for p in files:
        out_path = None
        if write_outputs:
            out_name = p.name.replace("_r.csv", "_DFC.csv")
            out_path = str(output_dir / out_name)
        jobs.append((str(p), out_path, write_outputs, skip_existing))

    summary_rows = []
    ok = skip = err = 0

    with ProcessPoolExecutor(max_workers=workers) as ex:
        futures = [ex.submit(_dfc_one_file_job, job) for job in jobs]
        for fut in tqdm(as_completed(futures), total=len(futures), desc="DFC Processing (MP)"):
            status, msg, row = fut.result()
            if status == "ok":
                ok += 1
                summary_rows.append(row)
            elif status == "skip":
                skip += 1
                # 스킵은 요약 제외(기존과 동일)
            else:
                err += 1
                print(f"[error] {msg}")
                summary_rows.append(row)

    # summary 저장
    if summary_csv_path is None:
        summary_csv_path = str(output_dir / "dfc_summary.csv") if write_outputs else str(Path(input_folder) / "dfc_summary.csv")

    summary_df = pd.DataFrame(summary_rows, columns=SUMMARY_COLUMNS)
    summary_df.to_csv(summary_csv_path, index=False, encoding="utf-8-sig")

    print(f"[done] ok={ok}, skip={skip}, error={err}")
    print(f"✅ 요약 저장: {summary_csv_path} (rows={len(summary_df)})")
    return summary_df

# ─────────────────────────────────────────────────────────────
# 실행 예시(필요한 부분만 주석 해제해서 사용)
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # 경로 설정
    input_folder_path = r'Z:\SamsungSTF\Processed_Data\DFC\EV6\DFC_수정용_251202'
    output_folder_path = r'Z:\SamsungSTF\Processed_Data\DFC\EV6\DFC_수정용_251202'
    summary_folder_path = r'Z:\SamsungSTF\Processed_Data\DFC\EV6\DFC_수정용_251202'
    summary_csv_path = os.path.join(summary_folder_path, 'dfc_features_summary.csv')
    os.makedirs(summary_folder_path, exist_ok=True)

    # ④ 파일 하나만 돌리기 (저장 O, 통계 확인)
    file_name = "bms_01241248900_2023-05_r.csv"  # ← 여기만 바꿔주면 됨
    file_path = os.path.join(input_folder_path, file_name)
    save_path = os.path.join(output_folder_path, file_name.replace('_r.csv', '_DFC.csv'))

    processed_df, stats = process_DFC_file(
        file_path,
        save_path=save_path,
        collect_stats=True,  # 통계도 보고 싶으면 True
        write_output=True  # DFC CSV 저장하고 싶으면 True
    )

    print(stats)

    # # 멀티프로세스
    # process_DFC_folder_mp(
    #     input_folder_path,
    #     output_folder_path,
    #     summary_csv_path=summary_csv_path,
    #     pattern="*.csv",
    #     write_outputs=True,
    #     skip_existing=True,
    #     workers=8,   # 네트워크 드라이브면 4~8 추천
    # )
