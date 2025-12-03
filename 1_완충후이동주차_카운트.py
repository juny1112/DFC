import os
import pandas as pd
from tqdm import tqdm

# ─────────────────────────────────────────────────────────────
# 설정
# ─────────────────────────────────────────────────────────────
MERGE_GAP_MINUTES = 5  # R_aftercharg 블록 끝과 R_uncharg 블록 시작 사이 허용 간격(분)

def get_blocks(series: pd.Series):
    """
    연속된 1 블록을 (start_idx, end_idx) 튜플 목록으로 반환 (end 포함)
    """
    blks = []
    in_block = False
    start = None
    n = len(series)
    for i, v in enumerate(series.astype(int).fillna(0)):
        if v == 1 and not in_block:
            start, in_block = i, True
        elif v == 0 and in_block:
            blks.append((start, i - 1))
            in_block = False
    if in_block:
        blks.append((start, n - 1))
    return blks

def summarize_fullcharge_parking_from_r_folder(input_folder: str, summary_path: str):
    """
    모드 OFF로 생성된 *_r.csv 파일만 읽어서,
    '완충 후 이동주차 모드'가 ON일 때 병합이 발생하는 파일과 인덱스 구간을 summary txt로 기록.
    """
    limit = pd.Timedelta(minutes=MERGE_GAP_MINUTES)
    summary_entries = []  # (filename, [logs...])

    files = [fn for fn in os.listdir(input_folder) if fn.lower().endswith(".csv")]
    files.sort()

    for filename in tqdm(files, desc="Scanning _r.csv files"):
        fpath = os.path.join(input_folder, filename)

        try:
            df = pd.read_csv(fpath)
        except Exception as e:
            tqdm.write(f"[skip] {filename} - read error: {e}")
            continue

        # 필요한 컬럼 확인
        required_cols = {"time", "R_aftercharg", "R_uncharg"}
        if not required_cols.issubset(df.columns):
            tqdm.write(f"[skip] {filename} - missing columns (need: {required_cols})")
            continue

        # 시간형
        if not pd.api.types.is_datetime64_any_dtype(df["time"]):
            try:
                df["time"] = pd.to_datetime(df["time"].astype(str).str.strip(),
                                            format="%Y-%m-%d %H:%M:%S", errors="raise")
            except Exception as e:
                tqdm.write(f"[skip] {filename} - time parse error: {e}")
                continue

        # 블록 검출
        after_blocks  = get_blocks(df["R_aftercharg"].fillna(0))
        uncharg_blocks = get_blocks(df["R_uncharg"].fillna(0))

        if not after_blocks or not uncharg_blocks:
            continue

        logs = []
        # 각 after 블록에 대해, 뒤에 오는 첫 uncharg 블록을 찾아 gap 체크
        for (aft_s, aft_e) in after_blocks:
            next_unc = next(((u_s, u_e) for (u_s, u_e) in uncharg_blocks if u_s > aft_e), None)
            if not next_unc:
                continue

            unc_s, unc_e = next_unc
            gap = df.loc[unc_s, "time"] - df.loc[aft_e, "time"]

            if gap <= limit:
                # 사이 구간(비-rest 포함) + uncharg 블록 전체가 aftercharg로 흡수되는 케이스
                logs.append(
                    f"merged={aft_e+1}-{unc_e} (after {aft_s}-{aft_e} → uncharg {unc_s}-{unc_e}, gap={int(gap.total_seconds())}s)"
                )

        if logs:
            summary_entries.append((filename, logs))

    # 요약 파일 기록 (병합 발생 파일만)
    with open(summary_path, "w", encoding="utf-8") as fp:
        fp.write("[Fullcharge Parking Mode — predicted merged index ranges from OFF-parsed files]\n")
        fp.write(f"(GAP_MIN={MERGE_GAP_MINUTES})\n\n")
        if summary_entries:
            for fname, logs in summary_entries:
                fp.write(f"{fname}: " + " | ".join(logs) + "\n")
        else:
            fp.write("None\n")

    print(f"Summary written to: {summary_path}")

# ─────────────────────────────────────────────────────────────
# 사용 예시
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # 모드 OFF로 생성된 *_r.csv 파일들이 들어있는 폴더
    r_folder =  r'Z:\SamsungSTF\Processed_Data\DFC\EV6\R_parsing_원본'
    # 요약 txt 저장 경로
    summary_txt = r"G:\공유 드라이브\BSG_DFC_result\EV6\DFC_완충후이동주차\FULLCHARGE_PARKING_SUMMARY_FROM_OFF.txt"

    summarize_fullcharge_parking_from_r_folder(r_folder, summary_txt)
