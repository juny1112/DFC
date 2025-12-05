import os
import pandas as pd
from tqdm import tqdm
from datetime import datetime

# ───────────────────────────────────────────────
# 경로 설정
CR_PARSING_DIR = r"Z:\SamsungSTF\Processed_Data\DFC\EV6\CR_parsing"

# 선택 실행 옵션
# 1) 리스트에 파일명을 직접 넣어 지정 (예: ['bms_..._2023-08_CR.csv', '...'])
SELECTED_FILES = []  # 비워두면 폴더 내 모든 .csv 대상

# 2) 줄마다 파일명이 적힌 .txt 파일 경로 (예: invalid_candidates_*.txt)
SELECTED_LIST_PATH = r"G:\공유 드라이브\BSG_DFC_result\EV6\SOC_missing_filtering_out.txt" # 없으면 빈 문자열

# 출력 인덱스 방식 (요청 포맷: "파일명: 행1,행2 ...")
ONE_BASED = False  # True로 바꾸면 1-based(엑셀/사람 눈 기준)로 출력
# ───────────────────────────────────────────────


def load_selected_files(dir_path, selected_list, selected_list_txt):
    """선택 파일 목록을 결정: 명시 리스트 > 텍스트 파일 > 전체 CSV"""
    if selected_list:
        # 사용자가 직접 지정한 리스트 우선
        files = selected_list[:]
    elif selected_list_txt and os.path.isfile(selected_list_txt):
        with open(selected_list_txt, "r", encoding="utf-8") as f:
            files = [line.strip() for line in f if line.strip()]
    else:
        files = [f for f in os.listdir(dir_path) if f.lower().endswith(".csv")]

    # 존재하는 파일만 필터
    files = [f for f in files if os.path.isfile(os.path.join(dir_path, f))]
    files.sort()
    return files


def find_offending_pairs(df):
    """
    연속된 두 행 (i-1, i) 에서
    ΔSOC >= 10  AND  Δtime >= 12h  를 만족하는 모든 (i-1, i) 쌍을 반환
    반환: List[(start_idx, end_idx)]  # 0-based
    """
    if df.empty or ("time" not in df.columns) or ("soc" not in df.columns):
        return []

    # 시간 포맷: 사용자가 준 코드와 동일하게 고정
    time = pd.to_datetime(df["time"].astype(str).str.strip(),
                          format="%Y-%m-%d %H:%M:%S", errors="raise")
    soc = pd.to_numeric(df["soc"], errors="coerce")

    dsoc = soc.diff().abs()
    dtime = time.diff()

    cond = (dsoc >= 10) & (dtime >= pd.Timedelta(hours=12))
    hit_idx = cond[cond].index  # i (i-1,i 쌍의 '뒤' 인덱스)

    pairs = []
    for i in hit_idx:
        if pd.isna(i) or i is None:
            continue
        if isinstance(i, (int,)) and i > 0:
            pairs.append((i-1, i))
        else:
            # pandas Index일 수 있으므로 int 변환 시도
            try:
                ii = int(i)
                if ii > 0:
                    pairs.append((ii-1, ii))
            except Exception:
                pass
    return pairs


def main():
    files = load_selected_files(CR_PARSING_DIR, SELECTED_FILES, SELECTED_LIST_PATH)

    lines_txt = []
    rows_csv = []  # (filename, start_idx, end_idx, start_idx_1b, end_idx_1b)
    errors = []

    for fname in tqdm(files, desc="Scanning files", unit="file"):
        fpath = os.path.join(CR_PARSING_DIR, fname)
        try:
            df = pd.read_csv(fpath)
            pairs = find_offending_pairs(df)

            if pairs:
                # 출력용 인덱스 변환
                if ONE_BASED:
                    pairs_str = "; ".join([f"{a+1},{b+1}" for (a, b) in pairs])
                else:
                    pairs_str = "; ".join([f"{a},{b}" for (a, b) in pairs])

                # 콘솔 출력 (요청 포맷)
                print(f"{fname}: {pairs_str}")

                # 파일 저장용
                lines_txt.append(f"{fname}: {pairs_str}")
                for a, b in pairs:
                    rows_csv.append((fname, a, b, a+1, b+1))
        except Exception as e:
            errors.append((fname, str(e)))

    # 결과 저장
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_txt = os.path.join(CR_PARSING_DIR, f"invalid_pairs_{ts}.txt")
    with open(out_txt, "w", encoding="utf-8") as f:
        for line in lines_txt:
            f.write(line + "\n")
    print(f"\n결과 목록 저장: {out_txt}")

    if rows_csv:
        import csv
        out_csv = os.path.join(CR_PARSING_DIR, f"invalid_pairs_{ts}.csv")
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["filename", "start_idx_0b", "end_idx_0b", "start_idx_1b", "end_idx_1b"])
            w.writerows(rows_csv)
        print(f"CSV도 저장했습니다: {out_csv}")

    if errors:
        err_log = os.path.join(CR_PARSING_DIR, f"errors_{ts}.log")
        with open(err_log, "w", encoding="utf-8") as f:
            for fname, msg in errors:
                f.write(f"{fname}\t{msg}\n")
        print(f"읽기 오류 {len(errors)}건 — 로그 저장: {err_log}")


if __name__ == "__main__":
    main()
