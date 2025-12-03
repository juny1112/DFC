import re
import pandas as pd
from pathlib import Path

def parse_fullcharge_summary_txt(input_txt: str, output_csv: str):
    """
    [Fullcharge Parking Mode — predicted merged index ranges from OFF-parsed files]
    (GAP_MIN=5)

    <filename1.csv>: merged=12-34 (after ...) | merged=78-120 (after ...)
    <filename2.csv>: merged=5-9 (after ...)
    ...
    또는 'None'
    형태의 요약 txt를 아래 컬럼의 CSV로 변환합니다.

    - filename
    - fullcharge_mode_count : 'merged=' 항목 개수
    - merged_ranges         : '12-34; 78-120' 처럼 merged= 뒤의 구간만 모아 문자열화
    """
    input_txt = Path(input_txt)
    lines = input_txt.read_text(encoding="utf-8").splitlines()

    data = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        # 헤더/주석 라인 스킵
        if line.startswith("[") or line.startswith("("):
            continue
        # "None"만 있는 경우(병합 없음)
        if line == "None":
            continue

        # 형식: "<filename>: <logs>"
        if ":" not in line:
            # 예상치 못한 라인은 건너뜀
            continue

        fname, logs = line.split(":", 1)
        fname = fname.strip()
        logs = logs.strip()

        if logs.lower() == "none":
            count = 0
            merged_ranges_joined = ""
        else:
            # 파이프 단위 분해 후 'merged=' 패턴만 추출
            parts = [p.strip() for p in logs.split("|")]
            merged_chunks = []
            for p in parts:
                # 'merged=...' 다음 공백 전까지를 추출 (예: 'merged=12-34' 한 토큰)
                m = re.search(r"merged=([^\s]+)", p)
                if m:
                    merged_chunks.append(m.group(1))
            count = len(merged_chunks)
            merged_ranges_joined = "; ".join(merged_chunks)

        data.append({
            "filename": fname,
            "fullcharge_mode_count": count,
            "merged_ranges": merged_ranges_joined
        })

    # 파일별로 정렬(원하는 다른 정렬 규칙 있으면 바꾸세요)
    df = pd.DataFrame(data).sort_values(by=["fullcharge_mode_count", "filename"], ascending=[False, True])
    df.to_csv(output_csv, index=False, encoding="utf-8-sig")
    print(f"Saved: {output_csv}")
    return df

if __name__ == "__main__":
    # ⬇️ 여기에 본인이 만든 txt 경로와 내보낼 csv 경로를 넣어주세요.
    input_txt = r"G:\공유 드라이브\BSG_DFC_result\EV6\DFC_완충후이동주차\FULLCHARGE_PARKING_SUMMARY_FROM_OFF.txt"
    output_csv = r"G:\공유 드라이브\BSG_DFC_result\EV6\DFC_완충후이동주차\FULLCHARGE_PARKING_SUMMARY_FROM_OFF.csv"
    parse_fullcharge_summary_txt(input_txt, output_csv)
