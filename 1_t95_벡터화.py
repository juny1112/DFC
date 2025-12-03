import pandas as pd
import os
from tqdm import tqdm
import numpy as np

# ──────────────────────────────────────────────────────────────────────
# 1) SOC≥95% 시간 계산 (벡터화)
#    - 연속 루프 없이 dt를 한 번에 계산해서 합산
#    - 측정오류 보정(시동 꺼짐 >5분 & 95% 상향 교차)을 마스크로 한 번에 합산
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
# 2) 폴더별 파일 → t95(h) 계산 (파일명 key로 딕셔너리 반환)
# ──────────────────────────────────────────────────────────────────────
def normalize_stem(filename: str) -> str:
    """확장자 제거한 기본 이름 (접미어 정규화)."""
    stem = os.path.splitext(os.path.basename(filename))[0]
    stem = stem.replace('_DFC', '').replace('_r', '')
    return stem

def compute_t95_by_file(folder_path: str) -> dict:
    results = {}
    if not os.path.isdir(folder_path):
        print(f"[WARN] 폴더 미존재: {folder_path}")
        return results

    for filename in tqdm(os.listdir(folder_path), desc=f"Scanning {os.path.basename(folder_path)}"):
        if not filename.lower().endswith(".csv"):
            continue
        file_path = os.path.join(folder_path, filename)
        try:
            # pyarrow 엔진 설치되어 있으면 더 빠름:
            # data = pd.read_csv(file_path, engine="pyarrow")
            data = pd.read_csv(file_path)
            t95_h = soc95_time(data)
            key = normalize_stem(filename)
            if key in results:
                print(f"[WARN] 중복 키 처리: {key} (기존 값 덮어씀)")
            results[key] = t95_h
        except Exception as e:
            print(f"[SKIP] {filename} - error: {e}")
    return results

# ──────────────────────────────────────────────────────────────────────
# 3) 실행부 예시
#    - ①/②: 폴더 전체 before/after 돌리기 (기존 방식)
#    - ③: 파일 하나만 t95 계산해서 확인
# ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # # ─ ① 폴더 전체: before / after 비교 (DFC 미적용 vs 적용) ─
    # folder_before = r'Z:\SamsungSTF\Processed_Data\DFC\EV6\R_parsing_원본'  # DFC 미적용
    # folder_after  = r'Z:\SamsungSTF\Processed_Data\DFC\EV6\DFC_원본'        # DFC 적용
    #
    # before_dict = compute_t95_by_file(folder_before)
    # after_dict  = compute_t95_by_file(folder_after)
    #
    # # ─ ② 파일명 기준으로 병합 테이블 생성 + Δt 계산 (없으면 NaN) ─
    # all_keys = sorted(set(before_dict.keys()) | set(after_dict.keys()))
    # df = pd.DataFrame({
    #     'file': all_keys,
    #     't95_before_h': [before_dict.get(k, np.nan) for k in all_keys],
    #     't95_after_h':  [after_dict.get(k,  np.nan) for k in all_keys],
    # })
    # df['delta_t_h'] = df['t95_before_h'] - df['t95_after_h']  # before - after
    #
    # # 저장
    # out_dir = r'G:\공유 드라이브\BSG_DFC_result\EV6\DFC_원본'
    # os.makedirs(out_dir, exist_ok=True)
    # csv_path = os.path.join(out_dir, 't95_before_after_delta.csv')
    # df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    # print(f"[SAVE] CSV -> {csv_path}")

    # ─ ③ (옵션) 파일 하나만 t95 계산해서 확인 ─
    # 필요할 때만 아래 주석 해제해서 사용
    single_file = r"Z:\SamsungSTF\Processed_Data\DFC\EV6\DFC_수정용_251202\bms_01241228086_2023-12_DFC.csv"
    t95_single_h = compute_t95_for_file(single_file)
    print(f"[INFO] single file t95 (h): {os.path.basename(single_file)} -> {t95_single_h:.5f} h")
