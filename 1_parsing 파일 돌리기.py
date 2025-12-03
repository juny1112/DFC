import os
import pandas as pd
from tqdm import tqdm
import numpy as np

# SOC=0 구간 보정
def fix_soc_zero(data):
    i = 0
    n = len(data)

    while i < n - 1:
        if data.loc[i, 'soc'] > 0.5 and data.loc[i+1, 'soc'] == 0:
            start = i + 1
            end = start
            while end < n and data.loc[end, 'soc'] == 0:
                end += 1

            # current = 0
            data.loc[start:end-1, 'pack_current'] = 0  # 여기만 0구간까지만

            # 뒤 앵커가 있을 때만 보간
            if end < n:
                data.loc[start:end, 'soc'] = np.linspace(
                    data.loc[i, 'soc'], data.loc[end, 'soc'], end - i + 1
                )[1:]
                data.loc[start:end, 'pack_volt'] = np.linspace(
                    data.loc[i, 'pack_volt'], data.loc[end, 'pack_volt'], end - i + 1
                )[1:]
        i += 1
    return data


# 이상 파일 제외
def invalid_data(data):
    data['time'] = pd.to_datetime(data['time'].str.strip(), format='%Y-%m-%d %H:%M:%S')
    data['soc'] = pd.to_numeric(data['soc'], errors='coerce')

    # ΔSOC, Δtime 벡터 계산
    dsoc = data['soc'].diff().abs()
    dtime = data['time'].diff()

    # 조건: SOC 변화 ≥ 10 % & 시간 간격 ≥ 12 h
    if ((dsoc >= 10) & (dtime >= pd.Timedelta(hours=12))).any():
        return True
    return False

# charging 구간 구하기
def parse_charging(data):
    cut = []

    # parse charging by current and speed
    if data.loc[0, 'pack_current'] < 0 and data.loc[0, 'speed'] == 0:
        cut.append(0)

    for i in range(1, len(data) - 1):
        if (data.loc[i, 'pack_current'] < 0 and data.loc[i, 'speed'] == 0) and \
                (data.loc[i + 1, 'pack_current'] >= 0 or data.loc[i + 1, 'speed'] != 0):
            cut.append(i + 1)

        elif (data.loc[i, 'pack_current'] >= 0 or data.loc[i, 'speed'] != 0) and \
                (data.loc[i + 1, 'pack_current'] < 0 and data.loc[i + 1, 'speed'] == 0):
            cut.append(i + 1)

    if data.loc[len(data) - 1, 'pack_current'] < 0 or data.loc[len(data) - 1, 'speed'] == 0:
        cut.append(len(data) - 1)

    cut = list(set(cut))
    cut.sort()


    # charging 여부 판단(charging column 생성)
    data['charging'] = 0
    for i in range(len(data) - 1):
        for j in range(len(cut) - 1):
            if data.loc[cut[j], 'pack_current'] < 0 and data.loc[cut[j], 'speed'] == 0:
                if cut[j] <= i <= cut[j + 1] - 1:
                    data.at[i, 'charging'] = 1


    # outlier 제거 - 충전구간과 충전구간 사이의 시간차가 1분 이내면 충전중으로 인식
    charging = []

    if data.loc[0, 'charging'] == 1:
        charging.append(0)
    for i in range(len(data) - 1):
        if data.loc[i, 'charging'] != data.loc[i + 1, 'charging']:
            charging.append(i + 1)
    if data.loc[len(data) - 1, 'charging'] == 1:
        charging.append(len(data) - 1)

    charging = list(set(charging))
    charging.sort()

    data['time'] = pd.to_datetime(data['time'], format='%y-%m-%d %H:%M:%S')
    for i in range(len(charging) - 1):
        if data.loc[charging[i], 'charging'] == 0 and \
                (data.loc[charging[i + 1], 'time'] - data.loc[charging[i], 'time']) <= pd.Timedelta(minutes=1):
            data.loc[charging[i]:charging[i + 1], 'charging'] = 1


    # outlier 제거 - 충전중이 5분 이하로 지속되면 충전중이 아닌 것으로 인식
    charging = []

    if data.loc[0, 'charging'] == 1:
        charging.append(0)
    for i in range(len(data) - 1):
        if data.loc[i, 'charging'] != data.loc[i + 1, 'charging']:
            charging.append(i + 1)
    if data.loc[len(data) - 1, 'charging'] == 1:
        charging.append(len(data) - 1)

    charging = list(set(charging))
    charging.sort()

    for i in range(len(charging) - 1):
        if data.loc[charging[i], 'charging'] == 1 and \
                (data.loc[charging[i + 1], 'time'] - data.loc[charging[i], 'time']) <= pd.Timedelta(minutes=5):
            data.loc[charging[i]:charging[i+1], 'charging'] = 0


    # print('charging_done')

    return data


# rest 구간 구하기
def parse_rest(data):
    rest = []

    # 충전중
    if data.loc[0, 'charging'] == 1:
        rest.append(0)
    for i in range(len(data) - 1):
        if data.loc[i, 'charging'] != data.loc[i + 1, 'charging']:
            rest.append(i + 1)
    if data.loc[len(data) - 1, 'charging'] == 1:
        rest.append(len(data) - 1)

    rest = list(set(rest))
    rest.sort()


    # rest 여부 판단(rest column 생성)
    data['rest'] = 0
    for i in range(len(data) - 1):
        for j in range(0, len(rest) - 1, 2):
            if rest[j] <= i <= rest[j + 1] - 1:
                data.at[i, 'rest'] = 1


    # 시동 꺼진 상태 rest에 포함
    limit_time = pd.Timedelta(minutes=5)  # 정차 시간 제한
    data['time'] = pd.to_datetime(data['time'], format='%y-%m-%d %H:%M:%S')

    rest_1 = []
    for i in range(len(data) - 1):
        if data.loc[i + 1, 'time'] - data.loc[i, 'time'] > limit_time:
            rest_1.extend([i, i + 1])

    rest_1 = list(set(rest_1))
    rest_1.sort()

    for i in range(len(rest_1) - 1):
        data.at[rest_1[i], 'rest'] = 1


    # 이상패턴 수정
    abnormal = []

    if 0 <= data.loc[0, 'pack_current'] <= 1:
        abnormal.append(0)
    for i in range(len(data) - 1):
        if 0 <= data.loc[i, 'pack_current'] <= 1 and (data.loc[i+1, 'pack_current'] < 0 or data.loc[i+1, 'pack_current'] >1):
            abnormal.append(i+1)
        elif (data.loc[i, 'pack_current'] < 0 or data.loc[i, 'pack_current'] > 1) and 0 <= data.loc[i+1, 'pack_current'] <= 1:
            abnormal.append(i + 1)
    if 0 <= data.loc[len(data) - 1, 'pack_current'] <= 1:
        abnormal.append(len(data) - 1)

    for i in range(len(abnormal) - 1):
        if 0 <= data.loc[abnormal[i], 'pack_current'] <= 1:
            if data.loc[abnormal[i+1]-1, 'time'] - data.loc[abnormal[i], 'time'] >= pd.Timedelta(minutes=10):
                data.loc[abnormal[i]:abnormal[i+1]-1, 'rest'] = 1


    # rest 떨어져있는 문제 수정
    for i in range(len(data) - 1):
        if data.loc[i, 'rest'] == 1 and data.loc[i + 1, 'rest'] == 0 and any(data.loc[i + 2:i + 31, 'rest'] == 1):
            change_idx = data.loc[i + 2: i + 31,
                          'rest'][data.loc[i + 2: i + 31,
                          'rest'] == 1].first_valid_index()  # 처음으로 결측치가 아닌값(= 정상적인 값. rest=1)이 나오는 행 인덱스 반환

            if data.loc[change_idx, 'time'] - data.loc[i, 'time'] <= pd.Timedelta(
                    minutes=1):  # rest가 떨어져있는 구간이 1분 이내면 rest로 간주
                data.loc[i + 1:change_idx, 'rest'] = 1

    #print('rest_done')
    return data


def process_folder(merge_folder, parsing_folder):
    os.makedirs(parsing_folder, exist_ok=True)

    for filename in tqdm(os.listdir(merge_folder), desc="Processing Files", unit="file"):
        if not filename.lower().endswith('.csv'):
            continue

        file_path = os.path.join(merge_folder, filename)
        save_path = os.path.join(parsing_folder, filename.replace('.csv', '_CR.csv'))

        try:
            data = pd.read_csv(file_path)

            # (1) SOC=0 보정
            data = fix_soc_zero(data)

            # (2) invalid → 저장 안 함 + 기존 산출물 삭제
            if invalid_data(data):
                '''if os.path.exists(save_path):
                    os.remove(save_path)'''
                print(f"[SKIP] {filename} - invalid (removed old output if existed)")
                continue

            # (3) 정상 파일만 파싱 후 저장
            data = parse_charging(data)
            data = parse_rest(data)
            data.to_csv(save_path, index=False)

        except Exception as e:
            print(f"[SKIP] {filename} - error: {e}")

def process_selected(merge_folder, parsing_folder, selected_filenames):
    os.makedirs(parsing_folder, exist_ok=True)

    for filename in tqdm(selected_filenames, desc="Processing Selected", unit="file"):
        if not filename.lower().endswith('.csv'):
            print(f"[MISS] {filename} - not a csv")
            continue

        file_path = os.path.join(merge_folder, filename)
        if not os.path.isfile(file_path):
            print(f"[MISS] {filename} - not found")
            continue

        save_path = os.path.join(parsing_folder, filename.replace('.csv', '_CR.csv'))

        try:
            data = pd.read_csv(file_path)
            data = fix_soc_zero(data)

            if invalid_data(data):
                if os.path.exists(save_path):
                    os.remove(save_path)
                print(f"[SKIP] {filename} - invalid (removed old output if existed)")
                continue

            data = parse_charging(data)
            data = parse_rest(data)
            data.to_csv(save_path, index=False)

        except Exception as e:
            print(f"[SKIP] {filename} - error: {e}")

# 선택 파일 이후 파일들 돌리기
def process_folder_resume(merge_folder, parsing_folder, resume_after=None, start_idx=None, skip_existing=True):
    """
    resume_after: 마지막으로 처리한 '파일명'을 넘기면 그 다음 파일부터 시작
                  (CR 출력 파일명을 넘겨도 자동으로 원본명으로 매핑)
    start_idx   : 정렬된 목록에서 시작할 인덱스(0-based). resume_after보다 우선순위 낮음.
    skip_existing: 이미 결과가 있으면 건너뛸지 여부
    """
    os.makedirs(parsing_folder, exist_ok=True)

    # 1) 이름순으로 고정
    files = [f for f in os.listdir(merge_folder) if f.lower().endswith('.csv')]
    files.sort()

    # 2) 시작 위치 결정
    start = 0
    if resume_after:
        # CR 이름을 넘긴 경우 원본 이름으로 변환
        raw_name = (resume_after
                    .replace('_CR.csv', '.csv')
                    .replace('_cr.csv', '.csv'))
        if raw_name in files:
            start = files.index(raw_name) + 1  # 그 다음 것부터
        else:
            print(f"[WARN] '{resume_after}'(원본 '{raw_name}')을 목록에서 찾지 못했어요. 처음부터 시작합니다.")
    elif start_idx is not None:
        if 0 <= start_idx < len(files):
            start = start_idx
        else:
            print(f"[WARN] start_idx {start_idx}가 범위를 벗어났어요. 0부터 시작합니다.")
            start = 0

    # 3) 이어서 처리
    for filename in tqdm(files[start:], desc=f"Processing Files (from {start}/{len(files)})", unit="file"):
        file_path = os.path.join(merge_folder, filename)
        save_path = os.path.join(parsing_folder, filename.replace('.csv', '_CR.csv'))

        # 이미 결과가 있으면 생략(옵션)
        if skip_existing and os.path.exists(save_path):
            # print(f"[SKIP-EXIST] {filename}")
            continue

        try:
            data = pd.read_csv(file_path)

            # (1) SOC=0 보정
            data = fix_soc_zero(data)

            # (2) invalid → 저장 안 함
            if invalid_data(data):
                print(f"[SKIP] {filename} - invalid")
                continue

            # (3) 파싱 & 저장
            data = parse_charging(data)
            data = parse_rest(data)
            data.to_csv(save_path, index=False)

        except Exception as e:
            print(f"[SKIP] {filename} - error: {e}")


# 파일 경로
merge_folder_path = 'Z:/SamsungSTF/Processed_Data/Merged/EV6/'
parsing_folder_path = r'Z:\SamsungSTF\Processed_Data\DFC\EV6\CR_parsing'



# # 선택 파일 이후 파일들 돌리기
# process_folder_resume(
#     merge_folder_path,
#     parsing_folder_path,
#     resume_after='bms_altitude_01241364627_2024-01.csv',  # 마지막 완료한 파일명(출력명/원본명 둘 다 OK)
#     skip_existing=True  # 이미 만들어진 _CR.csv는 건너뜀
# )


# 전체 폴더
# process_folder(merge_folder_path, parsing_folder_path)

# 선택 파일
some_files = ['bms_01241228055_2023-04.csv']
process_selected(merge_folder_path, parsing_folder_path, some_files)

