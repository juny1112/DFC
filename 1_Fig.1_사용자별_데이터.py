import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.dates import AutoDateLocator, ConciseDateFormatter
from pandas.tseries.offsets import MonthEnd

# ─────────────────────────────────────────────────────────────────────
# 경로/설정
# ─────────────────────────────────────────────────────────────────────
# DIR_BEFORE = r'Z:\SamsungSTF\Processed_Data\DFC\Ioniq5\R_parsing_완충후이동주차'   # _CR / _R 파일 (before)
# DIR_AFTER  = r'Z:\SamsungSTF\Processed_Data\DFC\Ioniq5\DFC_완충후이동주차'         # _DFC 파일
# OUT_DIR    = r'G:\공유 드라이브\BSG_DFC_result\combined\DFC_완충후이동주차'

DIR_BEFORE = r'Z:\SamsungSTF\Processed_Data\DFC\EV6\DFC_수정용_251202'   # _CR / _R 파일 (before)
DIR_AFTER  = r'Z:\SamsungSTF\Processed_Data\DFC\EV6\DFC_수정용_251202'         # _DFC 파일
OUT_DIR    = r'Z:\SamsungSTF\Processed_Data\DFC\EV6\DFC_수정용_251202'
os.makedirs(OUT_DIR, exist_ok=True)

# ▶ 여기서 사용자(ID) / 월 / 플롯 모드 지정
#   - USER_ID: 'bms_01241228094' 처럼 지정하면 그 사용자만
#              None 이면 모든 사용자
#   - TARGET_YM: '2023-04' 처럼 지정하면 해당 월만
#                None 이면 해당 사용자의 모든 월
#   - PLOT_MODE:
#       'both'   : BEFORE vs DFC 비교
#       'before' : BEFORE만 플롯 (CR/R 모두 포함)
USER_ID    = 'bms_01241228055'   # 필요 시 None로 변경
TARGET_YM  = '2023-04'           # 필요 시 None로 변경
PLOT_MODE  = 'both'            # 'both' 또는 'before'


# ─────────────────────────────────────────────────────────────────────
# 파일명 파싱: bms_<ID>_<YYYY-MM>_CR.csv / bms_<ID>_<YYYY-MM>_R.csv / bms_<ID>_<YYYY-MM>_DFC.csv
# ─────────────────────────────────────────────────────────────────────
PATTERN = re.compile(
    r'^(?P<id>bms_(?:altitude_)?\d+?)_(?P<ym>\d{4}-\d{2})_(?P<tag>CR|DFC|R)\.csv$',
    flags=re.IGNORECASE
)

def parse_name(filename: str):
    m = PATTERN.match(filename)
    if not m:
        return None
    id_token = m.group('id')
    ym = m.group('ym')
    tag = m.group('tag').upper()
    return id_token, ym, tag

def scan_dir_for_tag(root: str, expected_tag: str):
    mapping = {}
    for fn in os.listdir(root):
        if not fn.lower().endswith('.csv'):
            continue
        parsed = parse_name(fn)
        if not parsed:
            continue
        id_token, ym, tag = parsed
        if tag == expected_tag.upper():
            mapping[(id_token, ym)] = os.path.join(root, fn)
    return mapping

# BEFORE: CR + R 둘 다 사용
before_map = {}
before_map.update(scan_dir_for_tag(DIR_BEFORE, 'CR'))
before_map.update(scan_dir_for_tag(DIR_BEFORE, 'R'))

after_map  = scan_dir_for_tag(DIR_AFTER,  'DFC')

# ─────────────────────────────────────────────────────────────────────
# (ID, 월) 페어 리스트 생성 (모드에 따라 다름)
# ─────────────────────────────────────────────────────────────────────
if PLOT_MODE == 'both':
    common_keys_all = sorted(set(before_map.keys()) & set(after_map.keys()))
elif PLOT_MODE == 'before':
    common_keys_all = sorted(before_map.keys())
else:
    raise ValueError(f"[ERR] Unknown PLOT_MODE: {PLOT_MODE}")

# ─────────────────────────────────────────────────────────────────────
# 사용자 / 월 필터링
# ─────────────────────────────────────────────────────────────────────
common_keys = common_keys_all

if USER_ID is not None:
    common_keys = [k for k in common_keys if k[0].lower() == USER_ID.lower()]
    if not common_keys:
        raise SystemExit(f"[INFO] No matching (ID, month) pairs for USER_ID={USER_ID}.")

if TARGET_YM is not None:
    common_keys = [k for k in common_keys if k[1] == TARGET_YM]
    if not common_keys:
        raise SystemExit(f"[INFO] No matching (ID, month) pairs for TARGET_YM={TARGET_YM}.")

if not common_keys:
    raise SystemExit("[INFO] No matching (ID, month) pairs after filtering.")

# ─────────────────────────────────────────────────────────────────────
# 로드 & 전처리
# ─────────────────────────────────────────────────────────────────────
def load_bms_csv(path):
    df = pd.read_csv(path)
    if 'time' not in df.columns or 'soc' not in df.columns:
        raise ValueError(f"[ERR] required columns (time, soc) not found in: {path}")
    df['time'] = pd.to_datetime(df['time'])
    df['soc']  = pd.to_numeric(df['soc'], errors='coerce')
    df = df.dropna(subset=['time', 'soc']).sort_values('time')
    return df

# ─────────────────────────────────────────────────────────────────────
# 한 달을 4개 윈도우(1–7, 8–14, 15–21, 22–말일)로 분할
# ─────────────────────────────────────────────────────────────────────
def month_windows(year, month):
    start = pd.Timestamp(year=year, month=month, day=1)
    end   = (start + MonthEnd(1)).replace(hour=23, minute=59, second=59, microsecond=999999)
    w1_start, w1_end = start, (start + pd.Timedelta(days=7))  - pd.Timedelta(seconds=1)
    w2_start, w2_end = start + pd.Timedelta(days=7),  (start + pd.Timedelta(days=14)) - pd.Timedelta(seconds=1)
    w3_start, w3_end = start + pd.Timedelta(days=14), (start + pd.Timedelta(days=21)) - pd.Timedelta(seconds=1)
    w4_start, w4_end = start + pd.Timedelta(days=21), end
    return [(w1_start, w1_end), (w2_start, w2_end), (w3_start, w3_end), (w4_start, w4_end)]

def slice_range(df, start, end):
    m = (df['time'] >= start) & (df['time'] <= end)
    return df.loc[m]

# ─────────────────────────────────────────────────────────────────────
# 서브플롯 하나 그리기
# ─────────────────────────────────────────────────────────────────────
def draw_window(ax, df_cr, df_dfc=None, show_legend=False, plot_mode='both'):
    for sp in ax.spines.values():
        sp.set_linewidth(1.3)
    ax.tick_params(axis='both', width=1.1, length=4, labelsize=9)

    lw = 2.0

    if plot_mode == 'both':
        ax.plot(df_cr['time'],  df_cr['soc'],
                color='#3366cc', linestyle='-',  linewidth=lw, label='DFC not applied')
        ax.plot(df_dfc['time'], df_dfc['soc'],
                color='#cc0000', linestyle='--', linewidth=lw, label='DFC applied')
    elif plot_mode == 'before':
        ax.plot(df_cr['time'], df_cr['soc'],
                color='#3366cc', linestyle='-', linewidth=lw, label='DFC not applied')
    else:
        raise ValueError(f"[ERR] Unknown plot_mode in draw_window: {plot_mode}")

    ax.axhline(95, color='0.5', linestyle=':', linewidth=1.2)

    ax.set_ylim(0, 100)
    ax.set_yticks([0, 20, 40, 60, 80, 100])

    loc = AutoDateLocator()
    ax.xaxis.set_major_locator(loc)
    ax.xaxis.set_major_formatter(ConciseDateFormatter(loc))
    for lab in ax.get_xticklabels():
        lab.set_rotation(45)
        lab.set_horizontalalignment('right')

    if show_legend:
        leg = ax.legend(loc='upper left', bbox_to_anchor=(0.02, 0.98),
                        frameon=True, framealpha=0.9, fontsize=9, borderaxespad=0.0)
        for legline in leg.get_lines():
            legline.set_linewidth(lw)

    if plot_mode == 'both':
        no_data = (df_cr is None or df_cr.empty) and (df_dfc is None or df_dfc.empty)
    else:  # 'before'
        no_data = (df_cr is None or df_cr.empty)

    if no_data:
        ax.text(0.5, 0.5, 'No data', transform=ax.transAxes,
                ha='center', va='center', fontsize=10, color='0.4')

# ─────────────────────────────────────────────────────────────────────
# 메인 루프
# ─────────────────────────────────────────────────────────────────────
common_keys = sorted(common_keys, key=lambda x: (x[0], x[1]))

for (id_token, ym) in common_keys:
    path_cr = before_map[(id_token, ym)]
    df_cr   = load_bms_csv(path_cr)

    if PLOT_MODE == 'both':
        path_dfc = after_map[(id_token, ym)]
        df_dfc   = load_bms_csv(path_dfc)
    else:  # 'before'
        df_dfc = None

    year, month = map(int, ym.split('-'))
    windows = month_windows(year, month)

    fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(12, 8), dpi=300)
    axs = axs.flatten()

    for i, (start, end) in enumerate(windows):
        sub_cr  = slice_range(df_cr,  start, end)
        sub_dfc = slice_range(df_dfc, start, end) if (df_dfc is not None) else None

        draw_window(axs[i],
                    sub_cr,
                    sub_dfc,
                    show_legend=(i == 0),
                    plot_mode=PLOT_MODE)

        if i in (0, 2):
            axs[i].set_ylabel('SOC (%)')
        if i in (2, 3):
            axs[i].set_xlabel('Time (day)')

        axs[i].set_title(f'{start:%b %d} – {end:%b %d}', fontsize=10, pad=6)

    mode_title = 'Before vs DFC' if PLOT_MODE == 'both' else 'Before only'
    fig.suptitle(
        f'SOC vs Time by week — {ym}  (ID: {id_token}, {mode_title})',
        fontsize=13, y=0.98
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    if PLOT_MODE == 'both':
        out_name = f'SOC_weekly_{id_token}_{ym}.png'
    else:  # 'before'
        out_name = f'SOC_weekly_BEFOREonly_{id_token}_{ym}.png'

    out_path = os.path.join(OUT_DIR, out_name)
    fig.savefig(out_path, bbox_inches='tight', dpi=300)
    print(f"[SAVE] {out_path}")

# plt.show()
