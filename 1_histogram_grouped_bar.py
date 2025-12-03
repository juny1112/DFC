# ───────────────── Plot-only: read CSV → grouped bar hist + Δt inset ─────────────────
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.ticker import MultipleLocator, NullFormatter, FormatStrFormatter

# ──────────────────────────────── 설정 ────────────────────────────────
MAIN_LIMIT_MODE = 'auto'     # 'auto' or 'fixed'
MAIN_X_RANGE    = (0, 200)   # x축 범위 (fixed 모드일 때)
MAIN_Y_MAX      = 200       # y축 최댓값 (fixed 모드일 때)

BIN_WIDTH       = 5.0        # 메인 히스토 bin 폭 (시간, h)
LABEL_EVERY     = 10         # 몇 개 bin마다 x라벨 표시(2면 10h 간격: 0,10,20,…)

# ── 인셋(Δt)용 새 옵션들 ──
INSET_LIMIT_MODE = 'auto'          # 'auto' or 'fixed'
INSET_X_RANGE    = (0.0, 175.0)    # fixed 모드일 때 인셋 x축 범위
INSET_Y_MAX      = 150            # fixed 모드일 때 y축 최댓값 (None이면 자동)
INSET_BIN_STEP   = 5.0
INSET_TICK_STEP  = 50.0

EXTRA_DOWN      = 0.08

# 파스텔 팔레트
COLOR_APPLIED   = "#9ECAE1"
COLOR_NOT       = "#FDAE6B"
COLOR_DELTA     = "#E15759"
ALPHA_MAIN      = 0.90
ALPHA_INSET     = 0.90
# ──────────────────────────────────────────────────────────────────────

# ====== 입력/출력 경로 ======
csv_path = r'G:\공유 드라이브\BSG_DFC_result\combined\DFC_원본\t95_before_after_delta_combined.csv'
out_dir  = os.path.dirname(csv_path)

# ====== 데이터 읽기 ======
usecols = ['t95_before_h', 't95_after_h', 'delta_t_h']
df = pd.read_csv(csv_path, usecols=usecols)

after  = df['t95_after_h'].dropna().to_numpy()   # DFC applied
before = df['t95_before_h'].dropna().to_numpy()  # DFC not applied
all_data = np.concatenate([after, before]) if (after.size or before.size) else np.array([])
if all_data.size == 0:
    raise SystemExit("[INFO] No data to plot. Check CSV columns/values.]")

# ====== 메인 bin 계산 (0,5,10…로 깔끔) ======
dmin, dmax = float(np.min(all_data)), float(np.max(all_data))
left  = np.floor(dmin / BIN_WIDTH) * BIN_WIDTH
right = np.ceil(dmax / BIN_WIDTH)  * BIN_WIDTH

# fixed 모드일 때는 X_RANGE 상한까지 bins를 확장
if MAIN_LIMIT_MODE == 'fixed':
    left  = min(left,  MAIN_X_RANGE[0])
    right = max(right, np.ceil(MAIN_X_RANGE[1] / BIN_WIDTH) * BIN_WIDTH)

edges   = np.arange(left, right + BIN_WIDTH*0.999, BIN_WIDTH)  # bin 경계
centers = (edges[:-1] + edges[1:]) / 2                         # 막대 중심

# ====== 그룹 막대 카운트 ======
cnt_after, _  = np.histogram(after,  bins=edges)
cnt_before, _ = np.histogram(before, bins=edges)

# 막대 폭/오프셋: 한 bin(폭의 90%)을 두 그룹이 반씩 사용
bar_total_width = BIN_WIDTH * 0.90
bar_width_each  = bar_total_width / 2.0
offset          = bar_width_each / 2.0

# ====== 그림 ======
fig, ax = plt.subplots(figsize=(12, 5.5), dpi=150)

# grouped bar (좌: after, 우: before)
ax.bar(centers - offset, cnt_after,  width=bar_width_each, label='DFC applied',
       color=COLOR_APPLIED, alpha=ALPHA_MAIN, edgecolor='none', align='center')
ax.bar(centers + offset, cnt_before, width=bar_width_each, label='DFC not applied',
       color=COLOR_NOT,   alpha=ALPHA_MAIN, edgecolor='none', align='center')

ax.set_title('t_95% comparison')
ax.set_ylabel('Count')
ax.set_xlabel('Total t_95% (hours)')

# y축 최대치(peak)
peak = int(max(cnt_after.max() if cnt_after.size else 0,
               cnt_before.max() if cnt_before.size else 0))

# ── 메인 축 범위 설정 ──
if MAIN_LIMIT_MODE == 'fixed':
    ax.set_xlim(*MAIN_X_RANGE)
    ax.set_ylim(0, MAIN_Y_MAX)
else:
    ax.set_xlim(edges[0], edges[-1])
    ax.set_ylim(0, peak * 1.15 if peak > 0 else 1)

# ── x라벨(간격 줄이기) ──
if MAIN_LIMIT_MODE == 'fixed':
    tick_positions = np.arange(MAIN_X_RANGE[0], MAIN_X_RANGE[1] + 1e-9, BIN_WIDTH)
else:
    tick_positions = edges[:-1]

tick_idx = np.arange(0, len(tick_positions), LABEL_EVERY)
ax.set_xticks(tick_positions[tick_idx])
ax.set_xticklabels([f"{int(x)}" for x in tick_positions[tick_idx]], rotation=0)

# 보조 눈금/격자
ax.xaxis.set_minor_locator(MultipleLocator(BIN_WIDTH))
ax.grid(axis='y', alpha=0.25, linewidth=0.7)

# 레전드
legend = ax.legend(loc='upper right', frameon=False)

# ====== Δt 인셋 ======
fig.canvas.draw()
renderer = fig.canvas.get_renderer()
leg_bbox = legend.get_window_extent(renderer=renderer)
x_disp = ax.bbox.x1
y_disp = leg_bbox.y0
_, y0_ax = ax.transAxes.inverted().transform((x_disp, y_disp))

width_frac, height_frac, pad = 0.35, 0.35, 0.01
x0 = 1 - width_frac - pad
y0 = max(pad, y0_ax - height_frac - pad - EXTRA_DOWN)

axins = inset_axes(ax, width="100%", height="100%",
                   loc="lower left",
                   bbox_to_anchor=(x0, y0, width_frac, height_frac),
                   bbox_transform=ax.transAxes, borderpad=0.0)

delta = df['delta_t_h'].dropna().to_numpy()
if delta.size > 0:
    # ── 인셋 x-범위 & bins ──
    if INSET_LIMIT_MODE == 'fixed':
        lo, hi = INSET_X_RANGE
        lo = float(lo); hi = float(hi)
        # step에 맞춰 깔끔하게 가장자리 맞추기
        lo_edge = np.floor(lo / INSET_BIN_STEP) * INSET_BIN_STEP
        hi_edge = np.ceil(hi / INSET_BIN_STEP) * INSET_BIN_STEP
    else:
        dmin_i, dmax_i = float(np.min(delta)), float(np.max(delta))
        lo_edge = np.floor(dmin_i / INSET_BIN_STEP) * INSET_BIN_STEP
        hi_edge = np.ceil(dmax_i  / INSET_BIN_STEP) * INSET_BIN_STEP
        lo, hi = lo_edge, hi_edge

    bins_dt = np.arange(lo_edge, hi_edge + INSET_BIN_STEP*0.999, INSET_BIN_STEP)

    axins.hist(delta, bins=bins_dt, alpha=ALPHA_INSET, color=COLOR_DELTA, edgecolor='none')
    axins.axvline(0, color='k', linestyle='--', linewidth=0.8)
    axins.set_title('Δt_>95% distribution', fontsize=9)
    axins.set_xlabel('Δt_>95% (h)', fontsize=8)
    axins.set_ylabel('Count', fontsize=8)
    axins.tick_params(labelsize=8)

    # 축/눈금 설정
    axins.set_xlim(lo, hi)
    axins.xaxis.set_major_locator(MultipleLocator(INSET_TICK_STEP))
    axins.xaxis.set_minor_locator(MultipleLocator(INSET_BIN_STEP))
    axins.xaxis.set_minor_formatter(NullFormatter())
    axins.xaxis.set_major_formatter(FormatStrFormatter('%.0f'))

    # 인셋 y축도 fixed로 강제할 수 있게
    if INSET_LIMIT_MODE == 'fixed' and INSET_Y_MAX is not None:
        axins.set_ylim(0, float(INSET_Y_MAX))

# ====== 저장/보기 ======
fig_path = os.path.join(out_dir, 't95_hist_grouped_readable.png')
fig.savefig(fig_path, dpi=150, bbox_inches='tight')
print(f"[SAVE] figure -> {fig_path}")
plt.show()
