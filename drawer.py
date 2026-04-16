# -*- coding: utf-8 -*-
"""
Created on Wed Dec 29 09:37:12 2021

@author: hp
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import palettable

_BASE = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(_BASE, 'datasets', 'price_LMD_no_clip.csv')
RAW_PATH = os.path.join(_BASE, 'datasets', 'price.csv')
RESULT_PATH = os.path.join(_BASE, 'results', 'imgs')
# 五条预测序列：CSV 列名见下；若为 True 则从剪贴板读入（与 Excel 相同列顺序）
PRED_CSV = os.path.join(_BASE, 'results', 'model_predictions.csv')
LOAD_PREDS_FROM_CLIPBOARD = False
_PRED_COLS = (
    'DPSO_ELM',
    'EMD_DPSO_ELM',
    'LMD_DPSO_ELM',
    'CEEMDAN_DPSO_ELM',
    'SSA_DPSO_ELM',
)

plt.rcParams['text.usetex'] = True
plt.rcParams['font.size'] = 14
# 分解分量
data = pd.read_csv(DATA_PATH, index_col=0)
t = data.values.sum(axis=1)
# SSA
# fig = plt.figure(figsize=(12,7))
# for i in range(data.shape[1]):
#     ax = fig.add_subplot(4, 2, i + 1)
#     ax.plot(data.iloc[:, i], label='$x_{C_' + str(i+1) + '}$',
#             linewidth=1)
#     if i == 0:
#         ax.plot(t, label='Real', alpha=0.5, linewidth=0.5)
#         ax.legend(fontsize=12, ncol=2, bbox_to_anchor=(0.53, 0.95))
#     else:
#         ax.legend(loc='upper right', fontsize=12, ncol=2)

# EMD / CEEMDAN
# fig = plt.figure(figsize=(12,6))
# lwd = [0.3, 0.4, 0.5, 0.8, 1, 1, 1.5, 1.5, 2]
# for i in range(data.shape[1]):
#     ax = fig.add_subplot(5, 2, i + 1)
#     ax.plot(data.iloc[:, i], label='IMF' + str(i+1),
#             linewidth=lwd[i])
#     ax.legend(loc='upper right', fontsize=10)
#     # if i + 1 == 5:
#     #     ax.fill_between(np.arange(1166, 1313), y1=24, y2=-22, alpha=0.3, color='#FF7F0D')
#     #     ax.set_ylim(-22, 24)
#     # if i + 1 == 6:
#     #     ax.fill_between(np.arange(1494, 1731), y1=30, y2=-26, alpha=0.3, color='#2BA02C')
#     #     ax.set_ylim(-26, 30)
# fig.subplots_adjust(hspace=0.4)

# LMD
# fig = plt.figure(figsize=(12,5))
# lwd = [0.3, 0.5, 0.7, 0.9, 1.2, 1.5, 2]
# labels = ['PF' + str(i+1) for i in range(6)] + ['RES']
# for i in range(data.shape[1]):
#     ax = fig.add_subplot(4, 2, i + 1) 
#     ax.plot(data.iloc[:, i], label=labels[i], linewidth=lwd[i])
#     ax.legend(loc='upper right', fontsize=10)
# fig.subplots_adjust(hspace=0.4)
# fig.savefig(RESULT_PATH + 'LMD_DECOM.png', dpi=500)

# %% 原始数据展示
# data = pd.read_csv(RAW_PATH, index_col=1).loc[:, 'RRP']
# up_quantile = np.quantile(data, 0.99)
# data = np.clip(data, a_min=None, a_max=up_quantile)

# fig = plt.figure(figsize=(10,4))
# ax = fig.gca()
# ax.plot(data.values, linewidth=0.5)
# ax.set_xlabel('Days')
# ax.set_ylabel('Recommended Retail Price (RRP)')
# ax.set_xlim(0, 2106)
# plt.tight_layout()
# fig.savefig(RESULT_PATH + '99_data_show.png', dpi=500)
# %% WINDOW_SIZE
# rmse = [2.836660, 2.783656, 2.419471, 2.424601, 2.479641, 2.592957, 2.694143]

# fig = plt.figure()
# ax = fig.gca()
# ax.set_prop_cycle('color', palettable.mycarta.Cube1_10_r.mpl_colors)
# ax.plot(range(2, 9), rmse, 'o-',  linewidth=2)
# ax.set_xlabel('Window Size')
# ax.set_ylabel('RMSE on Train Set')
# for i in range(7):
#     ax.text(i + 2, rmse[i], '{:.3f}'.format(rmse[i]), ha='center', va='bottom')
# fig.savefig(RESULT_PATH + 'search_window_size.png', dpi=500)
# 模型效果
# data = pd.read_clipboard(index_col=0)

def draw_my_figure(data, gap, width, dialation=2, fig_dict=None):
    n, p = data.shape[0], data.shape[1]
    if fig_dict:
        fig = plt.figure(**fig_dict)
    else:
        fig = plt.figure()
    ax = fig.gca()
    ax.set_prop_cycle('color', palettable.mycarta.Cube1_10_r.mpl_colors)
    xs = np.arange(0, dialation*p, dialation)
    for index, i in enumerate(data.values):
        plt.bar(xs+((index-(n-1)/2) * (gap + width)), i, width, label=data.index[index])
        for j in range(len(i)):
            plt.text(xs[j]+((index-(n-1)/2) * (gap + width)), i[j], '{:.1f}'.format(i[j]),
                      ha='center', va='bottom', fontsize=12)
    ax.set_xticks(xs, data.columns)
    ax.legend(loc='upper left', ncol=2, fontsize=11)
    return fig, ax
        
# fig, ax = draw_my_figure(data.iloc[:, :-2], gap=0.06, width=0.17, dialation=3,
#                           fig_dict={'figsize':(10,6)})
# fig.savefig(RESULT_PATH + 'model_results_1.png', dpi=500)
# fig, ax = draw_my_figure(data.iloc[:, -2:], gap=0.06, width=0.17, dialation=3,
#                           fig_dict={'figsize':(10,6)})
# fig.savefig(RESULT_PATH + 'model_results_2.png', dpi=500)
# 预测效果
def _load_prediction_columns():
    if LOAD_PREDS_FROM_CLIPBOARD:
        pred = pd.read_clipboard(index_col=0)
    else:
        if not os.path.isfile(PRED_CSV):
            raise FileNotFoundError(
                '找不到预测结果文件：\n  {path}\n'
                '请先创建该 CSV（首列为索引，列名含 {cols}），'
                '或在本文件顶部设置 LOAD_PREDS_FROM_CLIPBOARD = True 后从 Excel 复制对应列再运行。'.format(
                    path=PRED_CSV,
                    cols=', '.join(_PRED_COLS)))
        pred = pd.read_csv(PRED_CSV, index_col=0)
    out = []
    for i, name in enumerate(_PRED_COLS):
        if name in pred.columns:
            s = pred[name].values.ravel()
        elif pred.shape[1] > i:
            s = pred.iloc[:, i].values.ravel()
        else:
            raise KeyError('缺少列 {!r} 或第 {} 列'.format(name, i))
        out.append(s.astype(float))
    return tuple(out)


temp, temp1, temp2, temp3, temp4 = _load_prediction_columns()
NUM_SHOW = 50
for _name, _arr in zip(_PRED_COLS, (temp, temp1, temp2, temp3, temp4)):
    if _arr.size < NUM_SHOW:
        raise ValueError('{} 长度 {} 小于 NUM_SHOW={}'.format(_name, _arr.size, NUM_SHOW))
os.makedirs(RESULT_PATH, exist_ok=True)
fig = plt.figure(figsize=(10,5))
ax = fig.gca()
ax.set_prop_cycle('color', palettable.mycarta.Cube1_5_r.mpl_colors)
ax.plot(temp[-NUM_SHOW:], label="DPSO-ELM", alpha=0.5)
ax.scatter(range(NUM_SHOW), temp[-NUM_SHOW:], s=10, alpha=0.5)
ax.plot(temp1[-NUM_SHOW:], label="EMD-DPSO-ELM", alpha=0.5)
ax.scatter(range(NUM_SHOW), temp1[-NUM_SHOW:], s=10, alpha=0.5)
ax.plot(temp2[-NUM_SHOW:], label="LMD-DPSO-ELM", alpha=0.5)
ax.scatter(range(NUM_SHOW), temp2[-NUM_SHOW:], s=10, alpha=0.5)
ax.plot(temp3[-NUM_SHOW:], label="CEEMDAN-DPSO-ELM", alpha=0.5)
ax.scatter(range(NUM_SHOW), temp3[-NUM_SHOW:], s=10, alpha=0.5)
ax.plot(temp4[-NUM_SHOW:], label="SSA-DPSO-ELM", alpha=0.5)
ax.scatter(range(NUM_SHOW), temp4[-NUM_SHOW:], s=10, alpha=0.5)
ax.plot(t[-NUM_SHOW:], label="Real", alpha=0.5, color='black')
ax.scatter(range(NUM_SHOW), t[-NUM_SHOW:], s=10, c='black', alpha=0.2)
ax.legend(fontsize=12)
ax.set_xlabel("Days")
ax.set_ylabel("Recommended Retail Price (RRP)")
ax.set_xticks([0, 16, 32, 49], ['2020-08-18', '2020-09-03', '2020-09-29' ,'2020-10-06'])
os.makedirs(RESULT_PATH, exist_ok=True)
fig.savefig(os.path.join(RESULT_PATH, 'show_50_prediction.png'), dpi=500)