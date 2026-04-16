# -*- coding: utf-8 -*-
"""
Created on Mon Dec 27 16:45:28 2021

@author: hp
"""
from numpy.ma import true_divide
import pandas as pd
import numpy as np
from hpelm import ELM
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error as MSE
from sklearn.metrics import mean_absolute_error as MAE
from sklearn.metrics import r2_score as R2
from scipy.stats import ttest_rel
from GWO import GWO

def SMAPE(y_true, y_pred):
    return 2.0 * np.mean(np.abs(y_pred - y_true) / (np.abs(y_pred) + np.abs(y_true)))


# 制表符分隔一行结果（便于粘贴 Excel）；SD 为各次重复 RMSE 的样本标准差；
# t-test：跑 GWO 时为「GWO 最优 vs FIXED_NEURONS 基线」；不跑 GWO 且 ELM_ONLY 时为「FIXED_NEURONS vs ELM_COMPARE_NEURONS」；
# 不跑 GWO 且分解模式时为「全体 FIXED_NEURONS vs 全体 ELM_COMPARE_NEURONS」
RESULT_TABLE_SEP = "\t"


def _format_test_summary_row(mae_l, rmse_l, smape_l, r2_l, tt_stat, tt_p):
    mae_a = np.asarray(mae_l, dtype=float)
    rmse_a = np.asarray(rmse_l, dtype=float)
    smape_a = np.asarray(smape_l, dtype=float)
    r2_a = np.asarray(r2_l, dtype=float)
    sd_rmse = np.std(rmse_a, ddof=1) if rmse_a.size > 1 else 0.0
    try:
        ts = float(np.asarray(tt_stat).ravel()[0])
        pv = float(np.asarray(tt_p).ravel()[0])
    except (TypeError, ValueError, IndexError):
        ts, pv = float("nan"), float("nan")
    t_cell = f"t={ts:.6g},p={pv:.6g}"
    return RESULT_TABLE_SEP.join(
        [
            f"{np.mean(mae_a):.6g}",
            f"{np.mean(rmse_a):.6g}",
            f"{np.mean(smape_a) * 100:.6g}",
            f"{np.mean(r2_a):.6g}",
            f"{sd_rmse:.6g}",
            t_cell,
        ]
    )

# ---------- 运行模式 ----------
# ELM_ONLY：False 读分解多列 csv；True 只读 LABEL_PATH 的 RRP，单列滑窗（不做分解）
# RUN_GWO： True 用 GWO 搜隐层神经元（单列时 dim=1；多列时每分量一个整数）
#         False 不跑 GWO，隐层用 FIXED_NEURONS（形状与列数一致）
# 组合示例：
#   分解 + GWO + ELM     ： ELM_ONLY=False, RUN_GWO=True
#   仅 RRP + GWO + ELM   ： ELM_ONLY=True,  RUN_GWO=True
#   仅 RRP + 固定 ELM    ： ELM_ONLY=True,  RUN_GWO=False
#   EMD + 固定 ELM       ： ELM_ONLY=False, TYPE='EMD', RUN_GWO=False（需先有 datasets/price_EMD_no_clip.csv）
#   EMD + GWO + ELM      ： ELM_ONLY=False, TYPE='EMD', RUN_GWO=True
ELM_ONLY = False
RUN_GWO = True

# 与 decomposition.py 的 DECOM_METHOD 保持一致；ELM_ONLY=False 时需先运行 decomposition 生成该文件
TYPE = 'SSA'
WINDOW_SIZE = 4
LABEL_PATH = 'datasets/demand.csv'
DATA_PATH = 'datasets/demand_' + TYPE + '.csv'

# 隐层神经元：RUN_GWO=False 时为必选；RUN_GWO=True 时作「基线」参与配对 t（与 GWO 最优比）
# 单列(ELM_ONLY=True)须为 int；多列可为 int（各列相同）或与列数相等的 list
FIXED_NEURONS = 20
# 不跑 GWO 且 ELM_ONLY=True 时：与 FIXED_NEURONS 配对的另一隐层规模；多列不跑 GWO 时为各列同一整数
ELM_COMPARE_NEURONS = 20

# 每个分量上训练 K 个 ELM，预测取平均（K>1 可明显降低 EMD+ELM 等因随机初值导致的高 SD）；K=1 与原先完全一致
ELM_ENSEMBLE_K = 1

# GWO 搜索范围与迭代（RUN_GWO=True 时有效；单列时仍用 NEURON_LB/UB 搜那一个数）
NEURON_LB = 5
NEURON_UB = 120
GWO_N_WOLVES = 30
GWO_MAX_ITER = 50
GWO_SEED = 42


def _fitness_random_state(neurons):
    """
    由神经元向量与 GWO_SEED 派生整数种子（不依赖 Python 内置 hash 的进程随机盐）。
    GWO 调用 fitness 时必须传入，否则 hpelm 用全局 np.random，每次新开进程结果都会变。
    """
    h = int(GWO_SEED) & 0xFFFFFFFF
    for k, n in enumerate(np.ravel(np.asarray(neurons, dtype=np.int64))):
        h = (h * 1_000_003 + int(n) * (k + 97) + k) & 0xFFFFFFFF
    # np.random.seed 接受 [0, 2**32) 的整数
    return int(h)


def _elm_ensemble_seed(random_state, col_idx, member_idx):
    """同一 random_state 下，分量 col、集成第 member 个 ELM 的独立种子。"""
    return int((int(random_state) + int(col_idx) * 100_003 + int(member_idx) * 1_000_003) & 0x7FFFFFFF)


def get_windowed_data(ser, window_size=WINDOW_SIZE):
    if window_size <= 0 or not isinstance(window_size, int):
        raise ValueError("window_size应该是正整数！~")
        
    n_sample = ser.shape[0]
    array = ser.values
    x, y = [], []
    for i in range(n_sample - window_size):
        x.append(array[i: i + window_size])
        y.append(array[i + window_size])
    return np.array(x), np.array(y)

def my_ELM(subseries, neurons, norm=None, test_size=0.2, pred_on_test=True, random_state=None):
    if len(neurons) != subseries.shape[1]:
        raise ValueError('神经元个数 与 子序列个数 不匹配 !~')
    k_ens = int(ELM_ENSEMBLE_K)
    if k_ens < 1:
        raise ValueError("ELM_ENSEMBLE_K 须 >= 1")
    if random_state is not None and k_ens == 1:
        np.random.seed(int(random_state))
    T_pred = []
    nums_neuron = np.array(neurons)
    for i in range(subseries.shape[1]):
        scaler = StandardScaler()
        temp_X, temp_t = get_windowed_data(subseries.iloc[:, i])
        temp_X = scaler.fit_transform(temp_X)

        X_train, X_test, t_train, t_test = train_test_split(temp_X, temp_t, test_size=test_size,
                                                            shuffle=False)
        test_size = len(t_test)

        if k_ens == 1:
            model = ELM(inputs=WINDOW_SIZE, outputs=1, norm=None, batch=20)
            model.add_neurons(number=int(nums_neuron[i]), func='sigm')
            model.train(X_train, t_train)
            if pred_on_test:
                t_pred = model.predict(X_test).flatten()
            else:
                t_pred = model.predict(X_train).flatten()
        else:
            acc = []
            for j in range(k_ens):
                if random_state is not None:
                    np.random.seed(_elm_ensemble_seed(random_state, i, j))
                model = ELM(inputs=WINDOW_SIZE, outputs=1, norm=None, batch=20)
                model.add_neurons(number=int(nums_neuron[i]), func='sigm')
                model.train(X_train, t_train)
                if pred_on_test:
                    acc.append(model.predict(X_test).flatten())
                else:
                    acc.append(model.predict(X_train).flatten())
            t_pred = np.mean(acc, axis=0)
        T_pred.append(t_pred)

    T_pred = np.array(T_pred).sum(axis=0)
    return T_pred, test_size

# %% 1 导入数据
t = pd.read_csv(LABEL_PATH, index_col=1).loc[:, 'demand'].values
up_quantile = np.quantile(t, 0.99)
t = np.clip(t, a_min=None, a_max=up_quantile)

if ELM_ONLY:
    data = pd.DataFrame({'demand': t})
else:
    data = pd.read_csv(DATA_PATH, index_col=0)
    num_fs = data.shape[1]
    if TYPE == 'LMD':
        column_names = ['PF' + str(i) for i in range(1, data.shape[1])] + ['RES']
    elif TYPE == 'SSA':
        column_names = ['Y' + str(i) for i in range(1, data.shape[1] + 1)]
    else:
        column_names = ['IMF' + str(i) for i in range(1, data.shape[1] + 1)]
    data.rename(columns=dict(zip(data.columns, column_names)), inplace=True)


def _elm_rmse_fitness(neurons):
    """GWO 目标：测试集 RMSE 越小越好。"""
    ns = np.asarray(neurons, dtype=int)
    T_pred, ts = my_ELM(
        data,
        ns,
        pred_on_test=True,
        random_state=_fitness_random_state(ns),
    )
    return MSE(t[-ts:], T_pred) ** 0.5


# 固定全局 numpy 随机流，避免 GWO 以外代码在未传 seed 时扰动可复现性
np.random.seed(GWO_SEED)

if RUN_GWO:
    if ELM_ONLY:
        print("模式: 原始 demand + GWO + ELM（无分解）")
    searcher = GWO(
        dim=data.shape[1],
        lb=NEURON_LB,
        ub=NEURON_UB,
        fitness=_elm_rmse_fitness,
        n_wolves=GWO_N_WOLVES,
        max_iter=GWO_MAX_ITER,
        seed=GWO_SEED,
    )
    searcher.optimize()
    print("GWO 最优神经元数:", searcher.gbest, "验证 RMSE:", searcher.gbest_score)
    gbest_arr = np.array(searcher.gbest, dtype=int)
elif ELM_ONLY:
    if not isinstance(FIXED_NEURONS, int):
        raise ValueError("ELM_ONLY=True 且 RUN_GWO=False 时 FIXED_NEURONS 请设为单个 int")
    gbest_arr = np.array([FIXED_NEURONS], dtype=int)
    print("纯 ELM（无分解、无 GWO），隐层神经元:", int(FIXED_NEURONS))
else:
    if isinstance(FIXED_NEURONS, int):
        gbest_arr = np.full(data.shape[1], FIXED_NEURONS, dtype=int)
    else:
        gbest_arr = np.asarray(FIXED_NEURONS, dtype=int)
    if gbest_arr.shape[0] != data.shape[1]:
        raise ValueError("FIXED_NEURONS 展开后长度须等于分量列数")
    print("分解序列 + 固定隐层神经元（无 GWO）:", gbest_arr)

# %% 2 训练
N_REPEAT = 50
STAT_PAIR_SEED0 = GWO_SEED + 10_000
mae, rmse, smape, r2 = [], [], [], []
for i in range(N_REPEAT):
    T_pred, test_size = my_ELM(data, gbest_arr, random_state=GWO_SEED + i)
    rmse.append(MSE(t[-test_size:], T_pred) ** 0.5)
    mae.append(MAE(t[-test_size:], T_pred))
    smape.append(SMAPE(t[-test_size:], T_pred))
    r2.append(R2(t[-test_size:], T_pred))

# 配对 t 检验：主配置 vs 对比配置（同次随机种子，RMSE 配对）
if RUN_GWO:
    if isinstance(FIXED_NEURONS, int):
        neurons_fix = np.full(data.shape[1], FIXED_NEURONS, dtype=int)
    else:
        neurons_fix = np.asarray(FIXED_NEURONS, dtype=int)
    if neurons_fix.shape[0] != data.shape[1]:
        raise ValueError("FIXED_NEURONS 展开后长度须等于分量列数")
elif ELM_ONLY:
    if not isinstance(ELM_COMPARE_NEURONS, int):
        raise ValueError("ELM_ONLY=True 且 RUN_GWO=False 时 ELM_COMPARE_NEURONS 请设为单个 int")
    neurons_fix = np.array([ELM_COMPARE_NEURONS], dtype=int)
else:
    if not isinstance(ELM_COMPARE_NEURONS, int):
        raise ValueError("分解模式且 RUN_GWO=False 时 ELM_COMPARE_NEURONS 请设为 int（各列同一基线）")
    neurons_fix = np.full(data.shape[1], ELM_COMPARE_NEURONS, dtype=int)
rmse_gwo, rmse_fix = [], []
for i in range(N_REPEAT):
    rs = STAT_PAIR_SEED0 + i
    T_gwo, ts_g = my_ELM(data, gbest_arr, random_state=rs)
    T_fix, ts_f = my_ELM(data, neurons_fix, random_state=rs)
    if ts_g != ts_f:
        raise RuntimeError("配对检验要求两次 my_ELM 的 test_size 一致")
    rmse_gwo.append(MSE(t[-ts_g:], T_gwo) ** 0.5)
    rmse_fix.append(MSE(t[-ts_f:], T_fix) ** 0.5)
rmse_gwo = np.asarray(rmse_gwo)
rmse_fix = np.asarray(rmse_fix)
tt_stat, tt_p = ttest_rel(rmse_gwo, rmse_fix)

print(RESULT_TABLE_SEP.join(["MAE", "RMSE", "SMAPE/(%)", "R2", "SD", "t-test"]))
print(_format_test_summary_row(mae, rmse, smape, r2, tt_stat, tt_p))
