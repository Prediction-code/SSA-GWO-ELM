# -*- coding: utf-8 -*-
"""
Created on Mon Dec 27 20:01:45 2021

@author: hp
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PyEMD import CEEMDAN, EMD, EEMD
from PyLMD import LMD
from SSA import SSA

WINDOW_SIZE = 6
DATA_PATH = './datasets/demand.csv'

def decomposition(ser, method, SSA_window=None, save=False, timer=True):   
    if timer:
        import time
        begin = time.time()
        
    if method in ['EMD', 'EEMD', 'CEEMDAN']:
        decomposer = eval(method)()
        result = pd.DataFrame(decomposer(ser).T)
        end = time.time()
    
    elif method == 'LMD':
        decomposer = LMD()
        PFs, residue = decomposer.lmd(ser)
        result = pd.DataFrame(np.row_stack((PFs, residue)).T)
        end = time.time()
    
    elif method == 'SSA' and not SSA_window:
        raise ValueError('使用SSA方法需要指定嵌入维数 SSA_window ~')
        
    elif method == 'SSA' and SSA_window:
        result = pd.DataFrame(SSA(ser, SSA_window))
        end = time.time()
        
    else:
        raise ValueError('不支持 ' + method + ' 这个分解方法.')
    
    if save:
        result.to_csv('./datasets/demand_' + method + '.csv')
    print("分解方法 " + method + " 所消耗时间为：{:.5f}".format(end - begin))
    return result
    
data = pd.read_csv(DATA_PATH, index_col=1).loc[:, 'demand']

up_quantile = np.quantile(data, 0.99)
data = np.clip(data, a_min=None, a_max=up_quantile)

if __name__ == '__main__':
    temp_df = decomposition(data.values, method='SSA', SSA_window=8, 
                            save=True, timer=True)
# 结果可视化
# n = temp_df.shape[1]
# fig, ax = plt.subplots(nrows=int(np.ceil(n/2)), ncols=2)
# ax = ax.flatten()
# for i in range(n):
    # ax[i].plot(temp_df.iloc[:, i])
    # if i != 8:
    #     ax[i].plot(pfs[i])
    #     ax[i].set_title('PF {}'.format(i+1))
    # else:
    #     ax[i].plot(res)
    #     ax[i].set_title('RES')
