# -*- coding: utf-8 -*-
"""
Created on Tue Dec 28 22:14:40 2021

@author: hp
"""
import numpy as np

def get_windowed_data(array, window_size):
    if window_size <= 0 or not isinstance(window_size, int):
        raise ValueError("window_size应该是正整数！~")
        
    n_sample = len(array)
    x = []
    for i in range(n_sample - window_size + 1):
        x.append(array[i: i + window_size])
    return np.array(x)

def get_inv_diag(mat):
    out = [np.mean(mat[::-1, :].diagonal(i)) for i in range(-mat.shape[0]+1, mat.shape[1])]
    return np.array(out)

def SSA(ser, com_dim, sig_vals=False):
    ser = get_windowed_data(ser, com_dim)
    u, s, vh = np.linalg.svd(ser.T)
    r = np.linalg.matrix_rank(ser)

    Xs = [s[i] * u[:, i][:, np.newaxis] @ vh[i, :][:, np.newaxis].T for i in range(r)]
    Xs = np.array(Xs)
    
    Y = np.array([get_inv_diag(i) for i in Xs]).T
    if sig_vals:
        return Y, s
    else:
        return Y

if __name__ == '__main__':
    import pandas as pd
    import matplotlib.pyplot as plt
    import time
    
    DECOM_WINDOW = 9
    DATA_PATH = './datasets/price.csv'
    
    data = pd.read_csv(DATA_PATH, index_col=1).loc[:, 'RRP'].values

    begin = time.time()
    Y, s = SSA(data, DECOM_WINDOW, sig_vals=True)
    end = time.time()
    print("SSA: {}".format(end - begin))
    
    ## 分解展示
    # fig = plt.figure()
    # for i in range(DECOM_WINDOW):
    #     ax = fig.add_subplot(eval('52' + str(i + 1)))
        # if i == 0:
            # ax.plot(data)
        # ax.plot(Y[:, i])
    
    ## SSA 结果存储
    # temp_df = pd.DataFrame(Y)
    # temp_df.to_csv('./datasets/SSA_search/SSA_L=9.csv')