import sys
sys.path.append('../test/')
import matplotlib as plt
import numpy as np
from numba import jit
import TestScripts

# 单通道为彩色，输入为灰度矩阵， 
# @TestScripts.timmer
@jit
def single_channel_fake_color(gray_img_a,gray_img_b,  func , size_w = 0,size_h = 0, data_num = 0,*args, **kwargs):
    w = gray_img_a.shape[0]
    h = gray_img_a.shape[1]
    if((w%size_w!=0)or(h%size_h!=0)):
        print('Please check slide window SIZE!')
        return
    ans_a = np.zeros((w-size_w, h-size_h))
    ans_b = np.zeros((w-size_w, h-size_h))

    
    for i in range(w-size_w):
        for j in range(h-size_h):
            print(i,j)
            ans_a[i][j] = func(gray_img_a[i:i+size_w, j:j+size_h], *args, *kwargs)[data_num]
            ans_b[i][j] = func(gray_img_b[i:i+size_w, j:j+size_h], *args, *kwargs)[data_num]
    ans = np.abs(ans_a-ans_b)
    print(ans)

    return ans
            
            
    
        