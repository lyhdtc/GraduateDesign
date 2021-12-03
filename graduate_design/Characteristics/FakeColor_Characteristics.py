import sys
sys.path.append('../test/')
from matplotlib import pyplot as plt
import numpy as np
from numba import jit
import cv2
from cv2 import dnn_superres
import TestScripts


#不支持的：
# cc.color_coherence_vector输出为二维向量
# tc.glcm_feature貌似存在越界情况
# tc.rotation_invariant_LBP本身输出就是灰度图了
# tc.tamura_feature需要多核优化
# tc.dwt_feature输出为全白，需要看下代码
# tc.laws_feature本身输出就是一组图像了
# loss function 输入为彩色图像，还需要重写一组

# 单通道伪彩色，输入为灰度矩阵， 
# @jit(nopython=True)
@TestScripts.timmer
def __single_channel_fake_color(gray_img_a,gray_img_b,  func , step = 8, size_w = 0, size_h = 0, data_num = 0,*args, **kwargs):
    w = gray_img_a.shape[0]
    h = gray_img_a.shape[1]
    if((w%size_w!=0)or(h%size_h!=0)):
        print('Please check slide window SIZE!')
        return
    ans_a = np.zeros((int(w/step), int(h/step)))
    ans_b = np.zeros((int(w/step), int(h/step)))

    
    for i in range(int((w-size_w)/step)):
        for j in range(int((h-size_h)/step)):
            # print(i,j)
            ans_a[i][j] = func(gray_img_a[i*step:(i*step+size_w), j*step:(j*step+size_h)], *args, **kwargs)[data_num]
            ans_b[i][j] = func(gray_img_b[i*step:(i*step+size_w), j*step:(j*step+size_h)], *args, **kwargs)[data_num]
    ans = np.abs(ans_a-ans_b)
    print(np.shape(ans))
    
    ans = (255*ans) / np.max(ans)
    ans = ans.astype(np.uint8)
    
    
    ans_highsolution = cv2.resize(ans, None, fx=step, fy=step, interpolation=cv2.INTER_LINEAR)
    # sr = dnn_superres.DnnSuperResImpl_create()
    # sr.readModel("./model/LapSRN_x4.pb")
    # sr.setModel("bilinear", step)
    # ans_highsolution = sr.upsample(ans)
    ans_fakecolor = cv2.applyColorMap(ans_highsolution, cv2.COLORMAP_JET)
    plt.figure()
    plt.imshow(ans_fakecolor)
    plt.axis('off')
    plt.show()
    # cv2.imshow('asdf', ans_fakecolor)
    return ans
   
@TestScripts.timmer
def single_channel_fake_color(gray_img_a,gray_img_b,  func , step = 8, size_w = 0, size_h = 0, data_num = 0,*args, **kwargs):
    w = gray_img_a.shape[0]
    h = gray_img_a.shape[1]
    if((w%size_w!=0)or(h%size_h!=0)):
        print('Please check slide window SIZE!')
        return
    ans_a = []
    ans_b = []    
    for i in range(int(w/step)):
        col_a = []
        col_b = []
        for j in range(int(h/step)):            
            col_a.append(func(gray_img_a[i*step:(i*step+size_w), j*step:(j*step+size_h)], *args, **kwargs))
            col_b.append(func(gray_img_b[i*step:(i*step+size_w), j*step:(j*step+size_h)], *args, **kwargs))
        ans_a.append(col_a)
        ans_b.append(col_b)    

    ans_a = np.array(ans_a)
    ans_b = np.array(ans_b)
    ans = np.abs(ans_a-ans_b)
    ans = ans.transpose(2,0,1)
    ans = (255*ans) / np.max(ans)

    ans_len = ans.shape[0]
    print(np.shape(ans[1]))
    for i in range(ans_len):
        plt.figure()
        plt.imshow(ans[i],vmin = 0, vmax = 255,cmap = "hot")
        plt.colorbar()
    
   
    return  
 
