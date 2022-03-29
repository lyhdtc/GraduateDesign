'''
Author: lyh
Date: 2022-03-21 19:10:36
LastEditors: lyh
LastEditTime: 2022-03-24 09:01:40
FilePath: /GraduateDesign/AlgoTest.py
Description: 

Copyright (c) 2022 by lyh, All Rights Reserved. 
'''

from email import iterators
import enum
from pyparsing import col
from Color import ColorAlgorithrm as ca
from Texture import TextureAlgorithrm as ta
import numpy as np
import cv2
from matplotlib import pyplot as plt
from skimage.feature import greycomatrix, greycoprops
from scipy import signal as sg
LAB_COLOR_CHANNEL = {
    0: 'l',
    1: 'a',
    2: 'b'
}
RGB_COLOR_CHANNEL = {
    0: 'b',
    1: 'g',
    2: 'r'
}


def path2labmat(path_origin, path_new):
    img_origin  = cv2.imread(path_origin)
    img_new     = cv2.imread(path_new)

    lab_img_origin = cv2.cvtColor(img_origin, cv2.COLOR_BGR2LAB)
    lab_img_new    = cv2.cvtColor(img_new, cv2.COLOR_BGR2LAB)

    lab_matrix_origin = np.array(cv2.split(lab_img_origin))
    lab_matrix_new    = np.array(cv2.split(lab_img_new))
    return lab_matrix_origin, lab_matrix_new

def color_characteristics_histogram(lab_img, figsize, folder):
    plt.figure(figsize=figsize)
    plt.title('histogram')
    path = folder + 'Color_Histogram.jpg'
    bgr_img = cv2.cvtColor(lab_img, cv2.COLOR_LAB2BGR)

    for i in range(3):
        ax1 = plt.subplot(3,1,i+1)
        hist_title = RGB_COLOR_CHANNEL.get(i)+' channel'
        ax1.set_title(hist_title)
        hist = ca.histogram(bgr_img[i])        
        plt.plot(hist,RGB_COLOR_CHANNEL.get(i))

    plt.tight_layout()
    plt.plot()
    # plt.show()
    plt.savefig(path)
    plt.close()
    return

def brightness_test(path_origin, path_new):
    lab_matrix_origin , lab_matrix_new = path2labmat(path_origin, path_new)
    ans_origin = ca.brightness(lab_matrix_origin)
    ans_new    = ca.brightness(lab_matrix_new)
    print(f'brightness result:\n  brightness_origin = {ans_origin}\n  brightness_new = {ans_new}\n  delta = {ans_new-ans_origin}')
# TODO:
def constract_test(path_origin, path_new):
    lab_matrix_origin , lab_matrix_new = path2labmat(path_origin, path_new)
    ans = ca.constract(lab_matrix_new, lab_matrix_origin)
    print(f'constract result:{ans}')

# 生成直方图
# color_characteristics_histogram(lab_img, (18,10), 'AlgoTest/')

def exposure_test(path_origin, path_new):
    lab_matrix_origin , lab_matrix_new = path2labmat(path_origin, path_new)
    ans = ca.exposure(lab_matrix_new, lab_matrix_origin)
    print(ans)

def saturation_test(path_origin, path_new):
    lab_matrix_origin , lab_matrix_new = path2labmat(path_origin, path_new)
    ans = ca.saturation(lab_matrix_new, lab_matrix_origin)
    print(f'saturation result:{ans}')
    
def white_balance_test(path_origin, path_new):
    lab_matrix_origin, lab_matrix_new = path2labmat(path_origin, path_new)
    print(ca.white_balance(lab_matrix_origin), ca.white_balance(lab_matrix_new))
    
def color_coherence_vector_test(path_origin):
    test_img = cv2.imread(path_origin)
    test_img = cv2.resize(test_img, (5,5), cv2.INTER_NEAREST)
    b,g,r = np.array(cv2.split(test_img))
    print(b)
    b_comp = (b/32).astype(int)
    print(b_comp)
    print(ca.color_coherence_vector(b, color_threshold=8, area_threshold=2))


def glcm_test():
    DIRECTION = {
        0: [1,0],
        1: [0,1],
        2: [1,1],
        3: [-1,1]
    }
    # test_mat = np.random.randint(0,3,(4,4)).astype(np.uint8)
    test_mat = np.array([[0,1,2,2],
                         [0,2,1,2],
                         [2,0,2,1],
                         [0,0,0,0]])
    print(test_mat)
    print('------------------------------------\n\n')
    for i in range(4):
        dx,dy = DIRECTION[i]
        glcm_ans = ta.glcm(test_mat, dx, dy, gray_level=3)
        print(glcm_ans)
        print('------------------------------------\n\n')
    glcm = greycomatrix(test_mat, [1], [0], levels=3)
    print(glcm) 

def specular_shadow_test(path_origin, path_new):
    lab_matrix_origin, lab_matrix_new = path2labmat(path_origin, path_new)
    ans = ca.specular_shadow(lab_matrix_origin, option='shadow')
    ans = ans*255
    ans = ans.astype(np.uint8)
    cv2.imshow('asdf',ans)
    cv2.waitKey(0)
    # img = cv2.imread(path_origin)

   

    # # 分离 RGB 三个通道，注意：openCV 中图像格式是 BGR
    # srcR = img[:, :, 2]
    # srcG = img[:, :, 1]
    # srcB = img[:, :, 0]

    # # 将原图转成灰度图
    # # grayImg = 0.299 * srcR + 0.587 * srcG + 0.114 * srcB
    # grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # img = img.astype(np.float)/255.0
    # # 高光选区
    # maskThreshold = 0.64
    # luminance = grayImg * grayImg
    # luminance = np.where(luminance > maskThreshold, luminance, 0)


    # # 阴影选区
    # # maskThreshold = 0.33
    # # luminance = (1 - grayImg) * (1 - grayImg)
    # # luminance = np.where(luminance > maskThreshold, luminance, 0)

    # mask = luminance > maskThreshold

 
    # # 显示正交叠底图
    # # img[:, :, 0] = luminance
    # # img[:, :, 1] = luminance
    # # img[:, :, 2] = luminance

    # # 显示选区内原图
    # img[:, :, 0][~mask] = 0
    # img[:, :, 1][~mask] = 0
    # img[:, :, 2][~mask] = 0

    # img = img * 255
    # img = img.astype(np.uint8)

    # mask = np.where(mask==True, 255, 0)
    # mask = mask.astype(np.uint8)
    
    # # 创建图片显示窗口
    # title = "ShadowHighlight"
    # cv2.namedWindow(title, cv2.WINDOW_NORMAL)   
    # cv2.resizeWindow(title, 800, 600)
    # cv2.moveWindow(title, 0, 0)
    # while True:
    #     # 循环显示图片，按 ‘q’ 键退出
    #     cv2.imshow(title, mask)
    #     if cv2.waitKey(1) == ord('q'):
    #         break
    # cv2.destroyAllWindows() 

def tamuratest(gray_img):
    kmax = 1
    gray_img = np.array(gray_img)
    gray_img = gray_img.astype(np.float64)
    w = gray_img.shape[0]
    h = gray_img.shape[1]
    kmax = kmax if (np.power(2,kmax) < w) else int(np.log(w) / np.log(2))
    kmax = kmax if (np.power(2,kmax) < h) else int(np.log(h) / np.log(2))
    
    horizon = np.zeros([kmax,w,h])
    vertical = np.zeros([kmax,w,h])
    Sbest = np.zeros([w,h])
    average_gray = np.zeros([kmax,w,h])
    horizon = np.zeros([kmax,w,h])
    horizon2 = np.zeros([kmax,w,h])
    vertical = np.zeros([kmax,w,h])
    Sbest = np.zeros([w,h])

    k=0
    
    window = np.power(2,k)
    
    nurcle = np.ones((2*window+1, 2*window+1))
    nurcle[:,0] = 0
    nurcle[0,:] = 0
    average_gray[k] = sg.convolve2d(gray_img, nurcle, 'same')
    for i in range(window):
        average_gray[k][i,:] = 0
        average_gray[k][:,i] = 0
        average_gray[k][w-i-1,:] = 0
        average_gray[k][:,h-i-1] = 0
    for wi in range(w)[window:(w-window-1)]:
        for hi in range(h)[window:(h-window-1)]:
            horizon[k][wi][hi] = average_gray[k][wi+window][hi] - average_gray[k][wi-window][hi]
            vertical[k][wi][hi] = average_gray[k][wi][hi+window] - average_gray[k][wi][hi-window]
    horizon[k] = horizon[k] * (1.0 / np.power(2, 2*(k+1)))
    vertical[k] = horizon[k] * (1.0 / np.power(2, 2*(k+1)))
            
    nurcle_h = np.zeros((2*window+1, 2*window+1))
    nurcle_h[0][window] = 1
    nurcle_h[2*window][window]=-1
    horizon2[k] = sg.convolve2d(average_gray[k], nurcle_h, 'same') 
    for i in range(window):
        horizon2[k][i,:] = 0
        horizon2[k][:,i] = 0
        horizon2[k][w-i-1,:] = 0
        horizon2[k][:,h-i-1] = 0    
    horizon2[k][w-window-1,:] = 0
    horizon2[k][:, h-window-1]= 0
    horizon2[k] = horizon2[k] * (1.0 / np.power(2, 2*(k+1)))
    ans = horizon- horizon2
    index = int(ans.argmin())
    x = int(index/h)
    y = int(index%h)
    print(horizon[k][x][y], horizon2[k][x][y])
    return ans
path_origin = '/home/lyh/000Dataset/DataSet_0325_lit/images/o/00000.png'
path_new    = '/home/lyh/000Dataset/DataSet_0325_lit/images/p1/00000.png'
brightness_test(path_origin, path_new)
# white_balance_test(path_origin, path_new)
# color_coherence_vector_test(path_origin)
# saturation_test(path_origin, path_new)
# 
# specular_shadow_test(path_origin, path_new)

# a, b = path2labmat(path_origin, path_new)
# f, theta = ta.tamura_directionality_lyh(a[0])
# ans = ta.tamura_linelikeness_2(a[0], theta,4)
# print(ans)
# ans2 = ta.tamura_linelikeness(a[0], theta,4)
# print(ans2)
# ans = ta.tamura_feature(a[0], 3, 4)

# print(ans)
# ans2 = ta.glcm_feature(a[0], 4)


