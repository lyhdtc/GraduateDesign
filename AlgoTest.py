'''
Author: lyh
Date: 2022-03-21 19:10:36
LastEditors: lyh
LastEditTime: 2022-03-22 12:47:10
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
    ca.constract(lab_matrix_new, lab_matrix_origin)
    # print(f'constract result:\n  constract_origin = {ans_origin}\n  constract_new = {ans_new}\n  delta = {ans_new-ans_origin}')

# 生成直方图
# color_characteristics_histogram(lab_img, (18,10), 'AlgoTest/')

def exposure_test(path_origin, path_new):
    lab_matrix_origin , lab_matrix_new = path2labmat(path_origin, path_new)
    ans = ca.exposure(lab_matrix_new, lab_matrix_origin)
    print(ans)

def saturation_test(path_origin, path_new):
    lab_matrix_origin , lab_matrix_new = path2labmat(path_origin, path_new)
    ans = ca.saturation(lab_matrix_new, lab_matrix_origin)
    print(ans)
    
    
    
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

path_origin = 'AlgoTest/AlgoTestPic.jpg'
path_new    = 'AlgoTest/AlgoTestPic_Constract+50.jpg'

# exposure_test(path_origin, path_new)
# color_coherence_vector_test(path_origin)
constract_test(path_origin, path_new)








