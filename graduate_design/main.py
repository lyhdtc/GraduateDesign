import sys
sys.path.append('/mnt/d/Graduate/GraduateDesign/graduate_design/Characteristics')
import cv2
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import ColorCharacteristics as cc
import TextureCharacteristics as tc
import LossAboutColor as lac
import DrawPic




def get_img(path):
    img = cv2.imread(path)
    img_r,img_g,img_b = cv2.split(img)
    return img,img_r,img_g,img_b

   
    


if __name__=="__main__":
    print('Hello World')
    path_a = '/mnt/d/Graduate/GraduateDesign/graduate_design/Data/eee.jpg'
    path_b = '/mnt/d/Graduate/GraduateDesign/graduate_design/Data/fff.jpg'
    img_a = cv2.imread(path_a)
    img_b = cv2.imread(path_b)    

    matrix_a =  cv2.split(img_a)
    matrix_b =  cv2.split(img_b)
    rgb_img_b, r_img_b, g_img_b, b_img_b = get_img(path_b)
    DrawPic.color_characteristics(matrix_a, matrix_b)
    m = [[255,245,255,2,1,3],
        [245,34,3,2,3,3],
        [3,3,3,3,3,3]]
    # print(r_img_b)
    # res = cc.color_coherence_vector(r_img_b)
    
    # print(res)
    
    # for i in range(3):   
    #     a_color_moments = cc.color_moments(matrix_a[i])
    #     b_color_moments = cc.color_moments(matrix_b[i])
    #     print(a_color_moments)
    #     print(b_color_moments)
    