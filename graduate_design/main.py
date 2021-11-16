import sys
from numpy.core.fromnumeric import shape

from numpy.matrixlib.defmatrix import matrix
sys.path.append('/mnt/d/GraduateDesign/graduate_design/Characteristics')
import cv2
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import ColorCharacteristics as cc
import TextureCharacteristics as tc
import LossAboutColor as lac


RGB_COLOR_CHANNEL = {
    0: 'r',
    1: 'g',
    2: 'b'
}

def get_img(path):
    img = cv2.imread(path)
    img_r,img_g,img_b = cv2.split(img)
    return img,img_r,img_g,img_b

def color_characteristics(matrix_a, matrix_b):
    fig = plt.figure(1)
    
    #直方图
    for i in range(3):
        ax1 = plt.subplot(4,1,i+1)
        hist_a = cc.histogram(matrix_a[i])        
        plt.plot(hist_a,RGB_COLOR_CHANNEL.get(i))
        hist_b = cc.histogram(matrix_b[i])        
        plt.plot(hist_b,RGB_COLOR_CHANNEL.get(i),linestyle='dashed')     
    
    color_moments = []
    ordinary_moments = []
    color_coherence_vector = []
   
    for i in range(3):
        
        color_moments_a = cc.color_moments(matrix_a[i])
        color_moments_b = cc.color_moments(matrix_b[i])        
        color_moments.append(color_moments_a)
        color_moments.append(color_moments_b)
        
        ordinary_moments_a = cc.ordinary_moments(matrix_a[i])
        ordinary_moments_b = cc.ordinary_moments(matrix_b[i])
        ordinary_moments.append(ordinary_moments_a)
        ordinary_moments.append(ordinary_moments_b)
        
        # color_coherence_vector_a = cc.color_coherence_vector(matrix_a[i])
        # color_coherence_vector_b = cc.color_coherence_vector(matrix_b[i])
        # color_coherence_vector.append(color_coherence_vector_a)
        # color_coherence_vector.append(color_coherence_vector_b)
        
    color_moments = np.transpose(color_moments)
    print(color_moments)
    plt.table(color_moments,loc="center")
    
    ordinary_moments = np.transpose(ordinary_moments)
    print(ordinary_moments)
    
    print(color_coherence_vector)

    # plt.table(ordinary_moments,loc="center")
    
    
    
    
    plt.tight_layout()
    plt.show()     
    



if __name__=="__main__":
    print('Hello World')
    path_a = '/mnt/d/GraduateDesign/graduate_design/Data/eee.jpg'
    path_b = '/mnt/d/GraduateDesign/graduate_design/Data/fff.jpg'
    img_a = cv2.imread(path_a)
    img_b = cv2.imread(path_b)    

    matrix_a =  cv2.split(img_a)
    matrix_b =  cv2.split(img_b)
    rgb_img_b, r_img_b, g_img_b, b_img_b = get_img(path_b)
    # color_characteristics(matrix_a, matrix_b)
    m = [[255,245,255,2,1,3],
        [245,34,3,2,3,3],
        [3,3,3,3,3,3]]
    # print(r_img_b)
    res = cc.color_coherence_vector(r_img_b)
    
    print(res)
    
    # for i in range(3):   
    #     a_color_moments = cc.color_moments(matrix_a[i])
    #     b_color_moments = cc.color_moments(matrix_b[i])
    #     print(a_color_moments)
    #     print(b_color_moments)
