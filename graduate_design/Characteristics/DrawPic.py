from collections import _OrderedDictItemsView
import sys
from typing import Tuple
sys.path.append('/mnt/d/Graduate/GraduateDesign/graduate_design/Characteristics')
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


def color_characteristics(matrix_a, matrix_b):
    fig = plt.figure(1)
    
    #直方图
    for i in range(3):
        ax1 = plt.subplot(5,1,i+1)
        hist_title = 'histogram ———— '+RGB_COLOR_CHANNEL.get(i)+' channel'
        ax1.set_title(hist_title)
        hist_a = cc.histogram(matrix_a[i])        
        plt.plot(hist_a,RGB_COLOR_CHANNEL.get(i))
        hist_b = cc.histogram(matrix_b[i])        
        plt.plot(hist_b,RGB_COLOR_CHANNEL.get(i),linestyle='dashed')  
    
    # 颜色矩      
    color_moments = []    
    color_moments_collabel = []
    color_moments_rowlabel = ['1st', '2nd', '3rd']
    for i in range(3):
        
        color_moments_a = cc.color_moments(matrix_a[i])
        color_moments_b = cc.color_moments(matrix_b[i])        
        color_moments.append(color_moments_a)
        color_moments.append(color_moments_b)
        label_a = RGB_COLOR_CHANNEL.get(i)+'_img_a'
        label_b = RGB_COLOR_CHANNEL.get(i)+'_img_b'
        color_moments_collabel.append(label_a)
        color_moments_collabel.append(label_b)            
       
    color_moments = np.transpose(color_moments)
    color_moments = np.round(color_moments,3)
    # print(color_moments)
    ax4 = plt.subplot(5,1,4)
    ax4.axis('tight')
    ax4.axis('off')    
    color_moments_table = plt.table(color_moments, colLabels=color_moments_collabel, rowLabels=color_moments_rowlabel, loc='center',  cellLoc='center', rowLoc='center')
    color_moments_table.auto_set_font_size(False)
    color_moments_table.set_fontsize(8)
    # color_moments_table.scale(0.5,0.5)
    
    # TODO：颜色聚合向量需要大改
    
    # 普通矩
    ordinary_moments = []
    for i in range(3):
        ordinary_moments_a = cc.ordinary_moments(matrix_a[i])
        ordinary_moments_b = cc.ordinary_moments(matrix_b[i])
        
        ordinary_moments.append(ordinary_moments_a)
        ordinary_moments.append(ordinary_moments_b)
    
    
    ordinary_moments = np.transpose(ordinary_moments)
    ordinary_moments = np.round(ordinary_moments)
    print(ordinary_moments)
    ax5 = plt.subplot(5,1,5)
    ax5.axis('tight')
    ax5.axis('off')
    ordinary_moments_table = plt.table(ordinary_moments,loc="center")
    ordinary_moments_table.auto_set_font_size(False)
    ordinary_moments_table.set_fontsize(8)
    
    # print(ordinary_moments)
    
    # print(color_coherence_vector)

    # plt.table(ordinary_moments,loc="center")
    
    
    
    
    plt.tight_layout()
    plt.show()    