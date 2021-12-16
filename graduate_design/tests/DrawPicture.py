import sys
sys.path.append('graduate_design/Characteristics')
import cv2
import numpy as np
from matplotlib import pyplot as plt
import ColorCharacteristics as cc
import TextureCharacteristics as tc
import LossAboutColor as lac
import MatPlob_Characteristics
import profile
import FakeColor_Characteristics
import time
import os
import tqdm


def get_img(path):
    img = cv2.imread(path)
    img_r,img_g,img_b = cv2.split(img)
    return img,img_r,img_g,img_b
    



if __name__=="__main__":
    # for i in tqdm.trange(10):
    #     for j in range(10):
    #         time.sleep(0.25)
    start_time = time.perf_counter()
    print('Start!')
    path_a = 'graduate_design/Data/Normal_Changed.jpg'
    path_b = 'graduate_design/Data/Normal_Unchanged.jpg'
    img_a = cv2.imread(path_a)
    img_b = cv2.imread(path_b)    



    matrix_a =  cv2.split(img_a)
    matrix_b =  cv2.split(img_b)
    print(np.shape(matrix_a))
    print(np.shape(matrix_b))
    
    step = 8
    size_w = 40 
    size_h = 40
    figsize = (18,10)
    folder = '/home/lyh/results/MultiThreadTest/'
    if(not os.path.exists(folder)):
        os.makedirs(folder)
        print("New Folder Created!")
    
     
    fakecolor_cc = FakeColor_Characteristics.FakeColor_Color_Characteristics(matrix_a, matrix_b, step = step, size_w = size_w, size_h = size_h, folder = folder, figsize=figsize)
    fakecolor_cc.fakecolor_color_characteristics()
    
    fakecolor_tc = FakeColor_Characteristics.FakeColor_Texture_Characteristecs(matrix_a, matrix_b, step = step, size_w = size_w, size_h = size_h, folder = folder, figsize=figsize)
    fakecolor_tc.fakecolor_texture_characteristics()

    fakecolor_lac = FakeColor_Characteristics.FakeColor_LossAboutColor_Characteristics(img_a, img_b, step = step, size_w = size_w, size_h = size_h, folder = folder, figsize=figsize)
    fakecolor_lac.fakecolor_loss_about_color()
    
    # drawpic_tc = MatPlob_Characteristics.Draw_Texture_Characteristics(matrix_a, matrix_b)
    # drawpic_tc.draw_texture_characteristics()
    # plt.show()
    end_time = time.perf_counter()  
    print('程序共运行 {_time_}秒'.format(_time_=(end_time - start_time)))
