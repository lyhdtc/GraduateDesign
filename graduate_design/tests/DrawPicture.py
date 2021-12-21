from genericpath import getsize
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
import csv


def get_img(path):
    img = cv2.imread(path)
    img_r,img_g,img_b = cv2.split(img)
    return img,img_r,img_g,img_b
    
def fakecolor_and_csv(path_a, path_b, step, size_w, size_h, figsize, foldername, folder, csv_path):
    img_a = cv2.imread(path_a)
    img_b = cv2.imread(path_b)    
    
    matrix_a =  cv2.split(img_a)
    matrix_b =  cv2.split(img_b)
    print(np.shape(matrix_a))
    print(np.shape(matrix_b))    
    
    if(not os.path.exists(folder)):
        os.makedirs(folder)
        print("New Folder Created!")
    
    csv_label = []
    csv_label.append('foldername')
    csv_data = []
    csv_data.append(foldername)
    
    
    
     
    fakecolor_cc = FakeColor_Characteristics.FakeColor_Color_Characteristics(matrix_a, matrix_b, step = step, size_w = size_w, size_h = size_h, folder = folder, figsize=figsize)
    fakecolor_cc.fakecolor_color_characteristics()
    csv_label = csv_label + fakecolor_cc.csv_label
    csv_data  = csv_data  + fakecolor_cc.csv_data
    # print(fakecolor_cc.csv_data)
    # print(fakecolor_cc.csv_label)
    # fakecolor_tc = FakeColor_Characteristics.FakeColor_Texture_Characteristecs(matrix_a, matrix_b, step = step, size_w = size_w, size_h = size_h, folder = folder, figsize=figsize)
    # fakecolor_tc.fakecolor_texture_characteristics()
    # csv_label = csv_label + fakecolor_tc.csv_label
    # csv_data  = csv_data  + fakecolor_tc.csv_data

    # fakecolor_lac = FakeColor_Characteristics.FakeColor_LossAboutColor_Characteristics(img_a, img_b, step = step, size_w = size_w, size_h = size_h, folder = folder, figsize=figsize)
    # fakecolor_lac.fakecolor_loss_about_color()
    # csv_label = csv_label + fakecolor_lac.csv_label
    # csv_data  = csv_data  + fakecolor_lac.csv_data    
    if not os.path.exists(csv_path):
        print("Create New CSV File!")
        with open(csv_path, "w") as csvfile:
            file = csv.writer(csvfile)
            file.writerow(csv_label)
            file.writerow(csv_data)
    else:
        print("Open Exist CSV File!")
        with open(csv_path, "a") as csvfile:
            file = csv.writer(csvfile)

            file.writerow(csv_data)

    



if __name__=="__main__":
    # for i in tqdm.trange(10):
    #     for j in range(10):
    #         time.sleep(0.25)
    start_time = time.perf_counter()
    print('Start!')
    
    
    path_a = 'graduate_design/Data/Model_LS.png'
    path_b = 'graduate_design/Data/Photo_LS.png'

    step = 8
    size_w = 40 
    size_h = 40
    figsize = (18,10)
    fakecolor_foldername = 'GlcmMultithreadTest'
    fakecolor_folder = '/home/lyh/results/'+fakecolor_foldername+'/'   
    csv_path = '/home/lyh/results/test.csv'
   
    fakecolor_and_csv(path_a, path_b, step, size_w, size_h, figsize, fakecolor_foldername, fakecolor_folder, csv_path)
    

    end_time = time.perf_counter()  
    print('程序共运行 {_time_}秒'.format(_time_=(end_time - start_time)))
