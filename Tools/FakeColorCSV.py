'''
Author: lyh
Date: 2022-03-23 19:36:51
LastEditors: lyh
LastEditTime: 2022-03-23 11:36:47
FilePath: /GraduateDesign/Tools/FakeColorCSV.py
Description: 

Copyright (c) 2022 by lyh, All Rights Reserved. 
'''
# from genericpath import getsize
import os
import sys
sys.path.append(os.path.pardir)
import cv2
import numpy as np
from Color import Color
from Texture import Texture
import os
import csv
from Tools import Crop




def fakecolor_and_csv(path_a, path_b, step, size_w, size_h, figsize, foldername, folder, csv_path, picpair_name, reshape_size):
    img_a = cv2.imread(path_a)
    img_b = cv2.imread(path_b)  
    
    img_a = cv2.resize(img_a,(400,600)) 
    img_b = cv2.resize(img_b,(400,600)) 
    
    # mask = Crop.create_mask(img_b)
    # img_a = Crop.use_mask(img_a, mask)
    img_a = cv2.cvtColor(img_a,cv2.COLOR_BGR2LAB)
    img_b = cv2.cvtColor(img_b,cv2.COLOR_BGR2LAB)
    
    matrix_a = np.array(cv2.split(img_a))
    matrix_b = np.array(cv2.split(img_b))
    # print(np.shape(matrix_a))
    # print(np.shape(matrix_b))    
    
    if(not os.path.exists(folder)):
        os.makedirs(folder)
        print("New Folder Created!")
    
    csv_label = []
    csv_label.append('picpair_name')
    csv_label.append('foldername')
    csv_data = []
    csv_data.append(picpair_name)
    csv_data.append(foldername)
         
    fakecolor_color = Color.FakeColor_Color_Characteristics(matrix_a, matrix_b, step = step, size_w = size_w, size_h = size_h, folder = folder, figsize=figsize, reshape_size=reshape_size)
    fakecolor_color.fakecolor_color_characteristics()
    csv_label = csv_label + fakecolor_color.csv_label
    csv_data  = csv_data  + fakecolor_color.csv_data
    # print(fakecolor_cc.csv_data)
    # print(fakecolor_cc.csv_label)
    
    fakecolor_texture = Texture.FakeColor_Texture_Characteristecs(matrix_a, matrix_b, step = step, size_w = size_w, size_h = size_h, folder = folder, figsize=figsize, reshape_size=reshape_size)
    fakecolor_texture.fakecolor_texture_characteristics()
    csv_label = csv_label + fakecolor_texture.csv_label
    csv_data  = csv_data  + fakecolor_texture.csv_data
   
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
