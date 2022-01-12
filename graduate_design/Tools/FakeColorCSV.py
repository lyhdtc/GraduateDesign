from genericpath import getsize
import sys
sys.path.append('graduate_design/Characteristics')
sys.path.append('graduate_design/Tools')
import cv2
import numpy as np
import FakeColor_Characteristics
import os
import csv
import Crop




def fakecolor_and_csv(path_a, path_b, step, size_w, size_h, figsize, foldername, folder, csv_path):
    img_a = cv2.imread(path_a)
    img_b = cv2.imread(path_b)  
    
    img_a = cv2.resize(img_a,(400,600)) 
    img_b = cv2.resize(img_b,(400,600)) 
    
    mask = Crop.create_mask(img_b)
    img_a = Crop.use_mask(img_a, mask)
    

    
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
    fakecolor_tc = FakeColor_Characteristics.FakeColor_Texture_Characteristecs(matrix_a, matrix_b, step = step, size_w = size_w, size_h = size_h, folder = folder, figsize=figsize)
    fakecolor_tc.fakecolor_texture_characteristics()
    csv_label = csv_label + fakecolor_tc.csv_label
    csv_data  = csv_data  + fakecolor_tc.csv_data

    fakecolor_lac = FakeColor_Characteristics.FakeColor_LossAboutColor_Characteristics(img_a, img_b, step = step, size_w = size_w, size_h = size_h, folder = folder, figsize=figsize)
    fakecolor_lac.fakecolor_loss_about_color()
    csv_label = csv_label + fakecolor_lac.csv_label
    csv_data  = csv_data  + fakecolor_lac.csv_data    
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
