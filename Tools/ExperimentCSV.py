from genericpath import getsize
import os
import sys
sys.path.append(os.path.pardir)
import cv2
import numpy as np
from Color import ColorExperiment
from Texture import TextureExperiment
from Texture import Texture
import os
import csv
from Tools import Crop
from Tools import Experiment_Pic_Transform



def experiment_and_csv(path_a, path_b, folder, csv_path, picpair_name):
    img_a = cv2.imread(path_a)
    img_b = cv2.imread(path_b)  

    
    # img_a = cv2.resize(img_a,(400,600)) 
    # img_b = cv2.resize(img_b,(400,600)) 
    
    # mask = Crop.create_mask(img_b)
    # img_a = Crop.use_mask(img_a, mask)
    img_a = cv2.cvtColor(img_a,cv2.COLOR_BGR2LAB)
    img_b = cv2.cvtColor(img_b,cv2.COLOR_BGR2LAB)

    
    
    matrix_a = np.array(cv2.split(img_a))
    matrix_b = np.array(cv2.split(img_b))

    print(np.shape(matrix_a))
    print(np.shape(matrix_b))    

    
    if(not os.path.exists(folder)):
        os.makedirs(folder)
        print("New Folder Created!")
    
    csv_label = []
    # csv_label.append('experiment_name')
    csv_label.append('picpair_name')
    csv_data = []
    # csv_data.append(experiment_name)
    csv_data.append(picpair_name)
    
         
    experiment_color = ColorExperiment.Experiment_Color_Characteristics(matrix_a, matrix_b)
    experiment_color.experiment_color_characteristics()
    csv_label = csv_label + experiment_color.csv_label
    csv_data  = csv_data  + experiment_color.csv_data
    # print(experiment_cc.csv_data)
    # print(experiment_cc.csv_label)
    
    experiment_texture = TextureExperiment.Experiment_Texture_Characteristecs(matrix_a, matrix_b)
    experiment_texture.experiment_texture_characteristics()
    csv_label = csv_label + experiment_texture.csv_label
    csv_data  = csv_data  + experiment_texture.csv_data
   
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
            
            
def experiment1(path_a, path_b, folder, csv_path, picpair_name):
    img_a = cv2.imread(path_a)
    img_b = cv2.imread(path_b)  
    img_a = cv2.cvtColor(img_a,cv2.COLOR_BGR2LAB)
    img_b = cv2.cvtColor(img_b,cv2.COLOR_BGR2LAB)
    img_b1 = Experiment_Pic_Transform.experiment1_transform(img_b)
    matrix_a = np.array(cv2.split(img_a))
    matrix_b = np.array(cv2.split(img_b))
    matrix_b1 = np.array(cv2.split(img_b1))