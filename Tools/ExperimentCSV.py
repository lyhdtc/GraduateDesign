from genericpath import getsize
import os
import sys

from torch import exp
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
from multiprocessing import Lock


def test_algothrim_experiment_and_csv(path_a, path_b,folder, csv_path, picpair_name):
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
    
    experiment_texture = TextureExperiment.Experiment_Texture_Characteristics(matrix_a, matrix_b)
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

def experiment_csv(i, random_num_a, random_num_b, pic_folder,  csv_folder, lock):
    name_a = str(random_num_a[i])
    name_b = str(random_num_b[i])
    if(not os.path.exists(csv_folder)):
        os.makedirs(csv_folder)
        print("New Folder Created!")
    
    
    path_a = pic_folder + name_a +'.jpg'
    path_b = pic_folder + name_b +'.jpg'
    # print(path_a, path_b)
    
    csv_path = csv_folder + '1.csv'
    picpair_name = name_a+'_'+name_b
    __experiment1(path_a, path_b, csv_path, picpair_name, lock)
    
    csv_path = csv_folder + '2.csv'
    picpair_name = name_a+'_'+name_b
    __experiment2(path_a, path_b, csv_path, picpair_name, lock)
    
    csv_path = csv_folder + '3.csv'
    picpair_name = name_a+'_'+name_b
    __experiment3(path_a, path_b, csv_path, picpair_name, lock)
   
    csv_path = csv_folder + '4.csv'
    picpair_name = name_a+'_'+name_b
    __experiment4(path_a, path_b, csv_path, picpair_name, lock)
    
    csv_path = csv_folder + '5.csv'
    picpair_name = name_a+'_'+name_b
    __experiment5(path_a, path_b, csv_path, picpair_name, lock)
    
    csv_path = csv_folder + '6.csv'
    picpair_name = name_a
    __experiment6(path_a, path_b, csv_path, picpair_name, lock)    

             

def __experiment1(path_a, path_b, csv_path, picpair_name, lock):
    img_a = cv2.imread(path_a)
    img_b = cv2.imread(path_b)  
    img_a = cv2.cvtColor(img_a,cv2.COLOR_BGR2LAB)
    img_b = cv2.cvtColor(img_b,cv2.COLOR_BGR2LAB)
    img_b1 = Experiment_Pic_Transform.experiment1_transform(img_b)
    # img_b1 = img_b
    matrix_a = np.array(cv2.split(img_a))
    matrix_b = np.array(cv2.split(img_b))
    matrix_b1 = np.array(cv2.split(img_b1))
    
    csv_label = []
    # csv_label.append('experiment_name')
    csv_label.append('picpair_name')
    csv_data = []
    # csv_data.append(experiment_name)
    csv_data.append(picpair_name)
    
    D_color_origin = ColorExperiment.Experiment_Color_Characteristics(matrix_a, matrix_b)
    D_color_origin.experiment_color_characteristics()    
    D_color_experiment = ColorExperiment.Experiment_Color_Characteristics(matrix_a, matrix_b1)
    D_color_experiment.experiment_color_characteristics()  
    D_experiment =   np.array(D_color_experiment.csv_data)
    D_origin = np.array(D_color_origin.csv_data)+1e-7
    S_color = D_experiment/D_origin
    S_color = S_color.tolist()
    
    D_texture_origin = TextureExperiment.Experiment_Texture_Characteristics(matrix_a, matrix_b)
    D_texture_origin.experiment_texture_characteristics()    
    D_texture_experiment = TextureExperiment.Experiment_Texture_Characteristics(matrix_a, matrix_b1)
    D_texture_experiment.experiment_texture_characteristics()    
    S_texture = np.array(D_texture_experiment.csv_data)/(np.array(D_texture_origin.csv_data)+1e-7)
    S_texture = S_texture.tolist()
    
    csv_label = csv_label + D_color_origin.csv_label + D_texture_origin.csv_label
    csv_data = csv_data + S_color + S_texture
    # print(csv_data)
    lock.acquire()
    # print('sdfas')
    if not os.path.exists(csv_path):
        # print("Create New CSV File!")
        with open(csv_path, "w") as csvfile:
            file = csv.writer(csvfile)
            file.writerow(csv_label)
            file.writerow(csv_data)
    else:
        # print("Open Exist CSV File!")
        with open(csv_path, "a") as csvfile:
            file = csv.writer(csvfile)

            file.writerow(csv_data)
    lock.release()         
            
def __experiment2(path_a, path_b, csv_path, picpair_name, lock):
    img_a = cv2.imread(path_a)
    img_b = cv2.imread(path_b)  
    img_a = cv2.cvtColor(img_a,cv2.COLOR_BGR2LAB)
    img_b = cv2.cvtColor(img_b,cv2.COLOR_BGR2LAB)
    img_b1 = Experiment_Pic_Transform.experiment2_transform(img_b)
    # img_b1 = img_b
    matrix_a = np.array(cv2.split(img_a))
    matrix_b = np.array(cv2.split(img_b))
    matrix_b1 = np.array(cv2.split(img_b1))
    
    csv_label = []
    # csv_label.append('experiment_name')
    csv_label.append('picpair_name')
    csv_data = []
    # csv_data.append(experiment_name)
    csv_data.append(picpair_name)
    
    D_color_origin = ColorExperiment.Experiment_Color_Characteristics(matrix_a, matrix_b)
    D_color_origin.experiment_color_characteristics()    
    D_color_experiment = ColorExperiment.Experiment_Color_Characteristics(matrix_a, matrix_b1)
    D_color_experiment.experiment_color_characteristics()    
    S_color = np.array(D_color_experiment.csv_data)/(np.array(D_color_origin.csv_data)+1e-7)
    S_color = S_color.tolist()
    
    D_texture_origin = TextureExperiment.Experiment_Texture_Characteristics(matrix_a, matrix_b)
    D_texture_origin.experiment_texture_characteristics()    
    D_texture_experiment = TextureExperiment.Experiment_Texture_Characteristics(matrix_a, matrix_b1)
    D_texture_experiment.experiment_texture_characteristics()    
    S_texture = np.array(D_texture_experiment.csv_data)/(np.array(D_texture_origin.csv_data)+1e-7)
    S_texture = S_texture.tolist()
    
    csv_label = csv_label + D_color_origin.csv_label + D_texture_origin.csv_label
    csv_data = csv_data + S_color + S_texture
    
    lock.acquire()
    if not os.path.exists(csv_path):
        # print("Create New CSV File!")
        with open(csv_path, "w") as csvfile:
            file = csv.writer(csvfile)
            file.writerow(csv_label)
            file.writerow(csv_data)
    else:
        # print("Open Exist CSV File!")
        with open(csv_path, "a") as csvfile:
            file = csv.writer(csvfile)

            file.writerow(csv_data)
    lock.release()
            
def __experiment3(path_a, path_b, csv_path, picpair_name, lock):
    img_a = cv2.imread(path_a)
    img_b = cv2.imread(path_b)  
    img_a = cv2.cvtColor(img_a,cv2.COLOR_BGR2LAB)
    img_b = cv2.cvtColor(img_b,cv2.COLOR_BGR2LAB)
    img_am = Experiment_Pic_Transform.experiment3_transform(img_a, 0.6)
    img_bm = Experiment_Pic_Transform.experiment3_transform(img_b, 0.6)
    img_as = Experiment_Pic_Transform.experiment3_transform(img_a, 0.3)
    img_bs = Experiment_Pic_Transform.experiment3_transform(img_b, 0.3)
    matrix_a = np.array(cv2.split(img_a))
    matrix_b = np.array(cv2.split(img_b))
    matrix_am = np.array(cv2.split(img_am))
    matrix_bm = np.array(cv2.split(img_bm))
    matrix_as = np.array(cv2.split(img_as))
    matrix_bs = np.array(cv2.split(img_bs))
    
    csv_label = []
    # csv_label.append('experiment_name')
    csv_label.append('picpair_name')
    csv_data_ml = []
    csv_data_sl = []
    # csv_data.append(experiment_name)
    csv_data_ml.append(picpair_name+'_ml')
    csv_data_sl.append(picpair_name+'_sl')
    
    D_color_large = ColorExperiment.Experiment_Color_Characteristics(matrix_a, matrix_b)
    D_color_large.experiment_color_characteristics()    
    D_color_medium = ColorExperiment.Experiment_Color_Characteristics(matrix_am, matrix_bm)
    D_color_medium.experiment_color_characteristics()
    D_color_small = ColorExperiment.Experiment_Color_Characteristics(matrix_as, matrix_bs)
    D_color_small.experiment_color_characteristics()
        
    S_color_ml = np.array(D_color_medium.csv_data)/(np.array(D_color_large.csv_data)+1e-7)
    S_color_ml = S_color_ml.tolist()
    S_color_sl = np.array(D_color_small.csv_data)/(np.array(D_color_large.csv_data)+1e-7)    
    S_color_sl = S_color_sl.tolist()
    
    D_texture_large = TextureExperiment.Experiment_Texture_Characteristics(matrix_a, matrix_b)
    D_texture_large.experiment_texture_characteristics()    
    D_texture_medium = TextureExperiment.Experiment_Texture_Characteristics(matrix_am, matrix_bm)
    D_texture_medium.experiment_texture_characteristics()
    D_texture_small = TextureExperiment.Experiment_Texture_Characteristics(matrix_as, matrix_bs)
    D_texture_small.experiment_texture_characteristics()
        
    S_texture_ml = np.array(D_texture_medium.csv_data)/(np.array(D_texture_large.csv_data)+1e-7)
    S_texture_ml = S_texture_ml.tolist()
    S_texture_sl = np.array(D_texture_small.csv_data)/(np.array(D_texture_large.csv_data)+1e-7)
    S_texture_sl = S_texture_sl.tolist()
    
    csv_label = csv_label + D_color_large.csv_label + D_texture_large.csv_label
    csv_data_ml = csv_data_ml + S_color_ml + S_texture_ml
    csv_data_sl = csv_data_sl + S_color_sl + S_texture_sl
    
    
    lock.acquire()
    if not os.path.exists(csv_path):
        # print("Create New CSV File!")
        with open(csv_path, "w") as csvfile:
            file = csv.writer(csvfile)
            file.writerow(csv_label)
            file.writerow(csv_data_ml)
            file.writerow(csv_data_sl)
    else:
        # print("Open Exist CSV File!")
        with open(csv_path, "a") as csvfile:
            file = csv.writer(csvfile)

            file.writerow(csv_data_ml)
            file.writerow(csv_data_sl)
    lock.release()       

def __experiment4(path_a, path_b, csv_path, picpair_name, lock):
    img_a = cv2.imread(path_a)
    img_b = cv2.imread(path_b)  
    img_a = cv2.cvtColor(img_a,cv2.COLOR_BGR2LAB)
    img_b = cv2.cvtColor(img_b,cv2.COLOR_BGR2LAB)
    matrix_a = np.array(cv2.split(img_a))
    matrix_b = np.array(cv2.split(img_b))
    
    img_b0 = Experiment_Pic_Transform.experiment4_transform(img_b, 0)
    matrix_b0 = np.array(cv2.split(img_b0))
    D_color_0 = ColorExperiment.Experiment_Color_Characteristics(matrix_a, matrix_b0)
    D_color_0.experiment_color_characteristics()
    D_texture_0 = TextureExperiment.Experiment_Texture_Characteristics(matrix_a, matrix_b0)
    D_texture_0.experiment_texture_characteristics()
    D = D_color_0.csv_data+D_texture_0.csv_data
    D = np.array(D)
    
    
    csv_label = []
    # csv_label.append('experiment_name')
    csv_label.append('picpair_name')
    csv_label = csv_label+D_color_0.csv_label+D_texture_0.csv_label
    csv_data = []
    # csv_data.append(experiment_name)
    csv_data.append(picpair_name)
    
    for i in range(1,20):
        i = i/20.
        img_bi = Experiment_Pic_Transform.experiment4_transform(img_b, i)
        matrix_bi = np.array(cv2.split(img_bi))
        D_color_i = ColorExperiment.Experiment_Color_Characteristics(matrix_a, matrix_bi)
        D_color_i.experiment_color_characteristics()
        D_texture_i = TextureExperiment.Experiment_Texture_Characteristics(matrix_a, matrix_bi)
        D_texture_i.experiment_texture_characteristics()
        D_i = D_color_i.csv_data+D_texture_i.csv_data
        D_i = np.array(D_i)
        D = np.vstack((D,D_i))
        
    
    D_grd = np.gradient(D)[0]
    max_grd = np.max(D_grd, axis=0)
    D_grd = np.transpose(D_grd)
    length = np.shape(max_grd)[0]
    for i in range(length):
        csv_data = csv_data + [min(np.where(D_grd[i]==max_grd[i])[0])]    

    lock.acquire()
    if not os.path.exists(csv_path):
        # print("Create New CSV File!")
        with open(csv_path, "w") as csvfile:
            file = csv.writer(csvfile)
            file.writerow(csv_label)
            file.writerow(csv_data)
    else:
        # print("Open Exist CSV File!")
        with open(csv_path, "a") as csvfile:
            file = csv.writer(csvfile)

            file.writerow(csv_data)
    lock.release()
    
def __experiment5(path_a, path_b, csv_path, picpair_name, lock):
    img_a = cv2.imread(path_a)
    img_b = cv2.imread(path_b)  
    img_a = cv2.cvtColor(img_a,cv2.COLOR_BGR2LAB)
    img_b = cv2.cvtColor(img_b,cv2.COLOR_BGR2LAB)
    matrix_a = np.array(cv2.split(img_a))
    matrix_b = np.array(cv2.split(img_b))
    
    img_b0 = Experiment_Pic_Transform.experiment5_transform(img_b, img_a,0)
    matrix_b0 = np.array(cv2.split(img_b0))
    D_color_0 = ColorExperiment.Experiment_Color_Characteristics(matrix_a, matrix_b0)
    D_color_0.experiment_color_characteristics()
    D_texture_0 = TextureExperiment.Experiment_Texture_Characteristics(matrix_a, matrix_b0)
    D_texture_0.experiment_texture_characteristics()
    D = D_color_0.csv_data+D_texture_0.csv_data
    D = np.array(D)
    
    
    csv_label = []
    # csv_label.append('experiment_name')
    csv_label.append('picpair_name')
    csv_label = csv_label+D_color_0.csv_label+D_texture_0.csv_label
    csv_data = []
    # csv_data.append(experiment_name)
    csv_data.append(picpair_name)
    
    for i in range(1,20):
        i = i/20.
        img_bi = Experiment_Pic_Transform.experiment5_transform(img_b, img_a,i)
        matrix_bi = np.array(cv2.split(img_bi))
        D_color_i = ColorExperiment.Experiment_Color_Characteristics(matrix_a, matrix_bi)
        D_color_i.experiment_color_characteristics()
        D_texture_i = TextureExperiment.Experiment_Texture_Characteristics(matrix_a, matrix_bi)
        D_texture_i.experiment_texture_characteristics()
        D_i = D_color_i.csv_data+D_texture_i.csv_data
        D_i = np.array(D_i)
        D = np.vstack((D,D_i))
        
    
    D_grd = np.gradient(D)[0]
    max_grd = np.max(D_grd, axis=0)
    D_grd = np.transpose(D_grd)
    length = np.shape(max_grd)[0]
    for i in range(length):
        csv_data = csv_data + [min(np.where(D_grd[i]==max_grd[i])[0])]    

    lock.acquire()
    if not os.path.exists(csv_path):
        # print("Create New CSV File!")
        with open(csv_path, "w") as csvfile:
            file = csv.writer(csvfile)
            file.writerow(csv_label)
            file.writerow(csv_data)
    else:
        # print("Open Exist CSV File!")
        with open(csv_path, "a") as csvfile:
            file = csv.writer(csvfile)

            file.writerow(csv_data)
    lock.release()    
         
             
def __experiment6(path_a, path_b,csv_path, picpair_name, lock):
    img_a = cv2.imread(path_a)
    img_b = cv2.imread(path_b)  
    img_a = cv2.cvtColor(img_a,cv2.COLOR_BGR2LAB)
    img_b = cv2.cvtColor(img_b,cv2.COLOR_BGR2LAB)
    img_b1 = Experiment_Pic_Transform.experiment6_transform(img_b)
    # img_b1 = img_b
    matrix_a = np.array(cv2.split(img_a))
    matrix_b = np.array(cv2.split(img_b))
    matrix_b1 = np.array(cv2.split(img_b1))
    
    csv_label = []
    # csv_label.append('experiment_name')
    csv_label.append('picpair_name')
    csv_data = []
    # csv_data.append(experiment_name)
    csv_data.append(picpair_name)
    
    D_color_origin = ColorExperiment.Experiment_Color_Characteristics(matrix_a, matrix_b)
    D_color_origin.experiment_color_characteristics()    
    D_color_experiment = ColorExperiment.Experiment_Color_Characteristics(matrix_a, matrix_b1)
    D_color_experiment.experiment_color_characteristics()    
    S_color = np.array(D_color_experiment.csv_data)/(np.array(D_color_origin.csv_data)+1e-7)
    S_color = S_color.tolist()
    
    D_texture_origin = TextureExperiment.Experiment_Texture_Characteristics(matrix_a, matrix_b)
    D_texture_origin.experiment_texture_characteristics()    
    D_texture_experiment = TextureExperiment.Experiment_Texture_Characteristics(matrix_a, matrix_b1)
    D_texture_experiment.experiment_texture_characteristics()    
    S_texture = np.array(D_texture_experiment.csv_data)/(np.array(D_texture_origin.csv_data)+1e-7)
    S_texture = S_texture.tolist()
    
    csv_label = csv_label + D_color_origin.csv_label + D_texture_origin.csv_label
    csv_data = csv_data + S_color + S_texture
    
    lock.acquire()
    if not os.path.exists(csv_path):
        # print("Create New CSV File!")
        with open(csv_path, "w") as csvfile:
            file = csv.writer(csvfile)
            file.writerow(csv_label)
            file.writerow(csv_data)
    else:
        # print("Open Exist CSV File!")
        with open(csv_path, "a") as csvfile:
            file = csv.writer(csvfile)

            file.writerow(csv_data)
    lock.release()