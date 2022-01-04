from genericpath import getsize
import sys
sys.path.append('graduate_design/Characteristics')
sys.path.append('graduate_design/Tools')
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
import Noise
import Crop
import FakeColorCSV


def get_img(path):
    img = cv2.imread(path)
    img_r,img_g,img_b = cv2.split(img)
    return img,img_r,img_g,img_b
    

    



if __name__=="__main__":

    
    start_time = time.perf_counter()
    print('Start!')
    
    
    path_a = 'graduate_design/Data/Model.png'
    path_b = 'graduate_design/Data/Photo.png'

    step = 8
    size_w = 40 
    size_h = 40
    figsize = (18,10)
    fakecolor_foldername = 'LightStageFullPic'
    fakecolor_folder = '/home/lyh/results/'+fakecolor_foldername+'/'   
    csv_path = '/home/lyh/results/lightstage.csv'
   
    FakeColorCSV.fakecolor_and_csv(path_a, path_b, step, size_w, size_h, figsize, fakecolor_foldername, fakecolor_folder, csv_path)
    

    end_time = time.perf_counter()  
    print('程序共运行 {_time_}秒'.format(_time_=(end_time - start_time)))

        
    
    