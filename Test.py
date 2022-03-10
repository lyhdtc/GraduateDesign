import os
import sys

from torch import uint8
sys.path.append(os.pardir)
from Tools import FakeColorCSV
from Tools import ExperimentCSV
from Tools import Experiment_Pic_Transform
import numpy as np
import time
import cv2
from Texture import TextureAlgorithrm as ta
from Color import ColorAlgorithrm as ca
from Color import Color
from Texture import Texture

# print('hello')
def general_run():
     
    start_time = time.perf_counter()
    print('Start!')
    
    
    path_a = 'Data/fff.jpg'
    path_b = 'Data/ffff.jpg'

    step = 8
    size_w = 40 
    size_h = 40
    figsize = (18,10)
    fakecolor_foldername = 'TextureChangeTest'
    fakecolor_folder = '/home/lyh/results/'+fakecolor_foldername+'/'   
    csv_path = '/home/lyh/results/TextureChangeTest.csv'
    picpair_name = 'default'
    FakeColorCSV.fakecolor_and_csv(path_a, path_b, step, size_w, size_h, figsize, fakecolor_foldername, fakecolor_folder, csv_path, picpair_name)
    

    end_time = time.perf_counter()  
    print('程序共运行 {_time_}秒'.format(_time_=(end_time - start_time)))
    
def general_experiment():
    start_time = time.perf_counter()
    print('Start!')
    
    
    path_a = 'Data/fff.jpg'
    path_b = 'Data/ffff.jpg'


    experiment_foldername = 'ExperimentTest'
    experiment_folder = '/home/lyh/results/'+experiment_foldername+'/'   
    csv_path = experiment_folder+experiment_foldername+'.csv'
    picpair_name = 'default'
    # ExperimentCSV.experiment_and_csv(path_a, path_b, experiment_foldername, experiment_folder, csv_path, picpair_name)
    
    ExperimentCSV.experiment1(path_a, path_b, 'sdf', csv_path, 'test')
    end_time = time.perf_counter()  
    print('程序共运行 {_time_}秒'.format(_time_=(end_time - start_time)))
# general_experiment()

# path_a = 'Data/fff.jpg'
# img = cv2.imread(path_a)
# b,g,r = cv2.split(img)
# # ans=ta.tamura_feature(b, 8, 4)
# # print(ans)
# b1 = b
# g1 = g
# r1 = r
# b1[::2,] = b[1::2,]
# g1[::2,] = g[1::2,]
# r1[::2,] = r[1::2,]
# b1[1::2,] = b[::2,]
# g1[1::2,] = g[::2,]
# r1[1::2,] = r[::2,]
# ans = cv2.merge((b1,g1,r1))
# # # # # cv2.imshow('aa',img)
# # # # # cv2.imshow('sdf', ans)
# # # # cv2.imwrite('/mnt/d/GraduateDesign2/GraduateDesign/Data/ffff.jpg', ans)
# # # # cv2.waitKey(0)
# # # step = 8
# # # size_w = 40 
# # # size_h = 40
# # # fakecolor_foldername = 'Test'
# # # folder = '/home/lyh/results/'+fakecolor_foldername+'/'  
# # # fakecolor_color = Color.FakeColor_Color_Characteristics(img, img, step = step, size_w = size_w, size_h = size_h, folder = folder, figsize=figsize)
# # # fakecolor_color.fakecolor_color_characteristics()
# start_time = time.perf_counter()

# print(ta.tamura_feature(b1, 8,4))
# end = time.perf_counter()
# print(end-start_time)


    
# print(ans)


path_a = 'Data/aaa.jpg'
path_b = 'Data/bbb.jpg'
img = cv2.imread(path_a)
img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
noise = cv2.imread(path_b)
noise = cv2.cvtColor(noise, cv2.COLOR_BGR2LAB)
ans = Experiment_Pic_Transform.experiment6_transform(img)
# ans = ans.astype(np.uint8)
cv2.imshow('asdf', ans)
cv2.waitKey(0)
