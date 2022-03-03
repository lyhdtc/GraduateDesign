import os
import sys
sys.path.append(os.pardir)
from Tools import FakeColorCSV

import numpy as np
import time
import cv2
from Texture import TextureAlgorithrm as ta

# print('hello')
def general_run():
     
    start_time = time.perf_counter()
    print('Start!')
    
    
    path_a = 'Data/Model.png'
    path_b = 'Data/Photo.png'

    step = 8
    size_w = 40 
    size_h = 40
    figsize = (18,10)
    fakecolor_foldername = 'NewWorkflowTest'
    fakecolor_folder = '/home/lyh/results/'+fakecolor_foldername+'/'   
    csv_path = '/home/lyh/results/NewWorkflowTest.csv'
    picpair_name = 'default'
    FakeColorCSV.fakecolor_and_csv(path_a, path_b, step, size_w, size_h, figsize, fakecolor_foldername, fakecolor_folder, csv_path, picpair_name)
    

    end_time = time.perf_counter()  
    print('程序共运行 {_time_}秒'.format(_time_=(end_time - start_time)))
    
general_run()

# path_a = 'Data/Model.png'
# img = cv2.imread(path_a)
# b,g,r = cv2.split(img)
# ans=ta.tamura_feature(b, 8, 4)
# print(ans)