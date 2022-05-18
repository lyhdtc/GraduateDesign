# from concurrent.futures import process
from email.mime import base
import os
# from pydoc import describe
import sys

# from torch import uint8
from tqdm import tqdm,trange
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
import random
# from multiprocessing import Process, Lock
import multiprocessing
from functools import partial


def calculate_dataset_inside(i, step, size_w, size_h, figsize, csv_path, reshape_size, o_folder, p0_folder, p1_folder, save_path):
    filename = str(i).zfill(5) + '.png'
    filepath_o = os.path.join(o_folder, filename)
    filepath_p0= os.path.join(p0_folder, filename)
    filepath_p1= os.path.join(p1_folder, filename)
    
    save_path_i = os.path.join(save_path, str(i).zfill(5))
    save_path_o_p0 = os.path.join(save_path_i, 'o_p0')
    save_path_o_p1 = os.path.join(save_path_i, 'o_p1')
    
    FakeColorCSV.fakecolor_and_csv(filepath_o, filepath_p0, step, size_w, size_h, figsize, str(i).zfill(5), save_path_o_p0, csv_path, 'o_p0', reshape_size)
    FakeColorCSV.fakecolor_and_csv(filepath_o, filepath_p1, step, size_w, size_h, figsize, str(i).zfill(5), save_path_o_p1, csv_path, 'o_p1', reshape_size)
    

def dataset_run():
     
    start_time = time.perf_counter()
    print('Start!')
    step = 8
    size_w = 40
    size_h = 40
    figsize = (18,10)
    reshape_size = (400,400)
    
    # 网络生成的文件夹路径，这个路径下面应该有o,p0,p1三个文件夹
    dataset_folder = '/home/lyh/002Experiment/Masked'
    
    o_folder = os.path.join(dataset_folder, 'o')
    p0_folder = os.path.join(dataset_folder, 'p0')
    p1_folder = os.path.join(dataset_folder, 'p1')
    # 保存结果的根路径
    save_path = '/home/lyh/002Experiment/NormTest_Shading'
    # csv
    csv_path = '/home/lyh/002Experiment/DataTest.csv'
    fileslen = len(os.listdir(o_folder))
    # for i in range(fileslen):
    #     filename = str(i).zfill(5) + '.png'
    #     filepath_o = os.path.join(o_folder, filename)
    #     filepath_p0= os.path.join(p0_folder, filename)
    #     filepath_p1= os.path.join(p1_folder, filename)
        
    #     save_path_i = os.path.join(save_path, str(i).zfill(5))
    #     save_path_o_p0 = os.path.join(save_path_i, 'o_p0')
    #     save_path_o_p1 = os.path.join(save_path_i, 'o_p1')
        
    #     FakeColorCSV.fakecolor_and_csv(filepath_o, filepath_p0, step, size_w, size_h, figsize, str(i).zfill(5), save_path_o_p0, csv_path, 'o_p0', reshape_size)
    #     FakeColorCSV.fakecolor_and_csv(filepath_o, filepath_p1, step, size_w, size_h, figsize, str(i).zfill(5), save_path_o_p1, csv_path, 'o_p1', reshape_size)
    pool = multiprocessing.Pool()
    func = partial(calculate_dataset_inside, step=step, size_w=size_w, size_h=size_h, figsize=figsize, csv_path=csv_path, reshape_size=reshape_size, o_folder=o_folder, p0_folder=p0_folder, p1_folder=p1_folder, save_path=save_path)
    list( tqdm(pool.imap(func,range(fileslen)), total = fileslen, desc='Progress'))

    # list((tqdm(p.imap(f, range(10)), total=10, desc='监视进度')))
    pool.close()
    pool.join()

    end_time = time.perf_counter()  
    print('程序共运行 {_time_}秒'.format(_time_=(end_time - start_time)))