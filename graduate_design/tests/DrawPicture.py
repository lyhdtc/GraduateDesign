from genericpath import getsize
import sys
sys.path.append('graduate_design/Characteristics')
sys.path.append('graduate_design/Tools')
sys.path.append('graduate_design/Calculator')
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
import ParsePicName as PPN


def get_img(path):
    img = cv2.imread(path)
    img_r,img_g,img_b = cv2.split(img)
    return img,img_r,img_g,img_b
    

    
def general_run():
     
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

def new_general_run():
    step = 8
    size_w = 40 
    size_h = 40
    figsize = (18,10)
    
    material = ['Lambert', 'Phong', 'Blinn', 'PBR', 'Surface']
    noise = ['gaussian', 'localvar', 'poisson', 'salt', 'pepper', 's&p', 'speckle']
    lighting = ['Up', 'Down', 'Left', 'Right']
    
    start_time = time.perf_counter()
    # print('Start!')
    print("\033[1;33;40mStart!\033[0m")
    
    realpic_path = '/mnt/d/001Graduate/Data/Pre/RealPic/00001.png'
    renderfolder_path = '/mnt/d/001Graduate/Data/Pre/RenderPic/Camera1'
    csv_path = '/mnt/d/001Graduate/Data/Res/Camera1/result.csv'
    renderpics = os.listdir(renderfolder_path)    
    renderpics.sort(key=lambda x:int(PPN.get_pic_info(x,'number')))
    path_a=''
    path_b=''
    list_path_b=[]
# TODO：把文件夹选取写成自动的
# TODO: 先写到json再从json里面读图片对
# TODO： CSV增加第一列图片对类型

# material comparement
    picpair_name = 'Material'
    for i in tqdm.tqdm(renderpics):
        if(PPN.get_pic_info(i, 'pre_noise')=='N' and PPN.get_pic_info(i, 'lighting')=='N' and PPN.get_pic_info(i, 'after_noise')=='N'):
            path_a = realpic_path
            path_b = renderfolder_path+'/'+i
            
            
            fakecolor_foldername = os.path.basename(path_a)[:-4]+"____"+os.path.basename(path_b)[:-4]
            fakecolor_folder = '/mnt/d/001Graduate/Data/Res/Camera1/'+picpair_name+'/'+fakecolor_foldername+'/'  
            FakeColorCSV.fakecolor_and_csv(path_a, path_b, step, size_w, size_h, figsize, fakecolor_foldername, fakecolor_folder, csv_path, picpair_name)
    print("\033[1;33;40mMaterial Finished!\033[0m")
    
    
# prenoise comparement  
    picpair_name = 'Prenoise'
    for i in material:
        for j in renderpics:
            if(PPN.get_pic_info(j, 'pre_noise')=='N' and PPN.get_pic_info(j,'material')==i and PPN.get_pic_info(j, 'lighting')=='N' and PPN.get_pic_info(j, 'after_noise')=='N'):
                path_a=renderfolder_path+'/'+j
            elif (PPN.get_pic_info(j, 'pre_noise')!='N' and PPN.get_pic_info(j,'material')==i and PPN.get_pic_info(j, 'lighting')=='N' and PPN.get_pic_info(j, 'after_noise')=='N'):
                list_path_b.append(j)
                                
        for k in tqdm.tqdm(list_path_b):
            path_b = renderfolder_path+'/'+k
            fakecolor_foldername = os.path.basename(path_a)[:-4]+"____"+os.path.basename(path_b)[:-4]
            fakecolor_folder = '/mnt/d/001Graduate/Data/Res/Camera1/'+picpair_name+'/'+fakecolor_foldername+'/'  
            FakeColorCSV.fakecolor_and_csv(path_a, path_b, step, size_w, size_h, figsize, fakecolor_foldername, fakecolor_folder, csv_path, picpair_name)

        list_path_b=[]
    print("\033[1;33;40mPreNoise Finished!\033[0m")
    
# lighting comparement
    picpair_name = 'Lighting'
    for j in renderpics:
        if(PPN.get_pic_info(j, 'pre_noise')=='N' and PPN.get_pic_info(j,'material')=='PBR' and PPN.get_pic_info(j, 'lighting')=='N' and PPN.get_pic_info(j, 'after_noise')=='N'):
            path_a=renderfolder_path+'/'+j
        elif (PPN.get_pic_info(j, 'pre_noise')=='N' and PPN.get_pic_info(j,'material')=='PBR' and PPN.get_pic_info(j, 'lighting')!='N' and PPN.get_pic_info(j, 'after_noise')=='N'):
            list_path_b.append(j)

    for k in tqdm.tqdm(list_path_b):
        path_b = renderfolder_path+'/'+k
        fakecolor_foldername = os.path.basename(path_a)[:-4]+"____"+os.path.basename(path_b)[:-4]
        fakecolor_folder = '/mnt/d/001Graduate/Data/Res/Camera1/'+picpair_name+'/'+fakecolor_foldername+'/'  
        FakeColorCSV.fakecolor_and_csv(path_a, path_b, step, size_w, size_h, figsize, fakecolor_foldername, fakecolor_folder, csv_path, picpair_name)

    list_path_b=[]    
    print("\033[1;33;40mLighting Finished!\033[0m")   
    
# afternoise comparement
    picpair_name = 'Afternoise'
    for i in material:
        for j in renderpics:
            if(PPN.get_pic_info(j, 'pre_noise')=='N' and PPN.get_pic_info(j,'material')==i and PPN.get_pic_info(j, 'lighting')=='N' and PPN.get_pic_info(j, 'after_noise')=='N'):
                path_a=renderfolder_path+'/'+j
            elif (PPN.get_pic_info(j, 'pre_noise')=='N' and PPN.get_pic_info(j,'material')==i and PPN.get_pic_info(j, 'lighting')=='N' and PPN.get_pic_info(j, 'after_noise')!='N'):
                list_path_b.append(j)
                
        for k in tqdm.tqdm(list_path_b):
            path_b = renderfolder_path+'/'+k
            fakecolor_foldername = os.path.basename(path_a)[:-4]+"____"+os.path.basename(path_b)[:-4]
            fakecolor_folder = '/mnt/d/001Graduate/Data/Res/Camera1/'+picpair_name+'/'+fakecolor_foldername+'/'  
            FakeColorCSV.fakecolor_and_csv(path_a, path_b, step, size_w, size_h, figsize, fakecolor_foldername, fakecolor_folder, csv_path, picpair_name)

        list_path_b=[]
    print("\033[1;33;40mAfterNoise Finished!\033[0m")
 

    

    end_time = time.perf_counter()  
    print("\033[1;33;40m程序共运行 {_time_}秒\033[0m".format(_time_=(end_time - start_time)))

if __name__=="__main__":
   
    # path_a = '/mnt/d/001Graduate/lyh_01/RenderPic/Camera1/36_N_Surface_N_N.jpg'
    # Noise.generate_noise_pictures(path_a)
    # print(os.path.basename(path_a))
    # ans = os.listdir(path_a)
    # a = ParsePicName.get_pic_info(ans[3], "material")
    # print(a)
    # b = ParsePicName.generate_pic_info(['323','adf','adsf','ewr','rtte'])
    # print(b)
    
    new_general_run()
    
    
    