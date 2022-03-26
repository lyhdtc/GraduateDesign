import sys
import os
sys.path.append(os.pardir)
from matplotlib import pyplot as plt
import numpy as np
import cv2
from Color import ColorAlgorithrm as ca
from Texture import TextureAlgorithrm as ta
# import LossAboutColor as lac
from Tools import TimeCount as tc
from Tools import SlideWindow as sw
import multiprocessing
# import tqdm
from functools import partial

LAB_COLOR_CHANNEL = {
    0: 'l',
    1: 'a',
    2: 'b'
}

# TODO:
# cc.color_coherence_vector输出为二维向量(fixed)
# tc.glcm_feature貌似存在越界情况(fixed，把窗口放大了，不知道为啥20*20跑不了)
# tc.rotation_invariant_LBP本身输出就是灰度图了(fixed，直接套用了表格的图)
# tc.tamura_feature需要多核优化，且输出为空白(fixed)
# tc.dwt_feature输出为全白，需要看下代码(fixed,__norm(ar):分母除0，增加了一个1e-7)
# tc.laws_feature本身输出就是一组图像了(fixed，直接套用了表格的图)
# loss function 输入为彩色图像，还需要重写一组
#       (fixed，出现了空白输出的情况，排查代码发现__msssim(img1,img2)最后计算overall_mssim的时候写成了
#       mcs_array[：level-1],weight[:level-1]，目前参照了其他几个人的代码把冒号去掉了（不排除是之前写错了））

# TemuraFeature 多核优化版本，可以跑满cpu，从原来的8h降低至1h
def __temura_inside(j, kmax, dist, step, w,h,size_w, size_h, gray_img_a, gray_img_b, i):
    if(i*step+size_w>w)or(j*step+size_h>h):return ([0,0,0,0,0,0],[0,0,0,0,0,0])
    raw_a = ta.tamura_feature(gray_img_a[i*step:(i*step+size_w), j*step:(j*step+size_h)], kmax, dist)
    raw_b = ta.tamura_feature(gray_img_b[i*step:(i*step+size_w), j*step:(j*step+size_h)], kmax, dist)
    return (raw_a, raw_b)
def multithread_temurafeture_single_channel_slide_window_parameters( gray_img_a, gray_img_b, step, size_w, size_h):
    w = gray_img_a.shape[0]
    h = gray_img_a.shape[1]
    kmax = 3    
    dist = 4
    if((w%size_w!=0)or(h%size_h!=0)):
        #print('Please check slide window SIZE!')
        return
    ans_a = []
    ans_b = []

    # for i in tqdm.trange(int(w/step)):
    for i in range(int(w/step)):
        pool = multiprocessing.Pool()
        func = partial(__temura_inside, kmax = kmax, dist = dist, step = step, w = w, h = h,size_w = size_w, size_h = size_h, gray_img_a = gray_img_a, gray_img_b = gray_img_b, i = i)
        raw= pool.map(func, range(int(h/step)))
        pool.close()
        pool.join()
        raw = list(map(list, zip(*raw)))
        raw_a = raw[0]
        raw_b = raw[1]

        if(raw_a!=[]):ans_a.append(raw_a)
        if(raw_b!=[]):ans_b.append(raw_b) 

    ans_a = np.array(ans_a)
    ans_b = np.array(ans_b)
    ans = np.abs(ans_a-ans_b)
    ans = ans.transpose(2,0,1)
    ans = (255*ans) / np.max(ans) 
    return ans

#!Abandoned ! 单任务耗时过少，开启多进程反而增加时间消耗
# # glcm feature 多核优化版本，可以跑满cpu
# def __glcm_inside(j, d_x, d_y, step, w,h,size_w, size_h, gray_img_a, gray_img_b, i):
#     #print(i,j)
#     if(i*step+size_w>w)or(j*step+size_h>h):return ([0,0,0,0],[0,0,0,0])
#     raw_a = tc.tamura_feature(gray_img_a[i*step:(i*step+size_w), j*step:(j*step+size_h)], d_x, d_y)
#     raw_b = tc.tamura_feature(gray_img_b[i*step:(i*step+size_w), j*step:(j*step+size_h)], d_x, d_y)
#     return (raw_a, raw_b)


# def multithread_glcmfeature_single_channel_slide_window_parameters(gray_img_a, gray_img_b, step, size_w, size_h, d_x, d_y):
#     w = gray_img_a.shape[0]
#     h = gray_img_a.shape[1]

#     if((w%size_w!=0)or(h%size_h!=0)):
#         #print('Please check slide window SIZE!')
#         return
#     ans_a = []
#     ans_b = []

#     for i in tqdm.trange(int(w/step)):
#         pool = multiprocessing.Pool(2)
#         func = partial(__glcm_inside, d_x=d_x, d_y=d_y, step = step, w = w, h = h,size_w = size_w, size_h = size_h, gray_img_a = gray_img_a, gray_img_b = gray_img_b, i = i)
#         raw= pool.map(func, range(int(h/step)))
#         pool.close()
#         pool.join()
#         raw = list(map(list, zip(*raw)))
#         raw_a = raw[0]
#         raw_b = raw[1]

#         if(raw_a!=[]):ans_a.append(raw_a)
#         if(raw_b!=[]):ans_b.append(raw_b) 

#     ans_a = np.array(ans_a)
#     ans_b = np.array(ans_b)
#     ans = np.abs(ans_a-ans_b)
#     ans = ans.transpose(2,0,1)
#     ans = (255*ans) / np.max(ans) 
#     return ans    




class FakeColor_Texture_Characteristecs(object):
    
    # !窗口是20*20跑不了，40*40可以。。。
    #@tc.timmer
    def __fakecolor_texture_glcm_feature(self):
        glcm_feature_label = ['energy','contrast','homogeneity','correlation','entropy']
        for i in range(3):
          
                ans = sw.single_channel_parameters(self.matrix_a[i], self.matrix_b[i], ta.glcm_feature, self.step, self.size_w, self.size_h,distance=1)
                
                for k in range(ans.shape[0]):
                    label = 'Texture_GLCMFeature_'+LAB_COLOR_CHANNEL.get(i)
                    # path = self.folder + label+'_'+glcm_feature_label[k]+'.jpg'
                    path = os.path.join(self.folder, label+'_'+glcm_feature_label[k]+'.jpg')
                    self.csv_generate(ans[k], label)
                    # ans_highsolution = cv2.resize(ans[k], None, fx=self.step, fy=self.step, interpolation=cv2.INTER_LINEAR)
                    # ans_highsolution = ans_highsolution.astype(np.uint8)
                    # ans_fakecolor = cv2.applyColorMap(ans_highsolution, cv2.COLORMAP_HOT)
                    ans_reshape = cv2.resize(ans[k], self.reshape_size, cv2.INTER_LINEAR)
                    ans_reshape = ans_reshape.astype(np.uint8)
                    cv2.imwrite(path, ans_reshape)
                    #print(path)
                    # plt.figure(figsize=self.figsize)
                    # plt.imshow(ans_highsolution,vmin = 0, vmax = 255,cmap = "hot")
                    # plt.colorbar()
                    # plt.savefig(path)
                    # plt.close()
        return
    
    #!Abandoned ! 单任务耗时过少，开启多进程反而增加时间消耗
    # #@TestScripts.timmer
    # def __multithread_fakecolor_texture_characteristics_glcm_feature(self):
    #     glcm_feature_label = ['Energy', 'Entropy', 'Contrast', 'IDM']
    #     DIRECTION = {
    #         0: [1,0],
    #         1: [0,1],
    #         2: [1,1],
    #         3: [-1,1]
    #     }
    #     for i in range(3):
    #         for j in range(4):
    #             ans = multithread_glcmfeature_single_channel_slide_window_parameters(self.matrix_a[i], self.matrix_b[i],  self.step, self.size_w, self.size_h,d_x=DIRECTION.get(j)[0],d_y=DIRECTION.get(j)[1])
                
    #             for k in range(ans.shape[0]):
    #                 path = self.folder + 'Texture_GLCMFeature_'+RGB_COLOR_CHANNEL.get(i) +'_direction'+str(DIRECTION.get(j))+'_'+glcm_feature_label[k]+'.jpg'
    #                 ans_highsolution = cv2.resize(ans[k], None, fx=self.step, fy=self.step, interpolation=cv2.INTER_LINEAR)
    #                 #print(path)
    #                 plt.figure(figsize=self.figsize)
    #                 plt.imshow(ans_highsolution,vmin = 0, vmax = 255,cmap = "hot")
    #                 plt.colorbar()
    #                 plt.savefig(path)
    #                 plt.close()
    #     return
    
    #@tc.timmer
    def __fakecolor_texture_lbp(self):
        #print("LBP的结构信息不适用于滑动窗口的局部计算,将计算整体信息")
        path = os.path.join(self.folder,'Texture_LBP.jpg')
        # path = self.folder + 'Texture_LBP.jpg'
        for i in range(3):
            # #print(i)
            label = 'Texture_LBP_' + LAB_COLOR_CHANNEL.get(i)
            lbp_a = ta.rotation_invariant_LBP(self.matrix_a[i])


            lbp_a = lbp_a.astype(np.uint8)


            lbp_b = ta.rotation_invariant_LBP(self.matrix_b[i])

            lbp_b = lbp_b.astype(np.uint8)
            ans = np.abs(lbp_b - lbp_a)
            ans = (255*ans) / np.max(ans)
            self.csv_generate(ans,label)
            # ans_highsolution = cv2.resize(ans, None, fx=self.step, fy=self.step, interpolation=cv2.INTER_LINEAR)
            ans_reshape = cv2.resize(ans, self.reshape_size, cv2.INTER_LINEAR)
            cv2.imwrite(path, ans_reshape)
            #print(path)
        return 
    
    # ! ABANDONED
    # #@TestScripts.timmer
    # def __fakecolor_texture_characteristics_tamura_feature(self):
    #     kmax = 3    
    #     dist = 4
    #     tamura_label = ['coarseness', 'contrast', 'directionality', 'linelikeness']
    #     # tamura_label = ['contrast']
    #     for i in range(3):
    #         ans = single_channel_slide_window_parameters(self.matrix_a[i], self.matrix_b[i], tc.tamura_feature, self.step, self.size_w, self.size_h, kmax=kmax, dist=dist)
          
    #         for j in range(ans.shape[0]):
    #             path = self.folder + 'Texture_TamuraFeature_'+RGB_COLOR_CHANNEL.get(i) +'_'+tamura_label[j]+'.jpg'
    #             ans_highsolution = cv2.resize(ans[j], None, fx=self.step, fy=self.step, interpolation=cv2.INTER_LINEAR)
    #             # ans_highsolution = ans_highsolution.astype(np.uint8)
    #             # ans_fakecolor = cv2.applyColorMap(ans_highsolution, cv2.COLORMAP_JET)
    #             # cv2.imwrite(path, ans_fakecolor)
    #             #print(path)
    #             plt.figure(figsize=self.figsize)
    #             plt.imshow(ans_highsolution,vmin = 0, vmax = 255,cmap = "hot")
    #             plt.colorbar()
    #             plt.savefig(path)
    #             plt.close()
    #     return
    
    #@tc.timmer
    def __multithread_fakecolor_texture_tamura_feature(self):
        kmax = 3    
        dist = 4
        tamura_label = ['coarseness', 'contrast', 'directionality', 'linelikeness', 'regularity', 'roughness']
        # tamura_label = ['contrast']
        for i in range(3):
            # ans = multithread_temurafeture_single_channel_slide_window_parameters(self.matrix_a[i], self.matrix_b[i], self.step, self.size_w, self.size_h)
            ans = sw.single_channel_parameters(self.matrix_a[i], self.matrix_b[i], ta.tamura_feature, self.step, self.size_w, self.size_h, kmax=3, dist=4)
            for j in range(ans.shape[0]):
                label = 'Texture_TamuraFeature_'+LAB_COLOR_CHANNEL.get(i) +'_'+tamura_label[j]
                path = os.path.join(self.folder,label+'.jpg')
                # path = self.folder + label+'.jpg'
                self.csv_generate(ans[j], label)
                # ans_highsolution = cv2.resize(ans[j], None, fx=self.step, fy=self.step, interpolation=cv2.INTER_LINEAR)
                # ans_highsolution = ans_highsolution.astype(np.uint8)
                ans_reshape = cv2.resize(ans[j], self.reshape_size, cv2.INTER_LINEAR)
                ans_reshape = ans_reshape.astype(np.uint8)
                # ans_fakecolor = cv2.applyColorMap(ans_highsolution, cv2.COLORMAP_HOT)
                cv2.imwrite(path, ans_reshape)
                #print(path)
                # plt.figure(figsize=self.figsize)
                # plt.imshow(ans_highsolution,vmin = 0, vmax = 255,cmap = "hot")
                # plt.colorbar()
                # plt.savefig(path)
                # plt.close()
        return    
    
    #@tc.timmer
    def __fakecolor_texture_dwt_feature(self):
        wave_func = 'haar'
        dwt_label = ['average_ca', 'entropy_ca', 'sigma_ca', 'energy_ca',
                        'average_ch', 'entropy_ch', 'sigma_ch', 'energy_ch',
                        'average_cv', 'entropy_cv', 'sigma_cv', 'energy_cv',
                        'average_cd', 'entropy_cd', 'sigma_cd', 'energy_cd']
        for i in range(3):
            ans = sw.single_channel_parameters(self.matrix_a[i], self.matrix_b[i], ta.dwt_feature, self.step, self.size_w, self.size_h, wave_func=wave_func)
            
            for j in range(ans.shape[0]):
                label = 'Texture_DWTFeature_'+LAB_COLOR_CHANNEL.get(i) +'_'+dwt_label[j]
                path = os.path.join(self.folder,label+'.jpg')
                # path = self.folder + label+'.jpg'
                self.csv_generate(ans[j], label)
                # ans_highsolution = cv2.resize(ans[j], None, fx=self.step, fy=self.step, interpolation=cv2.INTER_LINEAR)
                # ans_highsolution = ans_highsolution.astype(np.uint8)
                # ans_fakecolor = cv2.applyColorMap(ans_highsolution, cv2.COLORMAP_HOT)
                
                ans_reshape = cv2.resize(ans[j], self.reshape_size, cv2.INTER_LINEAR)
                ans_reshape = ans_reshape.astype(np.uint8)
                cv2.imwrite(path, ans_reshape)
                #print(path)
                # plt.figure(figsize=self.figsize)
                # plt.imshow(ans_highsolution,vmin = 0, vmax = 255,cmap = "hot")
                # plt.colorbar()
                # plt.savefig(path)
                # plt.close()
        return
    
    #@tc.timmer
    def __fakecolor_texture_laws_feature(self):
        laws_label = ['0', '1', '2', '3', '4', '5', '6', '7']
        for i in range(3):
            
            laws_feature_single_feature_a = ta.laws_feature(self.matrix_a[i])
            laws_feature_single_feature_b = ta.laws_feature(self.matrix_b[i])
            laws_feature_single_feature_a = np.array(laws_feature_single_feature_a)
            laws_feature_single_feature_b = np.array(laws_feature_single_feature_b)
            laws_feature_single_feature = np.absolute(laws_feature_single_feature_a-laws_feature_single_feature_b)
            for j in range(8):
                label = 'Texture_Laws_'+LAB_COLOR_CHANNEL.get(i)+'_'+laws_label[j]
                path = os.path.join(self.folder,label+'.jpg')
                # path = self.folder + label+'.jpg'
                self.csv_generate(laws_feature_single_feature[j], label)
                
                # ans_highsolution = laws_feature_single_feature[j].astype(np.uint8)
                # ans_fakecolor = cv2.applyColorMap(ans_highsolution, cv2.COLORMAP_HOT)
                ans_reshape = cv2.resize(laws_feature_single_feature[j], self.reshape_size, cv2.INTER_LINEAR)
                ans_reshape = ans_reshape.astype(np.uint8)
                cv2.imwrite(path, ans_reshape)
                #print(path)
                # plt.figure(figsize=self.figsize)
                # plt.imshow(laws_feature_single_feature[j],vmin = 0, vmax = 255,cmap = "hot")
                # plt.colorbar()
                # plt.savefig(path)
                # plt.close()
                
    def __fakecolor_texture_gabor(self):
        label = 'Texture_Gabor'
        path = self.folder +'Texture_Gabor.jpg'

        ans_a = ta.gabor_process(self.matrix_a)
        ans_b = ta.gabor_process(self.matrix_b)
        ans = np.abs(ans_a-ans_b)
        ans_highsolution = cv2.resize(ans, None, fx=self.step, fy=self.step, interpolation=cv2.INTER_LINEAR)
        self.csv_generate(ans, label)
        cv2.imwrite(path, ans_highsolution)
        #print(path)
    
    def csv_generate(self, ans, label):
        self.csv_label.append(label)
        cal = np.sum(ans>=10)
        total = np.size(ans)
        # #print(cal, total, cal/total)
        if(cal/total>0.01):
            self.csv_data.append(1)
        else:
            self.csv_data.append(0)
    
    
     
    def fakecolor_texture_characteristics(self):
        
        self.__fakecolor_texture_glcm_feature()
        
        self.__fakecolor_texture_lbp()
        
        self.__multithread_fakecolor_texture_tamura_feature()
        self.__fakecolor_texture_dwt_feature()
        self.__fakecolor_texture_laws_feature()
        # self.__fakecolor_texture_gabor()
        return
    
    def __init__(self, matrix_a, matrix_b, step, size_w, size_h, folder, figsize, reshape_size):
        self.matrix_a = matrix_a
        self.matrix_b = matrix_b
        self.step = step
        self.size_w = size_w
        self.size_h = size_h   
        self.folder = folder 
        self.figsize = figsize
        self.csv_data = []
        self.csv_label = []
        self.reshape_size = reshape_size