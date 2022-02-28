import sys
sys.path.append('graduate_design/tests')
from matplotlib import pyplot as plt
import numpy as np
# from numba import jit
import cv2
# from cv2 import dnn_superres
import ColorCharacteristics as cc
import TextureCharacteristics as tc
import LossAboutColor as lac
import TestScripts
import time
import multiprocessing
# import tqdm
from functools import partial

RGB_COLOR_CHANNEL = {
    0: 'b',
    1: 'g',
    2: 'r'
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
    if(i*step+size_w>w)or(j*step+size_h>h):return ([0,0,0,0],[0,0,0,0])
    raw_a = tc.tamura_feature(gray_img_a[i*step:(i*step+size_w), j*step:(j*step+size_h)], kmax, dist)
    raw_b = tc.tamura_feature(gray_img_b[i*step:(i*step+size_w), j*step:(j*step+size_h)], kmax, dist)
    return (raw_a, raw_b)
def multithread_temurafeture_single_channel_slide_window_parameters( gray_img_a, gray_img_b, step, size_w, size_h):
    w = gray_img_a.shape[0]
    h = gray_img_a.shape[1]
    kmax = 3    
    dist = 4
    if((w%size_w!=0)or(h%size_h!=0)):
        print('Please check slide window SIZE!')
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
#     print(i,j)
#     if(i*step+size_w>w)or(j*step+size_h>h):return ([0,0,0,0],[0,0,0,0])
#     raw_a = tc.tamura_feature(gray_img_a[i*step:(i*step+size_w), j*step:(j*step+size_h)], d_x, d_y)
#     raw_b = tc.tamura_feature(gray_img_b[i*step:(i*step+size_w), j*step:(j*step+size_h)], d_x, d_y)
#     return (raw_a, raw_b)


# def multithread_glcmfeature_single_channel_slide_window_parameters(gray_img_a, gray_img_b, step, size_w, size_h, d_x, d_y):
#     w = gray_img_a.shape[0]
#     h = gray_img_a.shape[1]

#     if((w%size_w!=0)or(h%size_h!=0)):
#         print('Please check slide window SIZE!')
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

# 三通道伪彩色，目前是为损失函数写的，故func调用了两张图片作为参数（与单通道不同）
def rgb_channel_slide_window_parameters(rgb_img_a, rgb_img_b, func , step = 8, size_w = 40, size_h = 40, *args, **kwargs):
    w = rgb_img_a.shape[0]
    h = rgb_img_a.shape[1]
    if((w%size_w!=0)or(h%size_h!=0)):
        print('Please check slide window SIZE!')
        return
    ans = []  
    for i in range(int(w/step)):
        raw = [] 
        for j in range(int(h/step)): 
            if(i*step+size_w>=w)or(j*step+size_h>=h):break           
            raw.append(func(rgb_img_a[i*step:(i*step+size_w), j*step:(j*step+size_h)], rgb_img_b[i*step:(i*step+size_w), j*step:(j*step+size_h)], *args, **kwargs))
        if(raw!=[]):ans.append(raw)
    ans = np.array(ans)
    ans = ans.transpose(2,0,1) 
    ans = (255*ans) / np.max(ans)    
    return ans


# 单通道伪彩色，输入为灰度矩阵， 输出为低分辨率参数差绝对值矩阵
def single_channel_slide_window_parameters(gray_img_a,gray_img_b,  func , step = 8, size_w = 40, size_h = 40, *args, **kwargs):
    w = gray_img_a.shape[0]
    h = gray_img_a.shape[1]
    if((w%size_w!=0)or(h%size_h!=0)):
        print('Please check slide window SIZE!')
        return
    ans_a = []
    ans_b = []   
    #!
    start_time = time.perf_counter()
    signal_single = False
    signal_inside = False
    # for i in tqdm.trange(int(w/step)):
    for i in range(int(w/step)):
        raw_a = []
        raw_b = []
        #!
        start_time_inside = time.perf_counter() 
        for j in range(int(h/step)): 
            # print(i,j)
            if(i*step+size_w>w)or(j*step+size_h>h):break   
            start_time_single = time.perf_counter()   
            # print('submat shape is {shape}'.format(shape=np.shape(gray_img_a[i*step:(i*step+size_w), j*step:(j*step+size_h)])))     
            raw_a.append(func(gray_img_a[i*step:(i*step+size_w), j*step:(j*step+size_h)], *args, **kwargs))
            end_time_single = time.perf_counter() 
            if(signal_single==False):
                # print('单指令共运行了 {_time_}秒'.format(_time_=(end_time_single - start_time_single)))
                signal_single=True
            raw_b.append(func(gray_img_b[i*step:(i*step+size_w), j*step:(j*step+size_h)], *args, **kwargs))
            # print("raw_a type is {}".format(type(raw_a)))
        #!
        end_time_inside = time.perf_counter() 
        if(signal_inside==False):
            # print('内循环共运行了 {_time_}秒'.format(_time_=(end_time_inside - start_time_inside)))
            signal_inside=True
        if(raw_a!=[]):ans_a.append(raw_a)
        if(raw_b!=[]):ans_b.append(raw_b) 
        
    #!
    end_time = time.perf_counter()  
    # print('循环共运行了 {_time_}秒'.format(_time_=(end_time - start_time)))
    ans_a = np.array(ans_a)
    ans_b = np.array(ans_b)
    ans = np.abs(ans_a-ans_b)
    # print(np.shape(ans))
    # print(ans_a)
    ans = ans.transpose(2,0,1)
    ans = (255*ans) / np.max(ans)    
    return ans

# 单通道伪彩色，输入为灰度矩阵，输出为低分辨率向量距离矩阵
def single_channel_slide_window_vectors(gray_img_a, gray_img_b, func , step = 8, size_w = 0, size_h = 0, *args, **kwargs):
    w = gray_img_a.shape[0]
    h = gray_img_a.shape[1]
    ans = [] 
    if((w%size_w!=0)or(h%size_h!=0)):
        print('Please check slide window SIZE!')
        return
    for i in range(int(w/step)):
        raw = [] 
        for j in range(int(h/step)):
            vec_a_first, vec_a_second = func(gray_img_a[i*step:(i*step+size_w), j*step:(j*step+size_h)], *args, **kwargs)
            vec_b_first, vec_b_second = func(gray_img_b[i*step:(i*step+size_w), j*step:(j*step+size_h)], *args, **kwargs)
            distance = np.sqrt(np.square(vec_a_first-vec_b_first)+np.square(vec_a_second-vec_b_second))
            raw.append(distance)
        ans.append(raw)
    ans = np.array(ans)
    ans = ans.transpose(2,0,1)
    ans = (255*ans) / np.max(ans)    
    return ans

# !:有大问题，输出的矩阵看起来是二维的，实际上是list拼list拼list拼出来的，画图或者计算max都报错
# 单通道伪彩色，输入为灰度图，输出为等大小灰度图结果
def single_channel_slide_window_pictures(gray_img_a, gray_img_b, func , step = 8, size_w = 0, size_h = 0, *args, **kwargs):
    w = gray_img_a.shape[0]
    h = gray_img_a.shape[1]
    ans_a = []
    ans_b = []  
    if((w%size_w!=0)or(h%size_h!=0)):
        print('Please check slide window SIZE!')
        return
    for i in range(int(w/step)):
        raw_a = []
        raw_b = []
        for j in range(int(h/step)):
            img_a = func(gray_img_a[i*step:(i*step+size_w), j*step:(j*step+size_h)], *args, **kwargs)
            img_b = func(gray_img_b[i*step:(i*step+size_w), j*step:(j*step+size_h)], *args, **kwargs)
            print(np.shape(img_a))
            # np.hstack((raw_a,img_a))
            # np.hstack((raw_b,img_b))
            raw_a.append(img_a)
            raw_b.append(img_b)
            # print(np.shape(col_a))
 
        ans_a.append(raw_a)
        ans_b.append(raw_b)
    print(type(ans_a))
    print(ans_a)
    ans_a = np.array(ans_a)
    ans_b = np.array(ans_b)
    # print(type(ans_a))
    # print(np.shape(ans_a))
    ans = np.abs(ans_a-ans_b)
    # ans = np.reshape(ans,(int(w/step),int(h/step)))
    # print(ans)
    # ans = ans.transpose(2,0,1)
    # ans = (255*ans) / np.max(ans)    
    return ans


class FakeColor_Texture_Characteristecs(object):
    
    # !窗口是20*20跑不了，40*40可以。。。
    @TestScripts.timmer
    def __fakecolor_texture_characteristics_glcm_feature(self):
        glcm_feature_label = ['Energy', 'Entropy', 'Contrast', 'IDM']
        DIRECTION = {
            0: [1,0],
            1: [0,1],
            2: [1,1],
            3: [-1,1]
        }
        for i in range(3):
            for j in range(4):
                ans = single_channel_slide_window_parameters(self.matrix_a[i], self.matrix_b[i], tc.glcm_feature, self.step, self.size_w, self.size_h,d_x=DIRECTION.get(j)[0],d_y=DIRECTION.get(j)[1])
                
                for k in range(ans.shape[0]):
                    label = 'Texture_GLCMFeature_'+RGB_COLOR_CHANNEL.get(i) +'_direction'+str(DIRECTION.get(j))
                    path = self.folder + label+'_'+glcm_feature_label[k]+'.jpg'
                    self.csv_generate(ans[k], label)
                    ans_highsolution = cv2.resize(ans[k], None, fx=self.step, fy=self.step, interpolation=cv2.INTER_LINEAR)
                    ans_highsolution = ans_highsolution.astype(np.uint8)
                    ans_fakecolor = cv2.applyColorMap(ans_highsolution, cv2.COLORMAP_HOT)
                    cv2.imwrite(path, ans_highsolution)
                    print(path)
                    # plt.figure(figsize=self.figsize)
                    # plt.imshow(ans_highsolution,vmin = 0, vmax = 255,cmap = "hot")
                    # plt.colorbar()
                    # plt.savefig(path)
                    # plt.close()
        return
    
    #!Abandoned ! 单任务耗时过少，开启多进程反而增加时间消耗
    # @TestScripts.timmer
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
    #                 print(path)
    #                 plt.figure(figsize=self.figsize)
    #                 plt.imshow(ans_highsolution,vmin = 0, vmax = 255,cmap = "hot")
    #                 plt.colorbar()
    #                 plt.savefig(path)
    #                 plt.close()
    #     return
    
    @TestScripts.timmer
    def __fakecolor_texture_characteristics_lbp(self):
        print("LBP的结构信息不适用于滑动窗口的局部计算,将计算整体信息")
        plt.figure(figsize=self.figsize)
        plt.title('lbp')
        path = self.folder + 'Texture_LBP.jpg'
        for i in range(3):
            # print(i)
            ax1 = plt.subplot(3,4,4*i+1)
            lbp_a = tc.rotation_invariant_LBP(self.matrix_a[i])
            ax1.imshow(lbp_a,cmap='gray')
            ax2 = plt.subplot(3,4,4*i+2)
            lbp_a = lbp_a.astype(np.uint8)
            hist_lbp_a = cc.histogram(lbp_a)
            ax2.plot(hist_lbp_a,RGB_COLOR_CHANNEL.get(i))
            ax3 = plt.subplot(3,4,4*i+3)
            lbp_b = tc.rotation_invariant_LBP(self.matrix_b[i])
            ax3.imshow(lbp_b,cmap='gray')
            ax4 = plt.subplot(3,4,4*i+4)
            lbp_b = lbp_b.astype(np.uint8)
            hist_lbp_b = cc.histogram(lbp_b)
            ax4.plot(hist_lbp_b,RGB_COLOR_CHANNEL.get(i))
        plt.tight_layout()
        plt.plot()
        plt.savefig(path)
        plt.close()
        return 
    
    # ! ABANDONED
    # @TestScripts.timmer
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
    #             print(path)
    #             plt.figure(figsize=self.figsize)
    #             plt.imshow(ans_highsolution,vmin = 0, vmax = 255,cmap = "hot")
    #             plt.colorbar()
    #             plt.savefig(path)
    #             plt.close()
    #     return
    
    @TestScripts.timmer
    def __multithread_fakecolor_texture_characteristics_tamura_feature(self):
        kmax = 3    
        dist = 4
        tamura_label = ['coarseness', 'contrast', 'directionality', 'linelikeness']
        # tamura_label = ['contrast']
        for i in range(3):
            ans = multithread_temurafeture_single_channel_slide_window_parameters(self.matrix_a[i], self.matrix_b[i], self.step, self.size_w, self.size_h)
          
            for j in range(ans.shape[0]):
                label = 'Texture_TamuraFeature_'+RGB_COLOR_CHANNEL.get(i) +'_'+tamura_label[j]
                path = self.folder + label+'.jpg'
                self.csv_generate(ans[j], label)
                ans_highsolution = cv2.resize(ans[j], None, fx=self.step, fy=self.step, interpolation=cv2.INTER_LINEAR)
                ans_highsolution = ans_highsolution.astype(np.uint8)
                ans_fakecolor = cv2.applyColorMap(ans_highsolution, cv2.COLORMAP_HOT)
                cv2.imwrite(path, ans_highsolution)
                print(path)
                # plt.figure(figsize=self.figsize)
                # plt.imshow(ans_highsolution,vmin = 0, vmax = 255,cmap = "hot")
                # plt.colorbar()
                # plt.savefig(path)
                # plt.close()
        return    
    
    @TestScripts.timmer
    def __fakecolor_texture_characteristics_dwt_feature(self):
        wave_func = 'haar'
        dwt_label = ['average_ca', 'entropy_ca', 'sigma_ca', 'energy_ca',
                        'average_ch', 'entropy_ch', 'sigma_ch', 'energy_ch',
                        'average_cv', 'entropy_cv', 'sigma_cv', 'energy_cv',
                        'average_cd', 'entropy_cd', 'sigma_cd', 'energy_cd']
        for i in range(3):
            ans = single_channel_slide_window_parameters(self.matrix_a[i], self.matrix_b[i], tc.dwt_feature, self.step, self.size_w, self.size_h, wave_func=wave_func)
            
            for j in range(ans.shape[0]):
                label = 'Texture_DWTFeature_'+RGB_COLOR_CHANNEL.get(i) +'_'+dwt_label[j]
                path = self.folder + label+'.jpg'
                self.csv_generate(ans[j], label)
                ans_highsolution = cv2.resize(ans[j], None, fx=self.step, fy=self.step, interpolation=cv2.INTER_LINEAR)
                ans_highsolution = ans_highsolution.astype(np.uint8)
                ans_fakecolor = cv2.applyColorMap(ans_highsolution, cv2.COLORMAP_HOT)
                cv2.imwrite(path, ans_highsolution)
                print(path)
                # plt.figure(figsize=self.figsize)
                # plt.imshow(ans_highsolution,vmin = 0, vmax = 255,cmap = "hot")
                # plt.colorbar()
                # plt.savefig(path)
                # plt.close()
        return
    
    @TestScripts.timmer
    def __fakecolor_texture_characteristics_laws_feature(self):
        laws_label = ['0', '1', '2', '3', '4', '5', '6', '7']
        for i in range(3):
            
            laws_feature_single_feature_a = tc.laws_feature(self.matrix_a[i])
            laws_feature_single_feature_b = tc.laws_feature(self.matrix_b[i])
            laws_feature_single_feature_a = np.array(laws_feature_single_feature_a)
            laws_feature_single_feature_b = np.array(laws_feature_single_feature_b)
            laws_feature_single_feature = np.absolute(laws_feature_single_feature_a-laws_feature_single_feature_b)
            for j in range(8):
                label = 'Texture_Laws_'+RGB_COLOR_CHANNEL.get(i)+'_'+laws_label[j]
                path = self.folder + label+'.jpg'
                self.csv_generate(laws_feature_single_feature[j], label)
                
                ans_highsolution = laws_feature_single_feature[j].astype(np.uint8)
                ans_fakecolor = cv2.applyColorMap(ans_highsolution, cv2.COLORMAP_HOT)
                cv2.imwrite(path, ans_highsolution)
                print(path)
                # plt.figure(figsize=self.figsize)
                # plt.imshow(laws_feature_single_feature[j],vmin = 0, vmax = 255,cmap = "hot")
                # plt.colorbar()
                # plt.savefig(path)
                # plt.close()
    
    def csv_generate(self, ans, label):
        self.csv_label.append(label)
        cal = np.sum(ans>=10)
        total = np.size(ans)
        # print(cal, total, cal/total)
        if(cal/total>0.01):
            self.csv_data.append(1)
        else:
            self.csv_data.append(0)
    
    
     
    def fakecolor_texture_characteristics(self):
        
        self.__fakecolor_texture_characteristics_glcm_feature()
        
        # self.__fakecolor_texture_characteristics_lbp()
        
        self.__multithread_fakecolor_texture_characteristics_tamura_feature()
        self.__fakecolor_texture_characteristics_dwt_feature()
        self.__fakecolor_texture_characteristics_laws_feature()
        return