import sys
sys.path.append('../test/')
from matplotlib import pyplot as plt
import numpy as np
from numba import jit
import cv2
from cv2 import dnn_superres
import ColorCharacteristics as cc
import TextureCharacteristics as tc
import LossAboutColor as lac
import TestScripts


RGB_COLOR_CHANNEL = {
    0: 'b',
    1: 'g',
    2: 'r'
}

#不支持的：
# cc.color_coherence_vector输出为二维向量
# tc.glcm_feature貌似存在越界情况
# tc.rotation_invariant_LBP本身输出就是灰度图了
# tc.tamura_feature需要多核优化
# tc.dwt_feature输出为全白，需要看下代码
# tc.laws_feature本身输出就是一组图像了
# loss function 输入为彩色图像，还需要重写一组

# 单通道伪彩色，输入为灰度矩阵， 输出为低分辨率参数差绝对值矩阵

def single_channel_slide_window_parameters(gray_img_a,gray_img_b,  func , step = 8, size_w = 0, size_h = 0, *args, **kwargs):
    w = gray_img_a.shape[0]
    h = gray_img_a.shape[1]
    if((w%size_w!=0)or(h%size_h!=0)):
        print('Please check slide window SIZE!')
        return
    ans_a = []
    ans_b = []    
    for i in range(int(w/step)):
        col_a = []
        col_b = []
        for j in range(int(h/step)):            
            col_a.append(func(gray_img_a[i*step:(i*step+size_w), j*step:(j*step+size_h)], *args, **kwargs))
            col_b.append(func(gray_img_b[i*step:(i*step+size_w), j*step:(j*step+size_h)], *args, **kwargs))
        ans_a.append(col_a)
        ans_b.append(col_b)   
    ans_a = np.array(ans_a)
    ans_b = np.array(ans_b)
    ans = np.abs(ans_a-ans_b)
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
        col = [] 
        for j in range(int(h/step)):
            vec_a_first, vec_a_second = func(gray_img_a[i*step:(i*step+size_w), j*step:(j*step+size_h)], *args, **kwargs)
            vec_b_first, vec_b_second = func(gray_img_b[i*step:(i*step+size_w), j*step:(j*step+size_h)], *args, **kwargs)
            distance = np.sqrt(np.square(vec_a_first-vec_b_first)+np.square(vec_a_second-vec_b_second))
            col.append(distance)
        ans.append(col)
    ans = np.array(ans)
    ans = ans.transpose(2,0,1)
    ans = (255*ans) / np.max(ans)    
    return ans

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
        col_a = [] 
        col_b = [] 
        for j in range(int(h/step)):
            img_a = func(gray_img_a[i*step:(i*step+size_w), j*step:(j*step+size_h)], *args, **kwargs)
            img_b = func(gray_img_b[i*step:(i*step+size_w), j*step:(j*step+size_h)], *args, **kwargs)
            print(np.shape(img_a))
            col_a.append(img_a)
            col_b.append(img_b)
        ans_a.append(col_a)
        ans_b.append(col_b)
    ans_a = np.array(ans_a)
    ans_b = np.array(ans_b)
    ans = np.abs(ans_a-ans_b)
    # ans = ans.transpose(2,0,1)
    ans = (255*ans) / np.max(ans)    
    return ans

class FakeColor_Color_Characteristics(object): 
    @TestScripts.timmer
    def __fakecolor_color_characteristics_histogram(self):
        plt.figure()
        plt.title('histogram')
        path = self.folder + 'Histogram.jpg'
        for i in range(3):
            ax1 = plt.subplot(3,1,i+1)
            hist_title = RGB_COLOR_CHANNEL.get(i)+' channel'
            ax1.set_title(hist_title)
            hist_a = cc.histogram(self.matrix_a[i])        
            plt.plot(hist_a,RGB_COLOR_CHANNEL.get(i))
            hist_b = cc.histogram(self.matrix_b[i])        
            plt.plot(hist_b,RGB_COLOR_CHANNEL.get(i),linestyle='dashed') 
        plt.tight_layout()
        plt.plot()
        plt.savefig(path)
        plt.close
        return
    
    @TestScripts.timmer
    def __fakecolor_color_characteristics_color_moments(self):        
        color_moments_label = ['1st', '2nd', '3rd']
        for i in range(3):
            ans = single_channel_slide_window_parameters(self.matrix_a[i], self.matrix_b[i], cc.color_moments, self.step, self.size_w, self.size_h)
            
            for j in range(ans.shape[0]):
                path = self.folder + 'ColorMoments_'+RGB_COLOR_CHANNEL.get(i) +'_'+color_moments_label[j]+'.jpg'
                ans_highsolution = cv2.resize(ans[j], None, fx=self.step, fy=self.step, interpolation=cv2.INTER_LINEAR)
                # ans_fakecolor = cv2.applyColorMap(ans_highsolution, cv2.COLORMAP_JET)
                # cv2.imwrite(path, ans_fakecolor)
                print(path)
                plt.figure()
                plt.imshow(ans_highsolution,vmin = 0, vmax = 255,cmap = "hot")
                plt.colorbar()
                plt.savefig(path)
                plt.close()
        return
    
    @TestScripts.timmer
    def __fakecolor_color_characteristics_ordinary_moments(self):
        ordinary_moments_label = ['m00', 'm10', 'm01', 'm20', 'm11', 'm02', 'm30', 'm21', 'm12',
                                     'm02', 'mu20', 'mu11', 'mu02', 'mu30', 'mu21', 'mu12', 'mu03',
                                     'nu20', 'nu11', 'nu02', 'nu30', 'nu21', 'nu12', 'nu03']
        for i in range(3):
            ans = single_channel_slide_window_parameters(self.matrix_a[i], self.matrix_b[i], cc.ordinary_moments, self.step, self.size_w, self.size_h)
            
            for j in range(ans.shape[0]):
                path = self.folder + 'OrdinaryMoments_'+RGB_COLOR_CHANNEL.get(i) +'_'+ordinary_moments_label[j]+'.jpg'
                ans_highsolution = cv2.resize(ans[j], None, fx=self.step, fy=self.step, interpolation=cv2.INTER_LINEAR)
                # ans_highsolution = ans_highsolution.astype(np.uint8)
                # ans_fakecolor = cv2.applyColorMap(ans_highsolution, cv2.COLORMAP_JET)
                # cv2.imwrite(path, ans_fakecolor)
                print(path)
                plt.figure()
                plt.imshow(ans_highsolution,vmin = 0, vmax = 255,cmap = "hot")
                plt.colorbar()
                plt.savefig(path)
                plt.close()
        return
    
    @TestScripts.timmer
    def __fakecolor_color_characteristics_color_coherence_vector(self):
        color_threshold = 8
        area_threshold = 30
        ccv_label = []
        for i in range(color_threshold):
            delta = int(256/color_threshold)
            buttom = delta * i
            top = delta*(i+1)-1
            cur_label = str(buttom) + "-" + str(top)
            ccv_label.append(cur_label)
        for i in range(3):
            ans = single_channel_slide_window_vectors(self.matrix_a[i], self.matrix_b[i], cc.color_coherence_vector, step = self.step, size_w=self.size_w, size_h=self.size_h, color_threshold=color_threshold, area_threshold=area_threshold)
            for j in range(ans.shape[0]):
                path = self.folder + 'ColorCoherenceVector_'+RGB_COLOR_CHANNEL.get(i) +'_'+ccv_label[j]+'.jpg'
                ans_highsolution = cv2.resize(ans[j], None, fx=self.step, fy=self.step, interpolation=cv2.INTER_LINEAR)
                # ans_highsolution = ans_highsolution.astype(np.uint8)
                # ans_fakecolor = cv2.applyColorMap(ans_highsolution, cv2.COLORMAP_JET)
                # cv2.imwrite(path, ans_fakecolor)
                print(path)
                plt.figure()
                plt.imshow(ans_highsolution,vmin = 0, vmax = 255,cmap = "hot")
                plt.colorbar()
                plt.savefig(path)
                plt.close()
        return
    
    
    def fakecolor_color_characteristics(self):
        self.__fakecolor_color_characteristics_histogram()
        self.__fakecolor_color_characteristics_color_moments()
        self.__fakecolor_color_characteristics_ordinary_moments()
        self.__fakecolor_color_characteristics_color_coherence_vector()
        return
    
    def __init__(self, matrix_a, matrix_b, step, size_w, size_h, folder):
        self.matrix_a = matrix_a
        self.matrix_b = matrix_b
        self.step = step
        self.size_w = size_w
        self.size_h = size_h   
        self.folder = folder     