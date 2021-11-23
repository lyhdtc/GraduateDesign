from collections import _OrderedDictItemsView
from operator import sub
import sys
from typing import Tuple

import cv2
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import ColorCharacteristics as cc
import TextureCharacteristics as tc
import LossAboutColor as lac

import time

RGB_COLOR_CHANNEL = {
    0: 'r',
    1: 'g',
    2: 'b'
}

def timmer(func):
    print('run timmer')
    def deco(*args, **kwargs):
        print('\n函数： {_funcname_} 开始运行：'.format(_funcname_ = func.__name__))
        start_time = time.time()
        res = func(*args, **kwargs)
        end_time = time.time()
        print('函数:{_funcname_}运行了 {_time_}秒'
              .format(_funcname_=func.__name__, _time_=(end_time - start_time)))
        return res
    return deco

class Draw_Color_Characteristics(object):
    
    #直方图
    @timmer
    def __draw_color_characteristics_histogram(self):
        plt.figure()
        plt.title('histogram')
        for i in range(3):
            ax1 = plt.subplot(3,1,i+1)
            hist_title = 'histogram ———— '+RGB_COLOR_CHANNEL.get(i)+' channel'
            ax1.set_title(hist_title)
            hist_a = cc.histogram(self.matrix_a[i])        
            plt.plot(hist_a,RGB_COLOR_CHANNEL.get(i))
            hist_b = cc.histogram(self.matrix_b[i])        
            plt.plot(hist_b,RGB_COLOR_CHANNEL.get(i),linestyle='dashed') 
        plt.tight_layout()
        plt.plot()
    
    # 颜色矩   
    @timmer
    def __draw_color_characteristics_color_moments(self):
        plt.figure()  
        plt.title('color_moments')   
        color_moments = []    
        color_moments_collabel = []
        color_moments_rowlabel = ['1st', '2nd', '3rd']
        for i in range(3):        
            color_moments_a = cc.color_moments(self.matrix_a[i])
            color_moments_b = cc.color_moments(self.matrix_b[i])        
            color_moments.append(color_moments_a)
            color_moments.append(color_moments_b)
            label_a = RGB_COLOR_CHANNEL.get(i)+'_img_a'
            label_b = RGB_COLOR_CHANNEL.get(i)+'_img_b'
            color_moments_collabel.append(label_a)
            color_moments_collabel.append(label_b)         
        color_moments = np.transpose(color_moments)
        color_moments = np.round(color_moments,3)
        # print(color_moments)

        color_moments_table = plt.table(color_moments, colLabels=color_moments_collabel, rowLabels=color_moments_rowlabel, loc='center',  cellLoc='center', rowLoc='center')
        color_moments_table.auto_set_font_size(False)
        color_moments_table.set_fontsize(8)
        plt.axis('off')
        # color_moments_table.scale(0.5,0.5) 
    
    
    @timmer  
    def __draw_color_characteristics_color_coherence_vector(self):
        plt.figure(figsize=(15,8))
        plt.title('color coherence vector')
        color_threshold=8
        area_threshold=100
        ccv = []
        ccv_collabel = []
        ccv_rowlabel = []
        for i in range(color_threshold):
            delta = int(256/color_threshold)
            buttom = delta * i
            top = delta*(i+1)-1
            cur_row_label = str(buttom) + "~" + str(top)
            ccv_rowlabel.append(cur_row_label)
        for i in range(3):
            ccv_a_smaller, ccv_a_bigger = cc.color_coherence_vector(self.matrix_a[i], color_threshold, area_threshold)
            ccv_b_smaller, ccv_b_bigger = cc.color_coherence_vector(self.matrix_b[i], color_threshold, area_threshold)
            ccv.append(ccv_a_smaller)
            ccv.append(ccv_a_bigger)
            ccv.append(ccv_b_smaller)
            ccv.append(ccv_b_bigger)
            label_a_smaller = RGB_COLOR_CHANNEL.get(i)+'_img_a_smaller'
            label_a_bigger = RGB_COLOR_CHANNEL.get(i)+'_img_a_bigger'            
            label_b_smaller = RGB_COLOR_CHANNEL.get(i)+'_img_b_smaller'
            label_b_bigger = RGB_COLOR_CHANNEL.get(i)+'_img_b_bigger'
            ccv_collabel.append(label_a_smaller)
            ccv_collabel.append(label_a_bigger)
            ccv_collabel.append(label_b_smaller)
            ccv_collabel.append(label_b_bigger)
        ccv = np.transpose(ccv)
        
        ccv_table = plt.table(ccv, colLabels=ccv_collabel, rowLabels=ccv_rowlabel, loc='center',  cellLoc='center', rowLoc='center')
        ccv_table.auto_set_font_size(False)
        ccv_table.set_fontsize(8)
        plt.axis('off')   
    # 普通矩
    @timmer
    def __draw_color_characteristics_ordinary_moments(self):
        plt.figure(figsize=(15,8))
        plt.title('ordinary_moments', verticalalignment = 'top')
        ordinary_moments = []
        ordinary_moments_collabel = []
        ordinary_moments_rowlabel = ['m00', 'm10', 'm01', 'm20', 'm11', 'm02', 'm30', 'm21', 'm12',
                                     'm02', 'mu20', 'mu11', 'mu02', 'mu30', 'mu21', 'mu12', 'mu03',
                                     'nu20', 'nu11', 'nu02', 'nu30', 'nu21', 'nu12', 'nu03']
        for i in range(3):
            ordinary_moments_a = cc.ordinary_moments(self.matrix_a[i])
            ordinary_moments_b = cc.ordinary_moments(self.matrix_b[i])
            
            ordinary_moments.append(ordinary_moments_a)
            ordinary_moments.append(ordinary_moments_b)        

            label_a = RGB_COLOR_CHANNEL.get(i)+'_img_a'
            label_b = RGB_COLOR_CHANNEL.get(i)+'_img_b'
            ordinary_moments_collabel.append(label_a)
            ordinary_moments_collabel.append(label_b) 
            
        ordinary_moments = np.transpose(ordinary_moments)
        # ordinary_moments = np.round(ordinary_moments,3)
        # print(ordinary_moments)

        ordinary_moments_table = plt.table(ordinary_moments,colLabels=ordinary_moments_collabel,rowLabels=ordinary_moments_rowlabel,loc='center',  cellLoc='center', rowLoc='center')
        ordinary_moments_table.auto_set_font_size(False)
        ordinary_moments_table.set_fontsize(8)
        plt.axis('off')
        
    
    def draw_color_characteristics(self):
        self.__draw_color_characteristics_histogram()
        self.__draw_color_characteristics_color_moments()
        self.__draw_color_characteristics_color_coherence_vector()
        self.__draw_color_characteristics_ordinary_moments()
        plt.tight_layout()
        plt.show()

    def __init__(self, matrix_a, matrix_b):
        self.matrix_a = matrix_a
        self.matrix_b = matrix_b
    
    
class Draw_Texture_Characteristics(object):
    @timmer
    def __draw_texture_characteristics_glcm_feature(self):
        plt.figure()
        plt.title('glcm_feature')
        DIRECTION = {
            0: [1,0],
            1: [0,1],
            2: [1,1],
            3: [-1,1]
        }
        glcm_feature = []
        glcm_feature_rowlabel = ['Energy', 'Entropy', 'Contrast', 'IDM']
        glcm_feature_collabel = []
        for i in range(3):
            for j in range(4):
                # print('i=='+str(i))
                # print('j=='+str(j))
                glcm_grad_single_channel_a = tc.glcm_feature(self.matrix_a[i], DIRECTION.get(j)[0], DIRECTION.get(j)[1])
                glcm_grad_single_channel_b = tc.glcm_feature(self.matrix_b[i], DIRECTION.get(j)[0], DIRECTION.get(j)[1])
                glcm_feature.append(glcm_grad_single_channel_a)
                glcm_feature.append(glcm_grad_single_channel_b)  
                label_a = RGB_COLOR_CHANNEL.get(i) + '_img_a '+ str(DIRECTION.get(j)[0])+str(DIRECTION.get(j)[1])
                label_b = RGB_COLOR_CHANNEL.get(i) + '_img_b '+ str(DIRECTION.get(j)[0])+str(DIRECTION.get(j)[1])
                glcm_feature_collabel.append(label_a)
                glcm_feature_collabel.append(label_b)
        glcm_feature = np.transpose(glcm_feature)
        glcm_feature = np.round(glcm_feature, 3)
        
        glcm_feaure_table = plt.table(glcm_feature, colLabels=glcm_feature_collabel, rowLabels=glcm_feature_rowlabel, loc='center',  cellLoc='center', rowLoc='center')
        glcm_feaure_table.auto_set_font_size(False)
        glcm_feaure_table.set_fontsize(8)
        plt.axis('off')
    
    @timmer
    def __draw_texture_characteristics_lbp(self):
        plt.figure(figsize=(10,5))
        plt.title('lbp')
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
    
    @timmer
    def __draw_texture_characteristics_tamura_feature(self):
        plt.figure()
        plt.title('tamura')
        tamura_feature = []
        tamura_rowlabel = ['coarseness', 'contrast', 'directionality', 'linelikeness']
        tamura_collabel = []
        kmax = 3    
        dist = 4
        for i in range(3):
            tamura_feature_single_feature_a = tc.tamura_feature(self.matrix_a[i], kmax, dist)
            tamura_feature_single_feature_b = tc.tamura_feature(self.matrix_b[i], kmax, dist)
            tamura_feature.append(tamura_feature_single_feature_a)
            tamura_feature.append(tamura_feature_single_feature_b)
            labela = RGB_COLOR_CHANNEL.get(i) + '_img_a'
            labelb = RGB_COLOR_CHANNEL.get(i) + '_img_b'
            tamura_collabel.append(labela)
            tamura_collabel.append(labelb)
        tamura_feature = np.transpose(tamura_feature)
        tamura_feature = np.round(tamura_feature,3)
        
        tamura_feature_table = plt.table(tamura_feature, colLabels= tamura_collabel, rowLabels=tamura_rowlabel, loc='center',  cellLoc='center', rowLoc='center')
        tamura_feature_table.auto_set_font_size(False)
        tamura_feature_table.set_fontsize(8)
        plt.axis('off')
     
    @timmer
    def __draw_texture_characteristics_dwt_feature(self):
        plt.figure()
        plt.title('dwt')
        dwt_feature = []
        dwt_rowlabel = ['average_ca', 'entropy_ca', 'sigma_ca', 'energy_ca',
                        'average_ch', 'entropy_ch', 'sigma_ch', 'energy_ch',
                        'average_cv', 'entropy_cv', 'sigma_cv', 'energy_cv',
                        'average_cd', 'entropy_cd', 'sigma_cd', 'energy_cd']
        dwt_collabel = []
        wave_func = 'haar'
        for i in range(3):
            dwt_feature_single_feature_a = tc.dwt_feature(self.matrix_a[i], wave_func)
            dwt_feature_single_feature_b = tc.dwt_feature(self.matrix_b[i], wave_func)
            dwt_feature.append(dwt_feature_single_feature_a)
            dwt_feature.append(dwt_feature_single_feature_b)
            labela = RGB_COLOR_CHANNEL.get(i) + '_img_a'
            labelb = RGB_COLOR_CHANNEL.get(i) + '_img_b'
            dwt_collabel.append(labela)
            dwt_collabel.append(labelb)
        dwt_feature = np.transpose(dwt_feature)
        dwt_feature = np.round(dwt_feature,3)
        
        dwt_feature_table = plt.table(dwt_feature, colLabels= dwt_collabel, rowLabels=dwt_rowlabel, loc='center',  cellLoc='center', rowLoc='center')
        dwt_feature_table.auto_set_font_size(False)
        dwt_feature_table.set_fontsize(8)
        plt.axis('off')
    @timmer   
    def __draw_texture_characteristics_laws_feature(self):
        plt.figure()
        plt.title('laws')
        
        laws_feature = []
        laws_rowlabel = ['0', '1', '2', '3', '4', '5', '6', '7']
        laws_collabel = []
        
        for i in range(3):
            laws_feature_single_feature_a = tc.laws_feature(self.matrix_a[i])
            laws_feature_single_feature_b = tc.laws_feature(self.matrix_b[i])
            laws_feature.append(laws_feature_single_feature_a)
            laws_feature.append(laws_feature_single_feature_b)
            labela = RGB_COLOR_CHANNEL.get(i) + '_img_a'
            labelb = RGB_COLOR_CHANNEL.get(i) + '_img_b'
            laws_collabel.append(labela)
            laws_collabel.append(labelb)
        for i in range(3):
            plt.subplot(9,6,2*i+1)
            origin_a_label = RGB_COLOR_CHANNEL.get(i) + '_img_a_original'
            plt.imshow(self.matrix_a[i], 'gray')
            plt.suptitle(origin_a_label)
            plt.subplot(9,6,2*i+2)
            origin_b_label = RGB_COLOR_CHANNEL.get(i) + '_img_b_original'
            plt.imshow(self.matrix_b[i], 'gray')
            plt.suptitle(origin_b_label)
        for i in range(6):
            for j in range(8):
                plt.subplot(9,6,6+6*j+i+1)
                plt.imshow(laws_feature[i][j], 'gray')
                # plt.suptitle(sub_label)  
                
        
        
        
    def draw_texture_characteristics(self):
        self.__draw_texture_characteristics_glcm_feature()
        # print('glcm finished!')
        self.__draw_texture_characteristics_lbp()
        # print('lbp finished')
        self.__draw_texture_characteristics_tamura_feature()
        # print('tamura feature finished')
        self.__draw_texture_characteristics_dwt_feature()
        # print('dwf feature finished')
        self.__draw_texture_characteristics_laws_feature()
        # print('laws feature finished')
        plt.show()
        
    def __init__(self, matrix_a, matrix_b):
        self.matrix_a = matrix_a
        self.matrix_b = matrix_b

     
    
     
    
class Draw_LossAboutColor_Characteristics(object):
    @timmer
    def __draw_loss_DSLRQualityPhotos_ICCV2017(self):
        plt.figure()
        plt.title('loss--DSLR-Quality Photos on Mobile \nDevices with Deep Convolutional Networks(ICCV 2017)')        
        loss = [lac.loss_DSLRQualityPhotos_ICCV2017(self.img_a, self.img_b)]           
        loss = np.round(loss,3)
        collabel = ['color loss', 'texture loss', 'total veriation loss']        
        loss_table = plt.table(loss, colLabels=collabel, loc='center',  cellLoc='center', rowLoc='center' )
        loss_table.auto_set_font_size(False)
        loss_table.set_fontsize(8)
        plt.axis('off')
        
    @timmer
    def __draw_loss_UnderexposedPhoto_CVPR2019(self):
        plt.figure()
        plt.title('loss--Underexposed Photo \n Enhancement using Deep Illumination Estimation(CVPR 2019)')
        loss = [lac.loss_UnderexposedPhoto_CVPR2019(self.img_a, self.img_b)]
        loss = np.round(loss,3)
        collabel = ['l2 loss', 'color loss']
        loss_table = plt.table(loss, colLabels=collabel, loc='center',  cellLoc='center', rowLoc='center' )
        loss_table.auto_set_font_size(False)
        loss_table.set_fontsize(8)
        plt.axis('off')
    
    @timmer
    def __draw_loss_RangeScalingGlobalUNet_ECCV2018(self):
        plt.figure()
        plt.title('loss--Range Scaling Global U-Net for Perceptual \nImage Enhancement on Mobile Devices(ECCV-PIRM2018)')        
        loss = [lac.loss_RangScalingGlobalUNet_ECCV2018(self.img_a, self.img_b)]           
        loss = np.round(loss,3)
        collabel = ['l1 loss', 'MS-SSIM loss--r channel','MS-SSIM loss--g channel','MS-SSIM loss--b channel', 'total veriation loss']        
        loss_table = plt.table(loss, colLabels=collabel, loc='center',  cellLoc='center', rowLoc='center' )
        loss_table.auto_set_font_size(False)
        loss_table.set_fontsize(8)
        plt.axis('off')
        
    @timmer    
    def __draw_loss_LossFunctions_IEEE2017(self):
        plt.figure()
        plt.title('loss--Loss Functions for \n Image Restoration with Neural Networks(IEEE2017)')        
        loss = [lac.loss_LossFunctions_IEEE2017(self.img_a, self.img_b)]           
        loss = np.round(loss,3)
        collabel = ['MS-SSIM+L1 loss -- r channel', 'MS-SSIM+L1 loss -- g channel', 'MS-SSIM+L1 loss -- b channel']        
        loss_table = plt.table(loss, colLabels=collabel, loc='center',  cellLoc='center', rowLoc='center' )
        loss_table.auto_set_font_size(False)
        loss_table.set_fontsize(8)
        plt.axis('off')
        
    
    def draw_loss_about_color(self):
        self.__draw_loss_DSLRQualityPhotos_ICCV2017()
        # print('loss--DSLR-Quality Photos on Mobile Devices with Deep Convolutional Networks(ICCV 2017) -- finished')
        self.__draw_loss_UnderexposedPhoto_CVPR2019()
        # print('loss--Underexposed Photo Enhancement using Deep Illumination Estimation(CVPR 2019) -- finished')
        self.__draw_loss_RangeScalingGlobalUNet_ECCV2018()
        # print('loss--Range Scaling Global U-Net for Perceptual Image Enhancement on Mobile Devices(ECCV-PIRM2018) -- finished')
        self.__draw_loss_LossFunctions_IEEE2017()
        # print('loss--Loss Functions for Image Restoration with Neural Networks(IEEE2017) -- finished')
        plt.show()
    def __init__(self, img_a, img_b):
        self.img_a = img_a
        self.img_b = img_b