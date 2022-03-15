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
class Experiment_Texture_Characteristics(object):
    
    # !窗口是20*20跑不了，40*40可以。。。
    # @tc.timmer
    def __experiment_texture_glcm_feature(self):
        glcm_feature_label = ['Energy', 'Entropy', 'Contrast', 'IDM']
        DIRECTION = {
            0: [1,0],
            1: [0,1],
            2: [1,1],
            3: [-1,1]
        }
        for i in range(3):
            for j in range(4):               
                ans = abs(np.array(ta.glcm_feature(self.matrix_a[i], d_x=DIRECTION.get(j)[0],d_y=DIRECTION.get(j)[1]))-np.array(ta.glcm_feature(self.matrix_b[i], d_x=DIRECTION.get(j)[0],d_y=DIRECTION.get(j)[1])))
                for k in range(ans.shape[0]):
                    label = 'Texture_GLCMFeature_'+LAB_COLOR_CHANNEL.get(i) +'_direction'+str(DIRECTION.get(j))+'_'+glcm_feature_label[k]
                    self.csv_generate(ans[k], label)

        return
    
   
    # TODO: 仍然是两张结果做差了。。。
    # @tc.timmer
    def __experiment_texture_lbp(self):

        for i in range(3):
            # print(i)

            lbp_a = ta.rotation_invariant_LBP(self.matrix_a[i])



            lbp_b = ta.rotation_invariant_LBP(self.matrix_b[i])
            lbp = np.abs(lbp_a-lbp_b)
            ans = np.count_nonzero(lbp)
            label = 'Texture_lbp_'+LAB_COLOR_CHANNEL.get(i)
            self.csv_generate(ans, label)


        return 

    
    # @tc.timmer
    def __experiment_texture_tamura_feature(self):
        kmax = 3    
        dist = 4
        tamura_label = ['coarseness', 'contrast', 'directionality', 'linelikeness', 'regularity', 'roughness']
        # tamura_label = ['contrast']
        for i in range(3):
           
            ans = abs(np.array(ta.tamura_feature(self.matrix_a[i], kmax, dist))-np.array(ta.tamura_feature(self.matrix_b[i], kmax, dist)))
            for j in range(ans.shape[0]):
                label = 'Texture_TamuraFeature_'+LAB_COLOR_CHANNEL.get(i) +'_'+tamura_label[j]    
                self.csv_generate(ans[j], label)

        return    
    
    # @tc.timmer
    def __experiment_texture_dwt_feature(self):
        wave_func = 'haar'
        dwt_label = ['average_ca', 'entropy_ca', 'sigma_ca', 'energy_ca',
                        'average_ch', 'entropy_ch', 'sigma_ch', 'energy_ch',
                        'average_cv', 'entropy_cv', 'sigma_cv', 'energy_cv',
                        'average_cd', 'entropy_cd', 'sigma_cd', 'energy_cd']
        for i in range(3):
            ans = abs(np.array(ta.dwt_feature(self.matrix_a[i],wave_func))-np.array(ta.dwt_feature(self.matrix_b[i], wave_func)))
            for j in range(ans.shape[0]):
                label = 'Texture_DWTFeature_'+LAB_COLOR_CHANNEL.get(i) +'_'+dwt_label[j]
                self.csv_generate(ans[j], label)

        return
    
    # @tc.timmer
    # TODO:这个也是直接出图的，所以也是就统计了不为0的像素的数量
    def __experiment_texture_laws_feature(self):
        laws_label = ['0', '1', '2', '3', '4', '5', '6', '7']
        for i in range(3):
            
            laws_feature_single_feature_a = ta.laws_feature(self.matrix_a[i])
            laws_feature_single_feature_b = ta.laws_feature(self.matrix_b[i])
            laws_feature_single_feature_a = np.array(laws_feature_single_feature_a)
            laws_feature_single_feature_b = np.array(laws_feature_single_feature_b)
            laws_feature_single_feature = np.absolute(laws_feature_single_feature_a-laws_feature_single_feature_b)
            for j in range(8):
                label = 'Texture_Laws_'+LAB_COLOR_CHANNEL.get(i)+'_'+laws_label[j]

                self.csv_generate(np.count_nonzero(laws_feature_single_feature[j]), label)


    def __experiment_texture_gabor(self):
        label = 'Texture_Gabor'
        ans_a = ta.gabor_process(self.matrix_a)
        ans_b = ta.gabor_process(self.matrix_b)
        ans_a = np.array(ans_a)
        ans_b = np.array(ans_b)
        ans = np.abs(ans_a-ans_b) 
        self.csv_generate(np.count_nonzero(ans), label)
    
    def csv_generate(self, ans, label):
        self.csv_label.append(label)
        self.csv_data.append(ans)
    
    
     
    def experiment_texture_characteristics(self):
        
        self.__experiment_texture_glcm_feature()
        
        self.__experiment_texture_lbp()
        
        self.__experiment_texture_tamura_feature()
        self.__experiment_texture_dwt_feature()
        self.__experiment_texture_laws_feature()
        
        self.__experiment_texture_gabor()
        return
    
    def __init__(self, matrix_a, matrix_b):
        self.matrix_a = matrix_a
        self.matrix_b = matrix_b
 

    
        self.csv_data = []
        self.csv_label = []