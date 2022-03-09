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
class Experiment_Texture_Characteristecs(object):
    
    # !窗口是20*20跑不了，40*40可以。。。
    @tc.timmer
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
    
   
    
    @tc.timmer
    def __experiment_texture_lbp(self):
        print("LBP的结构信息不适用于滑动窗口的局部计算,将计算整体信息")
        plt.figure(figsize=self.figsize)
        plt.title('lbp')
        path = self.folder + 'Texture_LBP.jpg'
        for i in range(3):
            # print(i)
            ax1 = plt.subplot(3,4,4*i+1)
            lbp_a = ta.rotation_invariant_LBP(self.matrix_a[i])
            ax1.imshow(lbp_a,cmap='gray')
            ax2 = plt.subplot(3,4,4*i+2)
            lbp_a = lbp_a.astype(np.uint8)
            hist_lbp_a = ca.histogram(lbp_a)
            ax2.plot(hist_lbp_a,LAB_COLOR_CHANNEL.get(i))
            ax3 = plt.subplot(3,4,4*i+3)
            lbp_b = ta.rotation_invariant_LBP(self.matrix_b[i])
            ax3.imshow(lbp_b,cmap='gray')
            ax4 = plt.subplot(3,4,4*i+4)
            lbp_b = lbp_b.astype(np.uint8)
            hist_lbp_b = ca.histogram(lbp_b)
            ax4.plot(hist_lbp_b,LAB_COLOR_CHANNEL.get(i))
        plt.tight_layout()
        plt.plot()
        plt.savefig(path)
        plt.close()
        return 

    
    @tc.timmer
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
    
    @tc.timmer
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
    
    @tc.timmer
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
                path = self.folder + label+'.jpg'
                self.csv_generate(laws_feature_single_feature[j], label)
                
                ans_highsolution = laws_feature_single_feature[j].astype(np.uint8)
                ans_experiment = cv2.applyColorMap(ans_highsolution, cv2.COLORMAP_HOT)
                cv2.imwrite(path, ans_highsolution)
                print(path)
                # plt.figure(figsize=self.figsize)
                # plt.imshow(laws_feature_single_feature[j],vmin = 0, vmax = 255,cmap = "hot")
                # plt.colorbar()
                # plt.savefig(path)
                # plt.close()
    
    def csv_generate(self, ans, label):
        self.csv_label.append(label)
        self.csv_data.append(ans)
    
    
     
    def experiment_texture_characteristics(self):
        
        self.__experiment_texture_glcm_feature()
        
        # self.__experiment_texture_characteristics_lbp()
        
        self.__experiment_texture_tamura_feature()
        self.__experiment_texture_dwt_feature()
        # self.__experiment_texture_laws_feature()
        return
    
    def __init__(self, matrix_a, matrix_b, folder):
        self.matrix_a = matrix_a
        self.matrix_b = matrix_b
 
        self.folder = folder 
    
        self.csv_data = []
        self.csv_label = []