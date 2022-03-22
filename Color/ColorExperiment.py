from Tools import TimeCount as tc
from matplotlib import pyplot as plt
from Color import ColorAlgorithrm as ca
import cv2
from Tools import SlideWindow as sw
import numpy as np
LAB_COLOR_CHANNEL = {
    0: 'l',
    1: 'a',
    2: 'b'
}
RGB_COLOR_CHANNEL = {
    0: 'b',
    1: 'g',
    2: 'r'
}
class Experiment_Color_Characteristics(object):
    def __experiment_color_brightness(self):
        label =  'Color_Brightness'
        ans = abs(ca.brightness(self.matrix_a)-ca.brightness(self.matrix_b))
        self.csv_generate(ans,label)


    def __experiment_color_constract(self):
        label = 'Color_Constract'
        ans = abs(ca.constract(self.matrix_a)-ca.constract(self.matrix_b))
        self.csv_generate(ans,label)

    def __experiment_color_exposure(self):
        label = 'Color_Exposure'
        ans = abs(ca.exposure(self.matrix_a,self.matrix_b))
        self.csv_generate(ans,label)

        
    def __experiment_color_saturation(self):
        label = 'Color_Saturation'
        ans = abs(ca.saturation(self.matrix_a)-ca.saturation(self.matrix_b))
        self.csv_generate(ans,label)

    
    def __experiment_color_white_balance(self):
        label = 'Color_WhiteBalance'
        ans = abs(ca.white_balance(self.matrix_a)-ca.white_balance(self.matrix_b))
        self.csv_generate(ans,label)

    # TODO: 这个是直接出图片的，应该做一下后期处理 
    # 加了一个count_nonzero，应该是计算了两张高光区域重合之后不为0的部分的像素数量
    def __experiment_color_specular_shadow(self):
        label = 'Color_SpecularShadow'
        
        mask_threshold = 0.33
        option = 'specular'
        ans1 = ca.specular_shadow(self.matrix_a, mask_threshold, option)
        ans2 = ca.specular_shadow(self.matrix_b, mask_threshold, option)
        ans = np.logical_xor(ans1,ans2).astype(int)
        ans = np.count_nonzero(ans)
       
        self.csv_generate(ans,label)


    # TODO: 灰度直方图还需要算吗？ 
    # 不算了，这个作为基本统计信息打算直接放在最后比较的结果里了
    # @tc.timmer
    def __experiment_color_characteristics_histogram(self):
        lab_img_a = cv2.merge(self.matrix_a)
        lab_img_b = cv2.merge(self.matrix_b)
        bgr_img_a = cv2.cvtColor(lab_img_a, cv2.COLOR_LAB2BGR)
        bgr_img_b = cv2.cvtColor(lab_img_b, cv2.COLOR_LAB2BGR)
       
        for i in range(3):
            ax1 = plt.subplot(3,1,i+1)
            hist_title = RGB_COLOR_CHANNEL.get(i)+' channel'
            ax1.set_title(hist_title)
            hist_a = ca.histogram(bgr_img_a[i])        
            plt.plot(hist_a,RGB_COLOR_CHANNEL.get(i))
            hist_b = ca.histogram(bgr_img_b[i])        
            plt.plot(hist_b,RGB_COLOR_CHANNEL.get(i),linestyle='dashed') 

        return
    
    # @tc.timmer
    def __experiment_color_color_moments(self):        
        color_moments_label = ['1st', '2nd', '3rd']
        for i in range(3):           
            ans = abs(ca.color_moments(self.matrix_a[i])-ca.color_moments(self.matrix_b[i]))
            for j in range(ans.shape[0]):                
                label =  'Color_ColorMoments_'+LAB_COLOR_CHANNEL.get(i) +'_'+color_moments_label[j]
                self.csv_generate(ans[j], label)


        return
    
    # @tc.timmer
    def __experiment_color_ordinary_moments(self):
        ordinary_moments_label = ['m00', 'm10', 'm01', 'm20', 'm11', 'm02', 'm30', 'm21', 'm12',
                                     'm02', 'mu20', 'mu11', 'mu02', 'mu30', 'mu21', 'mu12', 'mu03',
                                     'nu20', 'nu11', 'nu02', 'nu30', 'nu21', 'nu12', 'nu03']
        for i in range(3):
            
            ans = abs(np.array(ca.ordinary_moments(self.matrix_a[i]))-np.array(ca.ordinary_moments(self.matrix_b[i])))
            for j in range(ans.shape[0]):
                
                label = 'Color_OrdinaryMoments_'+LAB_COLOR_CHANNEL.get(i) +'_'+ordinary_moments_label[j]

                self.csv_generate(ans[j],label)
        return
    
    # @tc.timmer
    def __experiment_color_color_coherence_vector(self):
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
            vector_a = np.array(ca.color_coherence_vector(self.matrix_a[i], color_threshold, area_threshold))
            vector_b = np.array(ca.color_coherence_vector(self.matrix_b[i], color_threshold, area_threshold))

            for j in range(vector_a.shape[0]):
                
                label = 'Color_CoherenceVector_'+LAB_COLOR_CHANNEL.get(i) +'_'+ccv_label[j]
                ans = np.linalg.norm(vector_a[j]-vector_b[j])
                self.csv_generate(ans,label)


        return
    
    def csv_generate(self, ans, label):
        self.csv_label.append(label)
        self.csv_data.append(ans)
            
    
    def experiment_color_characteristics(self):
        self.__experiment_color_brightness()
        self.__experiment_color_constract()
        self.__experiment_color_exposure()
        self.__experiment_color_saturation()
        self.__experiment_color_white_balance()
        self.__experiment_color_specular_shadow()
        # self.__experiment_color_characteristics_histogram()
        self.__experiment_color_color_moments()
        self.__experiment_color_ordinary_moments()
        self.__experiment_color_color_coherence_vector()
        return
    
    def __init__(self, matrix_a, matrix_b):
        self.matrix_a = matrix_a
        self.matrix_b = matrix_b


        self.csv_data = []
        self.csv_label = []