from Tools import TimeCount as tc
from matplotlib import pyplot as plt
import ColorAlgorithrm as ca
import cv2
from Tools import SlideWindow as sw
import numpy as np
RGB_COLOR_CHANNEL = {
    0: 'b',
    1: 'g',
    2: 'r'
}

class FakeColor_Color_Characteristics(object): 
    def __fakecolor_color_brightness(self):
        label =  'Color_Brightness'
        path = self.folder +'Color_Brightness.jpg'
        plt.figure(figsize=self.figsize)
        plt.title('brightness')
        ans = sw.rgb_channel_parameters_1imgfunc(self.matrix_a,self.matrix_b, ca.brightness, self.step, self.size_w, self.size_h)
        ans_highsolution = cv2.resize(ans, None, fx=self.step, fy=self.step, interpolation=cv2.INTER_LINEAR)
        self.csv_generate(ans,label)
        print(path)
        plt.figure(figsize=self.figsize)
        plt.imshow(ans_highsolution,vmin = 0, vmax = 255,cmap = "hot")
        plt.colorbar()
        plt.savefig(path)
        plt.close()
    
    def __fakecolor_color_constract(self):
        label = 'Color_Constract'
        path = self.folder +'Color_Constract.jpg'
        plt.figure(figsize=self.figsize)
        plt.title('constract')
        ans = sw.rgb_channel_parameters_1imgfunc(self.matrix_a,self.matrix_b, ca.constract, self.step, self.size_w, self.size_h)
        ans_highsolution = cv2.resize(ans, None, fx=self.step, fy=self.step, interpolation=cv2.INTER_LINEAR)
        self.csv_generate(ans,label)
        print(path)
        plt.figure(figsize=self.figsize)
        plt.imshow(ans_highsolution,vmin = 0, vmax = 255,cmap = "hot")
        plt.colorbar()
        plt.savefig(path)
        plt.close()
    
    def __fakecolor_color_exposure(self):
        label = 'Color_Exposure'
        path = self.folder +'Color_Exposure.jpg'
        plt.figure(figsize=self.figsize)
        plt.title('exposure')
        ans = sw.rgb_channel_parameters_2imgfunc(self.matrix_a,self.matrix_b, ca.exposure, self.step, self.size_w, self.size_h)
        ans_highsolution = cv2.resize(ans, None, fx=self.step, fy=self.step, interpolation=cv2.INTER_LINEAR)
        self.csv_generate(ans,label)
        print(path)
        plt.figure(figsize=self.figsize)
        plt.imshow(ans_highsolution,vmin = 0, vmax = 255,cmap = "hot")
        plt.colorbar()
        plt.savefig(path)
        plt.close()   
        
    def __fakecolor_color_saturation(self):
        label = 'Color_Saturation'
        path = self.folder +'Color_Saturation.jpg'
        plt.figure(figsize=self.figsize)
        plt.title('saturation')
        ans = sw.rgb_channel_parameters_1imgfunc(self.matrix_a,self.matrix_b, ca.saturation, self.step, self.size_w, self.size_h)
        ans_highsolution = cv2.resize(ans, None, fx=self.step, fy=self.step, interpolation=cv2.INTER_LINEAR)
        self.csv_generate(ans, label)
        print(path)
        plt.figure(figsize=self.figsize)
        plt.imshow(ans_highsolution,vmin = 0, vmax = 255,cmap = "hot")
        plt.colorbar()
        plt.savefig(path)
        plt.close() 
    
    def __fakecolor_color_white_balance(self):
        label = 'Color_WhiteBalance'
        path = self.folder +'Color_WhiteBalance.jpg'
        plt.figure(figsize=self.figsize)
        plt.title('white balance')
        ans = sw.rgb_channel_parameters_1imgfunc(self.matrix_a,self.matrix_b, ca.white_balance, self.step, self.size_w, self.size_h)
        ans_highsolution = cv2.resize(ans, None, fx=self.step, fy=self.step, interpolation=cv2.INTER_LINEAR)
        self.csv_generate(ans, label)
        print(path)
        plt.figure(figsize=self.figsize)
        plt.imshow(ans_highsolution,vmin = 0, vmax = 255,cmap = "hot")
        plt.colorbar()
        plt.savefig(path)
        plt.close()
        
    def __fakecolor_color_specular_shadow(self):
        label = 'Color_SpecularShadow'
        path = self.folder +'Color_SpecularShadow.jpg'
        plt.figure(figsize=self.figsize)
        plt.title('specular shadow')
        mask_threshold = 0.33
        option = 'specular'
        ans1 = ca.specular_shadow(self.matrix_a, mask_threshold, option)
        ans2 = ca.specular_shadow(self.matrix_b, mask_threshold, option)
        ans = np.logical_xor(ans1,ans2)
        ans_highsolution = cv2.resize(ans, None, fx=self.step, fy=self.step, interpolation=cv2.INTER_LINEAR)
        self.csv_generate(ans,label)
        print(path)
        plt.figure(figsize=self.figsize)
        plt.imshow(ans_highsolution,vmin = 0, vmax = 255,cmap = "hot")
        plt.colorbar()
        plt.savefig(path)
        plt.close()
        
    @tc.timmer
    def __fakecolor_color_characteristics_histogram(self):
        plt.figure(figsize=self.figsize)
        plt.title('histogram')
        path = self.folder + 'Color_Histogram.jpg'
        for i in range(3):
            ax1 = plt.subplot(3,1,i+1)
            hist_title = RGB_COLOR_CHANNEL.get(i)+' channel'
            ax1.set_title(hist_title)
            hist_a = ca.histogram(self.matrix_a[i])        
            plt.plot(hist_a,RGB_COLOR_CHANNEL.get(i))
            hist_b = ca.histogram(self.matrix_b[i])        
            plt.plot(hist_b,RGB_COLOR_CHANNEL.get(i),linestyle='dashed') 
        plt.tight_layout()
        plt.plot()
        plt.savefig(path)
        plt.close()
        return
    
    @tc.timmer
    def __fakecolor_color_color_moments(self):        
        color_moments_label = ['1st', '2nd', '3rd']
        for i in range(3):
            ans = sw.single_channel_parameters(self.matrix_a[i], self.matrix_b[i], ca.color_moments, self.step, self.size_w, self.size_h)
            
            for j in range(ans.shape[0]):
                
                label =  'Color_ColorMoments_'+RGB_COLOR_CHANNEL.get(i) +'_'+color_moments_label[j]
                path = self.folder + label+'.jpg'
                self.csv_generate(ans[j], label)
                ans_highsolution = cv2.resize(ans[j], None, fx=self.step, fy=self.step, interpolation=cv2.INTER_LINEAR)
                # ans_fakecolor = cv2.applyColorMap(ans_highsolution, cv2.COLORMAP_JET)
                # cv2.imwrite(path, ans_fakecolor)
                print(path)
                plt.figure(figsize=self.figsize)
                plt.imshow(ans_highsolution,vmin = 0, vmax = 255,cmap = "hot")
                plt.colorbar()
                plt.savefig(path)
                plt.close()
        return
    
    @tc.timmer
    def __fakecolor_color_ordinary_moments(self):
        ordinary_moments_label = ['m00', 'm10', 'm01', 'm20', 'm11', 'm02', 'm30', 'm21', 'm12',
                                     'm02', 'mu20', 'mu11', 'mu02', 'mu30', 'mu21', 'mu12', 'mu03',
                                     'nu20', 'nu11', 'nu02', 'nu30', 'nu21', 'nu12', 'nu03']
        for i in range(3):
            ans = sw.single_channel_parameters(self.matrix_a[i], self.matrix_b[i], ca.ordinary_moments, self.step, self.size_w, self.size_h)
            
            for j in range(ans.shape[0]):
                
                label = 'Color_OrdinaryMoments_'+RGB_COLOR_CHANNEL.get(i) +'_'+ordinary_moments_label[j]
                path = self.folder + label +'.jpg'
                self.csv_generate(ans[j],label)
                ans_highsolution = cv2.resize(ans[j], None, fx=self.step, fy=self.step, interpolation=cv2.INTER_LINEAR)
                # ans_highsolution = ans_highsolution.astype(np.uint8)
                # ans_fakecolor = cv2.applyColorMap(ans_highsolution, cv2.COLORMAP_JET)
                # cv2.imwrite(path, ans_fakecolor)
                print(path)
                plt.figure(figsize=self.figsize)
                plt.imshow(ans_highsolution,vmin = 0, vmax = 255,cmap = "hot")
                plt.colorbar()
                plt.savefig(path)
                plt.close()
        return
    
    @tc.timmer
    def __fakecolor_color_color_coherence_vector(self):
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
            ans = sw.single_channel_vectors(self.matrix_a[i], self.matrix_b[i], ca.color_coherence_vector, step = self.step, size_w=self.size_w, size_h=self.size_h, color_threshold=color_threshold, area_threshold=area_threshold)
            
            for j in range(ans.shape[0]):
                
                label = 'Color_CoherenceVector_'+RGB_COLOR_CHANNEL.get(i) +'_'+ccv_label[j]
                path = self.folder + label +'.jpg'
                self.csv_generate(ans[j],label)
                ans_highsolution = cv2.resize(ans[j], None, fx=self.step, fy=self.step, interpolation=cv2.INTER_LINEAR)
                # ans_highsolution = ans_highsolution.astype(np.uint8)
                # ans_fakecolor = cv2.applyColorMap(ans_highsolution, cv2.COLORMAP_JET)
                # cv2.imwrite(path, ans_fakecolor)
                print(path)
                plt.figure(figsize=self.figsize)
                plt.imshow(ans_highsolution,vmin = 0, vmax = 255,cmap = "hot")
                plt.colorbar()
                plt.savefig(path)
                plt.close()
        return
    
    def csv_generate(self, ans, label):
        self.csv_label.append(label)
        cal = np.sum(ans>=10)
        total = np.size(ans)
        # print(cal, total, cal/total)
        if(cal/total>0.1):
            self.csv_data.append(1)
        else:
            self.csv_data.append(0)
            
    
    def fakecolor_color_characteristics(self):
        self.__fakecolor_color_characteristics_histogram()
        self.__fakecolor_color_color_moments()
        self.__fakecolor_color_ordinary_moments()
        self.__fakecolor_color_color_coherence_vector()
        return
    
    def __init__(self, matrix_a, matrix_b, step, size_w, size_h, folder, figsize):
        self.matrix_a = matrix_a
        self.matrix_b = matrix_b
        self.step = step
        self.size_w = size_w
        self.size_h = size_h   
        self.folder = folder 
        self.figsize = figsize
        self.csv_data = []
        self.csv_label = []