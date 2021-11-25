import sys
sys.path.append('graduate_design/Characteristics')
import cv2
import numpy as np
from matplotlib import pyplot as plt
import ColorCharacteristics as cc
import TextureCharacteristics as tc
import LossAboutColor as lac
import TotalCharacteristics
import profile



def get_img(path):
    img = cv2.imread(path)
    img_r,img_g,img_b = cv2.split(img)
    return img,img_r,img_g,img_b
    


if __name__=="__main__":
    print('Start!')
    path_a = 'graduate_design/Data/cornelbox_measured1.jpg'
    path_b = 'graduate_design/Data/cornelbox_measured1.jpg'
    img_a = cv2.imread(path_a)
    img_b = cv2.imread(path_b)    

    matrix_a =  cv2.split(img_a)
    matrix_b =  cv2.split(img_b)
    rgb_img_b, r_img_b, g_img_b, b_img_b = get_img(path_b)

    drawpic_cc = TotalCharacteristics.Draw_Color_Characteristics(matrix_a, matrix_b)    
    drawpic_cc.draw_color_characteristics()
    drawpic_tc = TotalCharacteristics.Draw_Texture_Characteristics(matrix_a, matrix_b)
    drawpic_tc.draw_texture_characteristics()
    # tc.test_linelikeness(r_img_b, r_img_b, 4)
    drawpic_lac = TotalCharacteristics.Draw_LossAboutColor_Characteristics(img_a, img_b)
    drawpic_lac.draw_loss_about_color()
    # profile.run('drawpic_cc.draw_color_characteristics()')

    plt.show()
    
    # dir = np.float32([[1,0,400],[0,1,400]])
    # print(type(dir))
    # print(dir)
    # ajs = cv2.warpAffine(img_a, dir, (1920,1080))
    # cv2.imshow("adsfa",ajs)
    # cv2.waitKey(0)