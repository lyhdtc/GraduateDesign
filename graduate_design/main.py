import sys
sys.path.append('graduate_design/Characteristics')
import cv2
import numpy as np
from matplotlib import pyplot as plt
import ColorCharacteristics as cc
import TextureCharacteristics as tc
import LossAboutColor as lac
import DrawPic
import profile



def get_img(path):
    img = cv2.imread(path)
    img_r,img_g,img_b = cv2.split(img)
    return img,img_r,img_g,img_b
    


if __name__=="__main__":
    print('Start!')
    path_a = 'graduate_design/Data/ccc.jpg'
    path_b = 'graduate_design/Data/eee.jpg'
    img_a = cv2.imread(path_a)
    img_b = cv2.imread(path_b)    

    matrix_a =  cv2.split(img_a)
    matrix_b =  cv2.split(img_b)
    rgb_img_b, r_img_b, g_img_b, b_img_b = get_img(path_b)

    # drawpic_cc = DrawPic.Draw_Color_Characteristics(matrix_a, matrix_b)    
    # drawpic_cc.draw_color_characteristics()
    drawpic_tc = DrawPic.Draw_Texture_Characteristics(matrix_a, matrix_b)
    drawpic_tc.draw_texture_characteristics()
    # drawpic_lac = DrawPic.Draw_LossAboutColor_Characteristics(img_a, img_b)
    # drawpic_lac.draw_loss_about_color()
    # profile.run('drawpic_cc.draw_color_characteristics()')
    # plt.show()
    