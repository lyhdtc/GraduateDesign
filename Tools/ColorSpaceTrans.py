import cv2
import numpy as np


'''
《Most apparent distortion: full-reference image quality assessment and the role of strategy》
https://s2.smu.edu/~eclarson/pubs/2010JEI_MAD.pdf
主要是对亮度进行了处理，首先通过公式 L = (b+kI)^gamma, (对于8bit rgb图像, b=0, k=0.02874, gamma=2.2)转换为亮度图像，后根据视觉特性对亮度进行立方根处理
三通道转换成一通道猜测直接rgb2gray实现的
'''
def bgr2MAD_lightness(bgr_img):
    gray_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)
    k = 0.02874
    gamma = 2.2
    lightness = (k * gray_img)**gamma
    lightness = lightness ** (1/3)
    return lightness