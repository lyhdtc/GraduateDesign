import cv2
import numpy as np



#直方图 输入某一通道的图片，直接返回灰度矩阵
def histogram(pic):    
    hist = cv2.calcHist([pic],[0],None, [256],[0,256])
    return hist

