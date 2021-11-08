import sys
sys.path.append('/mnt/d/GraduateDesign/graduate_design/Characteristics')
import cv2
import numpy as np
from matplotlib import pyplot as plt
import ColorCharacteristics as cc




RGB_COLOR_CHANNEL = {
    'r': 0,
    'g': 1,
    'b': 2
}

def get_pic(path):
    pic = cv2.imread(path)
    # cv2.imshow("asdf",pic)
    # key = cv2.waitKey(0)
    pic_r,pic_g,pic_b = cv2.split(pic)
    # cv2.imshow("adf",pic_r)
    # key = cv2.waitKey(0)
    return pic_r,pic_g,pic_b

def drawpicture(pic,color):    
    plt.plot(pic,color),    
    # plt.xlim([0,256]),
    # plt.title('test'),
    # plt.show()
    # ax.plot(pic,color,lab)
    
if __name__=="__main__":
    # get_pic("/mnt/d/GraduateDesign/graduate_design/Data/R-C.jpg")
    pic_r,pic_g,pic_b = get_pic("/mnt/d/GraduateDesign/graduate_design/Data/R-C.jpg")

    hist_r = cc.histogram(pic_r)
    hist_g = cc.histogram(pic_g)
    hist_b = cc.histogram(pic_b)
    print(np.sum(hist_r))
    
    # fig,ax = plt.subplots()
    drawpicture(hist_r,'r')
    drawpicture(hist_g,'g')
    drawpicture(hist_b,'b')
    # ax.set_xlabel("x")
    # ax.set_xlabel("y")
    # ax.set_title("test")
    # ax.legend()
    plt.xlim([0,256]),
    plt.title('test'),
    plt.show()