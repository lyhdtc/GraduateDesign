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

def get_img(path):
    img = cv2.imread(path)
    # cv2.imshow("asdf",img)
    # key = cv2.waitKey(0)
    img_r,img_g,img_b = cv2.split(img)
    # cv2.imshow("adf",img_r)
    # key = cv2.waitKey(0)
    return img_r,img_g,img_b

def drawimgture(img,color):    
    plt.plot(img,color),    
    # plt.xlim([0,256]),
    # plt.title('test'),
    # plt.show()
    # ax.plot(img,color,lab)
    
if __name__=="__main__":
    # get_img("/mnt/d/GraduateDesign/graduate_design/Data/R-C.jpg")
    img_r,img_g,img_b = get_img("/mnt/d/GraduateDesign/graduate_design/Data/R-C.jpg")

    # hist_r = cc.histogram(img_r)
    # hist_g = cc.histogram(img_g)
    # hist_b = cc.histogram(img_b)
    # print(np.sum(hist_r))
    
    # # fig,ax = plt.subplots()
    # drawimgture(hist_r,'r')
    # drawimgture(hist_g,'g')
    # drawimgture(hist_b,'b')
    # # ax.set_xlabel("x")
    # # ax.set_xlabel("y")
    # # ax.set_title("test")
    # # ax.legend()
    # plt.xlim([0,256]),
    # plt.title('test'),
    # plt.show()
    
    m = [[3,3,3,2,1,3],
         [3,3,3,2,3,3],
         [3,3,3,3,3,3]]
    # m = cc.metrix_addoneround(m,8)
    # c = [0]
    # cc.dfs(m,c,3,8,1,1)
    
    vec = cc.color_coherence_vector(m,8,3,3)
    
    print(vec)
    
    # ans = cc.img_quantify(img_r)
    # ans = cc.metrix_addoneround(ans,4)
    # ans = cc.metrix_minusoneround(ans)
    # print(len(ans))
    # print(len(ans[0]))