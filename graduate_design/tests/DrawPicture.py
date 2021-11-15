import sys
sys.path.append('/mnt/d/GraduateDesign/graduate_design/Characteristics')
import cv2
import numpy as np
from matplotlib import pyplot as plt
import ColorCharacteristics as cc
import TextureCharacteristics as tc
import LossAboutColor as lac



RGB_COLOR_CHANNEL = {
    'r': 0,
    'g': 1,
    'b': 2
}

def get_img(path):
    img = cv2.imread(path)
    img_r,img_g,img_b = cv2.split(img)
    return img_r,img_g,img_b

def drawpicture(img,color):    
    plt.plot(img,color)

    
if __name__=="__main__":
    # get_img("/mnt/d/GraduateDesign/graduate_design/Data/R-C.jpg")
    img_r,img_g,img_b = get_img("/mnt/d/GraduateDesign/graduate_design/Data/R-C.jpg")
    img = cv2.imread("/mnt/d/GraduateDesign/graduate_design/Data/R-C.jpg")
    ret  = np.zeros((8,2,2))
    # print(ret)
    # cv2.imshow("asdfa",ret)
    # cv2.waitKey(0)
    #print(ret)
    # for i in range (3,8):
    #     print(i)
    imga = cv2.imread("/mnt/d/GraduateDesign/graduate_design/Data/aaa.jpg")    
    imgb = cv2.imread("/mnt/d/GraduateDesign/graduate_design/Data/bbb.jpg")
    res=lac.loss_UnderexposedPhoto_CVPR2019_colorloss(imga, imgb)
    print(res)
    
    
    # HJL = np.array([
    #             [[1,2],
    #             [3,4]],
 
    #             [[5,6],
    #              [7,8]],
                
    #             [[5,6],
    #              [7,8]]
    #             ])
    # print(HJL[0][0][1])
    # print(HJL[1])
    # lac.loss_UnderexposedPhoto_CVPR2019_colorloss(HJL, HJL)
    # a = np.array([6,6,6])
    # b = np.array([3,3,3])
    # c = a/b
    # print(c)

    #res = tc.dwt(img_r)
    #cv2.imshow("daf", res)
    #cv2.waitKey(0)
   # print(res)
    """ 测试直方图
    hist_r = cc.histogram(img_r)
    hist_g = cc.histogram(img_g)
    hist_b = cc.histogram(img_b)
    print(np.sum(hist_r))
    
    # fig,ax = plt.subplots()
    drawimgture(hist_r,'r')
    drawimgture(hist_g,'g')
    drawimgture(hist_b,'b')
    # ax.set_xlabel("x")
    # ax.set_xlabel("y")
    # ax.set_title("test")
    # ax.legend()
    plt.xlim([0,256]),
    plt.title('test'),
    plt.show() """
    
    
""" 测试颜色聚合向量
    m = [[3,3,3,2,1,3],
         [3,3,3,2,3,3],
         [3,3,3,3,3,3]]
    m = cc.metrix_addoneround(m,8)
    c = [0]
    cc.dfs(m,c,3,8,1,1)
    
    vec = cc.color_coherence_vector(m,8,3,3)
    
    print(vec)
    
    ans = cc.img_quantify(img_r)
    ans = cc.metrix_addoneround(ans,4)
    ans = cc.metrix_minusoneround(ans)
    print(len(ans))
    print(len(ans[0])) """

