import sys


import cv2
import numpy as np



#直方图 输入某一通道的图片，直接返回灰度矩阵
def histogram(img):    
    hist = cv2.calcHist([img],[0],None, [256],[0,256])
    return hist


#颜色矩 输入某一通道的图片，返回1*3矩阵，分别为该通道的一阶矩二阶矩三阶矩
#颜色矩的计算建议使用HSV通道
def color_moments(img):
    color_feature = np.zeros(3)
    # N = channel_a.shape[0] * channel_a.shape[1]
    #一阶矩 - average
    first_moment = np.mean(img)# np.sum(channel_a)/float(N)        
    color_feature[0]=first_moment
    #二阶矩 - standard deviation
    second_moment = np.std(img)# np.sqrt(np.mean(abs(channel_a - channel_a.mean())**2))
    color_feature[1]=second_moment
    #三阶矩 - the third root of the skewness
    img_skewness = np.mean(abs(img - img.mean())**3)
    third_moment = img_skewness**(1./3)
    color_feature[2]=third_moment
    
    return color_feature



# 普通矩 输入某一通道图片，输出三阶及以下的几何矩（mpq）、中心矩(mupq)和归一化的矩(nupq)
def ordinary_moments(img):
    # print(cv2.moments(img))
    return list(cv2.moments(img).values())

# 这个是自己写的...网上没找到...
# 第二版，调用了opencv的查找连通域的函数
def color_coherence_vector(img,color_threshold = 8, area_threshold = 100, bit_depth = 8):
    # 高斯模糊，测试1920*1080的图片不需要做此操作提速
    # img = cv2.GaussianBlur(img, (3,3),0)
    # 使用opencv4自带函数之后就不需要手动量化图片了
    # img = img_quantify(img, color_threshold, bit_depth)
    vec_smaller = np.zeros(color_threshold, dtype=int)
    vec_bigger  = np.zeros(color_threshold, dtype=int)
    ret,th = cv2.threshold(img,127,255,0)
    ret, labeled, stat, centroids = cv2.connectedComponentsWithStats(th, None, cv2.CC_STAT_AREA, None,connectivity=8)
    
    areas = [[v[4],label_idx] for label_idx,v in enumerate(stat)]
    coord = [[v[0],v[1]] for label_idx,v in enumerate(stat)]
    for a,c in zip(areas, coord):
        area_size = a[0]
        x,y = c[0],c[1]
        if(x<img.shape[1])and(y<img.shape[0]):
            bin_idx = int(img[y,x]/(256/color_threshold))
            if(area_size >= area_threshold):
                vec_bigger[bin_idx] = vec_bigger[bin_idx]+1
            else:
                vec_smaller[bin_idx] = vec_smaller[bin_idx]+1
    return vec_smaller,vec_bigger


# 第一版，迭代过深，不好用
# 参照的这个https://blog.csdn.net/u014655590/article/details/25108297
# 颜色聚合向量 输入某一通道图片，量化级数，聚合阈值，色彩深度，输出该通道下的颜色聚合向量，其中后三个选项可不填
# def color_coherence_vector(img,color_threshold = 8, area_threshold = 100, bit_depth = 8):
#     img = cv2.GaussianBlur(img, (3,3),0)
#     img = img_quantify(img, color_threshold, bit_depth)
#     img = metrix_addoneround(img, color_threshold)
#     vec = np.zeros((color_threshold, 2), dtype = int)
#     for i in range(len(img)):
#         for j in range(len(img[0])):
#             if(img[i][j] != color_threshold):

#                 cur_color = img[i][j]                
#                 count = [0]
#                 coherence_dfs(img, count, cur_color,color_threshold, i, j)
#                 if(count[0]>=area_threshold):
#                     vec[int(cur_color)][1] = vec[int(cur_color)][1] + count[0]
#                 else:
#                     vec[int(cur_color)][0] = vec[int(cur_color)][1] + count[0]
#     return vec

        
 
#——————————————————————————————颜色聚合向量计算用的函数——————————————————— 
#dfs用方向向量，聚合向量的判断是周围八个元素            
__DIRECTION = [[-1,-1],[-1,0],[1,0],[0,-1],[0,1],[1,-1],[1,0],[1,1]]

#颜色聚合向量的dfs
def coherence_dfs(img, count, cur_color,color_threshold, pos_x, pos_y):    
    img[pos_x][pos_y] = color_threshold
    #python竟然不支持引用传参。。是根据变量类型自动决定深拷贝还是浅拷贝的，先用这个很丑陋的方法跑通再说
    count[0]  = count[0] + 1
    # print(count[0])
    for i in range(7):       
        if(img[pos_x+__DIRECTION[i][0]][pos_y+__DIRECTION[i][1]] == cur_color):            
            coherence_dfs(img, count, cur_color, color_threshold, pos_x+__DIRECTION[i][0], pos_y+__DIRECTION[i][1])
    return
    
 
 
#预处理用，把矩阵外面加一圈，加的值为num   
def metrix_addoneround(metrix, num):
    raw_matrix = [num]*len(metrix[0])    
    metrix     = np.vstack([raw_matrix,metrix])
    metrix     = np.vstack([metrix,raw_matrix])    
    col_matrix = [[num]]*(len(metrix))    
    metrix     = np.hstack([col_matrix,metrix])
    metrix     = np.hstack([metrix,col_matrix])
    return metrix
#预处理用，减掉矩阵外面的一圈，当矩阵小于3行或者3列时不处理
def metrix_minusoneround(metrix):
    if(len(metrix)<3 or len(metrix[0]<3)):return
    metrix = metrix[1:(len(metrix)-1), 1:(len(metrix[0])-1)]
    return metrix
    
#图像量化，color_threshold是量化的级数，bit_depth是图片的位数（通常为8位）
def img_quantify(img, color_threshold = 8, bit_depth = 8):
    color_range = pow(2, bit_depth)/color_threshold
    # print(color_range)
    img = np.floor(np.array(img)/color_range)
    return img
#——————————————————————————————————————————————————————————————————————
