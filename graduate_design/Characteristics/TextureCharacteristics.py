import sys
sys.path.append('graduate_design/tests')
import numpy as np
import math
import pywt
from scipy import signal as sg
from skimage.feature import local_binary_pattern
import cv2
import TestScripts
np.set_printoptions(suppress=True)
import time



# 工具函数
# 将数组转换为0-255
def __norm(ar):
    return 255.*np.absolute(ar)/np.max(ar)

#————————————————————————————灰度共生矩阵————————————————————————————————
# 输入图片和需要的灰度共生矩阵特征，输出对应的特征
# 先列出了四个比较常见的特征，如下：
# 能量（energy）是图像灰度分布均匀程度和纹理粗细的一个度量，反映了图像灰度分布均匀程度和纹理粗细度。当图像纹理均一规则时，能量值较大；反之灰度共生矩阵的元素值相近，能量值较小。
# 熵（entropy）度量了图像包含信息量的随机性，表现了图像的复杂程度。当共生矩阵中所有值均相等或者像素值表现出最大的随机性时，熵最大
# 对比度（contrast）反应了图像的清晰度和纹理的沟纹深浅。纹理越清晰反差越大对比度也就越大
# 反差分矩阵（Inverse Differential Moment, IDM）反映了纹理的清晰程度和规则程度，纹理清晰、规律性较强、易于描述的，值较大
def glcm_feature(gray_img,d_x, d_y,gray_level=16):
    glcm = __glcm(gray_img,d_x, d_y,gray_level)
   
    res = []
    # feature == "energy":
    matrix_energy = np.square(glcm)
    res.append(np.sum(matrix_energy)) 
    
    # feature == "entropy":
    w = glcm.shape[0]
    h = glcm.shape[1]
    matrix_delta = np.full((w,h), 1e-7)
    glcm += matrix_delta
    
    matrix_log = np.log(glcm)
    matrix_entropy = -1 * matrix_log * glcm
    res.append(np.sum(matrix_entropy)) 
    
    # feature == "contrast":
    contrast = 0
    for i in range (len(glcm)):
        for j in range (len(glcm[0])):
            contrast += (i-j)*(i-j)*glcm[i][j]
    res.append(contrast)
    # feature == "IDM":
    idm = 0
    for i in range (len(glcm)):
        for j in range (len(glcm[0])):
            idm += glcm[i][j]/(1+(i-j)*(i-j))
    res.append(idm)
    return res


# 计算并返回归一化后的灰度共生矩阵
def __glcm(arr, d_x, d_y, gray_level=16):
    max_gray = arr.max()
    height, width = arr.shape
    
    arr = arr.astype(np.float64)  # 将uint8类型转换为float64，以免数据失真
    arr = arr * (gray_level - 1) // max_gray  # 若灰度级数大于gray_level，则将图像的灰度级缩小至gray_level，减小灰度共生矩阵的大小。量化后灰度值范围：0 ~ gray_level - 1
    ret = np.zeros([gray_level, gray_level])
    for j in range(height -  abs(d_y)):
        for i in range(width - abs(d_x)):  # range(width - d_x)  #注释为源代码，经评论指出错误后修改
            rows = arr[j][i].astype(int)
            cols = arr[j + d_y][i + d_x].astype(int)
            
            ret[rows][cols] += 1
    if d_x >= d_y:
        ret = ret / float(height * (width - 1))  # 归一化, 水平方向或垂直方向
    else:
        ret = ret / float((height - 1) * (width - 1))  # 归一化, 45度或135度方向
    return ret
# glcm_0 = glcm(gray_gray_img_gray, 1, 0)  # 水平方向
# glcm_1  = glcm(gray_gray_img_gray, 0, 1)  # 垂直方向
# glcm_2  = glcm(gray_gray_img_gray, 1, 1)  # 45度方向
# glcm_3  = glcm(gray_gray_img_gray, -1, 1)  # 135度方向

#————————————————————————————————————————————————————————————————————————————————  

#————————————————————————局部二值（LBP）——————————————————————————————————————————
def rotation_invariant_LBP(gray_img, radius=3, neighbors=8):
    lbp = local_binary_pattern(gray_img, neighbors , radius)
    # print(lbp)
    return lbp
# https://zhuanlan.zhihu.com/p/91768977
# 下面的代码是根据原理来的，整个流程循环较多计算时间较长
# 在圆形选取框基础上，加入旋转不变操作
# def rotation_invariant_LBP(gray_gray_img, radius=3, neighbors=8):
#     h,w=gray_gray_img.shape
#     dst = np.zeros((h-2*radius, w-2*radius),dtype=gray_gray_img.dtype)
#     for i in range(radius,h-radius):
#         for j in range(radius,w-radius):
#             # 获得中心像素点的灰度值
#             center = gray_gray_img[i,j]
#             for k in range(neighbors):
#                 # 计算采样点对于中心点坐标的偏移量rx，ry
#                 rx = radius * np.cos(2.0 * np.pi * k / neighbors)
#                 ry = -(radius * np.sin(2.0 * np.pi * k / neighbors))
#                 # 为双线性插值做准备
#                 # 对采样点偏移量分别进行上下取整
#                 x1 = int(np.floor(rx))
#                 x2 = int(np.ceil(rx))
#                 y1 = int(np.floor(ry))
#                 y2 = int(np.ceil(ry))
#                 # 将坐标偏移量映射到0-1之间
#                 tx = rx - x1
#                 ty = ry - y1
#                 # 根据0-1之间的x，y的权重计算公式计算权重，权重与坐标具体位置无关，与坐标间的差值有关
#                 w1 = (1-tx) * (1-ty)
#                 w2 =    tx  * (1-ty)
#                 w3 = (1-tx) *    ty
#                 w4 =    tx  *    ty
#                 # 根据双线性插值公式计算第k个采样点的灰度值
#                 neighbor = gray_gray_img[i+y1,j+x1] * w1 + gray_gray_img[i+y2,j+x1] *w2 + gray_gray_img[i+y1,j+x2] *  w3 +gray_gray_img[i+y2,j+x2] *w4
#                 # LBP特征图像的每个邻居的LBP值累加，累加通过与操作完成，对应的LBP值通过移位取得
#                 dst[i-radius,j-radius] |= (neighbor>center)  <<  (np.uint8)(neighbors-k-1)
#     # 进行旋转不变处理
#     for i in range(dst.shape[0]):
#         for j in range(dst.shape[1]):
#             currentValue = dst[i,j]
#             minValue = currentValue
#             for k in range(1, neighbors):
#                 # 对二进制编码进行循环左移，意思即选取移动过程中二进制码最小的那个作为最终值
#                 temp = (np.uint8)(currentValue>>(neighbors-k)) |  (np.uint8)(currentValue<<k)
#                 if temp < minValue:
#                     minValue = temp
#             dst[i,j] = minValue

#     return dst    
#————————————————————————————————————————————————————————————————————————————————

# TODO:Malkov random field

#—————————————————————————————————————————————————————————————————————————————————

# Tamura特征
# https://github.com/MarshalLeeeeee/Tamura-In-Python/blob/master/tamura-numpy.py

def tamura_feature(gray_img, kmax, dist):
    tamura_feature = []
    tamura_feature.append(__tamura_coarseness(gray_img, kmax))
    # print('coarseness finished')
    tamura_feature.append(__tamura_contrast(gray_img))
    # print('contrast finished')
    fdir,theta = __tamura_directionality(gray_img)
    tamura_feature.append(fdir)
    # print('directionality finished')
    tamura_feature.append(__tamura_linelikeness(gray_img, theta, dist))
    # print('linelikeness finished')
    return tamura_feature


# 粗糙度 coarseness
# 用来反映纹理粒度
# 输入为图像，活动窗口的尺寸（边长为2^kmax）
def __tamura_coarseness(gray_img, kmax):
	gray_img = np.array(gray_img)
	w = gray_img.shape[0]
	h = gray_img.shape[1]
	kmax = kmax if (np.power(2,kmax) < w) else int(np.log(w) / np.log(2))
	kmax = kmax if (np.power(2,kmax) < h) else int(np.log(h) / np.log(2))
	average_gray = np.zeros([kmax,w,h])
	horizon = np.zeros([kmax,w,h])
	vertical = np.zeros([kmax,w,h])
	Sbest = np.zeros([w,h])

	for k in range(kmax):
		window = np.power(2,k)
		for wi in range(w)[window:(w-window)]:
			for hi in range(h)[window:(h-window)]:
				average_gray[k][wi][hi] = np.sum(gray_img[wi-window:wi+window, hi-window:hi+window])
		for wi in range(w)[window:(w-window-1)]:
			for hi in range(h)[window:(h-window-1)]:
				horizon[k][wi][hi] = average_gray[k][wi+window][hi] - average_gray[k][wi-window][hi]
				vertical[k][wi][hi] = average_gray[k][wi][hi+window] - average_gray[k][wi][hi-window]
		horizon[k] = horizon[k] * (1.0 / np.power(2, 2*(k+1)))
		vertical[k] = horizon[k] * (1.0 / np.power(2, 2*(k+1)))

	for wi in range(w):
		for hi in range(h):
			h_max = np.max(horizon[:,wi,hi])
			h_max_index = np.argmax(horizon[:,wi,hi])
			v_max = np.max(vertical[:,wi,hi])
			v_max_index = np.argmax(vertical[:,wi,hi])
			index = h_max_index if (h_max > v_max) else v_max_index
			Sbest[wi][hi] = np.power(2,index)

	fcrs = np.mean(Sbest)
	return fcrs

# 对比度
# 是通过对像素强度分布情况的统计得到的，其大小由四个因素决定：灰度动态范围、直方图上黑白部分两极分化程度、边缘锐度和重复模式的周期。一般情况下，对比度指前面两个因素。
# 输入为图像
def __tamura_contrast(gray_img):
	gray_img = np.array(gray_img)
	gray_img = np.reshape(gray_img, (1, gray_img.shape[0]*gray_img.shape[1]))
	m4 = np.mean(np.power(gray_img - np.mean(gray_img),4))
	v = np.var(gray_img)
	std = np.power(v, 0.5)
	alfa4 = m4 / np.power(v,2)
	fcon = std / np.power(alfa4, 0.25)
	return fcon

# 方向度
# 给定纹理区域的全局特性，描述纹理如何沿着某些方向发散或者集中的
# 输入为图像
def __tamura_directionality(gray_img):
	gray_img = np.array(gray_img, dtype = 'int64')
	h = gray_img.shape[0]
	w = gray_img.shape[1]
	convH = np.array([[-1,0,1],[-1,0,1],[-1,0,1]])
	convV = np.array([[1,1,1],[0,0,0],[-1,-1,-1]])
	deltaH = np.zeros([h,w])
	deltaV = np.zeros([h,w])
	theta = np.zeros([h,w])

	# calc for deltaH
	for hi in range(h)[1:h-1]:
		for wi in range(w)[1:w-1]:
			deltaH[hi][wi] = np.sum(np.multiply(gray_img[hi-1:hi+2, wi-1:wi+2], convH))
	for wi in range(w)[1:w-1]:
		deltaH[0][wi] = gray_img[0][wi+1] - gray_img[0][wi]
		deltaH[h-1][wi] = gray_img[h-1][wi+1] - gray_img[h-1][wi]
	for hi in range(h):
		deltaH[hi][0] = gray_img[hi][1] - gray_img[hi][0]
		deltaH[hi][w-1] = gray_img[hi][w-1] - gray_img[hi][w-2]

	# calc for deltaV
	for hi in range(h)[1:h-1]:
		for wi in range(w)[1:w-1]:
			deltaV[hi][wi] = np.sum(np.multiply(gray_img[hi-1:hi+2, wi-1:wi+2], convV))
	for wi in range(w):
		deltaV[0][wi] = gray_img[1][wi] - gray_img[0][wi]
		deltaV[h-1][wi] = gray_img[h-1][wi] - gray_img[h-2][wi]
	for hi in range(h)[1:h-1]:
		deltaV[hi][0] = gray_img[hi+1][0] - gray_img[hi][0]
		deltaV[hi][w-1] = gray_img[hi+1][w-1] - gray_img[hi][w-1]

	deltaG = (np.absolute(deltaH) + np.absolute(deltaV)) / 2.0
	deltaG_vec = np.reshape(deltaG, (deltaG.shape[0] * deltaG.shape[1]))

	# calc the theta
	for hi in range(h):
		for wi in range(w):
			if (deltaH[hi][wi] == 0 and deltaV[hi][wi] == 0):
				theta[hi][wi] = 0
			elif(deltaH[hi][wi] == 0):
				theta[hi][wi] = np.pi
			else:
				theta[hi][wi] = np.arctan(deltaV[hi][wi] / deltaH[hi][wi]) + np.pi / 2.0
	theta_vec = np.reshape(theta, (theta.shape[0] * theta.shape[1]))

	n = 16
	t = 12
	cnt = 0
	hd = np.zeros(n)
    
	dlen = deltaG_vec.shape[0]
	for ni in range(n):
		for k in range(dlen):
			if((deltaG_vec[k] >= t) and (theta_vec[k] >= (2*ni-1) * np.pi / (2 * n)) and (theta_vec[k] < (2*ni+1) * np.pi / (2 * n))):
				hd[ni] += 1
    
    
	hd = hd / np.mean(hd)
	hd_max_index = np.argmax(hd)
	fdir = 0
	for ni in range(n):
		fdir += np.power((ni - hd_max_index), 2) * hd[ni]
	return fdir,theta

# 线性度
# 输入为图片，tamura_directionality输出的矩阵，共生矩阵计算时的像素间隔距离 

# created by lyh 2021.11.24
# 之前的版本运行时间过长，重写了线性度的计算函数
def __tamura_linelikeness(gray_img, theta, dist):
    n = 16
    h = gray_img.shape[0]
    w = gray_img.shape[1]
    pi = 3.1415926
    dcm = np.zeros((8,n,n))
    dir = np.float32([   [[1,0,-dist],   [0,1,-dist]],
                    [[1,0,-dist],   [0,1,0]],
                    [[1,0,dist],    [0,1,0]],
                    [[1,0,0],       [0,1,-dist]],
                    [[1,0,0],       [0,1,dist]],
                    [[1,0,dist],    [0,1,-dist]],
                    [[1,0,dist],    [0,1,0]],
                    [[1,0,dist],    [0,1,dist]]
                 ])
    cooccurrence_matrixes = []
    for i in range(8):
        cooccurrence_matrixes.append(cv2.warpAffine(theta, dir[i], (w,h)))
        
    for m1 in range(1,n):
        for m2 in range(1,n):
            for d in range(8):
                m_theta_bottom = ( theta>=((2*(m1-1)*pi)/(2*n)) )
                m_theta_top = (theta<(((2*(m1-1)+1)*pi)/(2*n)) )  
                m_theta = np.logical_and(m_theta_bottom, m_theta_top)
                m_ccoccurrence_matrixes_bottom = ( cooccurrence_matrixes[d]>=((2*(m2-1)*pi)/(2*n)))
                m_ccoccurrence_matrixes_top = ( cooccurrence_matrixes[d]<(((2*(m2-1)+1)*pi)/(2*n)))
                m_ccoccurrence_matrixes = np.logical_and(m_ccoccurrence_matrixes_bottom, m_ccoccurrence_matrixes_top)
                dcm_matrix = np.logical_and(m_theta, m_ccoccurrence_matrixes)
                dcm_matrix = dcm_matrix.astype(int)
                dcm[d][m1][m2] = np.sum(dcm_matrix)                
    matrix_f = np.zeros((1,8))
    matrix_g = np.zeros((1,8))
    for i in range(n):
        for j in range(n):
            for d in range(8):
                matrix_f[0][d] += dcm[d][i][j]*(math.cos((i-j)*2*pi/n))
                matrix_g[0][d] += dcm[d][i][j]          
    matrix_res = matrix_f/matrix_g
    res = np.max(matrix_res)
    return res	

# @TestScripts.timmer
# def __tamura_linelikeness(gray_img, theta, dist):
#     # http://www.skcircle.com/?id=1496
#     # 建立方向向量，分别为左上、中上、右上、左中、右中、左下、中下、右下
#     DIRECTION = [[-dist,-dist],[-dist,0],[dist,0],[0,-dist],[0,dist],[dist,-dist],[dist,0],[dist,dist]]
#     # n这里默认取16了，具体原因还得再看看
#     n = 16
#     h = gray_img.shape[0]
#     w = gray_img.shape[1]
#     pi = 3.1415926
#     dcm = np.zeros((8,n,n))
#     round1begin = time.perf_counter()
#     # 遍历图像中间区域（因为考虑到步长的问题不能从最边缘开始）
#     for i in range(dist+1, h-dist-2):
#         for j in range(dist+1, w-dist-2):
#             # 共生矩阵遍历
#             for m1 in range(1,n):
#                 for m2 in range(1,n):
#                     # 每一个方向上判断，满足条件共生矩阵对应位置加一
#                     for d in range(8):                        
#                         if((theta[i][j]>=((2*(m1-1)*pi)/(2*n))) and (theta[i][j]<(((2*(m1-1)+1)*pi)/(2*n))) and (theta[i+DIRECTION[d][0]][j+DIRECTION[d][1]]>=((2*(m2-1)*pi)/(2*n))) and (theta[i+DIRECTION[d][0]][j+DIRECTION[d][1]]<(((2*(m2-1)+1)*pi)/(2*n)))):
#                             dcm[d][m1][m2] += 1
#     round1end = time.perf_counter()
#     print('part1 finished, total time is {_funcname_}s'.format(_funcname_ = round1end - round1begin))
#     matrix_f = np.zeros((1,8))
#     matrix_g = np.zeros((1,8))
#     round2begin = time.perf_counter()
#     for i in range(n):
#         for j in range(n):
#             for d in range(8):
#                 matrix_f[0][d] += dcm[d][i][j]*(math.cos((i-j)*2*pi/n))
#                 matrix_g[0][d] += dcm[d][i][j]
#     round2end = time.perf_counter()
#     print('part2 finished, total time is {_funcname_}s'.format(_funcname_ = round2end - round2begin))           
#     matrix_res = matrix_f/matrix_g
#     res = np.max(matrix_res)
#     return res


#规整度
def __tamura_regularity(gray_img, filter):
	pass

#粗略度
#粗糙度和对比度之和
def __tamura_roughness(fcrs, fcon):
	return fcrs + fcon

# ————————————————————————————————————————————————————————————————————————————————————

# ——————————————————————小波变换————————————————————————————————————————————————————————

# 输入为单通道图像以及对应的小波函数
# 小波函数种类如下，请输入括号中的字符串：
# Haar(haar)        Daubechies(dbN)   Biorthogonal(biorNr.Nd) 
# Coiflets(coifN)   Symlets(symN)     Morlet(morl) 
# Mexican Hat(mexh)
#获取小波变换的特征向量，输出为1*16的向量，直接调用这个函数就可以
def dwt_feature(gray_img, wave_func = "haar"):
    dwt_metrix = __dwt(gray_img, wave_func)
    feature = []
    for i in range(4):
        feature.append(__dwt_average(dwt_metrix[i]))
        feature.append(__dwt_entropy(dwt_metrix[i]))
        feature.append(__dwt_sigma(dwt_metrix[i]))
        feature.append(__dwt_energy(dwt_metrix[i]))
    return feature

# 小波变换函数，输出三维矩阵，0，1，2，3切片分别为低频分量，高频水平方向，高频垂直方向，高频对角线方向分量

def __dwt(gray_img, wave_func):
    ca,(ch, cv, cd) = pywt.dwt2(gray_img, wave_func)
    
    res = np.array([__norm(ca),__norm(ch),__norm(cv),__norm(cd)])
    np.nan_to_num(res)
    # print(res)
    # res = np.array([ca,ch,cv,cd])
    return res



# 一些小波变换中常用的特征，通常在分解之后将子带图像的下述特征组合构造向量，来达到好的分类效果

# 均值
def __dwt_average(gray_img):
    h = gray_img.shape[0]
    w = gray_img.shape[1]
    res = np.sum(gray_img) / (h*w)
    return res

# 熵
def __dwt_entropy(gray_img):
    h = gray_img.shape[0]
    w = gray_img.shape[1]
    
    # 用来避免除0操作
    w = gray_img.shape[0]
    h = gray_img.shape[1]   
    matrix_delta = np.full((w,h), 1e-7)  
    gray_img += matrix_delta
    
    matrix_log = np.log(gray_img)
    matrix_entropy = -1 * matrix_log * gray_img
    res = np.sum(matrix_entropy) / (h*w)
    return res
    
# 标准差  
def __dwt_sigma(gray_img):
    res = np.std(gray_img)
    return res

# 能量  
def __dwt_energy(gray_img):
    matrix_energy = np.square(gray_img)
    res = np.sum(matrix_energy)
    return res
# ——————————————————————————————————————————————————————————————————————————

#————————————————————————Laws 纹理测量————————————————————————————————————————
# https://github.com/yuvrajt/Laws-Texture/blob/master/laws.py
def  laws_feature(gray_img):
    gray_img2 = np.copy(gray_img.astype(np.float64))
    h = gray_img.shape[0]
    w = gray_img.shape[1]
    
    conv_maps = np.zeros((h, w, 16), np.float64)
    
    filter_vectors = np.array(  [[1, 4, 6,  4, 1],
                                [-1, -2, 0, 2, 1],
                                [-1, 0, 2, 0, 1],
                                [1, -4, 6, -4, 1]])
    filters = list()
    for i in range(4):
        for j in range(4):
            filters.append(np.matmul(filter_vectors[i][:].reshape(5,1),filter_vectors[j][:].reshape(1,5)))
    
    #Preprocess the image
    smooth_kernel = (1/25)*np.ones((5,5))
    gray_smooth = sg.convolve(gray_img2 ,smooth_kernel,"same")
    gray_processed = np.abs(gray_img2 - gray_smooth)

    #Convolve the Laws kernels
    for ii in range(len(filters)):
        conv_maps[:, :, ii] = sg.convolve(gray_processed,filters[ii],'same')

    #Create the 9 texture maps
    texture_maps = list()
    texture_maps.append(__norm((conv_maps[:, :, 1] +conv_maps[:, :, 4])//2))
    texture_maps.append(__norm((conv_maps[:, :, 3] +conv_maps[:, :, 12])//2))
    texture_maps.append(__norm( conv_maps[:, :, 10]))
    texture_maps.append(__norm( conv_maps[:, :, 15]))
    texture_maps.append(__norm((conv_maps[:, :, 2] +conv_maps[:, :, 8])//2))
    texture_maps.append(__norm( conv_maps[:, :, 5]))
    texture_maps.append(__norm((conv_maps[:, :, 7] +conv_maps[:, :, 13])//2))
    texture_maps.append(__norm((conv_maps[:, :, 11]+conv_maps[:, :, 14])//2))
    
    return texture_maps

