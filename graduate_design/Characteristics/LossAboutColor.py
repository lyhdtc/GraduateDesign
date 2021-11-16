import numpy as np
import cv2
import scipy
import tensorflow as tf


# 工具函数

# 两张图片的均方误差（mean squared error)
def __MSE(rgb_img_a, rgb_img_b):
    return 0.5 * np.sum((rgb_img_a-rgb_img_b)**2)

# 两张图片的交叉熵（cross entropy error）
def __cross_entropy_error(rgb_img_a, rgb_img_b):
    delta = 1e-7
    return -np.sum(rgb_img_b* np.log(rgb_img_a + delta))

# TV loss (Total Variation loss)
# 在最优化问题的模型中添加一些正则项来保持图像的光滑性，TV loss是常用的一种正则项（注意是正则项，配合其他loss一起使用，约束噪声）。图片中相邻像素值的差异可以通过降低TV loss来一定程度上解决。比如降噪，对抗checkerboard等等
def __tv_loss(rgb_img, weight):
    with tf.variable_scope('tv_loss'):
        return weight * tf.reduce_sum(tf.image.total_variation(rgb_img))
    
# L1 loss
def __l1_loss(rgb_img_a, rgb_img_b):
    return np.sum(np.abs(rgb_img_a-rgb_img_b))

# L2 loss
def __l2_loss(rgb_img_a, rgb_img_b):
    return np.sum(np.square(rgb_img_a - rgb_img_b))
  
  
# SSIM
def __ssim(img1,img2,K,win):

    M,N = img1.shape

    img1 = img1.astype(np.float32)
    img2 = img2.astype(np.float32)

    C1 = (K[0]*255)**2
    C2 = (K[1]*255) ** 2
    win = win/np.sum(win)

    mu1 = scipy.signal.convolve2d(img1,win,mode='valid')
    mu2 = scipy.signal.convolve2d(img2,win,mode='valid')
    mu1_sq = np.multiply(mu1,mu1)
    mu2_sq = np.multiply(mu2,mu2)
    mu1_mu2 = np.multiply(mu1,mu2)
    sigma1_sq = scipy.signal.convolve2d(np.multiply(img1,img1),win,mode='valid') - mu1_sq
    sigma2_sq = scipy.signal.convolve2d(np.multiply(img2, img2), win, mode='valid') - mu2_sq
    img12 = np.multiply(img1, img2)
    sigma12 = scipy.signal.convolve2d(np.multiply(img1, img2), win, mode='valid') - mu1_mu2

    if(C1 > 0 and C2>0):
        ssim1 =2*sigma12 + C2
        ssim_map = np.divide(np.multiply((2*mu1_mu2 + C1),(2*sigma12 + C2)),np.multiply((mu1_sq+mu2_sq+C1),(sigma1_sq+sigma2_sq+C2)))
        cs_map = np.divide((2*sigma12 + C2),(sigma1_sq + sigma2_sq + C2))
    else:
        numerator1 = 2*mu1_mu2 + C1
        numerator2 = 2*sigma12 + C2
        denominator1 = mu1_sq + mu2_sq +C1
        denominator2 = sigma1_sq + sigma2_sq +C2

        ssim_map = np.ones(mu1.shape)
        index = np.multiply(denominator1,denominator2)
        #如果index是真，就赋值，是假就原值
        n,m = mu1.shape
        for i in range(n):
            for j in range(m):
                if(index[i][j] > 0):
                    ssim_map[i][j] = numerator1[i][j]*numerator2[i][j]/denominator1[i][j]*denominator2[i][j]
                else:
                    ssim_map[i][j] = ssim_map[i][j]
        for i in range(n):
            for j in range(m):
                if((denominator1[i][j] != 0)and(denominator2[i][j] == 0)):
                    ssim_map[i][j] = numerator1[i][j]/denominator1[i][j]
                else:
                    ssim_map[i][j] = ssim_map[i][j]

        cs_map = np.ones(mu1.shape)
        for i in range(n):
            for j in range(m):
                if(denominator2[i][j] > 0):
                    cs_map[i][j] = numerator2[i][j]/denominator2[i][j]
                else:
                    cs_map[i][j] = cs_map[i][j]
    mssim = np.mean(ssim_map)
    mcs = np.mean(cs_map)
    return  mssim,mcs

# MS-SSIM
def __msssim(img1,img2):

    K = [0.01,0.03]
    win  = np.multiply(cv2.getGaussianKernel(11, 1.5), (cv2.getGaussianKernel(11, 1.5)).T)  # H.shape == (r, c)
    level = 5
    weight = [0.0448,0.2856,0.3001,0.2363,0.1333]
    method = 'product'

    M,N = img1.shape
    H,W = win.shape

    downsample_filter = np.ones((2,2))/4
    img1 = img1.astype(np.float32)
    img2 = img2.astype(np.float32)

    mssim_array = []
    mcs_array = []

    for i in range(0,level):
        mssim,mcs = __ssim(img1,img2,K,win)
        mssim_array.append(mssim)
        mcs_array.append(mcs)
        filtered_im1 = cv2.filter2D(img1,-1,downsample_filter,anchor = (0,0),borderType=cv2.BORDER_REFLECT)
        filtered_im2 = cv2.filter2D(img2,-1,downsample_filter,anchor = (0,0),borderType=cv2.BORDER_REFLECT)
        img1 = filtered_im1[::2,::2]
        img2 = filtered_im2[::2,::2]

    # print(np.power(mcs_array[:level-1],weight[:level-1]))
    # print(mssim_array[level-1]**weight[level-1])
    overall_mssim = np.prod(np.power(mcs_array[:level-1],weight[:level-1]))*(mssim_array[level-1]**weight[level-1])

    return overall_mssim

    

# ————————————————————————————————————————————————————————————————————————————————————————————————————————
# DSLR-Quality Photos on Mobile Devices with Deep Convolutional Networks（ICCV 2017)
# https://openaccess.thecvf.com/content_ICCV_2017/papers/Ignatov_DSLR-Quality_Photos_on_ICCV_2017_paper.pdf

# 评估图像之间的亮度、对比度和主要颜色的差异，同时消除纹理和内容的比较
def loss_DSLRQualityPhotos_ICCV2017_colorloss(rgb_img_a,rgb_img_b,sigma_x=3, sigma_y=3 ):
    dst_a = cv2.GaussianBlur(rgb_img_a,(0,0),sigma_x,None,sigma_y,None)
    dst_b = cv2.GaussianBlur(rgb_img_b,(0,0),sigma_x,None,sigma_y,None)
    loss = __MSE(dst_a,dst_b)
    return loss

# 专门针对纹理处理。它同时观察假（改进）和真实（目标）图像，其目标是预测输入图像是否真实
def loss_DSLRQualityPhotos_ICCV2017_textureloss(rgb_img_a, rgb_img_b):
    return __cross_entropy_error(rgb_img_a, rgb_img_b)

# 由于计算的是vgg网络中间输出的差异，故暂时不考虑
# 内容损失定义为增强图像和目标图像特征表示之间的欧几里德距离
def __loss_DSLRQualityPhotos_contentloss():
    pass

# 除了先前的损失之外，我们还添加了总变化 (TV) 损失 [1] 以加强所生成图像的空间平滑度
# TV Loss通常描述的单个图片中相邻像素值的差异，因此如果需要判断两张图片，感觉得同时算出来比较
# 原文是只计算了源图片的loss，目的是小图片变高清大图，所以感觉还是有点道理（但还是怪怪的）
def loss_DSLRQualityPhotos_ICCV2017_totalvariationloss(rgb_img,weight=1):
    return __tv_loss(rgb_img,weight)

# ——————————————————————————————————————————————————————————————————————————————————————
# Underexposed Photo Enhancement using Deep Illumination Estimation（CVPR 2019）
# https://openaccess.thecvf.com/content_CVPR_2019/papers/Wang_Underexposed_Photo_Enhancement_Using_Deep_Illumination_Estimation_CVPR_2019_paper.pdf

# 定义L2误差度量来测量重建误差
def loss_UnderexposedPhoto_CVPR2019_reconstructionloss(rgb_img_a, rgb_img_b):
    return __l2_loss(rgb_img_a, rgb_img_b)

# 根据先验光滑性，自然图像中的光照一般为局部光滑。Smoothness Loss为预测的全分辨率光照S的平滑度损失
# 这个是给他们计算光照函数的损失的，故暂不考虑
def __loss_UnderexposedPhoto_CVPR2019loss():
    pass

# 设计颜色损失来使生成的图像中的颜色与相应的标签图片中的颜色匹配，将RGB颜色作为三维向量计算两种颜色之间的夹角。
# 对每个像素对的颜色向量夹角求和

def loss_UnderexposedPhoto_CVPR2019_colorloss(rgb_img_a, rgb_img_b):
    # 通道分离
    r_img_a, g_img_a, b_img_a = cv2.split(rgb_img_a)
    r_img_b, g_img_b, b_img_b = cv2.split(rgb_img_b)
    # 计算各像素颜色向量长度
    img_a_length = np.sqrt(np.square(r_img_a)+np.square(g_img_a)+np.square(b_img_a))    
    img_b_length = np.sqrt(np.square(r_img_b)+np.square(g_img_b)+np.square(b_img_b))
    # 分通道计算内积
    r_img_dot = r_img_a * r_img_b
    g_img_dot = g_img_a * g_img_b
    b_img_dot = b_img_a * b_img_b    
    # 用来避免除0操作
    w = r_img_a.shape[0]
    h = r_img_a.shape[1]   
    matrix_delta = np.full((w,h), 1e-7)  
    # cos theta = (dot(a,b))/(|a||b|)
    matrix_cos   = (r_img_dot+g_img_dot+b_img_dot)/((img_a_length * img_b_length)+matrix_delta)
    matrix_cos   = np.clip(matrix_cos,-1,1)
    matrix_angle = np.arccos(matrix_cos)
    return np.sum(matrix_angle)

# ——————————————————————————————————————————————————————————————————————————————————————————
# Range Scaling Global U-Net for Perceptual Image Enhancement on Mobile Devices（ECCV-PIRM2018）
# https://openaccess.thecvf.com/content_ECCVW_2018/papers/11133/Huang_Range_Scaling_Global_U-Net_for_Perceptual_Image_Enhancement_on_Mobile_ECCVW_2018_paper.pdf

# L1 loss
def loss_RangeScalingGlobalUNet_ECCV2018_l1loss(rgb_img_a, rgb_img_b):
    return __l1_loss(rgb_img_a, rgb_img_b)

# MS-SSIM loss
def loss_RangeScalingGlobalUNet_ECCV2018_MSSSIMloss(rgb_img_a, rgb_img_b):
    return 1-__msssim(rgb_img_a, rgb_img_b)

# VGG损失，暂时跳过
def __loss_RangeScalingGlobalUNet_ECCV2018_vggloss():
    pass

# GAN损失，暂时跳过
def __loss_RangeScalingGlobalUNet_ECCV2018_ganloss():
    pass

# TV loss 和前面的TV loss一样
def _loss_RangeScalingGlobalUNet_ECCV2018_tvloss(rgb_img_a, weight = 1):
    return __tv_loss(rgb_img_a, weight)


# ————————————————————————————————————————————————————————————————————————————————————————————
# Loss Functions for Image Restoration with Neural Networks
# https://arxiv.org/pdf/1511.08861.pdf

# 前面几个损失函数跟上面的论文基本一致
# MS-SSIM + L1 loss， 论文中有一个G项不知道说的是啥，给出的源码里面就是两个损失函数的线性插值
def loss_LossFunctions_IEEE2017_l1andmsssimloss(rgb_img_a, rgb_img_b, alpha=0.84):
    return alpha*(1-__msssim(rgb_img_a, rgb_img_b)) + (1-alpha)*(__l1_loss(rgb_img_a, rgb_img_b))

