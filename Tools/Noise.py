from matplotlib.pyplot import gray
import skimage
import numpy as np
import cv2
import tqdm
from Tools import ParsePicName
import os


# 传统的噪声模式
# image：输入图片（将会被转换成浮点型），ndarray型
# mode： 可选择，str型，表示要添加的噪声类型
#  gaussian：高斯噪声
#  localvar：高斯分布的加性噪声，在“图像”的每个点处具有指定的局部方差。
#  poisson：泊松噪声
#  salt：盐噪声，随机将像素值变成1
#  pepper：椒噪声，随机将像素值变成0或-1，取决于矩阵的值是否带符号
#  s&p：椒盐噪声
#  speckle：均匀噪声（均值mean方差variance），out=image+n*image
# seed： 可选的，int型，如果选择的话，在生成噪声前会先设置随机种子以避免伪随机
# clip： 可选的，bool型，如果是True，在添加均值，泊松以及高斯噪声后，会将图片的数据裁剪到合适范围内。如果谁False，则输出矩阵的值可能会超出[-1,1]
# mean： 可选的，float型，高斯噪声和均值噪声中的mean参数，默认值=0
# var：  可选的，float型，高斯噪声和均值噪声中的方差，默认值=0.01（注：不是标准差）
# local_vars：可选的，ndarry型，用于定义每个像素点的局部方差，在localvar中使用
# amount： 可选的，float型，是椒盐噪声所占比例，默认值=0.05
# salt_vs_pepper：可选的，float型，椒盐噪声中椒盐比例，值越大表示盐噪声越多，默认值=0.5，即椒盐等量
def new_traditional_noise(img, mode, *args, **kwargs):
    img_b,img_g,img_r = cv2.split(img)
    noise_map = np.full_like(img_b, 255)
    ##print(noise_map)
    # print(np.shape(temp))
    noise_map = skimage.util.random_noise(noise_map, mode = mode, *args, **kwargs)
    # print(noise_map)
    ##print(img_b)
    ##print(noise_map)
    img_b =img_b* noise_map
    ##print(img_b)
    img_g =img_g* noise_map
    img_r =img_r* noise_map
    ans = cv2.merge([img_b, img_g, img_r])
    # ans = ans.astype(np.int)

    return ans
def traditional_noise(img, mode, *args, **kwargs):
    ans = skimage.util.random_noise(img, mode = mode, *args, **kwargs)
    ans = ans*255
    ans = ans.astype(np.uint8)

    return ans


# 将图片指定百分比的像素替换为随机值
def random_replace_element(gray_img, percent):
    w = gray_img.shape[0]
    h = gray_img.shape[1]
    num = int(percent*w*h)
    one_dim = np.reshape(gray_img,w*h)
    pos = np.random.choice(one_dim.shape[0], num, replace=False)
    noise = np.random.randint(0,255,size=num)    
    np.put(one_dim,pos,noise)
    ans = np.reshape(one_dim, (w,h))
    return ans


# 将图片指定百分比的像素替换为noise_img图片上随机的像素    
def random_replace_element_from_another_picture_gray(gray_img, noise_img, percent):
    w = gray_img.shape[0]
    h = gray_img.shape[1]
    num = int(percent*w*h)
    one_dim = np.reshape(gray_img,w*h)
    pos = np.random.choice(one_dim.shape[0], num, replace=False)    
    w_noise = noise_img.shape[0]
    h_noise = noise_img.shape[1]
    one_dim_noise = np.reshape(noise_img, w_noise*h_noise)
    noise_pos = np.random.choice(one_dim_noise.shape[0], num, replace=True)
    noise = np.take(noise_img, noise_pos)    
    np.put(one_dim, pos, noise)
    ans = np.reshape(one_dim, (w,h))
    return ans
# 将图片指定百分比的像素替换为img_noise上指定位置的像素
def random_replace_element_from_another_picture_bgr(img_data, img_noise, percent):
    h,w,c = np.shape(img_data)
    num = int(percent*w*h)
    b_data, g_data, r_data = cv2.split(img_data)
    b_noise, g_noise, r_noise = cv2.split(img_noise)
    
    b_one_dim = np.reshape(b_data,w*h)
    pos = np.random.choice(b_one_dim.shape[0], num, replace=False)
    b_noise = np.take(b_noise, pos)    
    np.put(b_one_dim, pos, b_noise)
    b_ans = np.reshape(b_one_dim, (h,w))
    
    g_one_dim = np.reshape(g_data, w*h)
    g_noise = np.take(g_noise, pos)    
    np.put(g_one_dim, pos, g_noise)
    g_ans = np.reshape(g_one_dim, (h,w))
    
    r_one_dim = np.reshape(r_data, w*h)
    r_noise = np.take(r_noise, pos)    
    np.put(r_one_dim, pos, r_noise)
    r_ans = np.reshape(r_one_dim, (h,w))
    
    ans = cv2.merge([b_ans, g_ans, r_ans])
    ans = ans.astype(np.uint8)
    return ans

# 输入绝对路径，在同一文件夹下按命名规则生成噪声图片
def generate_noise_pictures(absolute_path):
    img = cv2.imread(absolute_path)
    noise = ['gaussian', 'localvar', 'poisson', 'salt', 'pepper', 's&p', 'speckle']
    folder = os.path.dirname(absolute_path)
    pre_pic = os.path.basename(absolute_path)
    pre_num = len(os.listdir(folder))
    for i in tqdm.tqdm(noise):
        noise_img = traditional_noise(img, i)
        info=[str(pre_num), 
              ParsePicName.get_pic_info(pre_pic,'pre_noise'), 
              ParsePicName.get_pic_info(pre_pic,'material'),
              ParsePicName.get_pic_info(pre_pic,'lighting'),
              i]
        filename = ParsePicName.generate_pic_info(info)
        pre_num = pre_num + 1
        
        filepath = folder+'/'+filename
        # print(filepath)
        cv2.imwrite(filepath, noise_img)
                
