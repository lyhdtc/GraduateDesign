import numpy as np
import time

# 三通道伪彩色，func调用一张图片作为参数版本，与下面2imgfunc对应
def rgb_channel_parameters_1imgfunc(rgb_img_a, rgb_img_b, func , step = 8, size_w = 40, size_h = 40, *args, **kwargs):
    w = rgb_img_a[0].shape[0]
    h = rgb_img_a[0].shape[1]
    if((w%size_w!=0)or(h%size_h!=0)):
        print('Please check slide window SIZE!')
        return
    ans_a = []
    ans_b = []  
    for i in range(int(w/step)):
        raw_a = []
        raw_b = []
        for j in range(int(h/step)): 
            if(i*step+size_w>=w)or(j*step+size_h>=h):break  
            # print(type(rgb_img_a)) 
            # print(np.shape(rgb_img_a[:, i*step:(i*step+size_w), j*step:(j*step+size_h)]))  
            b = func(rgb_img_b[:,i*step:(i*step+size_w), j*step:(j*step+size_h)], *args, **kwargs)
            a =       func(rgb_img_a[:,i*step:(i*step+size_w), j*step:(j*step+size_h)], *args, **kwargs)
            raw_a.append(a)
            raw_b.append(b)
        if(raw_a!=[]):ans_a.append(raw_a)
        if(raw_b!=[]):ans_b.append(raw_b) 
    ans_a = np.array(ans_a)
    ans_b = np.array(ans_b)
    ans = np.abs(ans_a-ans_b)
    # print(ans)
    # print(np.shape(ans))
    # print(ans_a)
    # ans = ans.transpose(2,0,1)
    ans = (255*ans) / (np.max(ans)+1e-7)  
    # print(ans)  
    return ans


# 三通道伪彩色，目前是为损失函数写的，故func调用了两张图片作为参数（与单通道不同）
def rgb_channel_parameters_2imgfunc(rgb_img_a, rgb_img_b, func , step = 8, size_w = 40, size_h = 40, *args, **kwargs):
    w = rgb_img_a[0].shape[0]
    h = rgb_img_a[0].shape[1]
    if((w%size_w!=0)or(h%size_h!=0)):
        print('Please check slide window SIZE!')
        return
    ans = []  
    for i in range(int(w/step)):
        raw = [] 
        for j in range(int(h/step)): 
            if(i*step+size_w>=w)or(j*step+size_h>=h):break           
            raw.append(func(rgb_img_a[:,i*step:(i*step+size_w), j*step:(j*step+size_h)], rgb_img_b[:,i*step:(i*step+size_w), j*step:(j*step+size_h)], *args, **kwargs))
        if(raw!=[]):ans.append(raw)
    ans = np.array(ans)
    ans = np.abs(ans)
    # ans = ans.transpose(2,0,1) 
    ans = (255*ans) / (np.max(ans)+1e-7)  
    # print(ans)
    # ans = 255 * (ans/80)  
    return ans


# 单通道伪彩色，输入为灰度矩阵， 输出为低分辨率参数差绝对值矩阵
def single_channel_parameters(gray_img_a,gray_img_b,  func , step = 8, size_w = 40, size_h = 40, *args, **kwargs):
    w = gray_img_a.shape[0]
    h = gray_img_a.shape[1]
    if((w%size_w!=0)or(h%size_h!=0)):
        print('Please check slide window SIZE!')
        return
    ans_a = []
    ans_b = []   
    #!
    start_time = time.perf_counter()
    signal_single = False
    signal_inside = False
    # for i in tqdm.trange(int(w/step)):
    for i in range(int(w/step)):
        raw_a = []
        raw_b = []
        #!
        start_time_inside = time.perf_counter() 
        for j in range(int(h/step)): 
            # print(i,j)
            if(i*step+size_w>w)or(j*step+size_h>h):break   
            start_time_single = time.perf_counter()   
            # print('submat shape is {shape}'.format(shape=np.shape(gray_img_a[i*step:(i*step+size_w), j*step:(j*step+size_h)])))     
            raw_a.append(func(gray_img_a[i*step:(i*step+size_w), j*step:(j*step+size_h)], *args, **kwargs))
            end_time_single = time.perf_counter() 
            if(signal_single==False):
                # print('单指令共运行了 {_time_}秒'.format(_time_=(end_time_single - start_time_single)))
                signal_single=True
            raw_b.append(func(gray_img_b[i*step:(i*step+size_w), j*step:(j*step+size_h)], *args, **kwargs))
            # print("raw_a type is {}".format(type(raw_a)))
        #!
        end_time_inside = time.perf_counter() 
        if(signal_inside==False):
            # print('内循环共运行了 {_time_}秒'.format(_time_=(end_time_inside - start_time_inside)))
            signal_inside=True
        if(raw_a!=[]):ans_a.append(raw_a)
        if(raw_b!=[]):ans_b.append(raw_b) 
        
    #!
    end_time = time.perf_counter()  
    # print('循环共运行了 {_time_}秒'.format(_time_=(end_time - start_time)))
    ans_a = np.array(ans_a)
    ans_b = np.array(ans_b)
    ans = np.abs(ans_a-ans_b)
    # print(np.shape(ans))
    # print(ans_a)
    ans = ans.transpose(2,0,1)
    ans = (255*ans) / (np.max(ans)+1e-7)
    return ans

# 单通道伪彩色，输入为灰度矩阵，输出为低分辨率向量距离矩阵
def single_channel_vectors(gray_img_a, gray_img_b, func , step = 8, size_w = 0, size_h = 0, *args, **kwargs):
    w = gray_img_a.shape[0]
    h = gray_img_a.shape[1]
    ans = [] 
    if((w%size_w!=0)or(h%size_h!=0)):
        print('Please check slide window SIZE!')
        return
    for i in range(int(w/step)):
        raw = [] 
        for j in range(int(h/step)):
            vec_a_first, vec_a_second = func(gray_img_a[i*step:(i*step+size_w), j*step:(j*step+size_h)], *args, **kwargs)
            vec_b_first, vec_b_second = func(gray_img_b[i*step:(i*step+size_w), j*step:(j*step+size_h)], *args, **kwargs)
            distance = np.sqrt(np.square(vec_a_first-vec_b_first)+np.square(vec_a_second-vec_b_second))
            raw.append(distance)
        ans.append(raw)
    ans = np.array(ans)
    ans = ans.transpose(2,0,1)
    ans = (255*ans) / (np.max(ans)+1e-7)   
    return ans

# !:有大问题，输出的矩阵看起来是二维的，实际上是list拼list拼list拼出来的，画图或者计算max都报错
# 单通道伪彩色，输入为灰度图，输出为等大小灰度图结果
def single_channel_pictures(gray_img_a, gray_img_b, func , step = 8, size_w = 0, size_h = 0, *args, **kwargs):
    w = gray_img_a.shape[0]
    h = gray_img_a.shape[1]
    ans_a = []
    ans_b = []  
    if((w%size_w!=0)or(h%size_h!=0)):
        print('Please check slide window SIZE!')
        return
    for i in range(int(w/step)):
        raw_a = []
        raw_b = []
        for j in range(int(h/step)):
            img_a = func(gray_img_a[i*step:(i*step+size_w), j*step:(j*step+size_h)], *args, **kwargs)
            img_b = func(gray_img_b[i*step:(i*step+size_w), j*step:(j*step+size_h)], *args, **kwargs)
            print(np.shape(img_a))
            # np.hstack((raw_a,img_a))
            # np.hstack((raw_b,img_b))
            raw_a.append(img_a)
            raw_b.append(img_b)
            # print(np.shape(col_a))
 
        ans_a.append(raw_a)
        ans_b.append(raw_b)
    print(type(ans_a))
    print(ans_a)
    ans_a = np.array(ans_a)
    ans_b = np.array(ans_b)
    # print(type(ans_a))
    # print(np.shape(ans_a))
    ans = np.abs(ans_a-ans_b)
    # ans = np.reshape(ans,(int(w/step),int(h/step)))
    # print(ans)
    # ans = ans.transpose(2,0,1)
    # ans = (255*ans) / np.max(ans)    
    return ans