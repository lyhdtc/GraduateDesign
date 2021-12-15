from functools import partial
import time
import multiprocessing
from tqdm import tqdm
from scipy.spatial.distance import cdist
import numpy as np
import cv2
import os
 
# def doSomething(param):
#     # t1 = time.time()
#     # r=cdist([a],datas,"cosine")[0]#计算70W次cos值
#     # t2 = time.time()
#     # print('t2-t1:%4f' % (t2 - t1))
#     r=0
#     time.sleep(0.0001)
#     return r
# def do(param):
#     return doSomething(param)
# if __name__ == '__main__':
#     datas=[]
#     for param[0] in range(0,700000):
#         a = np.random.random((2,))
#         datas.append(a)
#     t1=time.time()
#     for e in tqdm(datas):
#         doSomething([e,datas])
#     t2=time.time()
#     print('t2-t1:%4f'%(t2-t1))
#     param=[]
#     for ele in tqdm(datas,desc='param:'):
#         t=(ele,datas)
#         param.append(t)
#     print('*'*10)
 
#     p=multiprocessing.Pool(4)
#     b = p.map(doSomething, param)
 
 
#     t1 = time.time()
#     b=p.map(do,param)
#     p.close()
#     p.join()
#     t2 = time.time()
#     print('t2-t1:%4f' % (t2 - t1))


# def single_channel_slide_window_parameters(gray_img_a,gray_img_b,  func , step = 8, size_w = 40, size_h = 40, *args, **kwargs):
#     w = gray_img_a.shape[0]
#     h = gray_img_a.shape[1]
#     if((w%size_w!=0)or(h%size_h!=0)):
#         print('Please check slide window SIZE!')
#         return
#     ans_a = []
#     ans_b = []   
#     #!
#     start_time = time.perf_counter() 
#     signal_single = False
#     signal_inside = False
#     for i in range(int(w/step)):
#         raw_a = []
#         raw_b = []
#         #!
#         start_time_inside = time.perf_counter() 
#         for j in range(int(h/step)): 
#             if(i*step+size_w>w)or(j*step+size_h>h):break   
#             start_time_single = time.perf_counter()   
#             # print('submat shape is {shape}'.format(shape=np.shape(gray_img_a[i*step:(i*step+size_w), j*step:(j*step+size_h)])))     
#             raw_a.append(func(gray_img_a[i*step:(i*step+size_w), j*step:(j*step+size_h)], *args, **kwargs))
#             end_time_single = time.perf_counter() 
#             if(signal_single==False):
#                 print('单指令共运行了 {_time_}秒'.format(_time_=(end_time_single - start_time_single)))
#                 signal_single=True
#             raw_b.append(func(gray_img_b[i*step:(i*step+size_w), j*step:(j*step+size_h)], *args, **kwargs))
#         #!
#         end_time_inside = time.perf_counter() 
#         if(signal_inside==False):
#             print('内循环共运行了 {_time_}秒'.format(_time_=(end_time_inside - start_time_inside)))
#             signal_inside=True
#         if(raw_a!=[]):ans_a.append(raw_a)
#         if(raw_b!=[]):ans_b.append(raw_b) 
#     #!
#     end_time = time.perf_counter()  
#     print('循环共运行了 {_time_}秒'.format(_time_=(end_time - start_time)))
#     ans_a = np.array(ans_a)
#     ans_b = np.array(ans_b)
#     ans = np.abs(ans_a-ans_b)
#     # print(np.shape(ans))
#     # print(ans_a)
#     ans = ans.transpose(2,0,1)
#     ans = (255*ans) / np.max(ans)    
#     return ans




# def __fakecolor_texture_characteristics_tamura_feature(self):
#     kmax = 3    
#     dist = 4
#     tamura_label = ['coarseness', 'contrast', 'directionality', 'linelikeness']
#     # tamura_label = ['contrast']
#     for i in range(3):
#         ans = single_channel_slide_window_parameters(self.matrix_a[i], self.matrix_b[i], tc.tamura_feature, self.step, self.size_w, self.size_h, kmax=kmax, dist=dist)
        
#         for j in range(ans.shape[0]):
#             path = self.folder + 'Texture_TamuraFeature_'+RGB_COLOR_CHANNEL.get(i) +'_'+tamura_label[j]+'.jpg'
#             ans_highsolution = cv2.resize(ans[j], None, fx=self.step, fy=self.step, interpolation=cv2.INTER_LINEAR)
#             # ans_highsolution = ans_highsolution.astype(np.uint8)
#             # ans_fakecolor = cv2.applyColorMap(ans_highsolution, cv2.COLORMAP_JET)
#             # cv2.imwrite(path, ans_fakecolor)
#             print(path)
#             plt.figure(figsize=self.figsize)
#             plt.imshow(ans_highsolution,vmin = 0, vmax = 255,cmap = "hot")
#             plt.colorbar()
#             plt.savefig(path)
#             plt.close()
#     return

# def __multithread_fakecolor_texture_characteristics_tamura_feature(self):
#     pool = multiprocessing.Pool(processes=4)
#     kmax = 3    
#     dist = 4
#     tamura_label = ['coarseness', 'contrast', 'directionality', 'linelikeness']
#     for i in range(3):
#         # param = [self.matrix_a[i], self.matrix_b[i], tc.tamura_feature, self.step, self.size_w, self.size_h, self.folder,kmax, dist]
#         param = [self, i, tamura_label, tc.tamura_feature,kmax, dist]
#         pool.apply_async(multithread_single_channel_slide_window_parameters, (param, ))
#     pool.close()
#     pool.join()


# def multithread_single_channel_slide_window_parameters(param):
#     w = param[0].matrix_a[param[1]].shape[0]
#     h = param[0].matrix_a[param[1]].shape[1]
#     if((w%param[0].size_w!=0)or(h%param[0].size_h!=0)):
#         print('Please check slide window SIZE!')
#         return
#     ans_a = []
#     ans_b = []   
#     #!
#     start_time = time.perf_counter() 
#     signal_single = False
#     signal_inside = False
#     for i in range(int(w/param[0].step)):
#         raw_a = []
#         raw_b = []
#         #!
#         start_time_inside = time.perf_counter() 
#         for j in range(int(h/param[0].step)): 
#             if(i*param[0].step+param[0].size_w>w)or(j*param[0].step+param[0].size_h>h):break   
#             start_time_single = time.perf_counter()   
#             # print('submat shape is {shape}'.format(shape=np.shape(param[0].matrix_a[param[1]][i*param[0].step:(i*param[0].step+param[0].size_w), j*param[0].step:(j*param[0].step+param[0].size_h)])))     
#             raw_a.append(param[3](param[0].matrix_a[param[1]][i*param[0].step:(i*param[0].step+param[0].size_w), j*param[0].step:(j*param[0].step+param[0].size_h)], *args, **kwargs))
#             end_time_single = time.perf_counter() 
#             if(signal_single==False):
#                 print('单指令共运行了 {_time_}秒'.format(_time_=(end_time_single - start_time_single)))
#                 signal_single=True
#             raw_b.append(param[3](param[0].matrix_b[param[1]][i*param[0].step:(i*param[0].step+param[0].size_w), j*param[0].step:(j*param[0].step+param[0].size_h)], *args, **kwargs))
#         #!
#         end_time_inside = time.perf_counter() 
#         if(signal_inside==False):
#             print('内循环共运行了 {_time_}秒'.format(_time_=(end_time_inside - start_time_inside)))
#             signal_inside=True
#         if(raw_a!=[]):ans_a.append(raw_a)
#         if(raw_b!=[]):ans_b.append(raw_b) 
#     #!
#     end_time = time.perf_counter()  
#     print('循环共运行了 {_time_}秒'.format(_time_=(end_time - start_time)))
#     ans_a = np.array(ans_a)
#     ans_b = np.array(ans_b)
#     ans = np.abs(ans_a-ans_b)
#     # print(np.shape(ans))
#     # print(ans_a)
#     ans = ans.transpose(2,0,1)
#     ans = (255*ans) / np.max(ans)    
#     for j in range(ans.shape[0]):
#             path = param[0].folder + 'Texture_TamuraFeature_'+RGB_COLOR_CHANNEL.get(param[1]) +'_'+param[2][j]+'.jpg'
#             ans_highsolution = cv2.resize(ans[j], None, fx=param[0].param[0].step, fy=param[0].param[0].step, interpolation=cv2.INTER_LINEAR)
#             # ans_highsolution = ans_highsolution.astype(np.uint8)
#             # ans_fakecolor = cv2.applyColorMap(ans_highsolution, cv2.COLORMAP_JET)
#             # cv2.imwrite(path, ans_fakecolor)
#             print(path)
#             plt.figure(figsize=param[0].figsize)
#             plt.imshow(ans_highsolution,vmin = 0, vmax = 255,cmap = "hot")
#             plt.colorbar()
#             plt.savefig(path)
#             plt.close()




def multithread_temurafeture_single_channel_slide_window_parameters( gray_img_a, gray_img_b, step, size_w, size_h, folder, figsize):
    w = gray_img_a.shape[0]
    h = gray_img_a.shape[1]
    tamura_label = ['coarseness', 'contrast', 'directionality', 'linelikeness']
    kmax = 3    
    dist = 4
    if((w%size_w!=0)or(h%size_h!=0)):
        print('Please check slide window SIZE!')
        return
    ans_a = []
    ans_b = []   

    for i in range(int(w/step)):

        pool = multiprocessing.Pool()
        func = partial(temura_inside, kmax = kmax, dist = dist, step = step, w = w, h = h,size_w = size_w, size_h = size_h, gray_img_a = gray_img_a, gray_img_b = gray_img_b, i = i)
        raw= pool.map(func, range(int(h/step)))
        # for j in range(int(h/step)): 
        #     if(i*step+size_w>w)or(j*step+size_h>h):break   

            # print('submat shape is {shape}'.format(shape=np.shape(gray_img_a[i*step:(i*step+size_w), j*step:(j*step+size_h)])))     
            # raw_a.append(tc.tamura_feature(gray_img_a[i*step:(i*step+size_w), j*step:(j*step+size_h)], kmax, dist))

           
            # raw_b.append(tc.tamura_feature(gray_img_b[i*step:(i*step+size_w), j*step:(j*step+size_h)], kmax, dist))
        # print(raw)
        raw_a = raw[0]
        raw_b = raw[1]

        if(raw_a!=[]):ans_a.append(raw_a)
        if(raw_b!=[]):ans_b.append(raw_b) 

    ans_a = np.array(ans_a)
    ans_b = np.array(ans_b)
    ans = np.abs(ans_a-ans_b)
    # print(np.shape(ans))
    # print(ans_a)
    ans = ans.transpose(2,0,1)
    
    


def temura_inside(j, kmax, dist, step, w,h,size_w, size_h, gray_img_a, gray_img_b, i):
    if(i*step+size_w>w)or(j*step+size_h>h):return 0,0
    raw_a = tc.tamura_feature(gray_img_a[i*step:(i*step+size_w), j*step:(j*step+size_h)], kmax, dist)
    raw_b = tc.tamura_feature(gray_img_b[i*step:(i*step+size_w), j*step:(j*step+size_h)], kmax, dist)
    return raw_a, raw_b
def get_img(path):
    img = cv2.imread(path)
    img_r,img_g,img_b = cv2.split(img)
    return img,img_r,img_g,img_b
    



if __name__=="__main__":
    # for i in tqdm.trange(10):
    #     for j in range(10):
    #         time.sleep(0.25)
    # start_time = time.perf_counter()
    # print('Start!')
    # path_a = 'graduate_design/Data/Normal_Changed.jpg'
    # path_b = 'graduate_design/Data/Normal_Unchanged.jpg'
    # img_a = cv2.imread(path_a)
    # img_b = cv2.imread(path_b)    



    # matrix_a =  cv2.split(img_a)
    # matrix_b =  cv2.split(img_b)
    # print(np.shape(matrix_a))
    # print(np.shape(matrix_b))
    
    # step = 8
    # size_w = 40 
    # size_h = 40
    # figsize = (18,10)
    # folder = '/home/lyh/results/MultiThreadTest/'
    # if(not os.path.exists(folder)):
    #     os.makedirs(folder)
    #     print("New Folder Created!")

    # multithread_temurafeture_single_channel_slide_window_parameters(matrix_a[1], matrix_b[1], step, size_w, size_h, folder, figsize)

    
    a = list([1,3,4,5,6])
    b = list([a,a])
    c = list([b,b])
    print(b)
    print(np.shape(b))
    b = list(map(list, zip(*b)))
    print(np.shape(b))
    print(b)
    f = b[0]
    print(f)
    c = np.array(c)
    d = np.zeros(4)
    d[0]=1
    d[1]=2
    # print(d)
    
# if __name__=="__main__":
#     c = 20
#     pool = multiprocessing.Pool()
#     func = partial(testinside, d = c)
#     res = pool.map(func,range(20))
#     print(res)