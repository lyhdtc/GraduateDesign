import numpy as np
import cv2
import random
from Tools import Noise

def experiment1_transform(img):
    h,w,c = np.shape(img)
    delta_h = (int)(h*0.005)
    delta_w = (int)(w*0.005)
    dir_index = random.randint(0,3)
    dir_matrix = np.float32([[[1,0,delta_w],[0,1,0]       ],
                           [[1,0,0]      ,[0,1,delta_h] ],
                           [[1,0,-delta_w],[0,1,0]      ],
                           [[1,0,0],[0,1,-delta_h]      ]                           
                        ])
    ans = cv2.warpAffine(img, dir_matrix[dir_index],(w,h))
    return ans
    
    
def experiment2_transform(img):
    h,w,c = np.shape(img)
    center = (w/2, h/2)
    angle_matrix = np.array([1,-1])
    angle_index = random.randint(0,1)
    M = cv2.getRotationMatrix2D(center, angle_matrix[angle_index], 1)
    ans = cv2.warpAffine(img, M, (w,h))
    return ans


def experiment3_transform(img,scale):
    ans = cv2.resize(img, None, fx = scale, fy=scale, interpolation=cv2.INTER_LINEAR)
    return ans

def experiment4_transform(img, amount):
    img = cv2.cvtColor(img, cv2.COLOR_LAB2BGR)
    ans = Noise.traditional_noise(img, 's&p', amount=amount)
    ans = cv2.cvtColor(ans, cv2.COLOR_BGR2LAB)
    return ans

def experiment5_transform(img_data, img_noise, percent):
    img_data = cv2.cvtColor(img_data, cv2.COLOR_LAB2BGR)
    img_noise = cv2.cvtColor(img_noise, cv2.COLOR_LAB2BGR)

    ans = Noise.random_replace_element_from_another_picture_bgr(img_data, img_noise, percent)
    ans = cv2.cvtColor(ans, cv2.COLOR_BGR2LAB)
    return ans

def experiment6_transform(img):
    img = cv2.cvtColor(img, cv2.COLOR_LAB2BGR)
    b,g,r = cv2.split(img)
    b1,g1,r1 = b,g,r
    b1[::2,] = b[1::2,]
    g1[::2,] = g[1::2,]
    r1[::2,] = r[1::2,]
    b1[1::2,] = b[::2,]
    g1[1::2,] = g[::2,]
    r1[1::2,] = r[::2,]
    ans = cv2.merge((b1,g1,r1))
    ans = ans.astype(np.uint8)
    ans = cv2.cvtColor(ans, cv2.COLOR_BGR2LAB)
    return ans
    