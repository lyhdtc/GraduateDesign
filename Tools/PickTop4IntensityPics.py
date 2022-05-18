import os
import cv2
import numpy as np

def by_value(t):
    return(-t[1])


folder = '/home/lyh/Chapter4Experiment/4_3Exp/NewCal_Data/Face/albedo'
f_list = os.listdir(folder)
data_list = []


for f in f_list:
    file = os.path.join(folder, f)
    img = cv2.imread(file)
    value = np.mean(img)
    t = (f, value)
    data_list.append(t)
    
    
ans_list = sorted(data_list, key=by_value)
for i in range(4):
    print(ans_list[i][0])