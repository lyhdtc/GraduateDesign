import os
import cv2
import numpy as np
from torch import float64

path_O = '/home/lyh/Chapter4Experiment/4_3Exp/NewCal_Data/O/op0_00002g.png'
path_Face = '/home/lyh/Chapter4Experiment/4_3Exp/NewCal_Data/Face/op0_00002g.png'
path_abs = '/home/lyh/Chapter4Experiment/4_3Exp/NewCal_Data/roughness_abs.png'
mask_path = '/home/lyh/Chapter4Experiment/4_3Exp/NewNetRes/Pic/mask.png'


o,a,b= cv2.split(cv2.imread(path_O))
# print(type(o))
o = o.astype(np.float64)
face,a,b = cv2.split(cv2.imread(path_Face))
face = face.astype(np.float64)

ans = o-face
ans = (ans-ans.min())
# 
# cv2.colormap

h = ans.shape[0]
w = ans.shape[1]
mask = cv2.imread(mask_path)
mask = cv2.resize(mask, (w,h))
mask = cv2.split(mask)[0]
mask = mask/255
ans = ans*mask
ans = np.uint8(ans)
ans = cv2.applyColorMap(ans, cv2.COLORMAP_JET)
cv2.imwrite(path_abs, ans)