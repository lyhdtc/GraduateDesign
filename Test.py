import os
import sys
sys.path.append(os.pardir)
from Color import ColorAlgorithrm as CA
from Tools import ColorSpaceTrans as CST
import cv2
import numpy as np

path = 'Data/批注 2022-03-01 153307.jpg'
path2 = 'Data/bbb.jpg'

bgr_img = cv2.imread(path)
bgr_img2 = cv2.imread(path2)

# print(CA.brightness(bgr_img))
# print(CA.constract(bgr_img))
# print(8**(1/3))
# print(np.log(2))

ans = CA.specular_shadow(bgr_img,0.3,'shadow')
print(ans)
ans = ans*255
ans = ans.astype(np.uint8)
cv2.imshow('asdf',ans)
cv2.waitKey(0)

a = np.array((3,2,1,0,2,1,1,0))

c = np.where(a<2, 1,0)
print(c)