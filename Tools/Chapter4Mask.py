import os
import cv2

mask_path = '/home/lyh/Chapter4Experiment/4_3Exp/NewNetRes/Pic/mask.png'
folder = '/home/lyh/Chapter4Experiment/res/H5pyGenFolder/00002/o_p1'
# /Saturation+45
pics = os.listdir(folder)
for p in pics:
    p = os.path.join(folder, p)
    img = cv2.imread(p)
    h = img.shape[0]
    w = img.shape[1]
    mask = cv2.imread(mask_path)
    mask = cv2.resize(mask, (w,h))
    mask = mask/255
    img = img*mask
    cv2.imwrite(p, img)
    