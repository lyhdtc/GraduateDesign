import cv2
import numpy as np
from matplotlib import pyplot as plt

def create_mask(color_img):
    res = np.zeros(np.shape(color_img))
    res[np.where(color_img!=0)] = 1
    print(np.shape(res))
    mask = np.multiply(res[:,:,0], res[:,:,1])
    mask = np.multiply(mask, res[:,:,2])

    return mask


def use_mask(color_img, mask):
    for i in range(3):
        color_img[:,:,i] = np.multiply(color_img[:,:,i], mask)
    return color_img