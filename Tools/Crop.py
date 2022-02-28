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

# TODO: Unfininshed
def bounding_box(mask):
    
    th, binary = cv2.threshold(mask, 0, 255, cv2.THRESH_OTSU)
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    
    bounding_boxes = [cv2.boundingRect(cnt) for cnt in contours]
    
    print(bounding_boxes)
