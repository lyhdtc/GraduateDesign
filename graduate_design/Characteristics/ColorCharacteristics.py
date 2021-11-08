import cv2
import numpy as np

color_channel = {
    "red"  : 0,
    "green": 1,
    "blue" : 2
}

#直方图 
def _histogram(img,channel):
    color = color_channel.get(channel,0)
    hist = cv2.calcHist([img],[color],None,[256],[0,256])
    return hist
