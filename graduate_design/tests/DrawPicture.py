import sys
sys.path.append('/mnt/d/GraduateDesign/graduate_design/Characteristics')
import cv2
import numpy as np
from matplotlib import pyplot as plt
import ColorCharacteristics as cc

def _drawpicture(hist_r,hist_g,hist_b):
    plt.plot(hist_r,'r'),  
    plt.plot(hist_g,'g'),
    plt.plot(hist_b,'b'),    
    plt.xlim([0,256]),
    plt.title('test'),
    plt.show()
    
    
    
img = cv2.imread('graduate_design/Data/R-C.jpg')


hist_r = cc._histogram(img,"red")
hist_g = cc._histogram(img,"green")
hist_b = cc._histogram(img,"blue")
print(np.sum(hist_r))
_drawpicture(hist_r,hist_g,hist_b)

