import os
from cv2 import sort, warpAffine
from matplotlib import pyplot as plt
import cv2
import numpy as np

O_or_Face = ['O', 'Face']
folder_list = ['albedo', 'normal', 'origin', 'roughness']

'''下面是O和Face两个文件夹，可能还有别的文件，不过不影响'''
base_folder = '/home/lyh/Chapter4Experiment/4_3Exp/NewCal_Data'

for oof in O_or_Face:
    for fl in folder_list:
        folder = os.path.join(base_folder, oof)
        folder = os.path.join(folder, fl)
        save_folder = os.path.join(base_folder, oof)
        save_folder = os.path.join(save_folder, 'Total_Pics')
        save_folder = os.path.join(save_folder, fl)
        
        img_list = os.listdir(folder)
        img_list = sorted(img_list)
        img_num = len(img_list)
        print(img_num)
        base = -1
        count = 0
        # save_folder = '/home/lyh/Chapter4Experiment/4_3Exp/NewCal_Data/O/Total_Pics/roughness'
        for n,i in enumerate(img_list):
            
            if n%35==0 or n==img_num-1:
                if n==0:
                    plt.figure(figsize=(10.8,19.2))
                    #    pass
                else:
                    # plt.show()
                    save_path = os.path.join(save_folder, str(count)+'.jpg')
                    plt.subplots_adjust(left = 0.05,right=0.95,bottom=0,top=0.95,hspace=0.1,wspace=0.1)
                    plt.savefig(save_path)
                    plt.close()
                    count = count+1
                    base = n-1
                    plt.figure(figsize=(10.8,19.2))
                    
            img_path = os.path.join(folder, i)
            img = cv2.imread(img_path)
            p = plt.subplot(7,5,n-base)
            p.axis('off')
            p.imshow(img)
            label = i[:-4]
            label = label.replace('_','\n',2 )
            p.set_title(label, fontsize=16)
