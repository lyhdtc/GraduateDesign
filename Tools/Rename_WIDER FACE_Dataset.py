import os
import shutil
import tqdm
import cv2


def run():
    folder_newpath = '/mnt/d/GraduateDesign2/WIDER_Data_lyh/'
    folder_root = '/mnt/d/GraduateDesign2/WIDER_train/images/'

    folder_varible = os.listdir(folder_root)
    # print(folder_varible)
    num=0
    for i in folder_varible:
        folder_img = os.path.join(folder_root, i)
        # print(folder_img)
        img_list = os.listdir(folder_img)
        for img_name in img_list:
            img_path = os.path.join(folder_img, img_name)
            
            
            new_name = str(num)+'.jpg'
            

            new_path = os.path.join(folder_newpath, new_name)
            img = cv2.imread(img_path)
            img = cv2.resize(img, (150,150), interpolation=cv2.INTER_LINEAR)
            cv2.imwrite(new_path, img)
            # shutil.copy(img_path, new_path)
            num = num+1
            if(num==1000):return
            print(num)
run()