# -*- coding: utf-8 -*-
"""
Created on Sat Dec  5 12:41:31 2020

@author: sheng
"""
import cv2
import os.path
from skimage import io


def calculate_opticalflow(data_dir):
    sub_folders = os.listdir(data_dir)
    
    inst = cv2.DISOpticalFlow.create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    
    for folder_name in sub_folders:
        
        opticalflow_sub_folder = os.path.join(data_dir, folder_name, 'opticalflow')
        if not os.path.exists(opticalflow_sub_folder):
            os.makedirs(opticalflow_sub_folder)
            
        sharp_sub_folder = os.path.join(data_dir, folder_name, 'sharp')
        sharp_file_names = os.listdir(sharp_sub_folder)
        
        for i in range(len(sharp_file_names) - 1):
            img1 = io.imread(os.path.join(sharp_sub_folder, sharp_file_names[i]))
            img2 = io.imread(os.path.join(sharp_sub_folder, sharp_file_names[i+1]))
            opticflow = cal_opticflow(img1, img2, inst)
            cv2.writeOpticalFlow(os.path.join(opticalflow_sub_folder, sharp_file_names[i+1].replace("png","flo")), opticflow)
            
        print(folder_name, ": Complete calculating ", len(sharp_file_names)-1, " opticalflow")
    print("Complete")

def cal_opticflow(img1, img2, inst):
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    flow = inst.calc(gray1, gray2, None)#flow:(720,1280,2)
    return flow
        
def delete_opticalflow(data_dir):
    sub_folders = os.listdir(data_dir)
    for folder_name in sub_folders:
        opticalflow_sub_folder = os.path.join(data_dir, folder_name, 'opticalflow')
        for file in os.listdir(opticalflow_sub_folder):
            os.remove(os.path.join(opticalflow_sub_folder, file))
            
            
if __name__ == '__main__':
    data_dir = '../../dataset/train/'
    calculate_opticalflow(data_dir)
        