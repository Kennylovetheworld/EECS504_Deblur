# -*- coding: utf-8 -*-
"""
Created on Tue Nov 24 12:06:30 2020

@author: sheng
"""
from __future__ import print_function, division
import random
import os
import shutil
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torchvision.transforms.functional as TF

class MyDeblurDataset(Dataset):#root_dir like: "./GOPRO_Large/train/"
    def __init__(self, root_path):
        """
        Args:
            root_dir (string): Directory with all the images.
        """
        self.root_path = root_path
    def __len__(self):
        blur_dir = self.root_path + "blur/"
        image = os.listdir(blur_dir)
        return len(image)
    
    def transform(self, image, mask, of1, of2):
        # Transform to tensor
        image = TF.to_tensor(image)
        mask = TF.to_tensor(mask)
        
        of1 = torch.Tensor(of1)
        of2 = torch.Tensor(of2)
        of1 = of1.permute(2,0,1)
        of2 = of2.permute(2,0,1)
        
        # Random horizontal flipping
        if random.random() > 0.5:
            image = TF.hflip(image)
            mask = TF.hflip(mask)
            of1 = TF.hflip(of1)
            of2 = TF.hflip(of2)

        # Random vertical flipping
        if random.random() > 0.5:
            image = TF.vflip(image)
            mask = TF.vflip(mask)
            of1 = TF.vflip(of1)
            of2 = TF.vflip(of2)
        
        # Random rotation
        angle = transforms.RandomRotation.get_params([-180, 180])
        image = TF.rotate(image, angle)   #pytorch:1.7.0, torchvision:0.8.1
        mask = TF.rotate(mask, angle)
        of1 = TF.rotate(of1, angle)
        of2 = TF.rotate(of2, angle)
        
        # Random crop
        i, j, h, w = transforms.RandomCrop.get_params(
            image, output_size=(256, 256))
        image = TF.crop(image, i, j, h, w)
        mask = TF.crop(mask, i, j, h, w)
        of1 = TF.crop(of1, i, j, h, w)
        of2 = TF.crop(of2, i, j, h, w)

        of1 = of1.permute(1,2,0)
        of2 = of2.permute(1,2,0)
        of1 = of1.numpy()
        of2 = of2.numpy()
        
        return image, mask, of1, of2

    def __getitem__(self, idx):
        blur_dir = self.root_path + "blur/"
        sharp_dir = self.root_path + "sharp/"
        opticflow_dir = self.root_path + "opticflow/"
        image = os.listdir(blur_dir)
        blur_image = io.imread(blur_dir + image[idx])
        sharp_image = io.imread(sharp_dir + image[idx])
        opticalflow_path_1 = image[idx][:-3] + 'flo'
        image_string = image[idx][:-3]
        index1 = image_string.find('e')
        index2 = image_string.find('.')
        number = int(image_string[index1+1:index2])
        opticalflow_path_2 = image_string[:index1+1] + str(number+1) + '.flo'
        opticalflow_1 = cv2.readOpticalFlow(opticflow_dir + opticalflow_path_1)
        opticalflow_2 = cv2.readOpticalFlow(opticflow_dir + opticalflow_path_2)
        """
        fig2 =plt.figure()
        fig2.add_subplot(3,1,1)
        plt.imshow(sharp_image)
        fig2.add_subplot(3,1,2)
        plt.imshow(draw_hsv(opticalflow_1))
        fig2.add_subplot(3,1,3)
        plt.imshow(draw_hsv(opticalflow_2))
        plt.show()
        """
        print("idx:", idx)
        print(blur_dir + image[idx])
        print(sharp_dir + image[idx])
        print(opticalflow_path_1)
        print(opticalflow_path_2)
        
        blur_image_transformed, sharp_image_transformed, of1, of2 = self.transform(blur_image,sharp_image,opticalflow_1,opticalflow_2)
            
        return blur_image_transformed, sharp_image_transformed, of1, of2
    
    
def ChangeDataset(root_path):# root_path like: "./GOPRO_Large/train/"
    blur_dir = root_path + "blur/"
    sharp_dir = root_path + "sharp/"
    opticflow_dir = root_path + "opticflow/"
    if (not os.path.isdir(blur_dir)):
        try:
            os.mkdir(root_path + "blur/")
        except OSError:
            print ("Creation of the directory %s failed" % (root_path + "blur/"))
        else:
            print ("Successfully created the directory %s " % (root_path + "blur/"))
    else:
        print ("Already created the directory %s " % (root_path + "blur/"))
    if (not os.path.isdir(sharp_dir)):
        try:
            os.mkdir(root_path + "sharp/")
        except OSError:
            print ("Creation of the directory %s failed" % (root_path + "sharp/"))
        else:
            print ("Successfully created the directory %s " % (root_path + "sharp/"))
    else:
        print ("Already created the directory %s " % (root_path + "sharp/"))
    if (not os.path.isdir(opticflow_dir)):
        try:
            os.mkdir(root_path + "opticflow/")
        except OSError:
            print ("Creation of the directory %s failed" % (root_path + "opticflow/"))
        else:
            print ("Successfully created the directory %s " % (root_path + "opticflow/"))
    else:
        print ("Already created the directory %s " % (root_path + "opticflow/"))
        
    dirs = os.listdir(root_path)
    v=1
    for dir in dirs:
        i=1
        if (dir != 'blur' and dir != 'sharp' and dir!= 'opticflow'):
            blur_image_path = root_path + dir + '/blur/'
            sharp_image_path = root_path + dir + '/sharp/'
            image_path = os.listdir(blur_image_path)
            for image in image_path:
                if (image != image_path[0] and image != image_path[-1]):
                    shutil.copyfile(blur_image_path + image, blur_dir + 'v' + str(v) + '_image' + str(i) + '.png')
                    shutil.copyfile(sharp_image_path + image, sharp_dir + 'v' + str(v) + '_image' + str(i) + '.png')
                    i += 1
            print("copy ",i," training images to blur and sharp folders from video ",v)
            for j in range(len(image_path) - 1):
                img1 = io.imread(sharp_image_path + image_path[j])
                img2 = io.imread(sharp_image_path + image_path[j+1])
                opticflow = cal_opticflow(img1, img2)
                cv2.writeOpticalFlow(opticflow_dir + 'v' + str(v) + '_image' + str(j+1) + '.flo', opticflow)
            print("caculate optical flow to optical flow folder from video ",v)
            v += 1
        
        
def RemoveAllFile(root_path):# root_path like: "./GOPRO_Large/train/"
    blur_dir = root_path + "blur/"
    sharp_dir = root_path + "sharp/"
    opticflow_dir = root_path + "opticflow/"
    for image in os.listdir(blur_dir):
        os.remove(blur_dir + image)
    for image in os.listdir(sharp_dir):
        os.remove(sharp_dir + image)
    for opticalflow in os.listdir(opticflow_dir):
        os.remove(opticflow_dir + opticalflow)
        
def cal_opticflow(img1, img2):
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # cv2 version: 4.4
    inst = cv2.DISOpticalFlow.create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    flow = inst.calc(gray1, gray2, None)#flow:(720,1280,2)
    return flow

def draw_hsv(flow):
    h, w = flow.shape[:2]
    fx, fy = flow[:,:,0], flow[:,:,1]
    ang = np.arctan2(fy, fx) + np.pi
    v = np.sqrt(fx*fx+fy*fy)
    hsv = np.zeros((h, w, 3), np.uint8)
    hsv[...,0] = ang*(180/np.pi/2)
    hsv[...,1] = 255
    hsv[...,2] = np.minimum(v*4, 255)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return bgr

def plot_dataset_item(image, mask, of1, of2):
    image = (255 * image).type(torch.IntTensor)
    mask = (255 * mask).type(torch.IntTensor)
    
    image = (image.permute(1, 2, 0)).numpy()
    mask = (mask.permute(1, 2, 0)).numpy()
    #of1 = (of1.permute(1, 2, 0)).numpy()
    #of2 = (of2.permute(1, 2, 0)).numpy()
    
    fig=plt.figure()
    
    fig.add_subplot(2,2,1)
    plt.imshow(image)
    fig.add_subplot(2,2,2)
    plt.imshow(mask)
    fig.add_subplot(2,2,3)
    plt.imshow(draw_hsv(of1))
    fig.add_subplot(2,2,4)
    plt.imshow(draw_hsv(of2))
    
    plt.show()
    
if __name__ == '__main__':
    
    dataset = MyDeblurDataset(root_path = './GOPRO_Large/train/')
    for i in range(len(dataset)):
        blur_image,sharp_image,of1,of2 = dataset[i]
    