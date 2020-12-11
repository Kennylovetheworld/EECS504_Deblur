# -*- coding: utf-8 -*-
"""
Created on Sat Dec  5 11:45:18 2020

@author: sheng
"""

import os.path
import random
import torch.utils.data as data
from torch.utils import *
import torch
from PIL import Image
from torchvision import transforms
import torchvision.transforms.functional as TF
import numpy as np
from skimage import io
import cv2

"""
def augment(img_input, img_target):
    degree = random.choice([0, 90, 180, 270])
    img_input = transforms.functional.rotate(img_input, degree)
    img_target = transforms.functional.rotate(img_target, degree)

    # color augmentation
    img_input = transforms.functional.adjust_gamma(img_input, 1)
    img_target = transforms.functional.adjust_gamma(img_target, 1)
    sat_factor = 1 + (0.2 - 0.4 * np.random.rand())
    img_input = transforms.functional.adjust_saturation(img_input, sat_factor)
    img_target = transforms.functional.adjust_saturation(img_target, sat_factor)

    return img_input, img_target


def getPatch(img_input, img_target, opticalflow_1, opticalflow_2, path_size):
    w, h = img_input.size
    p = path_size
    x = random.randrange(0, w - p + 1)
    y = random.randrange(0, h - p + 1)
    img_input = img_input.crop((x, y, x + p, y + p))
    img_target = img_target.crop((x, y, x + p, y + p))
    opticalflow_1 = img_target.crop((x, y, x + p, y + p))
    opticalflow_2 = img_target.crop((x, y, x + p, y + p))
    return img_input, img_target, opticalflow_1, opticalflow_2
"""

def transform(img_input, img_target, opticalflow_1, opticalflow_2, patchsize):
    opticalflow_1 = opticalflow_1.permute(2,0,1)
    opticalflow_2 = opticalflow_2.permute(2,0,1)
    
    # Random horizontal flipping
    if random.random() > 0.5:
        img_input = TF.hflip(img_input)
        img_target = TF.hflip(img_target)
        opticalflow_1 = TF.hflip(opticalflow_1)
        opticalflow_2 = TF.hflip(opticalflow_2)

    # Random vertical flipping
    if random.random() > 0.5:
        img_input = TF.vflip(img_input)
        img_target = TF.vflip(img_target)
        opticalflow_1 = TF.vflip(opticalflow_1)
        opticalflow_2 = TF.vflip(opticalflow_2)
        
    # Random rotation
    angle = transforms.RandomRotation.get_params([-180, 180])
    img_input = TF.rotate(img_input, angle)   #pytorch:1.7.0, torchvision:0.8.1
    img_target = TF.rotate(img_target, angle)
    opticalflow_1 = TF.rotate(opticalflow_1, angle)
    opticalflow_2 = TF.rotate(opticalflow_2, angle)
    
    # Random crop
    i, j, h, w = transforms.RandomCrop.get_params(
        img_input, output_size=(patchsize, patchsize))
    img_input = TF.crop(img_input, i, j, h, w)
    img_target = TF.crop(img_target, i, j, h, w)
    opticalflow_1 = TF.crop(opticalflow_1, i, j, h, w)
    opticalflow_2 = TF.crop(opticalflow_2, i, j, h, w)
    
    return img_input, img_target, opticalflow_1, opticalflow_2
        
class Gopro(data.Dataset):
    def __init__(self, data_dir, patch_size=256): #is_train=False, multi=True):
        super(Gopro, self).__init__()
        #self.is_train = is_train
        self.patch_size = patch_size
        #self.multi = multi

        self.sharp_file_paths = []

        sub_folders = os.listdir(data_dir)

        for folder_name in sub_folders:
            sharp_sub_folder = os.path.join(data_dir, folder_name, 'sharp')
            sharp_file_names = os.listdir(sharp_sub_folder)

            for file_name in sharp_file_names:
                if(sharp_file_names.index(file_name)!=0 and sharp_file_names.index(file_name)!=len(sharp_file_names)-1):
                    sharp_file_path = os.path.join(sharp_sub_folder, file_name)
                    self.sharp_file_paths.append(sharp_file_path)

        self.n_samples = len(self.sharp_file_paths)

    def get_img_pair_and_opticalflow(self, idx):
        sharp_file_path = self.sharp_file_paths[idx]
        blur_file_path = sharp_file_path.replace("sharp", "blur")
        opticalflow_file_path = sharp_file_path.replace("sharp", "opticalflow")
        opticalflow_file_path = opticalflow_file_path.replace("png", "flo")

        img_input = Image.open(blur_file_path).convert('RGB')
        img_target = Image.open(sharp_file_path).convert('RGB')
        opticalflow_1 = cv2.readOpticalFlow(opticalflow_file_path)
        
        opticalflow_folder = os.path.dirname(opticalflow_file_path)
        flow1_name = os.path.basename(opticalflow_file_path)
        files = os.listdir(opticalflow_folder)
        index = files.index(flow1_name)
        
        opticalflow_2 = cv2.readOpticalFlow(os.path.join(opticalflow_folder, files[index+1]))
            
        return img_input, img_target, opticalflow_1, opticalflow_2

    def __getitem__(self, idx):
        img_input, img_target, opticalflow_1, opticalflow_2 = self.get_img_pair_and_opticalflow(idx)
            
        """
        if self.is_train:
            img_input, img_target, opticalflow_1, opticalflow_2 = getPatch(
                img_input, img_target, opticalflow_1, opticalflow_2, self.patch_size)
            img_input, img_target = augment(img_input, img_target)
        """
        img_input = transforms.ToTensor()(img_input)
        img_target = transforms.ToTensor()(img_target)
        opticalflow_1 = torch.Tensor(opticalflow_1)
        opticalflow_2 = torch.Tensor(opticalflow_2)
        
        img_input, img_target, opticalflow_1, opticalflow_2 = transform(
            img_input, img_target, opticalflow_1, opticalflow_2, self.patch_size)
        """
        H = input_b1.size()[1]
        W = input_b1.size()[2]
       
        if self.multi:
            input_b1 = transforms.ToPILImage()(input_b1)
            target_s1 = transforms.ToPILImage()(target_s1)

            input_b2 = transforms.ToTensor()(transforms.Resize([int(H / 2), int(W / 2)])(input_b1))
            input_b3 = transforms.ToTensor()(transforms.Resize([int(H / 4), int(W / 4)])(input_b1))

            if self.is_train:
                target_s2 = transforms.ToTensor()(transforms.Resize([int(H / 2), int(W / 2)])(target_s1))
                target_s3 = transforms.ToTensor()(transforms.Resize([int(H / 4), int(W / 4)])(target_s1))
            else:
                target_s2 = []
                target_s3 = []

            input_b1 = transforms.ToTensor()(input_b1)
            target_s1 = transforms.ToTensor()(target_s1)
            return {'input_b1': input_b1, 'input_b2': input_b2, 'input_b3': input_b3,
                    'target_s1': target_s1, 'target_s2': target_s2, 'target_s3': target_s3}
        else:
            return {'input_b1': input_b1, 'target_s1': target_s1}
        """
        return img_input, img_target, opticalflow_1, opticalflow_2

    def __len__(self):
        return self.n_samples
    
    
if __name__ == '__main__':
    dataset = Gopro(data_dir = './GOPRO_Large/train/')

    training_generator = data.DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)

    for epoch in range(2):
        for i_batch, (img_input, img_target, opticalflow_1, opticalflow_2) in enumerate(training_generator):
            print(i_batch)