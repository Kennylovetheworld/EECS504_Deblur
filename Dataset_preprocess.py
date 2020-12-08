# -*- coding: utf-8 -*-
"""
Created on Tue Dec  8 15:41:08 2020

@author: sheng
"""
import os.path
import random
import torch.utils.data as data
import torch
from PIL import Image
from torchvision import transforms
import torchvision.transforms.functional as TF
import cv2
import pandas as pd

def transform(img_input, img_target, opticalflow_1, opticalflow_2, patchsize):

    opticalflow_1 = opticalflow_1.permute(2,0,1)
    opticalflow_2 = opticalflow_2.permute(2,0,1)
    
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
    
    opticalflow_1 = transforms.Resize((patchsize//2, patchsize//2))(opticalflow_1)
    opticalflow_2 = transforms.Resize((patchsize//2, patchsize//2))(opticalflow_2)

    return img_input, img_target, opticalflow_1, opticalflow_2    
    
def preprocess_dataset(data_dir):
    sub_folders = os.listdir(data_dir)
    sharp_file_paths = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    df = pd.DataFrame(columns=['sharp_path', 'blur_path','opticalflow_1_path','opticalflow_2_path'])
    
    folder_number = 0
    for folder_name in sub_folders:
        if (folder_name == 'blur_img_all.pt' or folder_name == 'sharp_img_all.pt' or folder_name == 'opticalflow_all.pt' or folder_name == 'log.xlsx' or folder_name == 'log.csv'):
            continue
        folder_number += 1
        sharp_sub_folder = os.path.join(data_dir, folder_name, 'sharp')
        sharp_file_names = os.listdir(sharp_sub_folder)

        for file_name in sharp_file_names:
            if(sharp_file_names.index(file_name)!=0 and sharp_file_names.index(file_name)!=len(sharp_file_names)-1):
                sharp_file_path = os.path.join(sharp_sub_folder, file_name)
                sharp_file_paths.append(sharp_file_path)

    n_samples = len(sharp_file_paths)
    
    blur_img_all = torch.zeros((n_samples, 3, 256, 256), dtype=torch.uint8)
    sharp_img_all = torch.zeros((n_samples, 3, 256, 256), dtype=torch.uint8)
    opticalflow_all = torch.zeros((n_samples, 2, 2, 128, 128), dtype=torch.float16)
    
    for idx in range(n_samples):
        sharp_path = sharp_file_paths[idx]
        blur_path = sharp_path.replace("sharp", "blur")
        
        opticalflow_1_path = sharp_path.replace("sharp", "opticalflow")
        opticalflow_1_path = opticalflow_1_path.replace("png", "flo")
        
        opticalflow_folder = os.path.dirname(opticalflow_1_path)
        flow1_name = os.path.basename(opticalflow_1_path)
        files = os.listdir(opticalflow_folder)
        index = files.index(flow1_name)
        
        opticalflow_2_path = os.path.join(opticalflow_folder, files[index+1])
        
        """
        print(os.path.basename(blur_path))
        print(os.path.basename(sharp_path))
        print(flow1_name)
        print(files[index+1])
        """
        print("prepocessing: ", idx)
        new_row = {'sharp_path':os.path.basename(sharp_path), 'blur_path':os.path.basename(blur_path), 'opticalflow_1_path':flow1_name, 
                           'opticalflow_2_path':files[index+1]}
                
        df = df.append(new_row, ignore_index=True)
        
        img_input = Image.open(blur_path).convert('RGB')
        img_target = Image.open(sharp_path).convert('RGB')
        opticalflow_1 = cv2.readOpticalFlow(opticalflow_1_path)
        opticalflow_2 = cv2.readOpticalFlow(opticalflow_2_path)
        
        img_input = transforms.ToTensor()(img_input).to(device)
        img_target = transforms.ToTensor()(img_target).to(device)
        opticalflow_1 = torch.Tensor(opticalflow_1).to(device)
        opticalflow_2 = torch.Tensor(opticalflow_2).to(device)
        
        img_input, img_target, opticalflow_1, opticalflow_2 = transform(
                img_input, img_target, opticalflow_1, opticalflow_2, 256)
        
        blur_img_all[idx, :, :, :] = img_input
        sharp_img_all[idx, :, :, :] = img_target
        opticalflow_all[idx, 0, :, :, :] = opticalflow_1
        opticalflow_all[idx, 1, :, :, :] = opticalflow_2
        
    torch.save(blur_img_all, os.path.join(data_dir, 'blur_img_all.pt'))
    torch.save(sharp_img_all, os.path.join(data_dir, 'sharp_img_all.pt'))
    torch.save(opticalflow_all, os.path.join(data_dir, 'opticalflow_all.pt'))
    
    df.to_excel(os.path.join(data_dir, 'log.xlsx'))
    df.to_csv((os.path.join(data_dir, 'log.csv')), index=False)
    
class Gopro_prepocessed(data.Dataset):
    def __init__(self, data_dir):
        self.blur_img_all = torch.load(os.path.join(data_dir, 'blur_img_all.pt'))
        self.sharp_img_all = torch.load(os.path.join(data_dir, 'sharp_img_all.pt'))
        self.opticalflow_all = torch.load(os.path.join(data_dir, 'opticalflow_all.pt'))
        
        print("blur tensor:", self.blur_img_all.size())
        print("sharp tensor:", self.sharp_img_all.size())
        print("opticalflow tensor:", self.opticalflow_all.size())
        
        self.n_samples = self.blur_img_all.size()[0]

    def __getitem__(self, idx):
        input_img = self.blur_img_all[idx, :, :, :]
        target_img = self.sharp_img_all[idx, :, :, :]
        opticalflow_1 = self.opticalflow_all[idx, 0, :, :, :]
        opticalflow_2 = self.opticalflow_all[idx, 1, :, :, :]
        
        return input_img, target_img, opticalflow_1, opticalflow_2

    def __len__(self):
        return self.n_samples
        
        
if __name__ == '__main__':
    data_dir = './GOPRO_Large/train/'
    
    
    #preprocess_dataset(data_dir)
    
    dataset = Gopro_prepocessed(data_dir)
    
    training_generator = data.DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)
    
    
    
    
    