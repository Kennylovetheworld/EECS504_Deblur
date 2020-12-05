"""
loss functions 1-3
2020.12.05
Chengzhi Peng
"""

import torch
import torchvision.models as models


def loss1(recon_img, original_img):
  # Pixel loss for reconstructed img and original img
  # Input:
  #   recon_img, tensor of (B, C, H, W)
  #   origial_img, tensor of (B, C, H, W)
  # Output:
  #   loss(B,)
  C, H, W = original_img.shape[1:4]
  loss1 = torch.sum(torch.sum(torch.sum((recon_img - original_img)**2,dim=3),dim=2),dim=1) / (2*C*H*W)
  return loss1



def loss2(recon_img, original_img):
  # Preceptual loss for feature map
  # Input:
  #   recon_img, tensor of (B, C, H, W)
  #   origial_img, tensor of (B, C, H, W)
  # Output:
  #   loss(B,)
  vgg16_model = models.vgg16(pretrained=True).features[:26]
  vgg16_model.eval()
  recon_img_feature = vgg16_model(recon_img) # (B,512,16,16) If input (B,3,256,256)
  original_img_feature = vgg16_model(original_img)

  C, H, W = original_img_feature.shape[1:4] # should be 512, 16, 16
  loss2 = torch.sum(torch.sum(torch.sum((recon_img_feature - original_img_feature)**2,dim=3),dim=2),dim=1) / (2*C*H*W)
  return loss2

def loss3(optical_flow, deformable_conv, M=10):
  # Position loss for deformable convolution
  # Input:
  #   optical_flow, tensor of (B, 2, 128, 128)
  #           Second layer is x and y, respectively
  #   deformable_conv, tensor of (B, 2, 128, 128)
  #           Second layer is u and v, respectively
  # Output:
  #   loss(B,)
  
  # The paper used sampling points = 25, which it didn't explain further
  # So I used the whole picture
  
  x0 = torch.add(optical_flow[:,1,:,:]*deformable_conv[:,0,:,:], optical_flow[:,0,:,:]*deformable_conv[:,1,:,:])*deformable_conv[:,1,:,:] / torch.sum(deformable_conv**2,dim=1)
  mask = x0.clone() #(B,H,W)
  lower_bound = torch.min(deformable_conv[:,0,:,:],torch.tensor([0.0]))
  upper_bound = torch.max(deformable_conv[:,0,:,:],torch.tensor([0.0]))
  mask[mask<lower_bound] = 0
  mask[mask>upper_bound] = 0
  # mask=1 if x0 in [min(0,u),max(0,u)]; else mask=0
  min_dis_in = torch.abs(torch.sub(optical_flow[:,0,:,:]*deformable_conv[:,0,:,:], optical_flow[:,1,:,:]*deformable_conv[:,1,:,:])) / torch.sqrt(torch.sum(deformable_conv**2,dim=1))
  min_dis_in[mask==0]=0

  d1 = torch.sqrt(torch.sum(optical_flow**2,dim=1))
  d2 = torch.sqrt(torch.add((optical_flow[:,0,:,:]-deformable_conv[:,0,:,:])**2,(optical_flow[:,1,:,:]-deformable_conv[:,1,:,:])**2))
  min_dis_out = torch.min(d1,d2)
  min_dis_out[mask!=0]=0
  loss3 = min_dis_in + min_dis_out
  m = torch.tensor([1.0])*M
  loss3 = torch.max(loss3,m)
  H,W = loss3.shape[1:3]
  loss3 = torch.sum(loss3,dim=(1,2)) / (H*W)
  return loss3