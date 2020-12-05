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

