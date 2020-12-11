"""
loss functions 1-3
2020.12.05
Chengzhi Peng
"""

import torch
import torchvision.models as models
from src.util import *

def loss1(recon_img, original_img):
  # Pixel loss for reconstructed img and original img
  # Input:
  #   recon_img, tensor of (B, C, H, W)
  #   origial_img, tensor of (B, C, H, W)
  # Output:
  #   loss(B,)
  device = recon_img.device
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
  device = recon_img.device
  vgg16_model = models.vgg16(pretrained=True).features[:26].to(device)
  if tensor_dtype == 'half':
    vgg16_model = vgg16_model.half()
  vgg16_model.eval()
  recon_img_feature = vgg16_model(recon_img) # (B,512,16,16) If input (B,3,256,256)
  original_img_feature = vgg16_model(original_img)
  
  C, H, W = original_img_feature.shape[1:4]
  loss2 = torch.sum((recon_img_feature - original_img_feature)**2,dim=(3,2,1)) / (2*C*H*W)
  return loss2
  
  
  
def loss3(optical_flow1, optical_flow2, deformable_conv, M=10):
  # Position loss for deformable convolution
  # Input:
  #   optical_flow1/2, tensor of (B, 2, 128, 128)
  #           Second layer is u and v, respectively
  #           Optical_flow1: the frame before and this frame
  #           Optical_flow2: this frame and the frame after
  #   deformable_conv, tensor of (B, 2*9, 128, 128)
  #           Second layer is dx and dy, respectively
  #           Sampling points will be 3*3, centered (31,31) (31,95) (95,31) (95,95)
  #           Assume 2*9 are dx11, dx12, dx13 ... dx23, dx33; dy11...
  # Output:
  #   loss(B,)

  device = deformable_conv.device
  # Sampling points, dxdy
  dx1 = deformable_conv[:,0:9,31,31] #(B,9)
  dy1 = deformable_conv[:,9:18,31,31]
  dx2 = deformable_conv[:,0:9,31,95]
  dy2 = deformable_conv[:,9:18,31,95]
  dx3 = deformable_conv[:,0:9,95,31]
  dy3 = deformable_conv[:,9:18,95,31]
  dx4 = deformable_conv[:,0:9,95,95]
  dy4 = deformable_conv[:,9:18,95,95]
  samp_x = torch.cat((dx1,dx2,dx3,dx4),1) #(B,4*9)
  samp_y = torch.cat((dy1,dy2,dy3,dy4),1)

  for i in range(2):
    if i == 0:
      optical_flow = optical_flow1.clone()
    if i == 1:
      optical_flow = optical_flow2.clone()

    # optical flow u,v
    op1_u = optical_flow[:,0,30:33,30:33].reshape(-1,9) #(B,9)
    op1_v = optical_flow[:,1,30:33,30:33].reshape(-1,9)
    op2_u = optical_flow[:,0,30:33,94:97].reshape(-1,9)
    op2_v = optical_flow[:,1,30:33,94:97].reshape(-1,9)
    op3_u = optical_flow[:,0,94:97,30:33].reshape(-1,9)
    op3_v = optical_flow[:,1,94:97,30:33].reshape(-1,9)
    op4_u = optical_flow[:,0,94:97,94:97].reshape(-1,9)
    op4_v = optical_flow[:,1,94:97,94:97].reshape(-1,9)
    opt_u = torch.cat((op1_u,op2_u,op3_u,op4_u),1) #(B,4*9)
    opt_v = torch.cat((op1_v,op2_v,op3_v,op4_v),1)


    # Algorithm 1
    x0 = opt_v*(opt_u*samp_y + opt_v*samp_x)/(opt_u**2 + opt_v**2) #(B,4*9)
    mask = x0.clone() #(B,4*9) # mask=1 if x0 in [min(0,u),max(0,u)]; else mask=0
    lower_bound = torch.min(opt_u,torch.tensor([0.0]).to(device))
    upper_bound = torch.max(opt_u,torch.tensor([0.0]).to(device))
    mask[mask<lower_bound] = 0
    mask[mask>upper_bound] = 0

    min_dis_in = torch.abs(opt_u*samp_x - opt_v*samp_y)/torch.sqrt(opt_u**2 + opt_v**2)
    min_dis_in[mask==0]=0

    d1 = torch.sqrt(samp_x**2+samp_y**2)
    d2 = torch.sqrt((samp_x-opt_u)**2 + (samp_y-opt_v)**2)
    min_dis_out = torch.min(d1,d2)
    min_dis_out[mask!=0]=0

    if i == 0:
      min_dis1 = min_dis_in + min_dis_out
    if i == 1:
      min_dis2 = min_dis_in + min_dis_out 

  loss3 = torch.min(min_dis1, min_dis2)
  m = torch.tensor([1.0]).to(device)*M
  loss3 = torch.max(loss3,m)
  loss3 = torch.sum(loss3,dim=1) / 36
  return loss3