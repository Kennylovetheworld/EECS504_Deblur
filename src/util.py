import torch
from torch import nn
from torchvision.ops import DeformConv2d

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
# device = torch.device("cpu")
torch.backends.cudnn.benchmark = True

CHECK_GPU_USAGE = False

tensor_dtype = "float"

def weights_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d) or isinstance(m, DeformConv2d):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)