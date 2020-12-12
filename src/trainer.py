from src.Dataset_preprocess import *
from src.loss import loss1, loss2, loss3
from src.model import Net
from src.util import *
import torch.utils.data as data

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torchvision.models as models

import time
import os
import pdb
from pynvml import *
import numpy as np


class DeblurTrainer(object):
  """
  Class to train a CNN

  Attributes
  ----------
  network: :py:class:`torch.nn.Module`
    The network to train
  batch_size: int
    The size of your minibatch
  use_gpu: boolean
    If you would like to use the gpu
  verbosity_level: int
    The level of verbosity output to stdout
  
  """
 
  def __init__(self, batch_size=4, use_gpu=False, verbosity_level=2):
    """ Init function

    Parameters
    ----------
    batch_size: int
      The size of your minibatch
    use_gpu: boolean
      If you would like to use the gpu
    verbosity_level: int
      The level of verbosity output to stdout

    """

    self.network = Net().to(device)
    self.vgg16_model = models.vgg16(pretrained=True).features[:26].to(device)
    if tensor_dtype == 'half':
        self.vgg16_model = self.vgg16_model.half()
    self.vgg16_model.eval()
    # pdb.set_trace()
    self.batch_size = batch_size
    if tensor_dtype == "half":
      self.network = self.network.half()

  def load_model(self, model_filename):
    """Loads an existing model

    Parameters
    ----------
    model_file: str
      The filename of the model to load

    Returns
    -------
    start_epoch: int
      The epoch to start with
    start_iteration: int
      The iteration to start with
    losses: list(float)
      The list of losses from previous training 
    
    """
    
    cp = torch.load(model_filename)
    self.network.load_state_dict(cp['state_dict'])
    start_epoch = cp['epoch']
    start_iter = cp['iteration']
    losses = cp['loss']
    return start_epoch, start_iter, losses


  def save_model(self, output_dir, epoch=0, iteration=0, losses=None):
    """Save the trained network

    Parameters
    ----------
    output_dir: str
      The directory to write the models to
    epoch: int
      the current epoch
    iteration: int
      the current (last) iteration
    losses: list(float)
        The list of losses since the beginning of training 
    
    """ 
    
    saved_filename = 'model_{}_{}.pth'.format(epoch, iteration)    
    saved_path = os.path.join(output_dir, saved_filename)    
    print('Saving model to {}'.format(saved_path))
    cp = {'epoch': epoch, 
          'iteration': iteration,
          'loss': losses, 
          'state_dict': self.network.cpu().state_dict()
          }
    torch.save(cp, saved_path)
  

  def train(self, n_epochs=20, learning_rate=0.01, output_dir='model_checkpoints', model=None):
    """Performs the training.

    Parameters
    ----------
    n_epochs: int
      The number of epochs you would like to train for
    learning_rate: float
      The learning rate for SGD optimizer.
    output_dir: str
      The directory where you would like to save models 
    
    """
    
    # if model exists, load it
    if model is not None:
      start_epoch, start_iter, losses = self.load_model(model)
      print('Starting training at epoch {}, iteration {} - last loss value is {}'.format(start_epoch, start_iter, losses[-1]))
    else:
      start_epoch = 0
      start_iter = 0
      losses = []
      print('Starting training from scratch')

    # setup optimizer
    optimizer = optim.Adam(self.network.parameters(), 1e-3, (0.9,0.9), 1e-8)
    #optimizer = optim.SGD(self.network.parameters(), 1e-4)

    # let's go
    dataset = Gopro_prepocessed(data_dir = 'dataset/train/')
    training_generator = data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True, num_workers=1)
    val_dataset = Gopro_prepocessed(data_dir = 'dataset/test/')
    validation_generator = data.DataLoader(val_dataset, batch_size=self.batch_size, shuffle=True, num_workers=1)

    for epoch in range(start_epoch, n_epochs):
      for i, (img_input, img_target, opticalflow_1, opticalflow_2) in enumerate(training_generator):
        # torch.cuda.empty_cache()
        if i >= start_iter:
        
          start = time.time()

          img_input, img_target, opticalflow_1, opticalflow_2 = img_input.to(device), img_target.to(device), opticalflow_1.to(device), opticalflow_2.to(device)

          # pdb.set_trace()
          recon_img, offset = self.network(img_input)
          # print(recon_img.shape)
          l1 = loss1(recon_img, img_target)
          l2 = loss2(recon_img, img_target, self.vgg16_model)
          l3 = loss3(opticalflow_1, opticalflow_2, offset)
          # l1.register_hook(lambda grad: print(grad))
          # l2.register_hook(lambda grad: print(grad))
          # l3.register_hook(lambda grad: print(grad))
          # offset.register_hook(lambda grad: print(grad))
          # recon_img.register_hook(lambda grad: print(grad))
          loss = l1.mean()
          loss_value = loss.item()
          # loss = l1.mean()
          optimizer.zero_grad()
          loss.backward()
          optimizer.step()

          end = time.time()
          print("[{}/{}][{}/{}] => Loss = {} (time spent: {})".format(epoch, n_epochs, i, len(training_generator), loss_value, (end-start)))
          losses.append(loss_value)

          if CHECK_GPU_USAGE:
            print(torch.cuda.is_available())
            nvmlInit()
            h = nvmlDeviceGetHandleByIndex(0)
            info = nvmlDeviceGetMemoryInfo(h)
            print(f'total    : {info.total}')
            print(f'free     : {info.free}')
            print(f'used     : {info.used}')
      
      # evaluation
      eval_loss = []
      for i, (img_input, img_target, opticalflow_1, opticalflow_2) in enumerate(validation_generator):
        self.network.eval()
        img_input, img_target, opticalflow_1, opticalflow_2 = img_input.to(device), img_target.to(device), opticalflow_1.to(device), opticalflow_2.to(device)
        recon_img, offset = self.network(img_input)
        l1 = loss1(recon_img, img_target)
        l2 = loss2(recon_img, img_target, self.vgg16_model)
        l3 = loss3(opticalflow_1, opticalflow_2, offset)
        loss = l1.mean()
        eval_loss.append(loss.item())

      print("Validation loss after epoch [{}/{}] => Loss = {}".format(epoch, n_epochs, np.mean(eval_loss)))

      # do stuff - like saving models
      print("EPOCH {} DONE".format(epoch+1))
      #self.save_model(output_dir, epoch=(epoch+1), iteration=0, losses=losses)
