from src.dataset import *
from src.loss import loss1, loss2, loss3
from src.model import Net

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

import time
import os

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
 
  def __init__(self, network, batch_size=64, use_gpu=False, verbosity_level=2):
    """ Init function

    Parameters
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

    self.network = Net()
    self.batch_size = batch_size
    
    self.use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    torch.backends.cudnn.benchmark = True


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
    logger.info('Saving model to {}'.format(saved_path))
    cp = {'epoch': epoch, 
          'iteration': iteration,
          'loss': losses, 
          'state_dict': self.network.cpu().state_dict()
          }
    torch.save(cp, saved_path)
    
    # moved the model back to GPU if needed
    self.network.to(device)


  def train(self, dataloader, n_epochs=20, learning_rate=0.01, output_dir='out', model=None):
    """Performs the training.

    Parameters
    ----------
    dataloader: :py:class:`torch.utils.data.DataLoader`
      The dataloader for your data
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
    optimizer = optim.Adam(self.network.parameters(), learning_rate)

    # let's go
    dataset = Gopro(data_dir = '~/dataset/train/')
    training_generator = data.DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)
    
    for epoch in range(start_epoch, n_epochs):
      for i, (img_input, img_target, opticalflow_1, opticalflow_2) in enumerate(training_generator):
   
        if i >= start_iter:
        
          start = time.time()
          
          batch_size = len(img_input)
          img_input, img_target, opticalflow_1, opticalflow_2 = img_input.to(device), img_target.to(device), opticalflow_1.to(device), opticalflow_2.to(device)

          

          recon_img, offset = self.network(img_input)
          l1 = loss1(recon_img, original_img)
          l2 = loss2(recon_img, original_img)
          l2 = loss3(optical_flow, offset)
          loss = l1 + l2 + l3
          optimizer.zero_grad()
          loss.backward()
          optimizer.step()

          end = time.time()
          print("[{}/{}][{}/{}] => Loss = {} (time spent: {})".format(epoch, n_epochs, i, len(dataloader), loss.item(), (end-start)))
          losses.append(loss.item())
      
      # evaluation


      # do stuff - like saving models
      print("EPOCH {} DONE".format(epoch+1))
      self.save_model(output_dir, epoch=(epoch+1), iteration=0, losses=losses)
