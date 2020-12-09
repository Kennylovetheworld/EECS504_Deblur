from src.dataset import dataloader
from src.loss import loss1, loss2, loss3

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

    self.network = network
    self.batch_size = batch_size
    
    self.use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    torch.backends.cudnn.benchmark = True

    self.criterion = nn.CrossEntropyLoss()

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
      logger.info('Starting training at epoch {}, iteration {} - last loss value is {}'.format(start_epoch, start_iter, losses[-1]))
    else:
      start_epoch = 0
      start_iter = 0
      losses = []
      logger.info('Starting training from scratch')

    # setup optimizer
    optimizer = optim.SGD(self.network.parameters(), learning_rate, momentum = 0.9, weight_decay = 0.0005)

    # let's go
    for epoch in range(start_epoch, n_epochs):
      for i, data in enumerate(dataloader, 0):
   
        if i >= start_iter:
        
          start = time.time()
          
          images = data['image']
          labels = data['label']
          batch_size = len(images)
          if self.use_gpu:
            images = images.cuda()
            labels = labels.cuda()
          imagesv = Variable(images)
          labelsv = Variable(labels)

          output, _ = self.network(imagesv)
          loss = self.criterion(output, labelsv)
          optimizer.zero_grad()
          loss.backward()
          optimizer.step()

          end = time.time()
          logger.info("[{}/{}][{}/{}] => Loss = {} (time spent: {})".format(epoch, n_epochs, i, len(dataloader), loss.item(), (end-start)))
          losses.append(loss.item())
      
      # do stuff - like saving models
      logger.info("EPOCH {} DONE".format(epoch+1))
      self.save_model(output_dir, epoch=(epoch+1), iteration=0, losses=losses)
