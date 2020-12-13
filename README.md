# EECS504_Deblur
This the our EECS 504 final project for image Deblur.


Code description
+ Calculate_opticalflow.py: This file mainly constructed some functions for bidirectional optical flow calculation and the dataset preprocessing on optical flow.
+ Dataset_preprocess.py: This file defines the Pytorch dataset class and the pre-process procedures, including data augmentations, tensor object forming and saving. The data augmentation included cropping, shifting, rotation and flipping.
+ dataset.py: This file is abandoned
+ loss.py: This file constructed three main loss we used in the model. Loss 1 defines the pixel loss. Loss 2 defines the perceptual loss. Loss 3 defines the position loss. 
+ model.py: This file defines the model in Net class and the network blocks.
+ trainer.py: This file defines a trainer which contains all the training, evaluation, model saving and reporting.
+ util.py: This file defines some utility functions and global variables used across through the repo.

## 03/12
+ Tianrong Zhang, Shukai Fan: Build `Net` (change the DeformConvBlock conv behaviour)
+ Chengzhi Peng: Loss function 1-3 initial version (by end of 05/12) 
+ Sheng Shen: Work on dataset (dataloader 06/12)
+ Next meeting: 05/12 (Sat) 10:00 AM +8

## 05/12
+ Tianrong Zhang, Shukai Fan: Trainer (Fan), utils functions in `Net` (save/load checkpoints) (09/12)
+ Chengzhi Peng: Loss function 1-3, check
+ Shang Shen: Work on dataset
+ Next meeting: 07/12 (Mon) 10:00 AM +8

## 07/12
+ Tianrong Zhang: Change DeformConvBlock conv behaviour (3N -> 2N + 64)
+ Chengzhi Peng: Loss 3, input 2N & chop input pic/sampling points
+ Shang Shen: Data augmentation (optional)
+ Shukai Fan: Trainer
+ Next meeting: 09/12 (Wed) 10:00 AM +8 (runable model)


