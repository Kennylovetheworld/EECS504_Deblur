# EECS504_Deblur
This the our EECS 504 final project for image Deblur.

## Branch description
This branch is intended for debugging the "CUDA: an illegal memory access was encountered" issue. 
The cause of the problem is likely that the malconfiged Adam optimizer produces NaN and this has made deformConv offsets points to illegal memory location.
Currently, the parameters of Adam is changed to match that in the paper and the problem seems to have been resolved.

Experiments on small batch overfitting is performed as well. With `lr` changed from `1e-3` to `1e-4` so as to adapt to the small data size, the reconstructed image is much shaper after 200 updates and converges (in terms of human observation) after around 400 updates. This validates the network design to some extant. However, note that `loss3` is not used as it is not correctly implemented. Even when the reconstructed image has been rather close to the ground truth target, `loss3` remains unchanged at its largest allowed value.

Hence, the next thing to fix is `loss3`.


## Group meeting summary

### 03/12
+ Tianrong Zhang, Shukai Fan: Build `Net` (change the DeformConvBlock conv behaviour)
+ Chengzhi Peng: Loss function 1-3 initial version (by end of 05/12) 
+ Sheng Shen: Work on dataset (dataloader 06/12)
+ Next meeting: 05/12 (Sat) 10:00 AM +8

### 05/12
+ Tianrong Zhang, Shukai Fan: Trainer (Fan), utils functions in `Net` (save/load checkpoints) (09/12)
+ Chengzhi Peng: Loss function 1-3, check
+ Shang Shen: Work on dataset
+ Next meeting: 07/12 (Mon) 10:00 AM +8

### 07/12
+ Tianrong Zhang: Change DeformConvBlock conv behaviour (3N -> 2N + 64)
+ Chengzhi Peng: Loss 3, input 2N & chop input pic/sampling points
+ Shang Shen: Data augmentation (optional)
+ Shukai Fan: Trainer
+ Next meeting: 09/12 (Wed) 10:00 AM +8 (runable model)
