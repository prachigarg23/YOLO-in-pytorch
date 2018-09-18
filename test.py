from __future__ import division

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import cv2

arr = torch.rand(1, 5, 85)
print(arr)

confidence = 0.5
print('array size: {}'.format(arr.size()))

conf_mask = (arr[:,:,4] > confidence).float().unsqueeze(2)
print(conf_mask)
print('conf_mask size: {}'.format(conf_mask.size()))

prediction = arr*conf_mask
print('prediction size: {}'.format(prediction.size()))
print(prediction)
img_pred = prediction[0]
non_zero_ind =  (torch.nonzero(img_pred[:,4]))
print(non_zero_ind)
print(non_zero_ind.size())
image_pred_ = img_pred[non_zero_ind.squeeze(),:].view(-1,85)
print(image_pred_)

'''
test = torch.tensor([1.9, 2.9, 3.9])
test1 = torch.unsqueeze(test, 0)
print("along 0 dim")
print(test1)
print(test1.size())
test2 = torch.unsqueeze(test, 1)
print('along 1 dim')
print(test2)
print(test2.size())
'''
