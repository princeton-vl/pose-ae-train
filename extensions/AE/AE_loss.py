import torch
import os
import time
from torch.autograd import Function
from torch import nn
from extensions.AE._ext import my_lib

class AElossFunction(Function):
    def forward(self, tags, keypoints):
        output = torch.zeros(torch.Size((tags.size()[0], 2)))
        mean_tags = torch.zeros(torch.Size((tags.size()[0], keypoints.size()[1], tags.size()[2]+1)))
        self.mean_tags = mean_tags

        my_lib.my_lib_loss_forward(tags, keypoints, output, mean_tags)
        self.save_for_backward(tags, keypoints)
        return output

    def backward(self, grad_output):
        tags, keypoints = self.saved_tensors
        grad_input = torch.zeros(tags.size()).cuda(tags.get_device())
        #grad_input = tags.new(tags.size()).zero_()
        my_lib.my_lib_loss_backward(tags, keypoints, self.mean_tags, grad_output, grad_input)
        self.mean_tags = None
        return grad_input, torch.zeros(keypoints.size())

class AEloss(nn.Module):
    def forward(self, input, input1):
        if not input.is_cuda:
            input = input.cuda()
        output = AElossFunction()(input, input1)
        return output
