import torch
import time
import numpy as np
from utils.misc import make_input

class HeatmapLoss(torch.nn.Module):
    """
    loss for detection heatmap
    mask is used to mask off the crowds in coco dataset
    """
    def __init__(self):
        super(HeatmapLoss, self).__init__()

    def forward(self, pred, gt, masks):
        assert pred.size() == gt.size()
        l = ((pred - gt)**2) * masks[:, None, :, :].expand_as(pred)
        l = l.mean(dim=3).mean(dim=2).mean(dim=1)
        return l

def singleTagLoss(pred_tag, keypoints):
    """
    associative embedding loss for one image
    """
    eps = 1e-6
    tags = []
    pull = 0
    for i in keypoints:
        tmp = []
        for j in i:
            if j[1]>0:
                tmp.append(pred_tag[j[0]])
        if len(tmp) == 0:
            continue
        tmp = torch.stack(tmp)
        tags.append(torch.mean(tmp, dim=0))
        pull = pull +  torch.mean((tmp - tags[-1].expand_as(tmp))**2)

    if len(tags) == 0:
        return make_input(torch.zeros([1]).float()), make_input(torch.zeros([1]).float())

    tags = torch.stack(tags)[:,0]

    num = tags.size()[0]
    size = (num, num, tags.size()[1])
    A = tags.unsqueeze(dim=1).expand(*size)
    B = A.permute(1, 0, 2)

    diff = A - B
    diff = torch.pow(diff, 2).sum(dim=2)[:,:,0]
    push = torch.exp(-diff)
    push = (torch.sum(push) - num)
    return push/((num - 1) * num + eps) * 0.5, pull/(num + eps)

def tagLoss(tags, keypoints):
    """
    accumulate the tag loss for each image in the batch
    """
    pushes, pulls = [], []
    keypoints = keypoints.cpu().data.numpy()
    for i in range(tags.size()[0]):
        push, pull = singleTagLoss(tags[i], keypoints[i%len(keypoints)])
        pushes.append(push)
        pulls.append(pull)
    return torch.stack(pushes), torch.stack(pulls)

def test_tag_loss():
    t = make_input( torch.Tensor((1, 2)), requires_grad=True )
    t.register_hook(lambda x: print('t', x))
    loss = singleTagLoss((t, [[[0,1]], [[1,1]]]))[0]
    loss.backward()
    print(loss)

if __name__ == '__main__':
    test_tag_loss()
