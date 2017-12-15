# Functions for grouping tags
import numpy as np
from munkres import Munkres
import torch

def py_max_match(scores):
    m = Munkres()
    tmp = m.compute(-scores)
    tmp = np.array(tmp).astype(np.int32)
    return tmp

class Params:
    def __init__(self):
        self.num_parts = 17
        self.detection_threshold = 0.2
        self.tag_threshold = 1.
        self.partOrder = [i-1 for i in [1,2,3,4,5,6,7,12,13,8,9,10,11,14,15,16,17]]
        self.max_num_people = 30
        self.use_detection_val = 0
        self.ignore_too_much = False

def match_by_tag(inp, params, pad=False):
    tag_k, loc_k, val_k = inp
    assert type(params) is Params
    default_ = np.zeros((params.num_parts, 3 + tag_k.shape[2]))

    dic = {}
    dic2 = {}
    for i in range(params.num_parts):
        ptIdx = params.partOrder[i]

        tags = tag_k[ptIdx]
        joints = np.concatenate((loc_k[ptIdx], val_k[ptIdx, :, None], tags), 1)
        mask = joints[:, 2] > params.detection_threshold
        tags = tags[mask]
        joints = joints[mask]
        if i == 0 or len(dic) == 0:
            for tag, joint in zip(tags, joints):
                dic.setdefault(tag[0], np.copy(default_))[ptIdx] = joint
                dic2[tag[0]] = [tag]
        else:
            actualTags = list(dic.keys())[:params.max_num_people]
            actualTags_key = actualTags
            actualTags = [np.mean(dic2[i], axis = 0) for i in actualTags]

            if params.ignore_too_much and len(actualTags) == params.max_num_people:
                continue
            diff = ((joints[:, None, 3:] - np.array(actualTags)[None, :, :])**2).mean(axis = 2) ** 0.5
            if diff.shape[0]==0:
                continue

            diff2 = np.copy(diff)

            if params.use_detection_val :
                diff = np.round(diff) * 100 - joints[:, 2:3]

            if diff.shape[0]>diff.shape[1]:
                diff = np.concatenate((diff, np.zeros((diff.shape[0], diff.shape[0] - diff.shape[1])) + 1e10), axis = 1)

            pairs = py_max_match(-diff) ##get minimal matching
            for row, col in pairs:
                if row<diff2.shape[0] and col < diff2.shape[1] and diff2[row][col] < params.tag_threshold:
                    dic[actualTags_key[col]][ptIdx] = joints[row]
                    dic2[actualTags_key[col]].append(tags[row])
                else:
                    key = tags[row][0]
                    dic.setdefault(key, np.copy(default_))[ptIdx] = joints[row]
                    dic2[key] = [tags[row]]

    ans = np.array([dic[i] for i in dic])
    if pad:
        num = len(ans)
        if num < params.max_num_people:
            padding = np.zeros((params.max_num_people-num, params.num_parts, default_.shape[1]))
            if num>0: ans = np.concatenate((ans, padding), axis = 0)
            else: ans = padding
        return np.array(ans[:params.max_num_people]).astype(np.float32)
    else:
        return np.array(ans).astype(np.float32)

class HeatmapParser():
    def __init__(self, detection_val=0.03, tag_val=1.):
        from torch import nn
        self.pool = nn.MaxPool2d(3, 1, 1)
        param = Params()
        param.detection_threshold = detection_val
        param.tag_threshold = tag_val
        param.ignore_too_much = True
        param.max_num_people = 30
        param.use_detection_val = True
        self.param = param

    def nms(self, det):
        # suppose det is a tensor
        maxm = self.pool(det)
        maxm = torch.eq(maxm, det).float()
        det = det * maxm
        return det

    def match(self, tag_k, loc_k, val_k):
        match = lambda x:match_by_tag(x, self.param)
        return list(map(match, zip(tag_k, loc_k, val_k)))

    def calc(self, det, tag):
        det = torch.autograd.Variable(torch.Tensor(det), volatile=True)
        tag = torch.autograd.Variable(torch.Tensor(tag), volatile=True)

        det = self.nms(det)
        h = det.size()[2]
        w = det.size()[3]
        det = det.view(det.size()[0], det.size()[1], -1)
        tag = tag.view(tag.size()[0], tag.size()[1], det.size()[2], -1)
        # ind (1, 17, 30)
        # val (1, 17, 128*128)
        # tag (1, 17, 128*128, -1)
        val_k, ind = det.topk(self.param.max_num_people, dim=2)
        tag_k = torch.stack([torch.gather(tag[:,:,:,i], 2, ind) for i in range(tag.size()[3])], dim=3)

        x = ind % w
        y = (ind / w).long()
        ind_k = torch.stack((x, y), dim=3)
        ans = {'tag_k': tag_k, 'loc_k': ind_k, 'val_k': val_k}
        return {key:ans[key].cpu().data.numpy() for key in ans}

    def adjust(self, ans, det):
        for batch_id, people in enumerate(ans):
            for people_id, i in enumerate(people):
                for joint_id, joint in enumerate(i):
                    if joint[2]>0:
                        y, x = joint[0:2]
                        xx, yy = int(x), int(y)
                        #print(batch_id, joint_id, det[batch_id].shape)
                        tmp = det[batch_id][joint_id]
                        if tmp[xx, min(yy+1, tmp.shape[1]-1)]>tmp[xx, max(yy-1, 0)]:
                            y+=0.25
                        else:
                            y-=0.25

                        if tmp[min(xx+1, tmp.shape[0]-1), yy]>tmp[max(0, xx-1), yy]:
                            x+=0.25
                        else:
                            x-=0.25
                        ans[batch_id][people_id, joint_id, 0:2] = (y+0.5, x+0.5)
        return ans

    def parse(self, det, tag, adjust=True):
        ans = self.match(**self.calc(det, tag))
        if adjust:
            ans = self.adjust(ans, det)
        return ans
