import torch
from torch import nn
from torch.autograd import Function


class AElossFunction(Function):
    @staticmethod
    def forward(ctx, tags, keypoints):
        with torch.cuda.device(tags.get_device()):
            batch_size = tags.size()[0]
            tag_dim = tags.size()[2]
            num_people = keypoints.size()[1]

            output = torch.zeros(torch.Size((batch_size, 2)), device=tags.get_device())
            mean_tags = torch.zeros(torch.Size((batch_size, num_people, tag_dim + 1)), device=tags.get_device())
            # mean_tags: (batch_size x num_people x (tag_dim + 1)),
            #            keeps both mean tag and number of joints (for backprop)
            for b in range(batch_size):
                cur_people_count = 0
                # pull loss
                for p in range(num_people):
                    valid_keypoints = keypoints[b, p, (keypoints[b, p, :, -1] == 1), 0]
                    len_valid_kpts = len(valid_keypoints)
                    if len_valid_kpts > 0:
                        valid_tags = tags[b, valid_keypoints, 0]
                        mean_tags[b, p, 0] = torch.mean(valid_tags)
                        mean_tags[b, p, 1] = len_valid_kpts
                        output[b, 1] += torch.sum(torch.square(valid_tags - mean_tags[b, p, 0])) / len_valid_kpts
                        cur_people_count += 1
                if cur_people_count == 0:
                    continue
                output[b, 1] /= cur_people_count

                # push loss
                for p1 in range(cur_people_count - 1):
                    for p2 in range(p1 + 1, cur_people_count):
                        output[b, 0] += torch.exp(-(torch.square(mean_tags[b, p1, 0] - mean_tags[b, p2, 0])))
                if cur_people_count > 1:
                    output[b, 0] /= cur_people_count * (cur_people_count - 1) / 2
                output[b, 0] *= 0.5
            ctx.save_for_backward(tags, keypoints, mean_tags)
            return output

    @staticmethod
    def backward(ctx, grad_output):
        tags, keypoints, mean_tags = ctx.saved_tensors
        with torch.cuda.device(tags.get_device()):
            batch_size = tags.size()[0]
            tag_dim = tags.size()[2]
            num_people = keypoints.size()[1]

            grad_input = torch.zeros(tags.size(), device=tags.get_device())
            mean_grad = torch.zeros(num_people, device=tags.get_device())

            for b in range(batch_size):
                cur_people_count = torch.sum(mean_tags[b, :, 1] != 0)
                if cur_people_count == 0:
                    continue

                mean_grad.fill_(0)
                if cur_people_count > 1:
                    factor = 0.5 * grad_output[b, 0] / (cur_people_count * (cur_people_count - 1) / 2)
                    for p1 in range(cur_people_count - 1):
                        for p2 in range(p1 + 1, cur_people_count):
                            diff = mean_tags[b, p1, 0] - mean_tags[b, p2, 0]
                            grad = 2 * factor * (-torch.exp(-torch.square(diff))) * diff
                            mean_grad[p1] += grad
                            mean_grad[p2] -= grad
                for p in range(num_people):
                    valid_keypoints = keypoints[b, p, (keypoints[b, p, :, -1] == 1), 0]
                    len_valid_kpts = len(valid_keypoints)
                    if len_valid_kpts > 0:
                        valid_tags = tags[b, valid_keypoints, 0]
                        factor = 1/tag_dim/mean_tags[b, p, 1]/cur_people_count * grad_output[b, 1]
                        grad = 2 * (valid_tags - mean_tags[b, p, 0]) * factor
                        mean_grad[p] -= torch.sum(grad)
                        grad_input[b, valid_keypoints, 0] += grad
                        grad_input[b, valid_keypoints, 0] += mean_grad[p] / mean_tags[b, p, 1]

            # use the following two lines if you need to debug the backward loss
            # import pdb
            # pdb.set_trace()
            return grad_input, torch.zeros(keypoints.size())


class AEloss(nn.Module):
    @staticmethod
    def forward(inp, input1):
        if not inp.is_cuda:
            inp = inp.cuda()
        output = AElossFunction.apply(inp, input1)
        return output
