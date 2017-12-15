int my_lib_loss_forward(THCudaTensor *Tag, THLongTensor *keypoints, THFloatTensor *output, THFloatTensor *mean_tags);
int my_lib_loss_backward(THCudaTensor *Tag, THLongTensor *keypoints, THFloatTensor *mean_tags, THFloatTensor *grad_output, THCudaTensor *grad_input);
