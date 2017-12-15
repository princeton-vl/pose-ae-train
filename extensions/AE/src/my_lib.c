#include <TH/TH.h>
#include <THC/THC.h>
#include "stdio.h"
#include "stdlib.h"
#include "my_lib_kernel.h"

#define real float
extern THCState *state;

int my_lib_loss_forward(THCudaTensor *Tag, THLongTensor *keypoints, THFloatTensor *output, THFloatTensor *mean_tags){
    //Tag: batch x (featurexwxh) x tag_dim
    //keypoints: batch x people x 17 x 2
    //ouptut: batch x 2
    const int batchsize = Tag->size[0];
    const int tag_dim = Tag->size[2];
    const int num_people = keypoints->size[1];
    const int num_joint = keypoints->size[2];


    const int kpt_strideBatch = keypoints->stride[0];
    const int kpt_stridePeople = keypoints->stride[1];
    const int kpt_strideJoint = keypoints->stride[2];

    const int tag_strideBatch = Tag->stride[0];
    const int tag_stridePoint = Tag->stride[1];

    const int output_strideBatch = output->stride[0];
    const int mean_strideBatch = mean_tags->stride[0];
    const int mean_stride = mean_tags->stride[1];

    int i, j, j1, j2, k, k1, l;
    real *tag_data, *output_data;

    long* kpt_data = THLongTensor_data(keypoints);
    tag_data = THCudaTensor_data(state, Tag);
    output_data = THFloatTensor_data(output);

    //real* mean = malloc( num_people * tag_dim * sizeof(real) );
    real* batch_mean = THFloatTensor_data(mean_tags);
    real* current_tags = malloc( num_joint * tag_dim * sizeof(real) );

    for(i=0; i<batchsize; ++i){
        //real* current_hm = tag_data + i*tag_strideBatch;
        real* output_tmp = output_data + i*output_strideBatch;
        real* mean = batch_mean + i*mean_strideBatch;
        output_tmp[0] = 0;
        output_tmp[1] = 0;

        int current_people=0;
        for(j=0; j<num_people; ++j){
            real* current_mean = mean + current_people*mean_stride;
            memset(current_mean, 0, tag_dim*sizeof(real));

            int current_joint=0;
            for(k=0; k<num_joint; ++k){
                int idx = i*kpt_strideBatch + j*kpt_stridePeople + k*kpt_strideJoint;
                int flag = kpt_data[idx+1];
                if(flag){
                    int tag_idx = ((int)kpt_data[idx]) * tag_stridePoint;
                    for(l=0;l<tag_dim;++l){
                        float t = get_cuda(tag_data, i*tag_strideBatch + tag_idx + l);
                        current_mean[l] += t;
                        current_tags[current_joint*tag_dim + l] = t;
                    }
                    current_joint++;
                }
            }
            if(current_joint==0)
                continue;
            for(l=0;l<tag_dim;++l)
                current_mean[l]/=current_joint;
            current_mean[tag_dim] = current_joint;//record the number of joints for bp 

            // calculate pull loss
            real people_pull_loss = 0;
            for(k1=0; k1<current_joint; ++k1){
                real tmp = 0;
                for(l=0;l<tag_dim;++l){
                    real diff = (current_tags[k1*tag_dim+l] - current_mean[l]);
                    tmp += diff * diff;
                }
                people_pull_loss += tmp/l;
            }
            output_tmp[1] += people_pull_loss/current_joint;
            current_people++;
        }
        if(current_people==0)
            continue;

        output_tmp[1]/=current_people;
        for(j1=0;j1<current_people;++j1)
            for(j2=j1+1;j2<current_people;++j2){
                real tmp = 0;
                for(l=0;l<tag_dim;++l){
                    real diff = mean[j1*mean_stride+l]-mean[j2*mean_stride+l];
                    tmp += diff * diff;
                }
                output_tmp[0] += exp(-tmp);
            }
        if(current_people>1)
            output_tmp[0]/= current_people*(current_people-1)/2;
        output_tmp[0] *= 0.5;
    }
    free(current_tags);
    return 1;
}

int my_lib_loss_backward(THCudaTensor *Tag, THLongTensor *keypoints, THFloatTensor *mean_tags, THFloatTensor *grad_output, THCudaTensor *grad_input){
    //Tag: batch x (featurexwxh) x tag_dim
    //keypoints: batch x people x 17 x 2
    //ouptut: batch x 2
    const int batchsize = Tag->size[0];
    const int tag_dim = Tag->size[2];
    const int num_people = keypoints->size[1];
    const int num_joint = keypoints->size[2];


    const int kpt_strideBatch = keypoints->stride[0];
    const int kpt_stridePeople = keypoints->stride[1];
    const int kpt_strideJoint = keypoints->stride[2];

    const int tag_strideBatch = Tag->stride[0];
    const int tag_stridePoint = Tag->stride[1];

    const int mean_strideBatch = mean_tags->stride[0];
    const int mean_stride = mean_tags->stride[1];
    const int grad_output_strideBatch = grad_output->stride[0];

    real* batch_mean = THFloatTensor_data(mean_tags);
    real* grad_output_data = THFloatTensor_data(grad_output);
    real* grad_input_data = THCudaTensor_data(state, grad_input);
    real* tag_data = THCudaTensor_data(state, Tag);
    real* mean_grad = malloc(mean_strideBatch * sizeof(real));
    long* kpt_data = THLongTensor_data(keypoints);

    int i, j, k, l, j1, j2;

    for(i=0; i<batchsize; ++i){
        // first calculate the gradient for mean_tags
        //real* current_hm = tag_data + i*tag_strideBatch;
        real* mean = batch_mean + i*mean_strideBatch;

        int current_people = 0;
        for(j=0; j<num_people; ++j)
            if(mean[j*mean_stride+tag_dim])
                current_people+= 1;

        if(current_people==0)
            continue;

        memset(mean_grad, 0, current_people*mean_stride*sizeof(real));

        //printf("%d\n", current_people);
        if(current_people>1){
            real factor = 1./(current_people*(current_people-1)/2) * 0.5 * grad_output_data[i*grad_output_strideBatch];
            for(j1=0;j1<current_people;++j1){
                for(j2=j1+1;j2<current_people;++j2){
                    //printf("%d %d\n", j1, j2);
                    real tmp = 0, tmp2;
                    for(l=0; l<tag_dim; ++l){
                        real diff = mean[j1*mean_stride+l] - mean[j2*mean_stride+l];
                        tmp += diff * diff;
                    }
                    //dL/d e^{-tmp} * d e^{-tmp}/d tmp * dtmp/d diff * d diff/d mean
                    tmp2 = factor * (-exp(-tmp));
                    for(l=0; l<tag_dim; ++l){
                        real diff = mean[j1*mean_stride+l] - mean[j2*mean_stride+l];
                        //d tmp/d diff = 2 * diff
                        mean_grad[j1 * mean_stride + l] += tmp2 * (2*diff);
                        mean_grad[j2 * mean_stride + l] -= tmp2 * (2*diff);
                    }
                }
            }
        }

        real* current_mean = mean;
        real* current_mean_grad = mean_grad;

        for(j=0; j<num_people; ++j){
            int joint_flag = 0;
            for(k=0; k<num_joint; ++k){
                int idx = i*kpt_strideBatch + j*kpt_stridePeople + k*kpt_strideJoint;
                int flag = kpt_data[idx+1];
                if(flag){
                    joint_flag += 1;
                    int go = ((int)kpt_data[idx]) * tag_stridePoint;
                    //real* tmp = current_hm + go;
                    //real* tmp_grad = grad_input_data + (i*tag_strideBatch + go);

                    real factor = 1./tag_dim/current_mean[tag_dim]/current_people * grad_output_data[i*grad_output_strideBatch+1];
                    for(l=0; l<tag_dim; ++l){
                        int tt = go + i*tag_strideBatch + l;
                        float tmp_l = get_cuda(tag_data, tt);
                        real grad = 2 * (tmp_l - current_mean[l]) * factor;
                        //d(a-b)^2/da=2(a-b) or -2(a-b)
                        current_mean_grad[l] -= grad;

                        float tmp_grad = get_cuda(grad_input_data, tt);
                        set_cuda(grad_input_data, tt, tmp_grad + grad);                        
                        //tmp_grad[l] += grad;
                    }
                }
            }
            if(joint_flag){
                for(k=0; k<num_joint; ++k){
                    int idx = i*kpt_strideBatch + j*kpt_stridePeople + k*kpt_strideJoint;
                    if(kpt_data[idx+1]){
                        int tt = i*tag_strideBatch + (int)kpt_data[idx] * tag_stridePoint;
                        for(l=0; l<tag_dim; ++l){
                            float tmp_grad = get_cuda(grad_input_data, tt+l);
                            set_cuda(grad_input_data, tt+l, tmp_grad + current_mean_grad[l]/current_mean[tag_dim]);                        
                            //tmp_grad[l] += current_mean_grad[l]/current_mean[tag_dim];
                        }
                    }
                }

                current_mean_grad += mean_stride;
                current_mean += mean_stride;
            }
        }
    }

    free(mean_grad);
    return 1;
}