#include "my_lib_kernel.h"
#include "stdio.h"

float get_cuda(float* ans, int idx){
    float t=0;
    cudaMemcpy(&t, ans+idx, sizeof(float), cudaMemcpyDeviceToHost);
    return t;
}

void set_cuda(float* ans, int idx, float t){
    cudaMemcpy(ans+idx, &t, sizeof(float), cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
}