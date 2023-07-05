#include "include/utils.h"
#include "assert.h"
#include <iostream>


// this only works for batchsize == 1
__global__ void broadCastMultiply(float* src1, dim_t src1Shape, float* src2, dim_t src2Shape, float* dst, dim_t dstShape){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    while (idx < dstShape.n * dstShape.c * dstShape.h * dstShape.w){
        int dstIdx = idx;

        int src1Idx = idx % (src1Shape.n * src1Shape.c * src1Shape.h * src1Shape.w);
        int h_ratio = src1Shape.h / src2Shape.h;
        int w_ratio = src1Shape.w / src2Shape.w;
        int n_ratio = src1Shape.n / src2Shape.n;
        int src2Idx = (src1Idx % (src1Shape.c * src1Shape.h * src1Shape.w * (src1Shape.n / n_ratio))) / (h_ratio * w_ratio);
        dst[dstIdx] = src1[src1Idx] * src2[src2Idx];
        idx += blockDim.x * gridDim.x;
    }
    
}

__global__ void broadCastAdd(float* src1, dim_t src1Shape, float* src2, dim_t src2Shape, float* dst, dim_t dstShape){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    while (idx < dstShape.n * dstShape.c * dstShape.h * dstShape.w){
        int dstIdx = idx;

        int src1Idx = idx % (src1Shape.n * src1Shape.c * src1Shape.h * src1Shape.w);
        int h_ratio = src1Shape.h / src2Shape.h;
        int w_ratio = src1Shape.w / src2Shape.w;
        int n_ratio = src1Shape.n / src2Shape.n;
        int src2Idx = (src1Idx % (src1Shape.c * src1Shape.h * src1Shape.w * (src1Shape.n / n_ratio))) / (h_ratio * w_ratio);
        dst[dstIdx] = src1[src1Idx] + src2[src2Idx];
        idx += blockDim.x * gridDim.x;
    }
    
}



void callBroadCastMultiply(float* src1, const dim_t src1Shape, float* src2, const dim_t src2Shape, float* dst, dim_t& dstShape){

    int blockSize = 512;
    int out_n = dstShape.n; int out_c = dstShape.c; int out_h = dstShape.h; int out_w = dstShape.w;
    

    int gridSize = (out_n * out_c * out_h * out_w + blockSize - 1) / blockSize;
    broadCastMultiply<<<gridSize, blockSize>>>(src1, src1Shape, src2, src2Shape, dst, dstShape);
    cudaDeviceSynchronize();
}

void callbroadCastAdd(float* src1, const dim_t src1Shape, float* src2, const dim_t src2Shape, float* dst, dim_t& dstShape){

    int blockSize = 512;

    int out_n = dstShape.n; int out_c = dstShape.c; int out_h = dstShape.h; int out_w = dstShape.w;

    int gridSize = (out_n * out_c * out_h * out_w + blockSize - 1) / blockSize;
    broadCastAdd<<<gridSize, blockSize>>>(src1, src1Shape, src2, src2Shape, dst, dstShape);
    cudaDeviceSynchronize();
}
