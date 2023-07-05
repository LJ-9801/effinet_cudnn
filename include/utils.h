#ifndef _UTILS_H_
#define _UTILS_H_
#include <cuda_runtime.h>
#include"device_launch_parameters.h"
#include "type.h"

__global__ void broadCastMultiply(float* src1, dim_t src1Shape, float* src2, dim_t src2Shape, float* dst, dim_t dstShape);
__global__ void broadCastAdd(float* src1, dim_t src1Shape, float* src2, dim_t src2Shape, float* dst, dim_t dstShape);

void callBroadCastMultiply(float* src1, const dim_t src1Shape, float* src2, const dim_t src2Shape, float* dst, dim_t& dstShape);
void callbroadCastAdd(float* src1, const dim_t src1Shape, float* src2, const dim_t src2Shape, float* dst, dim_t& dstShape);
#endif