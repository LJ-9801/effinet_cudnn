#ifndef _TYPE_H_
#define _TYPE_H_
#include <vector>
#include <string>
#include <cudnn.h>
#include <cublas_v2.h>

// General Parameter

struct file_info
{
    int num;
    std::string function;
    std::string layer;
    std::string filename;
    std::string path;
};

typedef std::vector<float> data_t;
typedef std::vector<long unsigned> shape_t;

struct DATA
{
    data_t data;
    shape_t shape;
};


struct GPU_DATA
{
    float* data;
    shape_t shape;
};

struct MODEL_DATA
{
    //std::string name;
    std::string data_type;
    shape_t shape;
    data_t data;
    float* gpu_data;
};

struct dim_t{
    int n;
    int c;
    int h;
    int w;
};

struct Tensor
{
    float* data;
    dim_t shape;
};


// CONVOLUTION PARAMETER
struct stride_t{
    int h;
    int w;
};

struct pad_t{
    int h;
    int w;
};

struct upscale_t{
    int h;
    int w;
};

struct kernel_t{
    int h;
    int w;
};


struct CONV_PARAM_T{
    int in_channels;
    int out_channels;
    kernel_t kernel;
    stride_t stride;
    pad_t pad;
    upscale_t upscale;
    bool is_bias;
    int groups;

    GPU_DATA gpu_weight;
    GPU_DATA gpu_bias;
};


struct CONV_DESC_T{
    cudnnConvolutionDescriptor_t convDesc;
    cudnnFilterDescriptor_t filterDesc;
    cudnnTensorDescriptor_t biasDesc;
    cudnnTensorDescriptor_t inputDesc;
    cudnnTensorDescriptor_t outputDesc;
    cudnnConvolutionFwdAlgo_t algo;

    cudnnActivationDescriptor_t actDesc;
    void* workSpace;
    size_t sizeInBytes;

    dim_t indim;
    //dim_t weightdim;
    //dim_t biasdim;
    dim_t outdim;
};


// BATCHNORM PARAMETER
struct BN_PARAM_T{

    int channels;
    float eps;
    float momentum;

    GPU_DATA gpu_weight;
    GPU_DATA gpu_bias;
    GPU_DATA gpu_mean;
    GPU_DATA gpu_var;

};

struct BN_DESC_T{
    cudnnBatchNormMode_t mode;
    cudnnTensorDescriptor_t inputDesc;
    cudnnTensorDescriptor_t outputDesc;
    cudnnTensorDescriptor_t bnWeightBiasMeanVarDesc;

    dim_t dim;
};


// ACTIVATION PARAMETER
struct ACTIVATION_DESC_T{
    cudnnActivationDescriptor_t actDesc;
    cudnnActivationMode_t mode;
    cudnnTensorDescriptor_t inputDesc;
    cudnnTensorDescriptor_t outputDesc;
    float coef;

    dim_t dim;
};

// AVGPOOL PARAMETER
struct AVGPOOL_PARAM_T{
    dim_t indim;
    int out_w;
    int out_h;
};

struct AVGP_DESC_T{
    cudnnPoolingDescriptor_t poolDesc;
    cudnnTensorDescriptor_t inputDesc;
    cudnnTensorDescriptor_t outputDesc;
    cudnnPoolingMode_t mode;
    cudnnNanPropagation_t maxpoolingNanOpt;
    //dim_t dim;
};

struct LINEAR_PARAM_T{
    int in_features;
    int out_features;
    bool is_bias;

    GPU_DATA gpu_weight;
    GPU_DATA gpu_bias;
};

struct LINEAR_DESC_T
{
    cudnnTensorDescriptor_t inputDesc;
    cudnnTensorDescriptor_t outputDesc;
    cudnnFilterDescriptor_t filterDesc;
    cudnnTensorDescriptor_t biasDesc;

    dim_t indim;
    dim_t outdim;
};

struct SOFTMAX_DESC_T{
    cudnnTensorDescriptor_t inputDesc;
    cudnnTensorDescriptor_t outputDesc;
    cudnnSoftmaxAlgorithm_t algo;
    cudnnSoftmaxMode_t mode;
    dim_t dim;
};

struct block_arg
{
    int num_repeat;
    int kernel_size;
    stride_t strides;
    int expand_ratio;
    int input_filters;
    int output_filters;
    float se_ratio;
};

#endif