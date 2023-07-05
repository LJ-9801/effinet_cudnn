#ifndef _EFFICIENTNET_H_
#define _EFFICIENTNET_H_
#include <cudnn.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <vector>
#include <unordered_map>
#include <assert.h>
#include <chrono>

#include "libnpy/include/npy.hpp"
#include "error_utils.h"
#include "utils.h"
#include "type.h"


class EfficientNet{
    public:
       enum ACTIVATION_TYPE{
           SWISH = 0x00,
           RELU = 0x01,
           SIGMOID = 0x02,
           TANH = 0x03,
           CLIPPED_RELU = 0x04,
           ELU = 0x05
       };

        typedef size_t ActivationType;

        typedef std::vector<MODEL_DATA> WEIGHT;

        typedef std::string CONV_NAME_T;
        typedef std::string BN_NAME_T;
        typedef std::string AVGP_NAME_T;
        typedef std::string LINEAR_NAME_T;
        typedef std::string ACTIVATION_NAME_T;
        typedef std::string OPS_NAME_T;
        typedef std::string SOFTMAX_NAME_T;

        EfficientNet();
        ~EfficientNet();

        // forward
        void forward(DATA input);

        void MBConvBlockInit(const int num, dim_t& dim, const block_arg& args);
        void MBConvBlockForward(const int num, float* src_data, float** dst_data, dim_t& indim,
                                float* tmp0, float* tmp1, float* tmp2);

        // Operation initialize utils
        void Convolution2dInit(const CONV_NAME_T name, CONV_DESC_T& desc, CONV_PARAM_T param, const WEIGHT& weight);
        void BatchNorm2dInit(const BN_NAME_T name, BN_DESC_T& desc, BN_PARAM_T param, const WEIGHT& weight);
        void ActivationInit(const ACTIVATION_NAME_T name, const ActivationType type, ACTIVATION_DESC_T& desc);
        void AdaptiveAvgPool2dInit(const AVGP_NAME_T name, AVGPOOL_PARAM_T& param);
        void LinearInit(const LINEAR_NAME_T name, const LINEAR_PARAM_T param, const WEIGHT& weight);
        void SoftmaxInit(const SOFTMAX_NAME_T name, const SOFTMAX_DESC_T desc);

        // Operation forward utils
        void Convolute2dForward(const CONV_NAME_T name, float* srcData, float** dstData);
        void Batchnorm2dForward(const BN_NAME_T name, float* srcData, float** dstData);
        void ActivationForward(const ACTIVATION_NAME_T name, float* srcData, float** dstData);
        void AdaptiveAvgPool2dForward(const AVGP_NAME_T name, float* srcData, float** dstData);
        void LinearForward(const LINEAR_NAME_T name, float* srcData, float** dstData);
        void SoftmaxForward(const SOFTMAX_NAME_T name, float* srcData, float** dstData);

        // general operation
        void BroadcastMul(const Tensor& in1, const Tensor& in2, Tensor& out);
        void Add(const Tensor& in1, const Tensor& in2, Tensor& out);

        // weights loader
        void loadWeights(std::vector<file_info>& fn);

        // init
        void init();

        // model structure
        void printstate();

    private:
        // create handle
        void createHandle();
        // destroy handle
        void destroyHandle();
        // destroy weights
        void destroyWeights();
        //destroy conv map
        void destroyConv();
        //destroy bn map
        void destroyBn();
        //destroy activation map
        void destroyActivation();
        //destroy avgp map
        void destroyAvgp();
        //destroy linear map
        void destoryLinear();
        //destroy softmax map
        void destroySoftmax();

        // memory utils
        void resize(int size, float **data);
        void copy(float* srcData, const dim_t srcDim, float** dstData);

        // EfficientNet utils 
        int round_filters(int filters);
        int round_repeats(int repeats);


        // weights
        std::unordered_map<std::string, WEIGHT> weights;

        std::vector<std::string> model_structure;

        std::unordered_map<CONV_NAME_T, std::pair<CONV_DESC_T, CONV_PARAM_T> > conv_map;
        std::unordered_map<BN_NAME_T, std::pair<BN_DESC_T, BN_PARAM_T> > bn_map;
        std::unordered_map<ACTIVATION_NAME_T, ACTIVATION_DESC_T> activation_map;
        std::unordered_map<AVGP_NAME_T, std::pair<AVGP_DESC_T, AVGPOOL_PARAM_T> > avgp_map;
        std::unordered_map<LINEAR_NAME_T, std::pair<LINEAR_DESC_T, LINEAR_PARAM_T> > linear_map;
        std::unordered_map<SOFTMAX_NAME_T, SOFTMAX_DESC_T> softmax_map;


        cudnnDataType_t dataType;
        cublasHandle_t cublasHandle;
        cudnnHandle_t cudnnHandle;
        cudnnTensorFormat_t tensorFormat;
};
#endif