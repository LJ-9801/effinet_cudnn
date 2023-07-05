#include "include/efficientnet.h"
//#define DEBUG

EfficientNet::EfficientNet(){
    this->tensorFormat = CUDNN_TENSOR_NCHW;
    this->dataType = CUDNN_DATA_FLOAT;
    this->createHandle();

    int nDevices;

    cudaGetDeviceCount(&nDevices);
    for (int i = 0; i < nDevices; i++) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);
        printf("Device Number: %d\n", i);
        printf("  Device name: %s\n", prop.name);
        printf("  Memory Clock Rate (KHz): %d\n",
            prop.memoryClockRate);
        printf("  Memory Bus Width (bits): %d\n",
            prop.memoryBusWidth);
        printf("  Peak Memory Bandwidth (GB/s): %f\n\n",
            2.0*prop.memoryClockRate*(prop.memoryBusWidth/8)/1.0e6);
        std::cout << "  Max Threads per Block: " << prop.maxThreadsPerBlock << std::endl;
  }
}

EfficientNet::~EfficientNet(){
    this->destroyHandle();
}


void EfficientNet::createHandle(){
    checkCUDNN( cudnnCreate(&this->cudnnHandle) );
    checkCublasErrors( cublasCreate(&cublasHandle) );
}

void EfficientNet::destroyHandle(){

    this->destroyWeights();
    this->destroyConv();
    this->destroyBn();
    this->destroyActivation();
    this->destroyAvgp();
    this->destoryLinear();
    this->destroySoftmax();

    checkCUDNN( cudnnDestroy(this->cudnnHandle) );
    checkCublasErrors( cublasDestroy(this->cublasHandle) );
}

void EfficientNet::destroyWeights(){
    // free stem weights
    
    if (this->weights.size() > 0){
        std::unordered_map<std::string, WEIGHT>::iterator it;
        for (it = this->weights.begin(); it != this->weights.end(); it++){
            for (int i = 0; i < it->second.size(); i++){
                if(it->second[i].gpu_data != NULL){
                    checkCudaErrors( cudaFree(it->second[i].gpu_data) );
                }
            }
        }
    }
    
}

void EfficientNet::destroyConv(){
    std::unordered_map<CONV_NAME_T, std::pair<CONV_DESC_T, CONV_PARAM_T> >::iterator it;
    for (it = this->conv_map.begin(); it != this->conv_map.end(); it++){
        if (it->second.first.convDesc != NULL){
            checkCUDNN( cudnnDestroyConvolutionDescriptor(it->second.first.convDesc) );
        }
        if (it->second.first.filterDesc != NULL){
            checkCUDNN( cudnnDestroyFilterDescriptor(it->second.first.filterDesc) );
        }
        if (it->second.first.inputDesc != NULL){
            checkCUDNN( cudnnDestroyTensorDescriptor(it->second.first.inputDesc) );
        }
        if (it->second.first.outputDesc != NULL){
            checkCUDNN( cudnnDestroyTensorDescriptor(it->second.first.outputDesc) );
        }
        if (it->second.second.is_bias){
            if (it->second.first.biasDesc != NULL){
                checkCUDNN( cudnnDestroyTensorDescriptor(it->second.first.biasDesc) );
            }
            if (it->second.first.actDesc != NULL){
                checkCUDNN( cudnnDestroyActivationDescriptor(it->second.first.actDesc) );
            }
        }
    }
}

void EfficientNet::destroyBn(){
    std::unordered_map<BN_NAME_T, std::pair<BN_DESC_T, BN_PARAM_T> >::iterator it;
    for (it = this->bn_map.begin(); it != this->bn_map.end(); it++){
        if (it->second.first.inputDesc != NULL){
            checkCUDNN( cudnnDestroyTensorDescriptor(it->second.first.inputDesc) );
        }
        if (it->second.first.outputDesc != NULL){
            checkCUDNN( cudnnDestroyTensorDescriptor(it->second.first.outputDesc) );
        }
        if (it->second.first.bnWeightBiasMeanVarDesc != NULL){
            checkCUDNN( cudnnDestroyTensorDescriptor(it->second.first.bnWeightBiasMeanVarDesc) );
        }
    }
}

void EfficientNet::destroyActivation(){
    std::unordered_map<ACTIVATION_NAME_T, ACTIVATION_DESC_T>::iterator it;
    for (it = this->activation_map.begin(); it != this->activation_map.end(); it++){
        if (it->second.actDesc != NULL){
            checkCUDNN( cudnnDestroyActivationDescriptor(it->second.actDesc) );
        }
        if (it->second.inputDesc != NULL){
            checkCUDNN( cudnnDestroyTensorDescriptor(it->second.inputDesc) );
        }
        if (it->second.outputDesc != NULL){
            checkCUDNN( cudnnDestroyTensorDescriptor(it->second.outputDesc) );
        }
    }
}

void EfficientNet::destoryLinear(){
    std::unordered_map<LINEAR_NAME_T, std::pair<LINEAR_DESC_T, LINEAR_PARAM_T> >::iterator it;
    for (it = this->linear_map.begin(); it != this->linear_map.end(); it++){
        if (it->second.first.inputDesc != NULL){
            checkCUDNN( cudnnDestroyTensorDescriptor(it->second.first.inputDesc) );
        }
        if (it->second.first.outputDesc != NULL){
            checkCUDNN( cudnnDestroyTensorDescriptor(it->second.first.outputDesc) );
        }
        if (it->second.first.filterDesc != NULL){
            checkCUDNN( cudnnDestroyFilterDescriptor(it->second.first.filterDesc) );
        }
    }
}

void EfficientNet::destroyAvgp(){
    std::unordered_map<AVGP_NAME_T, std::pair<AVGP_DESC_T, AVGPOOL_PARAM_T> >::iterator it;
    for (it = this->avgp_map.begin(); it != this->avgp_map.end(); it++){
        if (it->second.first.inputDesc != NULL){
            checkCUDNN( cudnnDestroyTensorDescriptor(it->second.first.inputDesc) );
        }
        if (it->second.first.outputDesc != NULL){
            checkCUDNN( cudnnDestroyTensorDescriptor(it->second.first.outputDesc) );
        }
        if (it->second.first.poolDesc != NULL){
            checkCUDNN( cudnnDestroyPoolingDescriptor(it->second.first.poolDesc) );
        }
    }
}

void EfficientNet::destroySoftmax(){
    std::unordered_map<SOFTMAX_NAME_T, SOFTMAX_DESC_T>::iterator it;
    for (it = this->softmax_map.begin(); it != this->softmax_map.end(); it++){
        if (it->second.inputDesc != NULL){
            checkCUDNN( cudnnDestroyTensorDescriptor(it->second.inputDesc) );
        }
        if (it->second.outputDesc != NULL){
            checkCUDNN( cudnnDestroyTensorDescriptor(it->second.outputDesc) );
        }
    }
}

void EfficientNet::resize(int size, float **data){
    if (*data != NULL)
    {
        checkCudaErrors( cudaFree(*data) );
    }
    checkCudaErrors( cudaMalloc(data, size*sizeof(float)) );
}

void EfficientNet::copy(float* srcData, const dim_t srcDim, float** dstData){
    if(*dstData != NULL){
        checkCudaErrors( cudaFree(*dstData) );
    }
    checkCudaErrors( cudaMalloc(dstData, srcDim.n*srcDim.c*srcDim.h*srcDim.w*sizeof(float)) );
    if(srcData != NULL){
        checkCudaErrors( cudaMemcpy(*dstData, srcData, srcDim.n*srcDim.c*srcDim.h*srcDim.w*sizeof(float), cudaMemcpyDeviceToDevice) );
    }
}

void EfficientNet::loadWeights(std::vector<file_info>& fn){

    // load stem weights
    for (int i = 0; i < 5; i++){
        std::string path = fn[i].path + "/" + fn[i].filename;
        std::string layername = fn[i].layer;
        std::string func = fn[i].function.substr(0, fn[i].function.find("_"));
        std::string datatype = fn[i].function.substr(fn[i].function.find("_")+1, fn[i].function.size());
        std::vector<unsigned long> shape;
        std::vector<float> data;
        bool fortran_order;
        //std::cout << "path: " << path << std::endl;
        npy::LoadArrayFromNumpy(path, shape, fortran_order, data);
        
        MODEL_DATA model_data = {datatype, shape, data};
        int size = 1;
        for (int j = 0; j < shape.size(); j++){
            size *= shape[j];
        }

        checkCudaErrors( cudaMalloc(&model_data.gpu_data, size*sizeof(float)) );
        checkCudaErrors( cudaMemcpy(model_data.gpu_data, data.data(), size*sizeof(float), cudaMemcpyHostToDevice) );

        weights[func].push_back(model_data);
    }

    // load block weights
    for (int i = 5; i < fn.size()-7; i++){
        
        std::string tmp = fn[i].filename.substr(fn[i].filename.find("_")+1, fn[i].filename.size());
        tmp = tmp.substr(tmp.find("_")+1, tmp.size());
        std::string blk = tmp.substr(0, tmp.find("_"));
        std::string num = blk.substr(5, tmp.size()-5);

        std::string path = fn[i].path + "/" + fn[i].filename;
        std::string func = fn[i].function.substr(fn[i].function.find("_")+1, fn[i].function.size()-fn[i].function.find("_"));
        std::string ops = func.substr(0, func.find("_"));
        std::string datatype;
        if (ops == "bn0" || ops == "bn1" || ops == "bn2" || ops == "depthwise"){
            datatype = func.substr(func.find("_")+1, func.size()-func.find("_"));
        }else{
            int pos1 = func.find("_");
            int pos2 = func.find("_", pos1+1);
            datatype = func.substr(pos2+1, func.size()-pos2);
            ops = func.substr(0, pos2);
        }


        //int num_int = std::stoi(num);
        std::vector<unsigned long> shape;
        std::vector<float> data;
        bool fortran_order;
        //std::cout << "path: " << path << std::endl;
        npy::LoadArrayFromNumpy(path, shape, fortran_order, data);
        MODEL_DATA model_data = {datatype, shape, data};


        // copy data to gpu
        int size = 1;
        for (int j = 0; j < shape.size(); j++){
            size *= shape[j];
        }
        checkCudaErrors( cudaMalloc(&model_data.gpu_data, size*sizeof(float)) );
        checkCudaErrors( cudaMemcpy(model_data.gpu_data, data.data(), size*sizeof(float), cudaMemcpyHostToDevice) );
        std::string key = blk + "_" + ops;

        weights[key].push_back(model_data);
    }

    // load head weights
    for (int i = fn.size() - 7; i < fn.size(); i++){
        std::string path = fn[i].path + "/" + fn[i].filename;
        std::string layername = fn[i].layer;
        std::string func = fn[i].function.substr(0, fn[i].function.find("_"));
        std::string datatype = fn[i].function.substr(fn[i].function.find("_")+1, fn[i].function.size());

        std::vector<unsigned long> shape;
        std::vector<float> data;
        bool fortran_order;
        npy::LoadArrayFromNumpy(path, shape, fortran_order, data);
        MODEL_DATA model_data = {datatype, shape, data};

        int size = 1;
        for (int j = 0; j < shape.size(); j++){
            size *= shape[j];
        }

        // copy data to gpu
        checkCudaErrors( cudaMalloc(&model_data.gpu_data, size*sizeof(float)) );
        checkCudaErrors( cudaMemcpy(model_data.gpu_data, data.data(), size*sizeof(float), cudaMemcpyHostToDevice) );

        weights[func].push_back(model_data);
    }
}


void EfficientNet::forward(DATA input){

    // get src data
    float* src_data = input.data.data();
    std::vector<unsigned long> shape = input.shape;
    std::vector<unsigned long> src_shape = input.shape;

    int n, c, h, w;
    if (src_shape.size() == 4){
        n = src_shape[0]; c = src_shape[1]; h = src_shape[2]; w = src_shape[3];
    }else if (src_shape.size() == 3){
        n = 1; c = src_shape[0]; h = src_shape[1]; w = src_shape[2];
    }

    dim_t dim = {n,c,h,w};

    // send src data to device
    float* srcData_d = NULL;
    float* dst_data = NULL;
    
    checkCudaErrors( cudaMalloc(&srcData_d, n*c*h*w*sizeof(float)) );
    checkCudaErrors( cudaMemcpy(srcData_d, src_data, n*c*h*w*sizeof(float), cudaMemcpyHostToDevice) );

    std::cout << "Start Inferencing..." << std::endl;
    std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();

    this->Convolute2dForward("conv1", srcData_d, &dst_data );
    this->Batchnorm2dForward("bn1"  , dst_data,  &srcData_d);
    this->ActivationForward ("swish", srcData_d, &dst_data );

    dim_t indim = this->activation_map["swish"].dim;

    
    float* tmp0 = NULL;
    float* tmp1 = NULL;
    float* tmp2 = NULL;
    
    
    for (int i = 0; i < 32; i++){
        //std::cout << indim.n << " " << indim.c << " " << indim.h << " " << indim.w << std::endl;
        this->MBConvBlockForward(i, dst_data, &srcData_d, indim, tmp0, tmp1, tmp2);
        cudaFree(dst_data);
        dst_data = srcData_d;
        srcData_d = NULL;
    }
    // above is correct
    
    this->Convolute2dForward("conv2", dst_data, &srcData_d);
    this->Batchnorm2dForward("bn2", srcData_d, &dst_data);
    this->ActivationForward("swish1", dst_data, &srcData_d);
    this->AdaptiveAvgPool2dForward("avg_pool", srcData_d, &dst_data);
    this->LinearForward("fc", dst_data, &srcData_d);
    this->SoftmaxForward("softmax", srcData_d, &dst_data);
    
    std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
    std::cout << "Done Inferencing..." << std::endl;
    double time = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
    std::cout << "Inference time: " << time*1e-6 << "s" << std::endl;
    
    checkCudaErrors (cudaDeviceSynchronize());

    std::vector<float> output;
    output.resize(1000);
    checkCudaErrors( cudaMemcpy(output.data(), dst_data, 1000*sizeof(float), cudaMemcpyDeviceToHost) );

    std::vector<float>::iterator biggest = std::max_element(std::begin(output), std::end(output));
    int max_idx = std::distance(std::begin(output), biggest);
    std::cout << "Predicted class: " << max_idx << std::endl;
    

#ifdef DEBUG

    int n1 = 1; int n2 = 1000;
    int N = n1*n2;
        float cputmp[N];
        checkCudaErrors( cudaMemcpy(cputmp, dst_data, N*sizeof(float), cudaMemcpyDeviceToHost) );
        for (int i = 0; i < n1; i++){
            for (int j = 0; j < n2; j++){
                std::cout << cputmp[i*n2+j] << " ";
            }
            std::cout << std::endl;
        }
        std::cout << std::endl;

#endif
    checkCudaErrors( cudaFree(tmp0) );
    checkCudaErrors( cudaFree(tmp1) );
    checkCudaErrors( cudaFree(tmp2) );
    checkCudaErrors( cudaFree(dst_data) );
    checkCudaErrors( cudaFree(srcData_d) );

}

void EfficientNet::MBConvBlockForward(const int num, float* src_data, float** dst_data, dim_t& indim,
                                    float* tmp0, float* tmp1, float* tmp2){

    int n = indim.n; int c = indim.c; int h = indim.h; int w = indim.w;

    this->copy(src_data, indim, &tmp0);

    std::string blk = "block" + std::to_string(num) + "_";

    if(this->weights.find(blk+"expand_conv") != this->weights.end()){
        this->Convolute2dForward(blk+"expand_conv", tmp0, &tmp1);
        this->Batchnorm2dForward(blk+"bn0", tmp1, &tmp2);
        this->ActivationForward(blk+"swish0", tmp2, &tmp0);
    }

    this->Convolute2dForward(blk+"depthwise", tmp0, &tmp1);
    this->Batchnorm2dForward(blk+"bn1", tmp1, &tmp2);
    this->ActivationForward(blk+"swish1", tmp2, &tmp1);
    // save tmp1 for later use

    this->AdaptiveAvgPool2dForward(blk+"avgp", tmp1, &tmp0);
    this->Convolute2dForward(blk+"se_reduce", tmp0, &tmp2);
    this->ActivationForward(blk+"swish2", tmp2, &tmp0);
    this->Convolute2dForward(blk+"se_expand", tmp0, &tmp2);
    this->ActivationForward(blk+"sigmoid", tmp2, &tmp0);

    ACTIVATION_DESC_T sig = this->activation_map[blk+"sigmoid"];
    ACTIVATION_DESC_T sw1 = this->activation_map[blk+"swish1"];
    Tensor in1 = {tmp0, sig.dim};
    Tensor in2 = {tmp1, sw1.dim};
    Tensor out = {tmp2, dim_t()};
    this->BroadcastMul(in2, in1, out);

    this->Convolute2dForward(blk+"project_conv", out.data, &tmp1);
    this->Batchnorm2dForward(blk+"bn2", tmp1, dst_data);
    

    Tensor out2 = {NULL, dim_t()};
    BN_DESC_T bn_desc = this->bn_map[blk+"bn2"].first;
    if(bn_desc.dim.n == n && bn_desc.dim.c == c && bn_desc.dim.h == h && bn_desc.dim.w == w){
        this->Add({src_data, indim}, {*dst_data, bn_desc.dim}, out2);
        cudaFree(*dst_data);
        *dst_data = out2.data;
        out2.data = NULL;
    }
    indim = bn_desc.dim;
}


void EfficientNet::init(){

    stride_t stride = {2,2}; pad_t pad = {1,1}; upscale_t upscale = {1,1}; kernel_t kernel = {3,3}; bool nobias = false;
    CONV_PARAM_T conv1_param = {3, this->round_filters(32), kernel, stride, pad, upscale, nobias, 1};
    CONV_DESC_T conv1_desc;
    CONV_NAME_T conv1 = "conv1";
    conv1_desc.indim = {1,3,244,244};
    WEIGHT conv1weight = this->weights[conv1];
    this->Convolution2dInit(conv1, conv1_desc, conv1_param, conv1weight);

    BN_NAME_T bn1 = "bn1";
    BN_PARAM_T bn1_param = {conv1_desc.outdim.c, 1e-3, 0.01};
    BN_DESC_T bn1_desc;
    bn1_desc.dim = conv1_desc.outdim;
    WEIGHT bn1weight = this->weights[bn1];
    this->BatchNorm2dInit(bn1, bn1_desc, bn1_param, bn1weight);

    ACTIVATION_NAME_T act1 = "swish";
    ACTIVATION_DESC_T act1_desc;
    act1_desc.dim = bn1_desc.dim;
    this->ActivationInit(act1, SWISH, act1_desc);

    
    std::vector<block_arg> args = 
            {{1, 3, {1,1}, 1, 32, 16, 0.25},
             {2, 3, {2,2}, 6, 16, 24, 0.25},
             {2, 5, {2,2}, 6, 24, 40, 0.25},
             {3, 3, {2,2}, 6, 40, 80, 0.25},
             {3, 5, {1,1}, 6, 80, 112, 0.25},
             {4, 5, {2,2}, 6, 112, 192, 0.25},
             {1, 3, {1,1}, 6, 192, 320, 0.25}};

    
    int counter = 0;
    dim_t dim = act1_desc.dim;
    for (int i = 0; i < args.size(); i++){
        args[i].input_filters = this->round_filters(args[i].input_filters);
        args[i].output_filters = this->round_filters(args[i].output_filters);
        int repeats = this->round_repeats(args[i].num_repeat);
        for (int j = 0; j < repeats; j++){
            this->MBConvBlockInit(counter, dim, args[i]);
            args[i].strides = {1,1};
            args[i].input_filters = args[i].output_filters;
            counter++;
        }
    }

    CONV_PARAM_T conv2_param = {dim.c, this->round_filters(1280), {1,1}, {1,1}, {0,0}, {1,1}, false, 1};
    CONV_DESC_T conv2_desc;
    conv2_desc.indim = dim;
    this->Convolution2dInit("conv2", conv2_desc, conv2_param, this->weights["conv2"]);

    BN_PARAM_T bn2_param = {conv2_desc.outdim.c, 1e-3, 0.01};
    BN_DESC_T bn2_desc;
    bn2_desc.dim = conv2_desc.outdim;
    this->BatchNorm2dInit("bn2", bn2_desc, bn2_param, this->weights["bn2"]);

    ACTIVATION_DESC_T swish1_desc;
    swish1_desc.dim = bn2_desc.dim;
    this->ActivationInit("swish1", SWISH, swish1_desc);

    AVGPOOL_PARAM_T avg_param = {bn2_desc.dim, 1, 1};
    this->AdaptiveAvgPool2dInit("avg_pool", avg_param);
    //std::cout << desc.outdim.n << " " << desc.outdim.c << " " << desc.outdim.h << " " << desc.outdim.w << std::endl;

    LINEAR_PARAM_T linear_param = {bn2_desc.dim.c, 1000, true};
    this->LinearInit("fc", linear_param, this->weights["fc"]);

    SOFTMAX_DESC_T softmax_desc;
    softmax_desc.dim = {1, 1000, 1, 1};
    this->SoftmaxInit("softmax", softmax_desc);
}

void EfficientNet::MBConvBlockInit(const int num, dim_t& dim, const block_arg& args){
    std::string block_name = "block" + std::to_string(num) + "_";

    //std::cout << block_name << std::endl;
    
    int kernel_size = args.kernel_size;
    stride_t stride = args.strides;
    int expand_ratio = args.expand_ratio;
    int input_filters = args.input_filters;
    int output_filters = args.output_filters;
    float se_ratio = args.se_ratio;
    pad_t pad = {0,0};

    int n = dim.n; int c = dim.c; int h = dim.h; int w = dim.w;

    int input = input_filters;
    int output = input_filters * expand_ratio;

    
    if(expand_ratio != 1){
        CONV_NAME_T expand_conv_name = block_name + "expand_conv";
        CONV_PARAM_T expand_conv_param = {input, output, {1,1}, {1,1}, {0,0}, {1,1}, false, 1};
        CONV_DESC_T expand_conv_desc;
        expand_conv_desc.indim = dim;
        this->Convolution2dInit(expand_conv_name, expand_conv_desc, expand_conv_param, this->weights[expand_conv_name]);

        BN_NAME_T bn0_name = block_name + "bn0";
        BN_PARAM_T bn0_param = {output, 1e-3, 0.01};
        BN_DESC_T bn0_desc;
        bn0_desc.dim = expand_conv_desc.outdim;
        this->BatchNorm2dInit(bn0_name, bn0_desc, bn0_param, this->weights[bn0_name]);

        ACTIVATION_NAME_T swish0_name = block_name + "swish0";
        ACTIVATION_DESC_T swish0_desc;
        swish0_desc.dim = bn0_desc.dim;
        this->ActivationInit(swish0_name, SWISH, swish0_desc);
        dim = bn0_desc.dim;
    }

    if (stride.h == 2 && stride.w == 2){
        pad.h = (kernel_size-1)/2-1; pad.w = (kernel_size-1)/2;
    }else{
        pad.h = (kernel_size-1)/2; pad.w = (kernel_size-1)/2;
    }

    CONV_NAME_T depthwise_conv_name = block_name + "depthwise";
    CONV_PARAM_T depthwise_conv_param = {output, output, {kernel_size, kernel_size}, stride, pad, {1,1}, false, output};
    CONV_DESC_T depthwise_conv_desc;
    depthwise_conv_desc.indim = dim;
    this->Convolution2dInit(depthwise_conv_name, depthwise_conv_desc, depthwise_conv_param, this->weights[depthwise_conv_name]);

    BN_NAME_T bn1_name = block_name + "bn1";
    BN_PARAM_T bn1_param = {output, 1e-3, 0.01};
    BN_DESC_T bn1_desc;
    bn1_desc.dim = depthwise_conv_desc.outdim;
    //std::cout << bn1_desc.dim.n << " " << bn1_desc.dim.c << " " << bn1_desc.dim.h << " " << bn1_desc.dim.w << std::endl;
    this->BatchNorm2dInit(bn1_name, bn1_desc, bn1_param, this->weights[bn1_name]);

    ACTIVATION_NAME_T swish1 = block_name + "swish1";
    ACTIVATION_DESC_T swish1_desc;
    swish1_desc.dim = bn1_desc.dim;
    this->ActivationInit(swish1, SWISH, swish1_desc);

    int num_squeezed_channels = std::max(1, (int)(input_filters * se_ratio));

    AVGP_NAME_T avgp_name = block_name + "avgp";
    AVGPOOL_PARAM_T avgp_param = {swish1_desc.dim, 1, 1};
    this->AdaptiveAvgPool2dInit(avgp_name, avgp_param);

    CONV_NAME_T se_reduce_name = block_name + "se_reduce";
    CONV_PARAM_T se_reduce_param = {output, num_squeezed_channels, {1,1}, {1,1}, {0,0}, {1,1}, true, 1};
    CONV_DESC_T se_reduce_desc;
    se_reduce_desc.indim = {swish1_desc.dim.n, swish1_desc.dim.c, avgp_param.out_h, avgp_param.out_w};
    this->Convolution2dInit(se_reduce_name, se_reduce_desc, se_reduce_param, this->weights[se_reduce_name]);

    ACTIVATION_NAME_T swish2_name = block_name + "swish2";
    ACTIVATION_DESC_T swish2_desc;
    swish2_desc.dim = se_reduce_desc.outdim;
    this->ActivationInit(swish2_name, SWISH, swish2_desc);
    
    CONV_NAME_T se_expand_name = block_name + "se_expand";
    CONV_PARAM_T se_expand_param = {num_squeezed_channels, output, {1,1}, {1,1}, {0,0}, {1,1}, true, 1};
    CONV_DESC_T se_expand_desc;
    se_expand_desc.indim = se_reduce_desc.outdim;
    this->Convolution2dInit(se_expand_name, se_expand_desc, se_expand_param, this->weights[se_expand_name]);

    ACTIVATION_NAME_T sigmoid = block_name + "sigmoid";
    ACTIVATION_DESC_T sigmoid_desc;
    sigmoid_desc.dim = se_expand_desc.outdim;
    this->ActivationInit(sigmoid, SIGMOID, sigmoid_desc);

    CONV_NAME_T project_conv_name = block_name + "project_conv";
    CONV_PARAM_T project_conv_param = {output, output_filters, {1,1}, {1,1}, {0,0}, {1,1}, false, 1};
    CONV_DESC_T project_conv_desc;
    project_conv_desc.indim = swish1_desc.dim;
    this->Convolution2dInit(project_conv_name, project_conv_desc, project_conv_param, this->weights[project_conv_name]);

    BN_NAME_T bn2_name = block_name + "bn2";
    BN_PARAM_T bn2_param = {output_filters, 1e-3, 0.01};
    BN_DESC_T bn2_desc;
    bn2_desc.dim = project_conv_desc.outdim;
    this->BatchNorm2dInit(bn2_name, bn2_desc, bn2_param, this->weights[bn2_name]);

    dim = bn2_desc.dim;
}

void EfficientNet::LinearInit(const LINEAR_NAME_T name, LINEAR_PARAM_T param, const WEIGHT& weight){
    //std::cout << "init " + name << std::endl;
    int in_features = param.in_features; int out_features = param.out_features;
    bool bias = param.is_bias;
    param.gpu_weight.data = weight[0].gpu_data;

    if(bias){
        param.gpu_bias.data = weight[1].gpu_data;
    }

    // describe input tensor
    LINEAR_DESC_T desc;
    cudnnCreateTensorDescriptor(&desc.inputDesc);
    cudnnSetTensor4dDescriptor(desc.inputDesc, this->tensorFormat, this->dataType, 1, in_features, 1, 1);

    // describe output tensor
    cudnnCreateTensorDescriptor(&desc.outputDesc);
    cudnnSetTensor4dDescriptor(desc.outputDesc, this->tensorFormat, this->dataType, 1, out_features, 1, 1);

    // describe weight tensor
    cudnnCreateFilterDescriptor(&desc.filterDesc);
    cudnnSetFilter4dDescriptor(desc.filterDesc, this->dataType, this->tensorFormat, in_features, out_features, 1, 1);

    // describe bias tensor
    if(bias){
        cudnnCreateTensorDescriptor(&desc.biasDesc);
        cudnnSetTensor4dDescriptor(desc.biasDesc, this->tensorFormat, this->dataType, 1, out_features, 1, 1);
    }

    desc.indim = {1, in_features, 1, 1};
    desc.outdim = {1, out_features, 1, 1};

    this->linear_map[name] = std::make_pair(desc, param);
}

void EfficientNet::Convolution2dInit(const CONV_NAME_T name, CONV_DESC_T& desc, CONV_PARAM_T param, const WEIGHT& weight){

    //std::cout << "init " + name << std::endl;
    int n = desc.indim.n; int c = desc.indim.c; int h = desc.indim.h; int w = desc.indim.w;
    int out_channels = param.out_channels; int in_channels = param.in_channels;
    int kernel_h = param.kernel.h; int kernel_w = param.kernel.w;
    int stride_h = param.stride.h; int stride_w = param.stride.w;
    int pad_h = param.pad.h; int pad_w = param.pad.w;
    int dilation_h = param.upscale.h; int dilation_w = param.upscale.w;
    bool bias = param.is_bias; int groups = param.groups;

    
    checkCUDNN(cudnnCreateTensorDescriptor(&desc.inputDesc));
    checkCUDNN(cudnnCreateFilterDescriptor(&desc.filterDesc));
    checkCUDNN(cudnnCreateConvolutionDescriptor(&desc.convDesc));
    checkCUDNN(cudnnCreateTensorDescriptor(&desc.outputDesc));

    // src tensor descriptor
    checkCUDNN(cudnnSetTensor4dDescriptor(desc.inputDesc,
                                      /*format=*/this->tensorFormat,
                                      /*dataType=*/this->dataType,
                                      /*batch_size=*/n,
                                      /*channels=*/c,
                                      /*image_height=*/h,
                                      /*image_width=*/w));
    // weight tensor descriptor
    int wh = param.kernel.h; int ww = param.kernel.w;
    checkCUDNN(cudnnSetFilter4dDescriptor(desc.filterDesc,
                                      /*dataType=*/this->dataType,
                                      /*format=*/this->tensorFormat,
                                      /*out_channels=*/param.out_channels,
                                      /*in_channels=*/param.in_channels/groups,
                                      /*kernel_height=*/wh,
                                      /*kernel_width=*/ww));
    // output tensor descriptor
    int outn = desc.indim.n;
    int outc = param.out_channels;
    int outh = (h + 2*param.pad.h - (param.upscale.h*(wh-1)+1))/param.stride.h + 1;
    int outw = (w + 2*param.pad.w - (param.upscale.w*(ww-1)+1))/param.stride.w + 1;
    desc.outdim = {outn, outc, outh, outw};

    checkCUDNN(cudnnSetTensor4dDescriptor(desc.outputDesc,
                                      /*format=*/this->tensorFormat,
                                      /*dataType=*/this->dataType,
                                      /*batch_size=*/outn,
                                      /*channels=*/outc,
                                      /*image_height=*/outh,
                                      /*image_width=*/outw));

    checkCUDNN(cudnnSetConvolutionGroupCount(desc.convDesc, groups));

    // 2d convolution descriptor
    checkCUDNN(cudnnSetConvolution2dDescriptor(desc.convDesc,
                                            /*pad_height=*/param.pad.h,
                                            /*pad_width=*/param.pad.w,
                                            /*vertical_stride=*/param.stride.h,
                                            /*horizontal_stride=*/param.stride.w,
                                            /*dilation_height=*/param.upscale.h,
                                            /*dilation_width=*/param.upscale.w,
                                            /*mode=*/CUDNN_CROSS_CORRELATION,
                                            /*computeType=*/this->dataType));

    int requestedAlgoCount = CUDNN_CONVOLUTION_FWD_ALGO_COUNT;
    int returnedAlgoCount = -1;
    cudnnConvolutionFwdAlgoPerf_t results[2 * CUDNN_CONVOLUTION_FWD_ALGO_COUNT];
    checkCUDNN(cudnnFindConvolutionForwardAlgorithm(this->cudnnHandle,
                                                desc.inputDesc,
                                                desc.filterDesc,
                                                desc.convDesc,
                                                desc.outputDesc,
                                                requestedAlgoCount,
                                                &returnedAlgoCount,
                                                results));
    desc.algo = (cudnnConvolutionFwdAlgo_t)results[CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM].algo;


    if (weight.size() == 0){
        std::cout << "weight not found" << std::endl;
        return;
    }
    if (weight.size() == 1 && param.is_bias == false){

        param.gpu_weight.data = weight[0].gpu_data;
        param.gpu_weight.shape = weight[0].shape;

        param.gpu_bias.data = NULL;
    }else if (weight.size() == 2 && param.is_bias == true){
        checkCUDNN(cudnnCreateTensorDescriptor(&desc.biasDesc));
        checkCUDNN(cudnnSetTensor4dDescriptor(desc.biasDesc,
                                      /*format=*/this->tensorFormat,
                                      /*dataType=*/this->dataType,
                                      /*batch_size=*/1,
                                      /*channels=*/param.out_channels,
                                      /*image_height=*/1,
                                      /*image_width=*/1));

        checkCUDNN(cudnnCreateActivationDescriptor(&desc.actDesc));
        checkCUDNN(cudnnSetActivationDescriptor(desc.actDesc,
                                            CUDNN_ACTIVATION_IDENTITY,
                                            CUDNN_PROPAGATE_NAN,
                                            0.0));
        param.gpu_weight.data = weight[0].gpu_data;
        param.gpu_weight.shape = weight[0].shape;

        param.gpu_bias.data = weight[1].gpu_data;
        param.gpu_bias.shape = weight[1].shape;
    }else{
        std::cout << "weight size error" << std::endl;
        return;
    }
    
    // get workspace size
    // get convolution forward workspace size
    desc.sizeInBytes=0;
    desc.workSpace=NULL;

    // save desc and param
    std::pair<CONV_DESC_T, CONV_PARAM_T> conv = std::make_pair(desc, param);
    if (this->conv_map.find(name) != this->conv_map.end()){
        std::cout << "convolution name already exists, overwrite parameters" << std::endl;
        this->conv_map[name] = conv;
    }else{
        this->conv_map.insert(std::make_pair(name, conv));
    }

    //std::cout << name +" init done" << std::endl;
}

 void EfficientNet::BatchNorm2dInit(const BN_NAME_T name, BN_DESC_T& desc, BN_PARAM_T param, const WEIGHT& weight){

    int n = desc.dim.n; int c = desc.dim.c; int h = desc.dim.h; int w = desc.dim.w;

    // create tensor descriptor
    checkCUDNN(cudnnCreateTensorDescriptor(&desc.inputDesc));
    checkCUDNN(cudnnCreateTensorDescriptor(&desc.outputDesc));
    checkCUDNN(cudnnCreateTensorDescriptor(&desc.bnWeightBiasMeanVarDesc));

    // input tensor descriptor
    checkCUDNN(cudnnSetTensor4dDescriptor(desc.inputDesc,
                                      /*format=*/this->tensorFormat,
                                      /*dataType=*/this->dataType,
                                      /*batch_size=*/n,
                                      /*channels=*/c,
                                      /*image_height=*/h,
                                      /*image_width=*/w));

    // output tensor descriptor
    checkCUDNN(cudnnSetTensor4dDescriptor(desc.outputDesc,
                                      /*format=*/this->tensorFormat,
                                      /*dataType=*/this->dataType,
                                      /*batch_size=*/n,
                                      /*channels=*/c,
                                      /*image_height=*/h,
                                      /*image_width=*/w));
    
    // batchnorm scale, bias, mean, and variance tensors descriptor
    checkCUDNN(cudnnSetTensor4dDescriptor(desc.bnWeightBiasMeanVarDesc,
                                      /*format=*/this->tensorFormat,
                                      /*dataType=*/this->dataType,
                                      /*batch_size=*/1,
                                      /*channels=*/c,
                                      /*image_height=*/1,
                                      /*image_width=*/1));
    

    // batchnorm mode
    desc.mode = CUDNN_BATCHNORM_SPATIAL;
    desc.dim.n = n; desc.dim.c = c; desc.dim.h = h; desc.dim.w = w;

    if (weight.size() < 4){
        std::cout << "weight size in 4 for batchnorm" << std::endl;
        return;
    }

    param.gpu_weight = {weight[0].gpu_data, weight[0].shape};
    param.gpu_bias = {weight[1].gpu_data, weight[1].shape};
    param.gpu_mean = {weight[2].gpu_data, weight[2].shape};
    param.gpu_var = {weight[3].gpu_data, weight[3].shape};

    // save desc and param
    std::pair<BN_DESC_T, BN_PARAM_T> bn = std::make_pair(desc, param);
    if (this->bn_map.find(name) != this->bn_map.end()){
        std::cout << "batchnorm name already exists, overwrite parameters" << std::endl;
        this->bn_map[name] = bn;
    }else{
        this->bn_map.insert(std::make_pair(name, bn));
    }
 }

void EfficientNet::ActivationInit(const ACTIVATION_NAME_T name, const ActivationType type, ACTIVATION_DESC_T& desc){
    if (type == RELU){
        desc.mode = CUDNN_ACTIVATION_RELU;
    }else if (type == SIGMOID){
        desc.mode = CUDNN_ACTIVATION_SIGMOID;
    }else if (type == TANH){
        desc.mode = CUDNN_ACTIVATION_TANH;
    }else if (type == CLIPPED_RELU){
        desc.mode = CUDNN_ACTIVATION_CLIPPED_RELU;
    }else if (type == ELU){
        desc.mode = CUDNN_ACTIVATION_ELU;
    }else if (type == SWISH){
        desc.mode = CUDNN_ACTIVATION_SWISH;
    }

    // create tensor descriptor
    checkCUDNN(cudnnCreateTensorDescriptor(&desc.inputDesc));
    checkCUDNN(cudnnCreateTensorDescriptor(&desc.outputDesc));

    int n = desc.dim.n; int c = desc.dim.c; int h = desc.dim.h; int w = desc.dim.w;

    checkCUDNN(cudnnSetTensor4dDescriptor(desc.inputDesc,
                                      /*format=*/this->tensorFormat,
                                      /*dataType=*/this->dataType,
                                      /*batch_size=*/n,
                                      /*channels=*/c,
                                      /*image_height=*/h,
                                      /*image_width=*/w));
    checkCUDNN(cudnnSetTensor4dDescriptor(desc.outputDesc,
                                      /*format=*/this->tensorFormat,
                                      /*dataType=*/this->dataType,
                                      /*batch_size=*/n,
                                      /*channels=*/c,
                                      /*image_height=*/h,
                                      /*image_width=*/w));

    // create activation descriptor
    checkCUDNN(cudnnCreateActivationDescriptor(&desc.actDesc));
    checkCUDNN(cudnnSetActivationDescriptor(desc.actDesc,
                                        /*mode=*/desc.mode,
                                        /*reluNanOpt=*/CUDNN_PROPAGATE_NAN,
                                        /*relu_coef=*/0));

    // save desc and param
    if (this->activation_map.find(name) != this->activation_map.end()){
        std::cout << "activation name already exists, overwrite parameters" << std::endl;
        this->activation_map[name] = desc;
    }else{
        this->activation_map.insert(std::make_pair(name, desc));
    }
}

void EfficientNet::SoftmaxInit(SOFTMAX_NAME_T name, SOFTMAX_DESC_T desc){
    checkCUDNN(cudnnCreateTensorDescriptor(&desc.inputDesc));
    checkCUDNN(cudnnCreateTensorDescriptor(&desc.outputDesc));
    
    desc.mode = CUDNN_SOFTMAX_MODE_INSTANCE;
    desc.algo = CUDNN_SOFTMAX_ACCURATE;

    int n = desc.dim.n; int c = desc.dim.c; int h = desc.dim.h; int w = desc.dim.w;

    checkCUDNN(cudnnSetTensor4dDescriptor(desc.inputDesc,
                                      /*format=*/this->tensorFormat,
                                      /*dataType=*/this->dataType,
                                      /*batch_size=*/n,
                                      /*channels=*/c,
                                      /*image_height=*/h,
                                      /*image_width=*/w));
    checkCUDNN(cudnnSetTensor4dDescriptor(desc.outputDesc,
                                        /*format=*/this->tensorFormat,
                                        /*dataType=*/this->dataType,
                                        /*batch_size=*/n,
                                        /*channels=*/c,
                                        /*image_height=*/h,
                                        /*image_width=*/w));

    // save desc
    if (this->softmax_map.find(name) != this->softmax_map.end()){
        std::cout << "softmax name already exists, overwrite parameters" << std::endl;
        this->softmax_map[name] = desc;
    }else{
        this->softmax_map.insert(std::make_pair(name, desc));
    }
}

void EfficientNet::AdaptiveAvgPool2dInit(const AVGP_NAME_T name, AVGPOOL_PARAM_T& param){
    int in_n = param.indim.n; int in_c = param.indim.c; int in_h = param.indim.h; int in_w = param.indim.w;
    int out_h = param.out_h; int out_w = param.out_w;
    
    // set adaptive pooling kernel, stride, and padding
    int stride_h = in_h/out_h; int stride_w = in_w/out_w;
    int kernel_h = in_h - (out_h-1)*stride_h; int kernel_w = in_w - (out_w-1)*stride_w; 
    int pad_h = 0; int pad_w = 0;

    // create pooling descriptor
    AVGP_DESC_T desc;
    desc.mode = CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING;
    desc.maxpoolingNanOpt = CUDNN_PROPAGATE_NAN;
    checkCUDNN(cudnnCreatePoolingDescriptor(&desc.poolDesc));
    checkCUDNN(cudnnCreateTensorDescriptor(&desc.inputDesc));
    checkCUDNN(cudnnCreateTensorDescriptor(&desc.outputDesc));

    checkCUDNN(cudnnSetPooling2dDescriptor(desc.poolDesc,
                                        /*mode=*/desc.mode,
                                        /*maxpoolingNanOpt=*/desc.maxpoolingNanOpt,
                                        /*windowHeight=*/kernel_h,
                                        /*windowWidth=*/kernel_w,
                                        /*verticalPadding=*/pad_h,
                                        /*horizontalPadding=*/pad_w,
                                        /*verticalStride=*/stride_h,
                                        /*horizontalStride=*/stride_w));
    
    checkCUDNN(cudnnSetTensor4dDescriptor(desc.inputDesc,
                                        /*format=*/this->tensorFormat,
                                        /*dataType=*/this->dataType,
                                        /*batch_size=*/in_n,
                                        /*channels=*/in_c,
                                        /*image_height=*/in_h,
                                        /*image_width=*/in_w));

    checkCUDNN(cudnnSetTensor4dDescriptor(desc.outputDesc,
                                        /*format=*/this->tensorFormat,
                                        /*dataType=*/this->dataType,
                                        /*batch_size=*/in_n,
                                        /*channels=*/in_c,
                                        /*image_height=*/out_h,
                                        /*image_width=*/out_w));

    // save desc and param
    if (this->avgp_map.find(name) != this->avgp_map.end()){
        std::cout << "adaptive avgpool name already exists, overwrite parameters" << std::endl;
        this->avgp_map[name] = std::make_pair(desc, param);
    }else{
        this->avgp_map.insert(std::make_pair(name, std::make_pair(desc, param)));
    }
}

void EfficientNet::BroadcastMul(const Tensor& in1, const Tensor& in2, Tensor& out){
    int n1 = in1.shape.n; int c1 = in1.shape.c; int h1 = in1.shape.h; int w1 = in1.shape.w;
    int n2 = in2.shape.n; int c2 = in2.shape.c; int h2 = in2.shape.h; int w2 = in2.shape.w;
    
    //assert(c1 == c2 && "broadcast mul error: channel mismatch");

    int out_n = n1 > n2 ? n1 : n2;
    int out_c = c1 > c2 ? c1 : c2;
    int out_h = h1 > h2 ? h1 : h2;
    int out_w = w1 > w2 ? w1 : w2;
    out.shape = {out_n, out_c, out_h, out_w};

    float* outData = NULL;
    this->resize(out_n*out_c*out_h*out_w, &out.data);

  
    callBroadCastMultiply(in1.data, {n1, c1, h1, w1}, in2.data, {n2, c2, h2, w2}, out.data, out.shape);
    
}


void EfficientNet::Add(const Tensor& in1, const Tensor& in2, Tensor& out){
    int n1 = in1.shape.n; int c1 = in1.shape.c; int h1 = in1.shape.h; int w1 = in1.shape.w;
    int n2 = in2.shape.n; int c2 = in2.shape.c; int h2 = in2.shape.h; int w2 = in2.shape.w;

    int out_n = n1 > n2 ? n1 : n2;
    int out_c = c1 > c2 ? c1 : c2;
    int out_h = h1 > h2 ? h1 : h2;
    int out_w = w1 > w2 ? w1 : w2;
    out.shape = {out_n, out_c, out_h, out_w};

    float* outData = NULL;
    this->resize(out_n*out_c*out_h*out_w, &out.data);

    
    callbroadCastAdd(in1.data, {n1, c1, h1, w1}, in2.data, {n2, c2, h2, w2}, out.data, out.shape);

}

void EfficientNet::Convolute2dForward( const CONV_NAME_T name, float* srcData, float** dstData){

#ifdef DEBUG
    std::cout << "convolute2dForward " + name << std::endl;
#endif

    // get desc and param
    if (this->conv_map.find(name) == this->conv_map.end()){
        std::cout << "convolution name not found" << std::endl;
        return;
    }

    CONV_DESC_T desc = this->conv_map[name].first;
    CONV_PARAM_T param = this->conv_map[name].second;

    int out_size = desc.outdim.n*desc.outdim.c*desc.outdim.h*desc.outdim.w;
    this->resize(out_size, dstData);
    float alpha = 1.0f;
    float beta = 0.0f;

    
    checkCUDNN( cudnnGetConvolutionForwardWorkspaceSize(this->cudnnHandle,
                                            desc.inputDesc,
                                            desc.filterDesc,
                                            desc.convDesc,
                                            desc.outputDesc,
                                            desc.algo,
                                            &desc.sizeInBytes) );

    if (desc.sizeInBytes!=0)
    {
        checkCudaErrors( cudaMalloc(&desc.workSpace,desc.sizeInBytes) );
    }


    float* weigth_d = NULL;
    float* bias_d = NULL;
    if (param.is_bias){
        weigth_d = param.gpu_weight.data;
        bias_d = param.gpu_bias.data;
        
        checkCUDNN(cudnnConvolutionBiasActivationForward(this->cudnnHandle, &alpha, 
                                        desc.inputDesc, srcData, 
                                        desc.filterDesc, weigth_d, 
                                        desc.convDesc, desc.algo, 
                                        desc.workSpace, desc.sizeInBytes, &beta, 
                                        desc.outputDesc, *dstData, 
                                        desc.biasDesc, bias_d, 
                                        desc.actDesc, desc.outputDesc, *dstData) );
        
    }else{
        weigth_d = param.gpu_weight.data;
        ( cudnnConvolutionForward(this->cudnnHandle, &alpha, 
                                        desc.inputDesc, srcData, 
                                        desc.filterDesc, weigth_d, 
                                        desc.convDesc, desc.algo, 
                                        desc.workSpace, desc.sizeInBytes, &beta, 
                                        desc.outputDesc, *dstData) );
    }
    

    cudaFree(desc.workSpace);
#ifdef DEBUG
    std::cout << name +" forward done\n" << std::endl;
#endif
}


void EfficientNet::Batchnorm2dForward( const BN_NAME_T name, float* srcData, float** dstData){
    
#ifdef DEBUG
        std::cout << "batchnorm2dForward " + name << std::endl;
#endif
    
        // get desc and param
        if (this->bn_map.find(name) == this->bn_map.end()){
            std::cout << "batchnorm name not found" << std::endl;
            return;
        }
    
        BN_DESC_T desc = this->bn_map[name].first;
        BN_PARAM_T param = this->bn_map[name].second;
    
        float beta = param.momentum;
        float alpha = 1.0f - param.momentum;


        int out_size = desc.dim.n*desc.dim.c*desc.dim.h*desc.dim.w;
        this->resize(out_size, dstData);

        checkCUDNN( cudnnBatchNormalizationForwardInference(this->cudnnHandle, desc.mode, &alpha, &beta, 
                                                            desc.inputDesc, srcData, desc.outputDesc, *dstData, 
                                                            desc.bnWeightBiasMeanVarDesc, param.gpu_weight.data, param.gpu_bias.data, 
                                                            param.gpu_mean.data, param.gpu_var.data, param.eps) );
#ifdef DEBUG
        std::cout << name +" forward done\n" << std::endl;
#endif
}

void EfficientNet::ActivationForward( const ACTIVATION_NAME_T name, float* srcData, float** dstData){
    if(this->activation_map.find(name) == this->activation_map.end()){
        std::cout << "activation name not found" << std::endl;
        return;
    }

#ifdef DEBUG
    std::cout << "activationForward " + name << std::endl;
#endif
    float alpha = 1.0f;
    float beta = 0.0f;

    ACTIVATION_DESC_T desc = this->activation_map[name];
    this->resize(desc.dim.n*desc.dim.c*desc.dim.h*desc.dim.w, dstData);
    checkCUDNN( cudnnActivationForward(this->cudnnHandle, 
                                        desc.actDesc, &alpha, 
                                        desc.inputDesc, srcData, 
                                        &beta, desc.outputDesc, *dstData) );
#ifdef DEBUG
    std::cout << name +" forward done\n" << std::endl;
#endif

}

void EfficientNet::AdaptiveAvgPool2dForward(const AVGP_NAME_T name, float* srcData, float** dstData){
    if(this->avgp_map.find(name) == this->avgp_map.end()){
        std::cout << "adaptive avgpool name not found" << std::endl;
        return;
    }
    
#ifdef DEBUG
    std::cout << "adaptiveAvgPool2dForward " + name << std::endl;
#endif

    float alpha = 1.0f;
    float beta = 0.0f;

    AVGP_DESC_T desc = this->avgp_map[name].first;
    AVGPOOL_PARAM_T param = this->avgp_map[name].second;
    this->resize(param.indim.c*param.indim.h*param.indim.w, dstData);
    checkCUDNN( cudnnPoolingForward(this->cudnnHandle, 
                                        desc.poolDesc, &alpha, 
                                        desc.inputDesc, srcData, 
                                        &beta, desc.outputDesc, *dstData) );

#ifdef DEBUG
    std::cout << name +" forward done\n" << std::endl;
#endif
}

void EfficientNet::LinearForward(const LINEAR_NAME_T name, float* srcData, float** dstData){
    if(this->linear_map.find(name) == this->linear_map.end()){
        std::cout << "linear name not found" << std::endl;
        return;
    }

#ifdef DEBUG
    std::cout << "linearForward " + name << std::endl;
#endif

    LINEAR_PARAM_T param = this->linear_map[name].second;
    LINEAR_DESC_T desc = this->linear_map[name].first;

    float alpha = 1.0f;
    float beta = 0.0f;

    int out_size = param.out_features;
    this->resize(out_size, dstData);

    checkCublasErrors( cublasSgemm_v2(this->cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 
                                        1, param.in_features, param.out_features,
                                        &alpha, srcData, 1,
                                        param.gpu_weight.data, param.in_features,
                                        &beta, *dstData, 1) );
    if(param.is_bias){
        checkCUDNN( cudnnAddTensor(this->cudnnHandle, &alpha, 
                                        desc.biasDesc, param.gpu_bias.data, 
                                        &alpha, desc.outputDesc, *dstData) );
    }
}

void EfficientNet::SoftmaxForward(const SOFTMAX_NAME_T name, float* srcData, float** dstData){
    if(this->softmax_map.find(name) == this->softmax_map.end()){
        std::cout << "softmax name not found" << std::endl;
        return;
    }

#ifdef DEBUG
    std::cout << "softmaxForward " + name << std::endl;
#endif

    SOFTMAX_DESC_T desc = this->softmax_map[name];
    float alpha = 1.0f;
    float beta = 0.0f;

    this->resize(desc.dim.n*desc.dim.c*desc.dim.h*desc.dim.w, dstData);
    checkCUDNN( cudnnSoftmaxForward(this->cudnnHandle, 
                                        desc.algo, desc.mode, &alpha, 
                                        desc.inputDesc, srcData, 
                                        &beta, desc.outputDesc, *dstData) );

#ifdef DEBUG
    std::cout << name +" forward done\n" << std::endl;
#endif
}


int EfficientNet::round_filters(int filters){
    float multiplier = 1.4; // v4 params
    int divisor = 8;
    int new_filters = std::max(divisor, int(filters * multiplier + divisor / 2) / divisor * divisor);
    if (new_filters < 0.9 * filters){
        new_filters += divisor;
    }
    return new_filters;
}

int EfficientNet::round_repeats(int repeats){
    float multiplier = 1.8; // v4 params
    return int(std::ceil(multiplier * repeats));
}

