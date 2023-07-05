#if !defined(_ERROR_UTIL_H_)
#define _ERROR_UTIL_H_
#include <sstream>
#include <stdlib.h>
#include <stdio.h>
#include <iostream>

#define FatalError(s) do {                                             \
    std::cout << std::flush << "ERROR: " << s << " in " <<             \
              __FILE__ << ':' << __LINE__ << "\nAborting...\n";        \
    cudaDeviceReset();                                                 \
    exit(-1);                                                          \
} while (0)

#define checkCUDNN(status) do {                                        \
    std::stringstream _err;                                            \
    if (status != CUDNN_STATUS_SUCCESS) {                              \
      _err << "cudnn failure (" << cudnnGetErrorString(status) << ')'; \
      FatalError(_err.str());                                          \
    }                                                                  \
} while(0)

#define checkCublasErrors(status) do {                                 \
    std::stringstream _err;                                            \
    if (status != 0) {                                                 \
      _err << "cublas failure (code=" << status << ')';                \
      FatalError(_err.str());                                          \
    }                                                                  \
} while(0)

#define checkCudaErrors(status) do {                                   \
    std::stringstream _err;                                            \
    if (status != 0) {                                                 \
      _err << "cuda failure (" << cudaGetErrorString(status) << ')';   \
      FatalError(_err.str());                                          \
    }                                                                  \
} while(0)
#endif