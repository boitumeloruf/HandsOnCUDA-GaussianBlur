#ifndef CUDA_UTILS_DEVICES_H
#define CUDA_UTILS_DEVICES_H

// CUDA
#include <cuda_runtime.h>
#include <helper_functions.h>
#include <helper_cuda.h>

#include "cudautils_common.hpp"

int findCapableDevice();
bool checkCUDAProfile(int dev, int min_runtime = MIN_RUNTIME_VERSION, int min_compute = MIN_COMPUTE_VERSION);


#endif // CUDA_UTILS_DEVICES_H
