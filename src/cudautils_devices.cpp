#include "cudautils_devices.h"

// CUDA
#include <cuda.h>
#include <cuda_gl_interop.h>

//==================================================================================================
int findCapableDevice()
{
    int dev;
    int bestDev = -1;

    int deviceCount = 0;
    cudaError_t error_id = cudaGetDeviceCount(&deviceCount);

    if (error_id != cudaSuccess)
    {
        fprintf(stderr, "cudaGetDeviceCount returned %d\n-> %s\n", (int)error_id, cudaGetErrorString(error_id));
        exit(EXIT_FAILURE);
    }

    if (deviceCount==0)
    {
        fprintf(stderr,"There are no CUDA capable devices.\n");
        exit(EXIT_SUCCESS);
    }
    else
    {
        fprintf(stdout,"Found %d CUDA Capable device(s) supporting CUDA\n", deviceCount);
    }

    for (dev = 0; dev < deviceCount; ++dev)
    {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, dev);

        if (checkCUDAProfile(dev, MIN_RUNTIME_VERSION, MIN_COMPUTE_VERSION))
        {
            fprintf(stdout,"\nFound CUDA Capable Device %d: \"%s\"\n", dev, deviceProp.name);

            if (bestDev == -1)
            {
                bestDev = dev;
                fprintf(stdout, "Setting active device to %d\n", bestDev);
            }
        }
    }

    if (bestDev == -1)
    {
        fprintf(stderr, "\nNo configuration with available capabilities was found.  Test has been waived.\n");
        fprintf(stderr, "The CUDA Sample minimum requirements:\n");
        fprintf(stderr, "\tCUDA Compute Capability >= %d.%d is required\n", MIN_COMPUTE_VERSION/16, MIN_COMPUTE_VERSION%16);
        fprintf(stderr, "\tCUDA Runtime Version    >= %d.%d is required\n", MIN_RUNTIME_VERSION/1000, (MIN_RUNTIME_VERSION%100)/10);
        exit(EXIT_SUCCESS);
    }

    return bestDev;
}

//==================================================================================================
bool checkCUDAProfile(int dev, int min_runtime, int min_compute)
{
    int runtimeVersion = 0;

    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, dev);

    fprintf(stdout,"\nDevice %d: \"%s\"\n", dev, deviceProp.name);
    cudaRuntimeGetVersion(&runtimeVersion);
    fprintf(stdout,"  CUDA Runtime Version     :\t%d.%d\n", runtimeVersion/1000, (runtimeVersion%100)/10);
    fprintf(stdout,"  CUDA Compute Capability  :\t%d.%d\n", deviceProp.major, deviceProp.minor);

    if (runtimeVersion >= min_runtime && ((deviceProp.major<<4) + deviceProp.minor) >= min_compute)
    {
        return true;
    }
    else
    {
        return false;
    }
}
