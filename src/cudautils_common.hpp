#ifndef CUDA_UTILS_COMMON_H
#define CUDA_UTILS_COMMON_H

// CUDA defines
#define MIN_RUNTIME_VERSION 1000
#define MIN_COMPUTE_VERSION 0x10

//==================================================================================================
// This will output the proper CUDA error strings in the event that a CUDA host call returns an error
#ifndef checkCudaErrors
#define checkCudaErrors(err)           __checkCudaErrors (err, __FILE__, __LINE__)
#endif

inline void __checkCudaErrors(cudaError err, const char *file, const int line)
{
    if (cudaSuccess != err)
    {
        fprintf(stderr, "%s(%i) : CUDA Runtime API error %d: %s.\n",
                file, line, (int)err, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

//==================================================================================================
inline int iDivUp(int a, int b)
{
    return ((a % b) != 0) ? (a / b + 1) : (a / b);
}


#endif // CUDA_UTILS_COMMON_H
