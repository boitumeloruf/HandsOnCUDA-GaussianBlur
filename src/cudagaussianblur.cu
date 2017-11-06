////////////////////////////////////////////////////////////////////////////////
//! Copyright 2017 Boitumelo Ruf. All rights reserved.
////////////////////////////////////////////////////////////////////////////////

#include "cudagaussianblur.cuh"

// std
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

// opencv
#include <opencv2/imgproc.hpp>

// cuda runtime
#include <cuda_runtime.h>
#include <texture_types.h>

// cuda sdk
#include <helper_functions.h>  // helper for shared that are common to CUDA Samples
#include <helper_cuda.h>       // helper for checking cuda initialization and error checking
#include <helper_string.h>     // helper functions for string parsing

#include "cudautils_common.hpp"
#include "cudautils_devices.h"
#include "cudautils_memory.h"
#include "gaussianblur.kernel.cuh"


//==================================================================================================
cv::Mat runCudaGaussianBlur(const cv::Mat& inputImg)
{
  findCapableDevice();

  //--- adjust input image to Blocksize ---
  cv::Size imgSize = inputImg.size();
  imgSize.width -= ((imgSize.width % BLOCKSIZE_X) != 0) ?
                                 (imgSize.width % BLOCKSIZE_X) : 0;
  imgSize.height -= ((imgSize.height % BLOCKSIZE_Y) != 0) ?
                                (imgSize.height % BLOCKSIZE_Y) : 0;
  cv::resize(inputImg, inputImg, imgSize);


  //--- allocate memory ---
  //--- normalized texture coordinates are needed in order to use mirror border handler ---
  cudaTextureObject_t inputImgTex = uploadImageToTexture<uint>(inputImg, cudaAddressModeMirror, false);
  cudaArray* outputSurf_data;
  cudaSurfaceObject_t outputSurf = createSurfaceObject<uint>(imgSize, outputSurf_data);

  //--- run kernel ---
  dim3 numThreads = dim3(BLOCKSIZE_X, BLOCKSIZE_Y, 1);
  dim3 numBlocks = dim3(imgSize.width / numThreads.x,
                        imgSize.height/ numThreads.y);


  // First run the warmup kernel (which we'll use to get the GPU in the correct max power state
  applyGaussianBlur<<<numBlocks, numThreads>>>(inputImgTex, outputSurf, imgSize.width, imgSize.height);
  cudaDeviceSynchronize();

  //--- Allocate CUDA events that we'll use for timing ---
  cudaEvent_t start, stop;
  checkCudaErrors(cudaEventCreate(&start));
  checkCudaErrors(cudaEventCreate(&stop));

  printf("Launching CUDA Kernel\n");

  //--- Record the start event ---
  checkCudaErrors(cudaEventRecord(start, NULL));

  //--- launch kernel ---
  applyGaussianBlur<<<numBlocks, numThreads>>>(inputImgTex, outputSurf, imgSize.width, imgSize.height);

  //--- Record the stop event ---
  checkCudaErrors(cudaEventRecord(stop, NULL));

  //--- Wait for the stop event to complete ---
  checkCudaErrors(cudaEventSynchronize(stop));

  //--- Check to make sure the kernel didn't fail ---
  getLastCudaError("Kernel execution failed");

  float msecTotal = 0.0f;
  checkCudaErrors(cudaEventElapsedTime(&msecTotal, start, stop));

  //--- download result ---
  cv::Mat outputImg = downloadCudaArrayToImage<uint, uchar>(outputSurf_data, imgSize);

  printf("Input Size  [%dx%d], ", imgSize.width, imgSize.height);
  printf("GPU processing time : %.4f (ms)\n", msecTotal);
  printf("Pixel throughput    : %.3f Mpixels/sec\n",
         ((float)(imgSize.width * imgSize.height*1000.f)/msecTotal)/1000000);
  printf("------------------------------------------------------------------\n");

  // free memory
  checkCudaErrors(
        cudaDestroyTextureObject(inputImgTex));
  checkCudaErrors(
        cudaDestroySurfaceObject(outputSurf));

  return outputImg;
}
