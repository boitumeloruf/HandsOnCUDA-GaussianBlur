////////////////////////////////////////////////////////////////////////////////
//! Copyright 2017 Boitumelo Ruf. All rights reserved.
////////////////////////////////////////////////////////////////////////////////

#ifndef CUDA_UTILS_MEMORY_H
#define CUDA_UTILS_MEMORY_H

// std
#include <vector>

// OpenCV
#include <opencv2/core.hpp>

// cuda runtime
#include <cuda_runtime.h>
#include <texture_types.h>

// cuda sdk
#include <helper_functions.h>
#include <helper_cuda.h>

#include "cudautils_common.hpp"

/**
 * @brief Allocate Memory on device.
 * @param[in] iSize Size of array to allocate.
 * @return Pointer allocated memory on device.
 * @tparam memT Type of memory that is to be allocated. (uchar, int, float)
 * @note Note that this function may also return error codes.
 * If so, return value is undefined.
 */
template<typename memT>
memT* allocateDeviceMem(const int iSize);

/**
 * @brief Download data from device into array on host.
 * @param[in] iDevArray Pointer to array on device.
 * @param[in] iSize Size of array to download.
 * @return Pointer of array on host.
 * @tparam memT Type of memory that is to be allocated. (uchar, int, float)
 * @note Note that this function may also return error codes.
 * If so, return value is undefined.
 */
template<typename memT>
memT* downloadArray(const memT* iDevArray, const int iSize);

/**
 * @brief Download array data from device into single channel image (cv::Mat) on host.
 * @param[in] iDevArray Pointer to array on device.
 * @param[in] iImgSize Size of image.
 * @return Image as cv::Mat.
 * @tparam memT Type of memory that is to be allocated. (uchar, int, float)
 * @note Note that this function may also return error codes.
 * If so, return value is undefined.
 */
template<typename memT>
cv::Mat downloadArrayToImage(const memT* iDevArray, const cv::Size iImgSize);

/**
 * @brief Method to create array on device memory.
 * @param[in] iWidth Width of array.
 * @param[in] iHeight Height of array. Default = 0, 1D Array.
 * @return Pointer to allocated cudaArray.
 * @tparam memT Type of memory that is to be allocated. (uchar, int, float)
 * @note Note that this function may also return error codes.
 * If so, return value is undefined.
 */
template<typename memT>
cudaArray* createCudaArray(const int iWidth, const int iHeight = 0);

/**
 * @brief Download cuda array data from device into single channel image (cv::Mat) on host.
 * Can be used to download data from surface object.
 * @param[in] iCuArray Pointer to cuda array on device.
 * @param[in] iImgSize Size of image.
 * @return Image as cv::Mat.
 * @tparam imgT Type of memory that is to be allocated. (uchar, int, float)
 * @note Note that this function may also return error codes.
 * If so, return value is undefined.
 * @note The Cuda Array will not be destroyed after download.
 * @see createSurfaceObject()
 */
template<typename memT, typename imgT>
cv::Mat downloadCudaArrayToImage(cudaArray* iCuArray, const cv::Size iImgSize);

/**
 * @brief Method to upload image (cv::Mat) to texture memory.
 * @param[in] iImgToUpload Image to upload.
 * @param[in] iAddrMode Addres mode to access texture. Default: Clamp
 * @param[in] iUseNormalizedCoords Bool to set if normalized coordinates are to be used.
 * Default = true.
 * @return Texture object to which image is uploaded.
 * @tparam imgT Type of image data. (uchar)
 */
template<typename imgT>
cudaTextureObject_t uploadImageToTexture(const cv::Mat& iImgToUpload,
                                         const cudaTextureAddressMode iAddrMode = cudaAddressModeClamp,
                                         const bool iUseNormalizedCoords = true);

/**
 * @brief Method to upload singe channel image (cv::Mat) int vectorized texture memory.
 * One pixel in texture memory holds multiple pixel of host memory
 * @param[in] iImgToUpload Image to upload.
 * @param[in] iUseNormaliedCoords Bool to set if normalized coordinates are to be used.
 * Default = true.
 * @return Texture object to which image is uploaded.
 * @tparam hostT Type of image data on host. (uchar)
 * @tparam deviceT Type of image data on device. (short, unsined int)
 */
template<typename hostT, typename deviceT>
cudaTextureObject_t uploadImageToTextureVectorized(const cv::Mat& iImgToUpload,
                                                    const bool iUseNormalizedCoords = true);

/**
 * @brief Method to upload single channel image (cv::Mat) to surface memory.
 * @param[in] iImgToUpload Image to upload.
 * @return Surface object to which image is uploaded.
 * @tparam imgT Type of image data. (uchar)
 */
template<typename imgT>
cudaSurfaceObject_t uploadImageToSurface(const cv::Mat& iImgToUpload);

/**
 * @brief Method to create surface object.
 * @param[in] iSurfSize Size of surface.
 * @param[out] CUDA Array holding surface data
 * @tparam surfT Type of image data. (uchar, float, int, uint)
 * @see downloadCudaArrayToImage
 */
template<typename surfT>
cudaSurfaceObject_t createSurfaceObject(const cv::Size& iSurfSize, cudaArray* &opCuArray);

#endif // CUDA_UTILS_MEMORY_H
