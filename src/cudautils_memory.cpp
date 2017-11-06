////////////////////////////////////////////////////////////////////////////////
//! Copyright 2017 Boitumelo Ruf. All rights reserved.
////////////////////////////////////////////////////////////////////////////////

#include "cudautils_memory.h"

// std
#include <assert.h>

// CUDA
#include <cuda.h>

//==================================================================================================
template< typename memT >
memT* allocateDeviceMem(const int iSize)
{
  memT* deviceMem;
  checkCudaErrors(
        cudaMalloc(&deviceMem, iSize * sizeof(memT)));

  return deviceMem;
}
template uchar* allocateDeviceMem<uchar>(const int);
template int* allocateDeviceMem<int>(const int);
template uint* allocateDeviceMem<uint>(const int);
template float* allocateDeviceMem<float>(const int);

//==================================================================================================
template<typename memT>
memT *downloadArray(const memT *iDevArray, const int iSize)
{
  memT* hostMem = new memT[iSize];
  checkCudaErrors(
        cudaMemcpy(hostMem, iDevArray, iSize * sizeof(memT), cudaMemcpyDeviceToHost));

  return hostMem;
}
template uchar* downloadArray<uchar>(const uchar *, const int);
template int* downloadArray<int>(const int *, const int);
template float* downloadArray<float>(const float *, const int);

//==================================================================================================
template<typename memT>
cv::Mat downloadArrayToImage(const memT* iDevArray, const cv::Size iImgSize)
{
  cv::Mat image = cv::Mat(iImgSize, cv::DataType<memT>::type);

  checkCudaErrors(
          cudaMemcpy(image.data, iDevArray, iImgSize.width * iImgSize.height * sizeof(memT),
                     cudaMemcpyDeviceToHost));

  return image;
}
template cv::Mat downloadArrayToImage<uchar>(const uchar*, const cv::Size);
template cv::Mat downloadArrayToImage<int>(const int*, const cv::Size);
template cv::Mat downloadArrayToImage<float>(const float*, const cv::Size);

//==================================================================================================
template<typename memT>
cudaArray* createCudaArray(const int iWidth, const int iHeight)
{
  cudaArray* cuArray;

  assert( iWidth > 0 && iHeight >= 0);

  //--- Allocate CUDA array in device memory ---
  cudaChannelFormatDesc channelDesc =
      cudaCreateChannelDesc<memT>();

  checkCudaErrors(
        cudaMallocArray(&cuArray, &channelDesc, iWidth, iHeight));

  return cuArray;
}
template cudaArray* createCudaArray<uchar>(const int, const int);
template cudaArray* createCudaArray<int>(const int, const int);
template cudaArray* createCudaArray<float>(const int, const int);

//==================================================================================================
template< typename memT, typename imgT >
cv::Mat downloadCudaArrayToImage(cudaArray* iCuArray, const cv::Size iImgSize)
{
  assert(sizeof(memT) >= sizeof (imgT));
  assert(iImgSize.width > 0 && iImgSize.height > 0);
  assert(iCuArray != 0);

  cv::Mat image = cv::Mat::zeros(iImgSize, CV_MAKETYPE(cv::DataType<imgT>::depth,
                                                       sizeof(memT) / sizeof (imgT)));

  //--- downlaod data ---
  checkCudaErrors(
        cudaMemcpyFromArray(image.data, iCuArray, 0,0, iImgSize.width * iImgSize.height * sizeof(memT),
                    cudaMemcpyDeviceToHost));

  return image;
}
template cv::Mat downloadCudaArrayToImage<uchar, uchar>(cudaArray*, const cv::Size);
template cv::Mat downloadCudaArrayToImage<int, uchar>(cudaArray*, const cv::Size);
template cv::Mat downloadCudaArrayToImage<uint, uchar>(cudaArray*, const cv::Size);
template cv::Mat downloadCudaArrayToImage<float, uchar>(cudaArray*, const cv::Size);

//==================================================================================================
template< typename imgT >
cudaTextureObject_t uploadImageToTexture(const cv::Mat& iImgToUpload,
                                         const cudaTextureAddressMode iAddrMode,
                                         const bool iUseNormalizedCoords)
{
  cudaTextureObject_t texObj = 0;

  cv::Size imgSize = iImgToUpload.size();
  assert(imgSize.width > 0 && imgSize.height > 0);

  //--- Allocate CUDA array in device memory ---
  cudaChannelFormatDesc channelDesc =
      cudaCreateChannelDesc<imgT>();

  cudaArray* cuArray;
  checkCudaErrors(
        cudaMallocArray(&cuArray, &channelDesc, imgSize.width, imgSize.height));

  //--- copy image data to device memory ---
  checkCudaErrors(
        cudaMemcpyToArray(cuArray, 0,0, iImgToUpload.data, imgSize.width * imgSize.height * sizeof(imgT),
                    cudaMemcpyHostToDevice));

  //--- Specify texture ---
  struct cudaResourceDesc resDesc;
  memset(&resDesc, 0, sizeof(resDesc));
  resDesc.resType = cudaResourceTypeArray;
  resDesc.res.array.array = cuArray;


  struct cudaTextureDesc texDesc;
  memset(&texDesc, 0, sizeof(texDesc));
  texDesc.addressMode[0] = iAddrMode;
  texDesc.addressMode[1] = iAddrMode;
  texDesc.filterMode = (iUseNormalizedCoords) ? cudaFilterModeLinear : cudaFilterModePoint;
  texDesc.readMode = cudaReadModeElementType;
  texDesc.normalizedCoords = iUseNormalizedCoords;

  // Create texture object
  checkCudaErrors(
        cudaCreateTextureObject(&texObj, &resDesc, &texDesc, NULL));

  return texObj;
}
template cudaTextureObject_t uploadImageToTexture<uchar>(const cv::Mat&, const cudaTextureAddressMode, const bool);
template cudaTextureObject_t uploadImageToTexture<uint>(const cv::Mat&, const cudaTextureAddressMode, const bool);

//==================================================================================================
template<typename hostT, typename deviceT>
cudaTextureObject_t uploadImageToTextureVectorized(const cv::Mat& iImgToUpload,
                                                   const bool iUseNormalizedCoords)
{
  cudaTextureObject_t texObj = 0;

  cv::Size imgSize = iImgToUpload.size();
  assert(imgSize.width > 0 && imgSize.height > 0);

  //--- Allocate CUDA array in device memory ---
  cudaChannelFormatDesc channelDesc =
      cudaCreateChannelDesc<deviceT>();

  int vectorizationFactor = sizeof(deviceT) / sizeof(hostT);

  cudaArray* cuArray;
  checkCudaErrors(
        cudaMallocArray(&cuArray, &channelDesc, imgSize.width, imgSize.height));

  //--- copy image data to device memory ---
  checkCudaErrors(
        cudaMemcpyToArray(cuArray, 0,0, iImgToUpload.data, imgSize.width * imgSize.height * sizeof(hostT) * vectorizationFactor,
                    cudaMemcpyHostToDevice));

  //--- Specify texture ---
  struct cudaResourceDesc resDesc;
  memset(&resDesc, 0, sizeof(resDesc));
  resDesc.resType = cudaResourceTypeArray;
  resDesc.res.array.array = cuArray;


  struct cudaTextureDesc texDesc;
  memset(&texDesc, 0, sizeof(texDesc));
  texDesc.addressMode[0] = cudaAddressModeClamp;
  texDesc.addressMode[1] = cudaAddressModeClamp;
  texDesc.filterMode = (iUseNormalizedCoords) ? cudaFilterModeLinear : cudaFilterModePoint;
  texDesc.readMode = cudaReadModeElementType;
  texDesc.normalizedCoords = iUseNormalizedCoords;

  // Create texture object
  checkCudaErrors(
        cudaCreateTextureObject(&texObj, &resDesc, &texDesc, NULL));

  return texObj;
}
template cudaTextureObject_t uploadImageToTextureVectorized<uchar, unsigned int>(const cv::Mat&, const bool);
template cudaTextureObject_t uploadImageToTextureVectorized<uchar, short>(const cv::Mat&, const bool);

//==================================================================================================
template< typename imgT >
cudaSurfaceObject_t uploadImageToSurface(const cv::Mat& iImgToUpload)
{
  cudaSurfaceObject_t surfObj = 0;

  cv::Size imgSize = iImgToUpload.size();
  assert(imgSize.width > 0 && imgSize.height > 0);
  assert(iImgToUpload.type() == cv::DataType<imgT>::type);

  //--- Allocate CUDA array in device memory ---
  cudaChannelFormatDesc channelDesc =
      cudaCreateChannelDesc<imgT>();

  cudaArray* cuArray;
  checkCudaErrors(
        cudaMallocArray(&cuArray, &channelDesc, imgSize.width, imgSize.height,
                        cudaArraySurfaceLoadStore));

  //--- copy image data to device memory ---
  checkCudaErrors(
        cudaMemcpyToArray(cuArray, 0,0, iImgToUpload.data, imgSize.width * imgSize.height * sizeof(imgT),
                    cudaMemcpyHostToDevice));

  //--- Specify sruface ---
  struct cudaResourceDesc resDesc;
  memset(&resDesc, 0, sizeof(resDesc));
  resDesc.resType = cudaResourceTypeArray;
  resDesc.res.array.array = cuArray;

  // Create texture object
  checkCudaErrors(cudaCreateSurfaceObject(&surfObj, &resDesc));

  return surfObj;
}
template cudaSurfaceObject_t uploadImageToSurface<uchar>(const cv::Mat&);
template cudaSurfaceObject_t uploadImageToSurface<float>(const cv::Mat&);

//==================================================================================================
template<typename surfT>
cudaSurfaceObject_t createSurfaceObject(const cv::Size& iSurfSize, cudaArray* &opCuArray)
{
  cudaSurfaceObject_t surfObj = 0;

  assert(iSurfSize.width > 0 && iSurfSize.height > 0);

  //--- Allocate CUDA array in device memory ---
  cudaChannelFormatDesc channelDesc =
      cudaCreateChannelDesc<surfT>();

  checkCudaErrors(
        cudaMallocArray(&opCuArray, &channelDesc, iSurfSize.width, iSurfSize.height,
                        cudaArraySurfaceLoadStore));

  //--- Specify sruface ---
  struct cudaResourceDesc resDesc;
  memset(&resDesc, 0, sizeof(resDesc));
  resDesc.resType = cudaResourceTypeArray;
  resDesc.res.array.array = opCuArray;

  // Create texture object
  checkCudaErrors(cudaCreateSurfaceObject(&surfObj, &resDesc));

  return surfObj;
}
template cudaSurfaceObject_t createSurfaceObject<uchar>(const cv::Size&, cudaArray*&);
template cudaSurfaceObject_t createSurfaceObject<uint>(const cv::Size&, cudaArray*&);
template cudaSurfaceObject_t createSurfaceObject<int>(const cv::Size&, cudaArray*&);
template cudaSurfaceObject_t createSurfaceObject<float>(const cv::Size&, cudaArray*&);
