////////////////////////////////////////////////////////////////////////////////
//! Copyright 2017 Boitumelo Ruf. All rights reserved.
////////////////////////////////////////////////////////////////////////////////

#ifndef CUDAGAUSSIANBLUR_CUH_
#define CUDAGAUSSIANBLUR_CUH_

// OpenCV
#include <opencv2/core.hpp>

/**
 * @brief Entry point to run gaussion blur on CUDA.
 * @param[in] inputImg
 * @return output image
 */
cv::Mat runCudaGaussianBlur(const cv::Mat& inputImg);

#endif // CUDAGAUSSIAN_CUH_
