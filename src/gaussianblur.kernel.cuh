#ifndef GAUSSIANBLUR_KERNEL_H_
#define GAUSSIANBLUR_KERNEL_H_

#define BLOCKSIZE_X 16
#define BLOCKSIZE_Y 16

#define KERNELRAD 2

////////////////////////////////////////////////////////////////////////////////
//!
////////////////////////////////////////////////////////////////////////////////
__global__ void applyGaussianBlur(cudaTextureObject_t inputImgTex, cudaSurfaceObject_t outputImgSurf,
                                  const int iWidth, const int iHeight)
{
  //--- access thread id ---
  const int tidx = blockDim.x * blockIdx.x + threadIdx.x;
  const int tidy = blockDim.y * blockIdx.y + threadIdx.y;

  //--- variables for accesing texture ---
  int texPxId_x, texPxId_y;

  //--- pixel variables ---
  unsigned int inputTexel;                  // assign memory to download texel to
  unsigned int outputTexel = 0xff000000;
  uchar* inputPx = (uchar*) &inputTexel;    // assign memory pointer to pixel memory uchar[4]
  uchar* outputPx = (uchar*) &outputTexel;  // assign memory pointer to pixel memory uchar[4]
  float tmpKernelVal;

  //--- define shared array to hold image data ---
  const uchar l_array_width = BLOCKSIZE_X + 2 * KERNELRAD;
  const uchar l_array_height = BLOCKSIZE_Y + 2 * KERNELRAD;
  __shared__ unsigned int l_imgData[BLOCKSIZE_X + 2*KERNELRAD][BLOCKSIZE_Y + 2*KERNELRAD];

  //--- define gaussian Kernel in shared memory ---
  __shared__ float l_Kernel[2*KERNELRAD + 1][2*KERNELRAD + 1];
  l_Kernel[0][0] = 1; l_Kernel[1][0] = 4;  l_Kernel[2][0] = 6;  l_Kernel[3][0] = 4;  l_Kernel[4][0] = 1;
  l_Kernel[0][1] = 4; l_Kernel[1][1] = 16; l_Kernel[2][1] = 24; l_Kernel[3][1] = 16; l_Kernel[4][1] = 4;
  l_Kernel[0][2] = 6; l_Kernel[1][2] = 24; l_Kernel[2][2] = 36; l_Kernel[3][2] = 24; l_Kernel[4][2] = 6;
  l_Kernel[0][3] = 4; l_Kernel[1][3] = 16; l_Kernel[2][3] = 24; l_Kernel[3][3] = 16; l_Kernel[4][3] = 4;
  l_Kernel[0][4] = 1; l_Kernel[1][4] = 4;  l_Kernel[2][4] = 6;  l_Kernel[3][4] = 4;  l_Kernel[4][4] = 1;

  //--- copy data from global to local memory ---
  texPxId_y = tidy - KERNELRAD;
#pragma unroll
  for(uchar n = threadIdx.y; n < l_array_height; n += BLOCKSIZE_Y)
  {
    texPxId_x = tidx - KERNELRAD;

#pragma unroll
    for(uchar m = threadIdx.x; m < l_array_width; m += BLOCKSIZE_X)
    {
      //--- read data from texture ---
      l_imgData[m][n] = tex2D<unsigned int>(inputImgTex, texPxId_x, texPxId_y);

      texPxId_x += BLOCKSIZE_X;
    }
    texPxId_y += BLOCKSIZE_X;
  }

  __syncthreads();

  //--- convolve kernel ---
#pragma unroll
  for(char n = -KERNELRAD; n <= KERNELRAD; n++)
  {
#pragma unroll
    for(char m = -KERNELRAD; m <= KERNELRAD; m++)
    {
      //--- read data from sahred memory ---
      inputTexel = l_imgData[threadIdx.x + KERNELRAD + m][threadIdx.y + KERNELRAD + n];

      //--- apply kernel to each chanel ---
      tmpKernelVal = l_Kernel[KERNELRAD + m][KERNELRAD + n];
      outputPx[0] += (uchar)((float)inputPx[0] * tmpKernelVal/256.f );
      outputPx[1] += (uchar)((float)inputPx[1] * tmpKernelVal/256.f );
      outputPx[2] += (uchar)((float)inputPx[2] * tmpKernelVal/256.f );
    }
  }

  //--- write to output ---
  surf2Dwrite(outputTexel, outputImgSurf, tidx * 4, tidy);
}

#endif // #ifndef GAUSSIANBLUR_KERNEL_H_
