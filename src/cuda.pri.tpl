CUDA_VERS_STR = <edit-here>
GCC_VERS_STR = <edit-here>
CUDA_ROOT_DIR = <edit-here>
CUDA_SDK_DIR = <edit-here>


# add directories and libraries to variables
unix {
  message("Using CUDA "$${CUDA_VERS_STR}", GCC "$${GCC_VERS_STR})

  INCLUDEPATH += $${CUDA_ROOT_DIR}/include/ \
               $${CUDA_SDK_DIR}/common/inc/ \
  DEPENDPATH += ${CUDA_ROOT_DIR}/include/ \
               $${CUDA_SDK_DIR}/common/inc/ \

  LIBS += -L$${CUDA_ROOT_DIR}/lib64
  message("-- export LD_LIBRARY_PATH="$${CUDA_ROOT_DIR}"/lib64:$LD_LIBRARY_PATH")

  LIBS += -lnppif \
          -lcublas \
          -lcusolver \
          -lnvblas \
          -lnppig \
          -lnvgraph \
          -lcusparse \
          -lcudart \
          -lnppim \
          -lnvrtc-builtins \
          -lnppc \
          -lnppi \
          -lcufft \
          -lnvrtc \
          -lnppial \
          -lnppist \
          -lcufftw \
          -lnvToolsExt \
          -lnppicc \
          -lnppisu \
          -lcuinj64 \
          -lnppicom \
          -lnppitc \
          -lcurand  \
          -lnppidei \
          -lnpps
}
