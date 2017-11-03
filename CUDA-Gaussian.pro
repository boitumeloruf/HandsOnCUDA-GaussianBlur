####################################################################################################
# CUDA-Gaussian
####################################################################################################

# Qt Config
QT += gui widgets

# configuration
TARGET = CUDA-Gaussian
TEMPLATE = app

message("Building "$$TARGET" ("$$TEMPLATE")")

# build configuration
include($$PWD/build_config.pri)

#---------------------------------------------------------------------------------------------------
# Compiler defines

# define target name
DEFINES += TARGET_NAME=\\\"$$TARGET\\\"

#---------------------------------------------------------------------------------------------------
# Compiler Configuration

CONFIG += c++11
unix: QMAKE_CXXFLAGS += -Wno-unused-function -fPIC -fopenmp

#---------------------------------------------------------------------------------------------------
# Libraries

unix {
  if(!include($$PWD/src/opencv.pri)) {
    error("Couldn't find opencv.pri.")
  }

  if(!include($$PWD/src/cuda.pri)) {
    error("Couldn't find cuda.pri.")
  }
}

#---------------------------------------------------------------------------------------------------
# C++ Sources

HEADERS += \
    $$PWD/src/cudautils_memory.h \
    $$PWD/src/cudautils_common.hpp \
    $$PWD/src/cudautils_devices.h

SOURCES += \
    $$PWD/src/main.cpp \
    $$PWD/src/cudautils_memory.cpp \
    $$PWD/src/cudautils_devices.cpp \

#---------------------------------------------------------------------------------------------------
# CUDA build

CUDA_SOURCES += \
    $$PWD/src/*.cu

CUDA_HEADERS += \
    $$PWD/src/*.cuh

CUDA_ARCH = compute_50
NVCCFLAGS     = --compiler-options -fno-strict-aliasing -use_fast_math --ptxas-options=-v \
 -Wno-deprecated-gpu-targets --shared -Xcompiler -fPIC -c
CUDA_INC = $$join(INCLUDEPATH,' -I','-I',' ')

cuda_build.commands = $$CUDA_ROOT_DIR/bin/nvcc -m64 -O3  -arch=$$CUDA_ARCH -c $$NVCCFLAGS \
                $$CUDA_INC $$LIBS  ${QMAKE_FILE_NAME} -o ${QMAKE_FILE_OUT}
cuda_build.dependency_type = TYPE_C
cuda_build.depend_command = $$CUDA_ROOT_DIR/bin/nvcc -O3 -M $$CUDA_INC $$NVCCFLAGS ${QMAKE_FILE_NAME}

cuda_build.input = CUDA_SOURCES
cuda_build.output = ${OBJECTS_DIR}${QMAKE_FILE_BASE}_cuda.o

QMAKE_EXTRA_COMPILERS += cuda_build
OTHER_FILES += $$CUDA_SOURCES $$CUDA_HEADERS
