cmake_minimum_required(VERSION 3.12)

if(${CMAKE_VERSION} VERSION_LESS 3.12)
    cmake_policy(VERSION ${CMAKE_MAJOR_VERSION}.${CMAKE_MINOR_VERSION})
endif()

message(STATUS "Using CMake version ${CMAKE_VERSION}")

set(DNN_VERSION_MAJOR 1)
set(DNN_VERSION_MINOR 0)
set(DNN_VERSION_PATCH 0)
set(DNN_VERSION_NOTE "")

set(DNN_VERSION ${DNN_VERSION_MAJOR}.${DNN_VERSION_MINOR}.${DNN_VERSION_PATCH})
math(EXPR DNN_VERSION_NUMBER "10000*${DNN_VERSION_MAJOR} + 100*${DNN_VERSION_MINOR} + ${DNN_VERSION_PATCH}")

#####
# Enables link_directories() treat paths relative
# to the source dir.
if(POLICY CMP0015)
    cmake_policy(SET CMP0015 NEW)
endif(POLICY CMP0015)

#####
# Enables project() command manages VERSION variables.
if(POLICY CMP0048)
    cmake_policy(SET CMP0048 NEW)
endif(POLICY CMP0048)

if("${CMAKE_BUILD_TYPE}" STREQUAL "")
    message(STATUS "CMAKE_BUILD_TYPE is unset, defaulting to Debug")
    set(CMAKE_BUILD_TYPE "Debug" CACHE STRING
        "Choose the type of build, options are: None Debug Release RelWithDebInfo MinSizeRel RelWithAssert ...")
endif()

project(dnn VERSION ${DNN_VERSION} LANGUAGES C CXX)

INCLUDE(GNUInstallDirs)
INCLUDE(CheckCSourceCompiles)
INCLUDE(CheckCXXSourceCompiles)

option(BUILD_SHARED_LIBS "Build shared library" ON)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(WIN32 OR MSVC)
  set(BUILD_SHARED_LIBS ON CACHE BOOL "" FORCE)
  set(flags /arch:AVX512 /fp:contract /favor:AMD64)
  set(linkflags "")
  add_compile_options(
    "$<$<COMPILE_LANGUAGE:CXX>:${flags}>"
    "$<$<COMPILE_LANGUAGE:C>:${flags}>")
  set(debug_flags /bigobj /openmp:llvm /W4 /Od)
  set(release_flags /bigobj /openmp:llvm /W3 /O2 /Ob2 /Oi /Ot /GL /GF /Gy /fp:fast)
else()
  if (CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    set(flags -mrdseed -mrdrnd -mfma -mavx512f -mavx512dq -mavx512vl -mavx512bw -mavx512ifma -mavx512vbmi -mavx512vbmi2 -mavx512bitalg -mavx512vpopcntdq -mavx512vnni -mavx512bf16 -mavx512vp2intersect -mavxvnni -mmovdiri -mmovdir64b -mprefetchi -march=native -mtune=native -fopenmp)
    set(linkflags -fopenmp=libiomp5 -z noexecstack)
  elseif(CMAKE_CXX_COMPILER_ID STREQUAL "IntelLLVM")
    set(flags -mrdseed -mrdrnd -mfma -mavx512f -mavx512dq -mavx512vl -mavx512bw -mavx512ifma -mavx512vbmi -mavx512vbmi2 -mavx512bitalg -mavx512vpopcntdq -mavx512vnni -mavx512bf16 -mavx512vp2intersect -mavxvnni -mmovdiri -mmovdir64b -mprefetchi -march=native -mtune=native -qopenmp)
    set(linkflags -qopenmp -z noexecstack)
  else()
    set(flags -mrdseed -mrdrnd -mfma -mavx512f -mavx512dq -mavx512vl -mavx512bw -mavx512ifma -mavx512vbmi -mavx512vbmi2 -mavx512bitalg -mavx512vpopcntdq -mavx512vnni -mavx512bf16 -mavx512vp2intersect -mavxvnni -mmovdiri -mmovdir64b -mprefetchi -march=native -mtune=native -fopenmp)
    set(linkflags -fopenmp -z noexecstack)
  endif()
  add_compile_options(
    "$<$<COMPILE_LANGUAGE:CXX>:${flags}>"
    "$<$<COMPILE_LANGUAGE:C>:${flags}>")
  set(debug_flags -g -Wextra)
  if (CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    set(release_flags -O3 -ffp-contract=on -funroll-loops -ftree-vectorize -Wno-error)
  elseif(CMAKE_CXX_COMPILER_ID STREQUAL "IntelLLVM")
    set(release_flags -O3 -ffp-contract=on -funroll-loops -ftree-vectorize -Wno-error)
  else()
    set(release_flags -O3 -ffp-contract=on -funroll-loops -ftree-vectorize -Wno-error)
  endif()
endif()

MACRO(DNN_TARGET_ENABLE_CXX17 target)
  SET_TARGET_PROPERTIES(${target} PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS ON)
  target_compile_options(${target} PRIVATE
    "$<$<CONFIG:RELEASE>:${release_flags}>"
    "$<$<CONFIG:DEBUG>:${debug_flags}>")
  target_link_options(${target} PRIVATE
    "$<$<COMPILE_LANGUAGE:CXX>:${linkflags}>"
    "$<$<COMPILE_LANGUAGE:C>:${linkflags}>")
ENDMACRO()

set(DNNL_LIBRARY_TYPE "STATIC" CACHE STRING "" FORCE)
set(DNNL_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(DNNL_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(DNNL_ENABLE_CONCURRENT_EXEC ON CACHE BOOL "" FORCE)
set(DNNL_ENABLE_PRIMITIVE_CACHE ON CACHE BOOL "" FORCE)
set(DNNL_EXPERIMENTAL ON CACHE BOOL "" FORCE)
set(ONEDNN_EXPERIMENTAL_UKERNEL ON CACHE BOOL "" FORCE)

if(WIN32 OR MSVC)
  set(DNNL_ARCH_OPT_FLAGS "/arch:AVX512 /openmp:llvm" CACHE STRING "" FORCE)
else()
  if (CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    set(DNNL_ARCH_OPT_FLAGS "-O3 -ffp-contract=on -funroll-loops -ftree-vectorize -Wno-error -mfma -mavx512f -mavx512dq -mavx512vl -mavx512bw -mavx512ifma -mavx512vbmi -mavx512vbmi2 -mavx512bitalg -mavx512vpopcntdq -mavx512vnni -mavx512bf16 -mavx512vp2intersect -mavxvnni -mmovdiri -mmovdir64b -mprefetchi -march=native -mtune=native -fopenmp" CACHE STRING "" FORCE)
  elseif(CMAKE_CXX_COMPILER_ID STREQUAL "IntelLLVM")
    set(DNNL_ARCH_OPT_FLAGS "-O3 -ffp-contract=on -funroll-loops -ftree-vectorize -Wno-error -mfma -mavx512f -mavx512dq -mavx512vl -mavx512bw -mavx512ifma -mavx512vbmi -mavx512vbmi2 -mavx512bitalg -mavx512vpopcntdq -mavx512vnni -mavx512bf16 -mavx512vp2intersect -mavxvnni -mmovdiri -mmovdir64b -mprefetchi -march=native -mtune=native -qopenmp" CACHE STRING "" FORCE)
  else()
    set(DNNL_ARCH_OPT_FLAGS "-O3 -ffp-contract=on -funroll-loops -ftree-vectorize -Wno-error -mfma -mavx512f -mavx512dq -mavx512vl -mavx512bw -mavx512ifma -mavx512vbmi -mavx512vbmi2 -mavx512bitalg -mavx512vpopcntdq -mavx512vnni -mavx512bf16 -mavx512vp2intersect -mavxvnni -mmovdiri -mmovdir64b -mprefetchi -march=native -mtune=native -fopenmp" CACHE STRING "" FORCE)
  endif()
endif()
set(WITH_TURBOJPEG FALSE CACHE BOOL "" FORCE)
#set(PNG_EXECUTABLES FALSE CACHE BOOL "" FORCE)
set(PNG_TESTS FALSE CACHE BOOL "" FORCE)
set(PNG_SHARED FALSE CACHE BOOL "" FORCE)
set(PNG_STATIC TRUE CACHE BOOL "" FORCE)
#set(PNG_BUILD_ZLIB ON CACHE BOOL "" FORCE)
set(PNG_TOOLS FALSE CACHE BOOL "" FORCE)
set(ENABLE_SHARED FALSE CACHE BOOL "" FORCE)
set(ENABLE_STATIC TRUE CACHE BOOL "" FORCE)
set(ZLIB_BUILD_SHARED TRUE CACHE BOOL "" FORCE)
set(ZLIB_BUILD_STATIC TRUE CACHE BOOL "" FORCE)
set(ZLIB_BUILD_EXAMPLES FALSE CACHE BOOL "" FORCE)

set(libdnn_headers
  include/Activation.h
  include/Add.h
  include/AlignedAllocator.h
  include/Average.h
  include/AvgPooling.h
  include/BatchNorm.h
  include/BatchNormActivation.h
  include/BatchNormActivationDropout.h
  include/BatchNormRelu.h
  include/ChannelSplit.h
  include/ChannelSplitRatioLeft.h
  include/ChannelSplitRatioRight.h
  include/ChannelZeroPad.h
  include/Concat.h
  include/Convolution.h
  include/ConvolutionTranspose.h
  include/Cost.h
  include/CsvFile.h
  include/Dataprovider.h
  include/Definition.h
  include/Dense.h
  include/DepthwiseConvolution.h
  include/Divide.h
  include/Dropout.h
  include/GlobalAvgPooling.h
  include/GlobalMaxPooling.h
  include/GroupNorm.h
  include/Image.h
  include/Input.h
  include/Layer.h
  include/LayerNorm.h
  include/LocalResponseNorm.h
  include/LogSoftmax.h
  include/Max.h
  include/MaxPooling.h
  include/Min.h
  include/Model.h
  include/Multiply.h
  include/ParallelFor.h
  include/PartialDepthwiseConvolution.h
  include/PRelu.h
  include/Reduction.h
  include/Resampling.h
  include/Scripts.h
  include/Shuffle.h
  include/Softmax.h
  include/stdafx.h
  include/Substract.h
  include/Utils.h
  include/targetver.h
)

set(libdnn_sources
  ${libdnn_headers}
  src/dllmain.cpp
  src/stdafx.cpp
)

set(libdnn_test
  src/test.cpp
)

# ---[ Download deps
SET(DNN_DEPENDENCIES_SOURCE_DIR ${CMAKE_SOURCE_DIR}/deps
  CACHE PATH "Confu-style dependencies source directory")
SET(DNN_DEPENDENCIES_BINARY_DIR ${CMAKE_BINARY_DIR}/deps
  CACHE PATH "Confu-style dependencies binary directory")

IF(NOT DEFINED BITSERY_SOURCE_DIR)
  MESSAGE(STATUS "Downloading bitsery to ${DNN_DEPENDENCIES_SOURCE_DIR}/bitsery (define BITSERY_SOURCE_DIR to avoid it)")
  CONFIGURE_FILE(cmake/DownloadBitsery.cmake "${DNN_DEPENDENCIES_BINARY_DIR}/bitsery-download/CMakeLists.txt")
  EXECUTE_PROCESS(COMMAND "${CMAKE_COMMAND}" -G "${CMAKE_GENERATOR}" .
    WORKING_DIRECTORY "${DNN_DEPENDENCIES_BINARY_DIR}/bitsery-download")
  EXECUTE_PROCESS(COMMAND "${CMAKE_COMMAND}" --build .
    WORKING_DIRECTORY "${DNN_DEPENDENCIES_BINARY_DIR}/bitsery-download")
  SET(BITSERY_SOURCE_DIR "${DNN_DEPENDENCIES_SOURCE_DIR}/bitsery" CACHE STRING "bitsery source directory")
ENDIF()

IF(NOT DEFINED ZLIB_SOURCE_DIR)
  MESSAGE(STATUS "Downloading zlib to ${DNN_DEPENDENCIES_SOURCE_DIR}/zlib (define ZLIB_SOURCE_DIR to avoid it)")
  CONFIGURE_FILE(cmake/DownloadZLIB.cmake "${DNN_DEPENDENCIES_BINARY_DIR}/zlib-download/CMakeLists.txt")
  EXECUTE_PROCESS(COMMAND "${CMAKE_COMMAND}" -G "${CMAKE_GENERATOR}" .
    WORKING_DIRECTORY "${DNN_DEPENDENCIES_BINARY_DIR}/zlib-download")
  EXECUTE_PROCESS(COMMAND "${CMAKE_COMMAND}" --build .
    WORKING_DIRECTORY "${DNN_DEPENDENCIES_BINARY_DIR}/zlib-download")
  SET(ZLIB_SOURCE_DIR "${DNN_DEPENDENCIES_SOURCE_DIR}/zlib" CACHE STRING "zlib source directory")
  SET(ZLIB_INCLUDE_DIR "${DNN_DEPENDENCIES_BINARY_DIR}/zlib" CACHE STRING "zlib include directory")
  SET(ZLIB_INCLUDE_DIRS "${DNN_DEPENDENCIES_BINARY_DIR}/zlib" CACHE STRING "zlib include directory")
  SET(ZLIB_ROOT "${DNN_DEPENDENCIES_SOURCE_DIR}/zlib;${DNN_DEPENDENCIES_BINARY_DIR}/zlib")
  SET(ZLIB_LIBRARY $<TARGET_FILE:zlib>)
  EXECUTE_PROCESS(COMMAND "${CMAKE_COMMAND}" -E copy "${ZLIB_SOURCE_DIR}/zlib.h" "${ZLIB_INCLUDE_DIRS}/zlib.h")
 ENDIF()
 
 IF(NOT DEFINED PNG_SOURCE_DIR)
  MESSAGE(STATUS "Downloading png to ${DNN_DEPENDENCIES_SOURCE_DIR}/libpng (define PNG_SOURCE_DIR to avoid it)")
  CONFIGURE_FILE(cmake/DownloadPNG.cmake "${DNN_DEPENDENCIES_BINARY_DIR}/png-download/CMakeLists.txt")
  EXECUTE_PROCESS(COMMAND "${CMAKE_COMMAND}" -G "${CMAKE_GENERATOR}" .
    WORKING_DIRECTORY "${DNN_DEPENDENCIES_BINARY_DIR}/png-download")
  EXECUTE_PROCESS(COMMAND "${CMAKE_COMMAND}" --build .
    WORKING_DIRECTORY "${DNN_DEPENDENCIES_BINARY_DIR}/png-download")
  SET(PNG_SOURCE_DIR "${DNN_DEPENDENCIES_SOURCE_DIR}/libpng" CACHE STRING "PNG source directory")
  EXECUTE_PROCESS(COMMAND "${CMAKE_COMMAND}" -E copy "${PNG_SOURCE_DIR}/scripts/pnglibconf.h.prebuilt" "${DNN_DEPENDENCIES_BINARY_DIR}/libpng/scripts/pnglibconf.h")
 ENDIF()

IF(NOT DEFINED VECTORCLASS_SOURCE_DIR)
  MESSAGE(STATUS "Downloading vectorclass to ${DNN_DEPENDENCIES_SOURCE_DIR}/version2 (define VECTORCLASS_SOURCE_DIR to avoid it)")
  CONFIGURE_FILE(cmake/DownloadVectorClass.cmake "${DNN_DEPENDENCIES_BINARY_DIR}/version2-download/CMakeLists.txt")
  EXECUTE_PROCESS(COMMAND "${CMAKE_COMMAND}" -G "${CMAKE_GENERATOR}" .
    WORKING_DIRECTORY "${DNN_DEPENDENCIES_BINARY_DIR}/version2-download")
  EXECUTE_PROCESS(COMMAND "${CMAKE_COMMAND}" --build .
    WORKING_DIRECTORY "${DNN_DEPENDENCIES_BINARY_DIR}/version2-download")
  SET(VECTORCLASS_SOURCE_DIR "${DNN_DEPENDENCIES_SOURCE_DIR}/version2" CACHE STRING "vectorclass source directory")
 ENDIF()

 IF(NOT DEFINED JPEG_SOURCE_DIR)
  MESSAGE(STATUS "Downloading jpeg to ${DNN_DEPENDENCIES_SOURCE_DIR}/libjpeg-turbo (define JPEG_SOURCE_DIR to avoid it)")
  CONFIGURE_FILE(cmake/DownloadJPEG.cmake "${DNN_DEPENDENCIES_BINARY_DIR}/libjpeg-turbo-download/CMakeLists.txt")
  EXECUTE_PROCESS(COMMAND "${CMAKE_COMMAND}" -G "${CMAKE_GENERATOR}" .
    WORKING_DIRECTORY "${DNN_DEPENDENCIES_BINARY_DIR}/libjpeg-turbo-download")
  EXECUTE_PROCESS(COMMAND "${CMAKE_COMMAND}" --build .
    WORKING_DIRECTORY "${DNN_DEPENDENCIES_BINARY_DIR}/libjpeg-turbo-download")
  SET(JPEG_SOURCE_DIR "${DNN_DEPENDENCIES_SOURCE_DIR}/libjpeg-turbo" CACHE STRING "JPEG source directory")
  # --- EXECUTE_PROCESS(COMMAND "${CMAKE_COMMAND}" -E copy "${DNN_DEPENDENCIES_SOURCE_DIR}/libjpeg-turbo/jconfig.h.in" "${DNN_DEPENDENCIES_SOURCE_DIR}/libjpeg-turbo/jconfig.h")
  # --- EXECUTE_PROCESS(COMMAND "${CMAKE_COMMAND}" -E copy "${DNN_DEPENDENCIES_SOURCE_DIR}/libjpeg-turbo/jconfig.h.in" "${DNN_DEPENDENCIES_BINARY_DIR}/libjpeg-turbo/jconfig.h")
  # --- EXECUTE_PROCESS(COMMAND "${CMAKE_COMMAND}" -E copy "${JPEG_SOURCE_DIR}/jpeglib.h" "${DNN_DEPENDENCIES_BINARY_DIR}/libjpeg-turbo/jpeglib.h")
 ENDIF()

 IF(NOT DEFINED MAGIC_ENUM_SOURCE_DIR)
  MESSAGE(STATUS "Downloading MagicEnum to ${DNN_DEPENDENCIES_SOURCE_DIR}/magic_enum (define MAGIC_ENUM_SOURCE_DIR to avoid it)")
  CONFIGURE_FILE(cmake/DownloadMagicEnum.cmake "${DNN_DEPENDENCIES_BINARY_DIR}/magic_enum-download/CMakeLists.txt")
  EXECUTE_PROCESS(COMMAND "${CMAKE_COMMAND}" -G "${CMAKE_GENERATOR}" .
    WORKING_DIRECTORY "${DNN_DEPENDENCIES_BINARY_DIR}/magic_enum-download")
  EXECUTE_PROCESS(COMMAND "${CMAKE_COMMAND}" --build .
    WORKING_DIRECTORY "${DNN_DEPENDENCIES_BINARY_DIR}/magic_enum-download")
  SET(MAGIC_ENUM_SOURCE_DIR "${DNN_DEPENDENCIES_SOURCE_DIR}/magic_enum" CACHE STRING "MagicEnum source directory")
 ENDIF()

 IF(NOT DEFINED ONEDNN_SOURCE_DIR)
  MESSAGE(STATUS "Downloading oneDNN to ${DNN_DEPENDENCIES_SOURCE_DIR}/oneDNN (define ONEDNN_SOURCE_DIR to avoid it)")
  CONFIGURE_FILE(cmake/DownloadOneDNN.cmake "${DNN_DEPENDENCIES_BINARY_DIR}/oneDNN-download/CMakeLists.txt")
  EXECUTE_PROCESS(COMMAND "${CMAKE_COMMAND}" -G "${CMAKE_GENERATOR}" .
    WORKING_DIRECTORY "${DNN_DEPENDENCIES_BINARY_DIR}/oneDNN-download")
  EXECUTE_PROCESS(COMMAND "${CMAKE_COMMAND}" --build .
    WORKING_DIRECTORY "${DNN_DEPENDENCIES_BINARY_DIR}/oneDNN-download")
  SET(ONEDNN_SOURCE_DIR "${DNN_DEPENDENCIES_SOURCE_DIR}/oneDNN" CACHE STRING "oneDNN source directory")
 ENDIF()

 IF(NOT DEFINED CIMG_SOURCE_DIR)
  MESSAGE(STATUS "Downloading CImg to ${DNN_DEPENDENCIES_SOURCE_DIR}/CImg (define CIMG_SOURCE_DIR to avoid it)")
  CONFIGURE_FILE(cmake/DownloadCImg.cmake "${DNN_DEPENDENCIES_BINARY_DIR}/cimg-download/CMakeLists.txt")
  EXECUTE_PROCESS(COMMAND "${CMAKE_COMMAND}" -G "${CMAKE_GENERATOR}" .
    WORKING_DIRECTORY "${DNN_DEPENDENCIES_BINARY_DIR}/cimg-download")
  EXECUTE_PROCESS(COMMAND "${CMAKE_COMMAND}" --build .
    WORKING_DIRECTORY "${DNN_DEPENDENCIES_BINARY_DIR}/cimg-download")
  SET(CIMG_SOURCE_DIR "${DNN_DEPENDENCIES_SOURCE_DIR}/CImg" CACHE STRING "CImg source directory")
 ENDIF()

add_library(${PROJECT_NAME} ${libdnn_sources})
DNN_TARGET_ENABLE_CXX17(${PROJECT_NAME})
if(BUILD_SHARED_LIBS)
  target_compile_definitions(${PROJECT_NAME} PRIVATE DNN_DLL DNN_EXPORTS DNN_CACHE_PRIMITIVES DNN_AVX512 cimg_use_openmp cimg_use_cpp11 cimg_use_jpeg cimg_use_png cimg_use_zlib)
else()
  target_compile_definitions(${PROJECT_NAME} PRIVATE DNN_EXPORTS DNN_CACHE_PRIMITIVES DNN_AVX512 cimg_use_openmp cimg_use_cpp11 cimg_use_jpeg cimg_use_png cimg_use_zlib)
endif()
target_include_directories(${PROJECT_NAME}
    PUBLIC
       $<INSTALL_INTERFACE:include>
       $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    PRIVATE
       ${CMAKE_CURRENT_SOURCE_DIR}/src)

add_executable(test ${libdnn_test})
DNN_TARGET_ENABLE_CXX17(test)
if(BUILD_SHARED_LIBS)
  target_compile_definitions(test PRIVATE DNN_EXPORTS DNN_DLL DNN_CACHE_PRIMITIVES DNN_AVX512 cimg_use_openmp cimg_use_cpp11 cimg_use_jpeg cimg_use_png cimg_use_zlib)
else()
  target_compile_definitions(test PRIVATE DNN_EXPORTS DNN_CACHE_PRIMITIVES DNN_AVX512 cimg_use_openmp cimg_use_cpp11 cimg_use_jpeg cimg_use_png cimg_use_zlib)
endif()
target_include_directories(test 
    PUBLIC
       $<INSTALL_INTERFACE:include>
       $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    PRIVATE
       ${CMAKE_CURRENT_SOURCE_DIR}/src)

include_directories(${DNN_DEPENDENCIES_SOURCE_DIR}/bitsery)
include_directories(${DNN_DEPENDENCIES_SOURCE_DIR}/zlib)
include_directories(${DNN_DEPENDENCIES_BINARY_DIR}/zlib)
include_directories(${DNN_DEPENDENCIES_SOURCE_DIR}/libpng)
include_directories(${DNN_DEPENDENCIES_BINARY_DIR}/libpng)
include_directories(${DNN_DEPENDENCIES_SOURCE_DIR}/libjpeg-turbo)
include_directories(${DNN_DEPENDENCIES_BINARY_DIR}/libjpeg-turbo)
include_directories(${DNN_DEPENDENCIES_SOURCE_DIR}/oneDNN/include)
include_directories(${DNN_DEPENDENCIES_BINARY_DIR}/oneDNN/include)
include_directories(${DNN_DEPENDENCIES_SOURCE_DIR}/oneDNN/src)
include_directories(${DNN_DEPENDENCIES_BINARY_DIR}/oneDNN/src)
include_directories(${DNN_DEPENDENCIES_SOURCE_DIR}/version2)
include_directories(${DNN_DEPENDENCIES_SOURCE_DIR}/magic_enum/include)
include_directories(${DNN_DEPENDENCIES_SOURCE_DIR}/CImg)


# ---[ Configure Bisery
IF(NOT TARGET bitsery)
  ADD_SUBDIRECTORY(
    "${BITSERY_SOURCE_DIR}"
    "${DNN_DEPENDENCIES_BINARY_DIR}/bitsery")
ENDIF()
TARGET_LINK_LIBRARIES(${PROJECT_NAME} PUBLIC bitsery)

# ---[ Configure ZLIB
IF(NOT TARGET zlib)
  ADD_SUBDIRECTORY(
    "${ZLIB_SOURCE_DIR}"
    "${DNN_DEPENDENCIES_BINARY_DIR}/zlib")
ENDIF()
TARGET_LINK_LIBRARIES(${PROJECT_NAME} PUBLIC zlibstatic)

IF(WIN32 OR MSVC)
  SET(SKIP_INSTALL_ALL ON)
ENDIF()
# ---[ Configure PNG
IF(NOT TARGET png)
  ADD_SUBDIRECTORY(
    "${PNG_SOURCE_DIR}"
    "${DNN_DEPENDENCIES_BINARY_DIR}/libpng")
ENDIF()
TARGET_LINK_LIBRARIES(${PROJECT_NAME} PUBLIC png_static)

# ---[ Configure JPEG
IF(NOT TARGET jpeg)
  ADD_SUBDIRECTORY(
    "${JPEG_SOURCE_DIR}"
    "${DNN_DEPENDENCIES_BINARY_DIR}/libjpeg-turbo")
ENDIF()
TARGET_LINK_LIBRARIES(${PROJECT_NAME} PUBLIC jpeg-static)

# ---[ Configure oneDNN
IF(NOT TARGET dnnl)
  ADD_SUBDIRECTORY(
    "${ONEDNN_SOURCE_DIR}"
    "${DNN_DEPENDENCIES_BINARY_DIR}/oneDNN")
ENDIF()
TARGET_LINK_LIBRARIES(${PROJECT_NAME} PUBLIC dnnl)

# ---[ Configure vectorclass
IF(NOT TARGET vectorclass)
  ADD_SUBDIRECTORY(
    "${VECTORCLASS_SOURCE_DIR}"
    "${DNN_DEPENDENCIES_BINARY_DIR}/version2")
ENDIF()
TARGET_LINK_LIBRARIES(${PROJECT_NAME} PUBLIC vectorclass)

# ---[ Configure MagicEnum
IF(NOT TARGET magic_enum)
  ADD_SUBDIRECTORY(
    "${MAGIC_ENUM_SOURCE_DIR}"
    "${DNN_DEPENDENCIES_BINARY_DIR}/magic_enum")
ENDIF()

TARGET_LINK_LIBRARIES(test PUBLIC ${PROJECT_NAME})

install(TARGETS test DESTINATION bin)
install(TARGETS ${PROJECT_NAME} ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR} LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
