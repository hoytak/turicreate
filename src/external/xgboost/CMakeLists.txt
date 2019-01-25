cmake_minimum_required (VERSION 3.2)
project(xgboost)
include(cmake/Utils.cmake)
list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake/modules")

set_default_configuration_release()

set(USE_CUDA OFF)
set(USE_NCCL OFF)

set(GPU_COMPUTE_VER "" CACHE STRING
  "Space separated list of compute versions to be built against, e.g. '35 61'")

# Devs
set(USE_SANITIZER OFF)

# Compiler flags
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
find_package(OpenMP)

if(OpenMP_CXX_FOUND OR OPENMP_FOUND)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(MSVC)
  # Multithreaded compilation
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /MP")
else()
  # Correct error for GCC 5 and cuda
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_MWAITXINTRIN_H_INCLUDED -D_FORCE_INLINES")
  # Performance
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -funroll-loops")
endif()

if(WIN32 AND MINGW)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -static-libstdc++")
endif()

# Check existence of software pre-fetching
include(CheckCXXSourceCompiles)
check_cxx_source_compiles("
#include <xmmintrin.h>
int main() {
  char data = 0;
  const char* address = &data;
  _mm_prefetch(address, _MM_HINT_NTA);
  return 0;
}
" XGBOOST_MM_PREFETCH_PRESENT)
check_cxx_source_compiles("
int main() {
  char data = 0;
  const char* address = &data;
  __builtin_prefetch(address, 0, 0);
  return 0;
}
" XGBOOST_BUILTIN_PREFETCH_PRESENT)

# Sanitizer
if(USE_SANITIZER)
  include(cmake/Sanitizer.cmake)
  enable_sanitizers("${ENABLED_SANITIZERS}")
endif(USE_SANITIZER)

# dmlc-core
add_subdirectory(dmlc-core)
set(LINK_LIBRARIES dmlc rabit)

# enable custom logging
add_definitions(-DDMLC_LOG_CUSTOMIZE=1)

# compiled code customizations for R package
if(R_LIB)
  add_definitions(
    -DXGBOOST_STRICT_R_MODE=1
    -DXGBOOST_CUSTOMIZE_GLOBAL_PRNG=1
    -DDMLC_LOG_BEFORE_THROW=0
    -DDMLC_DISABLE_STDIN=1
    -DDMLC_LOG_CUSTOMIZE=1
    -DRABIT_CUSTOMIZE_MSG_
    -DRABIT_STRICT_CXX98_
  )
endif()

# Gather source files
include_directories (
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/dmlc-core/include
    ${PROJECT_SOURCE_DIR}/rabit/include
)

#Generate configurable header
set(CMAKE_LOCAL "${PROJECT_SOURCE_DIR}/cmake")
set(INCLUDE_ROOT "${PROJECT_SOURCE_DIR}/include")
message(STATUS "${CMAKE_LOCAL}/build_config.h.in -> ${INCLUDE_ROOT}/xgboost/build_config.h")
configure_file("${CMAKE_LOCAL}/build_config.h.in" "${INCLUDE_ROOT}/xgboost/build_config.h")

file(GLOB_RECURSE SOURCES
    src/*.cc
    src/*.h
    include/*.h
)

# Only add main function for executable target
list(REMOVE_ITEM SOURCES ${PROJECT_SOURCE_DIR}/src/cli_main.cc)

file(GLOB_RECURSE TEST_SOURCES "tests/cpp/*.cc")

file(GLOB_RECURSE CUDA_SOURCES
    src/*.cu
    src/*.cuh
)

# rabit
# TODO: Create rabit cmakelists.txt
set(RABIT_SOURCES
    rabit/src/allreduce_base.cc
    rabit/src/allreduce_robust.cc
    rabit/src/engine.cc
    rabit/src/c_api.cc
)
set(RABIT_EMPTY_SOURCES
    rabit/src/engine_empty.cc
    rabit/src/c_api.cc
)
if(MINGW OR R_LIB)
  # build a dummy rabit library
  add_library(rabit STATIC ${RABIT_EMPTY_SOURCES})
else()
  add_library(rabit STATIC ${RABIT_SOURCES})
endif()

if(USE_CUDA)
  find_package(CUDA 8.0 REQUIRED)
  cmake_minimum_required(VERSION 3.5)

  add_definitions(-DXGBOOST_USE_CUDA)

  include_directories(cub)

  if(USE_NCCL)
    find_package(Nccl REQUIRED)
    include_directories(${NCCL_INCLUDE_DIR})
    add_definitions(-DXGBOOST_USE_NCCL)
  endif()

  set(GENCODE_FLAGS "")
  format_gencode_flags("${GPU_COMPUTE_VER}" GENCODE_FLAGS)
  message("cuda architecture flags: ${GENCODE_FLAGS}")

  set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS};--expt-extended-lambda;--expt-relaxed-constexpr;${GENCODE_FLAGS};-lineinfo;")
  if(NOT MSVC)
    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS};-Xcompiler -fPIC; -Xcompiler -Werror; -std=c++11")
  endif()

  cuda_add_library(gpuxgboost ${CUDA_SOURCES} STATIC)

  if(USE_NCCL)
    link_directories(${NCCL_LIBRARY})
    target_link_libraries(gpuxgboost ${NCCL_LIB_NAME})
  endif()
  list(APPEND LINK_LIBRARIES gpuxgboost)
endif()


# flags and sources for R-package
if(R_LIB)
  file(GLOB_RECURSE R_SOURCES
    R-package/src/*.h
    R-package/src/*.c
    R-package/src/*.cc
  )
  list(APPEND SOURCES ${R_SOURCES})
endif()

add_library(objxgboost OBJECT ${SOURCES})


# Shared library
make_library(xgboost SOURCES $<TARGET_OBJECTS:objxgboost> REQUIRES ${LINK_LIBRARIES})

# Group sources
auto_source_group("${SOURCES}")
