cmake_minimum_required(VERSION 3.16)
project(beam LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(SWEG_ENABLE_CUDA "Enable CUDA scoring backend build" OFF)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

find_package(OpenMP)
include(CheckLanguage)

set(SWEG_CUDA_SCORING_SOURCE src/cuda_scoring_stub.cpp)
if(SWEG_ENABLE_CUDA)
  check_language(CUDA)
  if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    set(SWEG_CUDA_SCORING_SOURCE src/cuda_scoring.cu)
    message(STATUS "CUDA compiler detected; building CUDA scoring backend skeleton.")
  else()
    message(WARNING "SWEG_ENABLE_CUDA=ON, but no CUDA compiler was found; falling back to stub backend.")
  endif()
endif()

add_library(sweg_lib
  src/graph_io/graph_io.cpp
  src/scoring_cpu.cpp
  src/sweg.cpp
  ${SWEG_CUDA_SCORING_SOURCE}
)

target_include_directories(sweg_lib PUBLIC include)
if(OpenMP_CXX_FOUND)
  target_link_libraries(sweg_lib PUBLIC OpenMP::OpenMP_CXX)
endif()
if(SWEG_ENABLE_CUDA AND CMAKE_CUDA_COMPILER)
  set_target_properties(sweg_lib PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
endif()

add_executable(beam src/main.cpp)
target_link_libraries(beam sweg_lib)
