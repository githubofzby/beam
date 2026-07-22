cmake_minimum_required(VERSION 3.16)
project(beam LANGUAGES CXX)

include(CTest)

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
  src/candidate_index.cpp
  src/cost_oracle.cpp
  src/graph_io/graph_io.cpp
  src/quotient_graph.cpp
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

if(BUILD_TESTING)
  add_executable(cost_oracle_test tests/cost_oracle_test.cpp)
  target_link_libraries(cost_oracle_test sweg_lib)
  add_test(NAME cost_oracle COMMAND cost_oracle_test)
  add_executable(compression_metrics_test tests/compression_metrics_test.cpp)
  target_link_libraries(compression_metrics_test sweg_lib)
  add_test(NAME compression_metrics COMMAND compression_metrics_test)
  add_executable(quotient_graph_test tests/quotient_graph_test.cpp)
  target_link_libraries(quotient_graph_test sweg_lib)
  add_test(NAME quotient_graph COMMAND quotient_graph_test)
  add_executable(quotient_row_view_test tests/quotient_row_view_test.cpp)
  target_link_libraries(quotient_row_view_test sweg_lib)
  add_test(NAME quotient_row_view COMMAND quotient_row_view_test)
  add_executable(candidate_index_recall_test tests/candidate_index_recall_test.cpp)
  target_link_libraries(candidate_index_recall_test sweg_lib)
  add_test(NAME candidate_index_recall COMMAND candidate_index_recall_test)
  find_package(Python3 COMPONENTS Interpreter)
  if(Python3_Interpreter_FOUND)
    add_test(NAME profiling_equivalence
             COMMAND ${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/tests/profiling_equivalence_test.py
                     --binary $<TARGET_FILE:beam>)
    add_test(NAME cost_oracle_integration
             COMMAND ${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/tests/cost_oracle_integration_test.py
                     --binary $<TARGET_FILE:beam>)
    add_test(NAME quotient_graph_integration
             COMMAND ${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/tests/quotient_graph_integration_test.py
                     --binary $<TARGET_FILE:beam>)
  endif()
endif()
