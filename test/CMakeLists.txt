cmake_minimum_required(VERSION 3.26)
project(senk
  VERSION 3.1.2
  LANGUAGES CXX
)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Choose the type of build." FORCE)
endif()

option(WERROR "Treat warnings as errors" OFF)

set(SENK_TARGETS_TO_BUILD "cpu" CACHE STRING "Targets to build (cpu;cuda;hip)")
separate_arguments(SENK_TARGETS_TO_BUILD)

set(COMMON_INCLUDES
  "${CMAKE_CURRENT_SOURCE_DIR}"
  "${CMAKE_CURRENT_SOURCE_DIR}/.."
  "${CMAKE_CURRENT_SOURCE_DIR}/../doctest"
)

foreach(target IN LISTS SENK_TARGETS_TO_BUILD)
  message(STATUS "Configuring target: ${target}")

  if(target STREQUAL "cpu")
    add_executable(cpu_test
      main.cpp
      tensor.cpp
      core.cpp
      matrix.cpp
      solver.cpp
    )
    target_compile_definitions(cpu_test PRIVATE SENK_CPU)
    target_include_directories(cpu_test PRIVATE ${COMMON_INCLUDES})
    target_compile_options(cpu_test PRIVATE
      $<$<CONFIG:Debug>:-g>
      $<$<CONFIG:Release>:-O3>
    )
    find_package(OpenMP REQUIRED COMPONENTS CXX)
    if (OpenMP_CXX_FOUND)
      target_link_libraries(cpu_test PRIVATE OpenMP::OpenMP_CXX)
    endif()

  elseif(target STREQUAL "cuda")
    find_package(CUDAToolkit)
    enable_language(CUDA)
    message(STATUS "CUDA compiler detected: ${CMAKE_CUDA_COMPILER}, enabling GPU tests")
    add_executable(cuda_test
      main.cpp
      core.cu
      tensor.cu
      matrix.cu
      solver.cu
    )
    target_compile_definitions(cuda_test PRIVATE SENK_CUDA)
    set_target_properties(cuda_test PROPERTIES
      CUDA_SEPARABLE_COMPILATION OFF
      CUDA_STANDARD 20
    )
    target_compile_options(cuda_test PRIVATE
      $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda --expt-relaxed-constexpr -Xcompiler=-fopenmp,-O3,-std=c++20>
    )
    target_include_directories(cuda_test PRIVATE ${COMMON_INCLUDES})
    find_package(OpenMP REQUIRED COMPONENTS CXX)
    if (OpenMP_CXX_FOUND)
      target_link_libraries(cuda_test PRIVATE OpenMP::OpenMP_CXX)
    endif()
    target_link_libraries(cuda_test PRIVATE cusparse_static cudart)

  elseif(target STREQUAL "hip")
    find_package(hip REQUIRED)
    enable_language(HIP)
    add_executable(hip_test
      main.cpp
      core.cu
      tensor.cu
      matrix.cu
      solver.cu
    )
    set_source_files_properties(main.cpp PROPERTIES LANGUAGE HIP)
    set_source_files_properties(core.cu PROPERTIES LANGUAGE HIP)
    set_source_files_properties(tensor.cu PROPERTIES LANGUAGE HIP)
    set_source_files_properties(matrix.cu PROPERTIES LANGUAGE HIP)
    set_source_files_properties(solver.cu PROPERTIES LANGUAGE HIP)
    target_compile_definitions(hip_test PRIVATE SENK_HIP)
    set_target_properties(hip_test PROPERTIES
      HIP_STANDARD 20
    )
    target_include_directories(hip_test PRIVATE ${COMMON_INCLUDES})
    find_package(OpenMP REQUIRED COMPONENTS CXX)
    if (OpenMP_CXX_FOUND)
      target_link_libraries(hip_test PRIVATE OpenMP::OpenMP_CXX)
    endif()
    target_link_libraries(hip_test PRIVATE hip::device)

  else()
    message(WARNING "Unknown target: ${target}")
  endif()
endforeach()
