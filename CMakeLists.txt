cmake_minimum_required(VERSION 3.18 FATAL_ERROR)

project(LBM_sim LANGUAGES C CXX)

# Set output folder
set(EXECUTABLE_OUTPUT_PATH "${CMAKE_CURRENT_SOURCE_DIR}/bin")

# set up fictional target using an INTERFACE library
add_library(LBM_sim INTERFACE)
target_compile_features(LBM_sim INTERFACE cxx_std_11)
target_include_directories(LBM_sim INTERFACE "include"
  $<INSTALL_INTERFACE:include>)

set(CMAKE_CXX_FLAGS "-O3")

# Check for GPU support
option(USE_GPU "Enabling GPU" ON)
if(USE_GPU)
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
        set(CMAKE_CUDA_ARCHITECTURES 61)
    endif()
    enable_language(CUDA)
  set( CMAKE_EXE_LINKER_FLAGS ${CMAKE_EXE_LINKER_FLAGS})
endif()

# Define option to activate LES
option(LES "Large eddy simulation" ON)
if(LES)
    target_compile_definitions(LBM_sim INTERFACE LES=1)
else()
    target_compile_definitions(LBM_sim INTERFACE LES=0)
endif()

# Define option to activate MRT
option(MRT "Multi-Relaxation Time," ON)
if(MRT)
    target_compile_definitions(LBM_sim INTERFACE MRT=1)
else()
    target_compile_definitions(LBM_sim INTERFACE MRT=0)
endif()

include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/609281088cfefc76f9d0ce82e1ff6c30cc3591e5.zip
)
# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

# Build Targets
enable_testing()
add_subdirectory(src)
add_subdirectory(tests)
file(MAKE_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/out ${directory})


