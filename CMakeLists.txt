cmake_minimum_required(VERSION 3.8 FATAL_ERROR)

project(LBM_sim LANGUAGES C CXX)

# GoogleTest requires at least C++11
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_FLAGS "-O3")

include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/609281088cfefc76f9d0ce82e1ff6c30cc3591e5.zip
)
# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

# Check for GPU support
#option(USE_GPU "Enabling GPU" ON)
#if(USE_GPU)
#  enable_language(CUDA)
#  set( CMAKE_EXE_LINKER_FLAGS ${CMAKE_EXE_LINKER_FLAGS})
#endif()

# Build Targets
enable_testing()
add_subdirectory(src)
add_subdirectory(tests)
include_directories(include)
file(MAKE_DIRECTORY build/out ${directory})
#add_subdirectory(doc)


