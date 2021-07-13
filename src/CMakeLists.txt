include_directories(../include)

add_executable(
  LBM_sim
  main.cpp
  #main_gpu.cu
  core.cpp
  #core_gpu.cu
  init.cpp
  utils.cpp
)