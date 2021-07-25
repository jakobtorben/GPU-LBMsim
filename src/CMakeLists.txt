
# set up CUDA target if using GPU
if(USE_GPU)
    add_custom_target(GPU_LBM COMMENT "Builds GPU targets.")
    add_executable(GPU_target main_gpu.cu core_gpu.cu init_gpu.cu utils.cpp)
    target_link_libraries(GPU_target PRIVATE LBM_sim)
    target_compile_definitions(GPU_target PRIVATE)
    set_target_properties(GPU_target PROPERTIES CUDA_SEPARABLE_COMPILATION OFF)
    add_dependencies(GPU_LBM GPU_target)
endif()

# set up CPU target
add_custom_target(CPU_LBM COMMENT "Builds CPU targets.")
add_executable(CPU_target main.cpp core.cpp init.cpp utils.cpp)
target_link_libraries(CPU_target PRIVATE LBM_sim)
add_dependencies(CPU_LBM CPU_target)