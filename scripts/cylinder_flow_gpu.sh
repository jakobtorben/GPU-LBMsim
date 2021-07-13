cd ../src
nvcc -arch sm_61 -v --ptxas-options=-v -I../include -O3 core_gpu.cu init_gpu.cu utils.cpp main_gpu.cu -o LBM_sim
#nvcc -gencode arch=compute_61,code=sm_61 saxpy.cu
#cmake .. -DUSE_GPU=ON 
#cd ../build
cd ..
src/LBM_sim input/benchmark.in