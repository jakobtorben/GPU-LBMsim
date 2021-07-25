cd ..
mkdir build
cd build
cmake -DPERIODIC=OFF ..
make GPU_target
cd ..
bin/GPU_target input/benchmark.in
