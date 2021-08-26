cd ..
mkdir build
cd build
cmake -DMRT=OFF -DLES=OFF ..
cd ..

# SRT benchmark
(cd build && make GPU_target)
bin/GPU_target input/GPU_benchmark.in

# MRT benchmark
cmake -DMRT=ON -DLES=OFF build
(cd build && make GPU_target)
bin/GPU_target input/GPU_benchmark.in

# MRT-LES benchmark
cmake -DMRT=ON -DLES=ON build
(cd build && make GPU_target)
bin/GPU_target input/GPU_benchmark.in
