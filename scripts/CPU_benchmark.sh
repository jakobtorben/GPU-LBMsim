cd ..
mkdir build
cd build
cmake -DMRT=OFF -DLES=OFF -DUSE_GPU=OFF ..
cd ..

# SRT benchmark
(cd build && make CPU_target)
bin/CPU_target input/CPU_benchmark.in

# MRT benchmark
cmake -DMRT=ON -DLES=OFF -DUSE_GPU=OFF build
(cd build && make CPU_target)
bin/CPU_target input/CPU_benchmark.in

# MRT-LES benchmark
cmake -DMRT=ON -DLES=ON -DUSE_GPU=OFF build
(cd build && make CPU_target)
bin/CPU_target input/CPU_benchmark.in
