cd ..
mkdir build
cd build
cmake -DMRT=OFF -DLES=OFF ..
cd ..

# SRT benchmark
(cd build && make GPU_target)
bin/GPU_target input/velocity_validation.in

# MRT benchmark
cmake -DMRT=ON -DLES=OFF build
(cd build && make GPU_target)
bin/GPU_target input/velocity_validation.in

# MRT-LES benchmark
cmake -DMRT=ON -DLES=ON build
(cd build && make GPU_target)
bin/GPU_target input/velocity_validation.in

# generate plots with python file, plots saved in current folder
cd scripts
python ./velocity_validation.py