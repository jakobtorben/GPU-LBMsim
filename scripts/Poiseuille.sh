cd ..
mkdir build out results
cd build
cmake ..
make all
./src/LBM_sim ../input/Poiseuille.in