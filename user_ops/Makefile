
TF_INC=/usr/local/lib/python2.7/dist-packages/tensorflow/include

TEST_KERNEL:test_kernel.cpp
	g++ -o test test_kernel.cpp shift_corr_kernel.cc

CPU:shift_corr_cpu.cc shift_corr_kernel.cc
	g++ -std=c++11 -shared shift_corr_cpu.cc shift_corr_kernel.cc -o shift_corr.so -fPIC -I $(TF_INC) -O2 -D_GLIBCXX_USE_CXX11_ABI=0
	
GPU:shift_corr_gpu.cc shift_corr_kernel.cu
	nvcc -std=c++11 -c -o shift_corr_kernel.cu.o shift_corr_kernel.cu -I $(TF_INC) -D GOOGLE_CUDA=1 -x cu -Xcompiler -fPIC
	g++ -std=c++11 -shared shift_corr_gpu.cc shift_corr_kernel.cu.o -o shift_corr.so -fPIC -I $(TF_INC) -O2 -D_GLIBCXX_USE_CXX11_ABI=0

