
TF_INC=/usr/local/lib/python2.7/dist-packages/tensorflow/include

TEST_KERNEL:test_kernel.cpp
	g++ -o test test_kernel.cpp shift_corr_kernel.cc

SHIFT_OP:shift_corr.cc shift_corr_kernel.cc
	g++ -std=c++11 -shared shift_corr.cc shift_corr_kernel.cc -o shift_corr.so -fPIC -I $(TF_INC) -O2 -D_GLIBCXX_USE_CXX11_ABI=0
	

