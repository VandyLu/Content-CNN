

ALL:dispmap.cpp
	gcc -shared -o dispmap.so -fPIC -I/usr/include/python2.7/ dispmap.cpp -lpython2.7 -lboost_python -lpng
	#gcc -shared -o dispmap.so -fPIC -I/usr/include/python2.7/ dispmap.cpp -lpython2.7 -lboost_python `libpng-config --cflags --ldflags` -std=c++11

