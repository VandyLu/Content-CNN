#define BOOST_PYTHON_SOURCE
#include <boost/python.hpp>
#include <string>
#include <iostream>
#include <math.h>
#include "io_disp.h"
#include "utils.h"

using namespace std;
using namespace boost::python;

// wrap 
class DisparityMap_{
public:
	DisparityImage disp;
	
	DisparityMap_(){}
	void readPNG(string fp){ disp.read(fp);}
	void write(string s){ disp.write(s);}
	void writeColor(string s,float max_disp){ disp.writeColor(s,max_disp);}


	int width(){ return disp.width(); }
	int height(){ return disp.height(); }
	float get_pixel(int w,int h){ return disp.data()[h*disp.width()+w]; }

	float maxDisp(){ return disp.maxDisp();}
	void interpolateBackground(){ disp.interpolateBackground();}

	void setSize(int width,int height){ disp = DisparityImage(width,height);}
	void setData(float x,int width,int height){ disp.data_[width+height*disp.width_]=x;}

};
BOOST_PYTHON_MODULE(dispmap)
{
	class_<DisparityMap_>("DisparityMap","this is a doc")
		.def("readPNG",&DisparityMap_::readPNG,"doc")
		.def("width",&DisparityMap_::width)
		.def("height",&DisparityMap_::height)
		.def("get_pixel",&DisparityMap_::get_pixel)
		.def("write",&DisparityMap_::write)
		.def("writeColor",&DisparityMap_::writeColor)
		.def("maxDisp",&DisparityMap_::maxDisp)
		.def("interpolateBackground",&DisparityMap_::interpolateBackground)
		.def("setSize",&DisparityMap_::setSize)
		.def("setData",&DisparityMap_::setData);
}
