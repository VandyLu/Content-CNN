#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include <iostream>

REGISTER_OP("ShiftCorr")
	.Attr("dispmax: int=1")
	.Input("a: float")
	.Input("b: float")
	.Output("distrib: float")
	.Doc("");

using namespace tensorflow;

void ShiftCorrKernel(const float* a,const float *b,float *out,const int dispmax,
		const int n_in,const int h_in,const int w_in,const int c_in);
void ShiftCorrKernel_GPU(const float* a,const float *b,float *out,const int dispmax,
		const int n_in,const int h_in,const int w_in,const int c_in);

class ShiftCorrOp: public OpKernel{
public:
	explicit ShiftCorrOp(OpKernelConstruction*context):OpKernel(context){
		OP_REQUIRES_OK(context,context->GetAttr("dispmax",&dispmax_));
		OP_REQUIRES(context,dispmax_>0,errors::InvalidArgument("Need dispmax > 0, got",dispmax_));
	}
	void Compute(OpKernelContext* context) override{

		const Tensor& a_tensor = context->input(0);
		const Tensor& b_tensor = context->input(1);
		auto a = a_tensor.flat<float>();
		auto b = b_tensor.flat<float>();
//		OP_REQUIRES(context,a_tensor.shape()==b_tensor.shape(),"");
		
		TensorShape in_shape = a_tensor.shape();
		TensorShape out_shape = in_shape;
		out_shape.set_dim(3,dispmax_);

		Tensor* output_tensor = NULL;
		OP_REQUIRES_OK(context,context->allocate_output(0,out_shape,&output_tensor));
		auto out = output_tensor->flat<float>();
		
		ShiftCorrKernel(a.data(),b.data(),out.data(),dispmax_,
				in_shape.dim_size(0),in_shape.dim_size(1),in_shape.dim_size(2),in_shape.dim_size(3));
		//ShiftCorrKernel_GPU(a.data(),b.data(),out.data(),dispmax_,
		//		in_shape.dim_size(0),in_shape.dim_size(1),in_shape.dim_size(2),in_shape.dim_size(3));
	}
private:
	int dispmax_;
};

REGISTER_KERNEL_BUILDER(Name("ShiftCorr").Device(DEVICE_CPU),ShiftCorrOp);
//REGISTER_KERNEL_BUILDER(Name("ShiftCorr").Device(DEVICE_GPU),ShiftCorrOp);
