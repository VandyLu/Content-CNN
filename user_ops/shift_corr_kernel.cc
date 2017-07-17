#include <iostream>


void ShiftCorrKernel(const float* a,const float *b,float *out,const int dispmax,
		const int n_in,const int h_in,const int w_in,const int c_in)
{
	int n_step = h_in*w_in*c_in;
	int h_step = w_in*c_in;
	int w_step = c_in;

	int out_n_step = h_in*w_in*dispmax;
	int out_h_step = w_in*dispmax;
	int out_w_step = dispmax;
	
	for (int n=0;n<n_in;n++)
	{
	int n_diff = n*n_step;	
	int out_n_diff = n*out_n_step;
	for(int h=0;h<h_in;h++)
	{
		int h_diff = h*h_step;
		int out_h_diff = h*out_h_step;
		for(int w=0;w<w_in;w++)
		{
			int w_diff = w*w_step;
			int out_w_diff = w*out_w_step;

			int in_diff = n_diff+h_diff+w_diff;
			int out_diff = out_n_diff+out_h_diff+out_w_diff;
			for(int d=0;d<dispmax;d++)
			{
				if(w-d>=0)
				{
					float sum = 0.0;
					for(int c=0;c<c_in;c++)
						sum += a[in_diff+c]*b[in_diff-d*c_in+c];
					out[out_diff+d] = sum; 
				}else{
					out[out_diff+d] = -1.0;
				}
			}
		}
	}
	}
}
