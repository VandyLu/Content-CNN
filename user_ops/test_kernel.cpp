
#include <iostream>

using namespace std;

void ShiftCorrKernel(const float* a,const float *b,float *out,const int dispmax,
		const int n_in,const int h_in,const int w_in,const int c_in);

int main()
{
	// a[2][2][2][2]
	// d = 3
	float a[16],b[16],y[24];
	for(int i=0;i<16;i++)
	{
		a[i]=i;
		b[i]=i;
	}
	const int n=2;
	const int h=2,w=2,c=2;
	const int d=3;
	ShiftCorrKernel(a,b,y,d,n,h,w,c);
	for(int i=0;i<24;i++)
		cout << y[i]<<' ';
	cout << endl;
	return 0;
}
