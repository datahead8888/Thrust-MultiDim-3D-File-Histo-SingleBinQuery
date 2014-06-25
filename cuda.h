#ifndef CUDA_H
#define CUDA_H

#include <thrust/device_vector.h>
#include <vector>
#include <iostream>
#include <opencv2/opencv.hpp>


using namespace std;

void printData(int rows, int cols, int printWidth, thrust::device_vector<int> & data);
void printData(int rows, int cols, int printWidth, thrust::host_vector<int> & data);
bool generateRandomData(int rows, int cols, int max, thrust::host_vector<int> & data);
bool loadImage(string fileName, cv::Mat & image);
thrust::host_vector<int> doHistogramGPU(int ROWS, int COLS, int MAX);
std::vector<int> doHistogramCPU(int ROWS, int COLS, int MAX);


#define IS_LOGGING 1

typedef thrust::tuple<int, int, int> Int3;
typedef thrust::tuple<int> Int;

struct BinFinder
{
	//This kernel assigns each element to a bin group
	__host__ __device__ Int operator()(const Int & param1, const Int & param2) const
	{
		//return x + y;
		

		//if (x >= 0 && x <=5)
		//{
		//	return 1;
		//}
		//else if (x >=6 && x <= 10)
		//{
		//	return 2;
		//}
		//else if (x >= 11 && x <= 15)
		//{
		//	return 3;
		//}
		//else
		//{
		//	return 4;
		//}

		//return (x >= 0 && x <= 5) * 1 +
		//	(x >=6 && x <= 10) * 2 +
		//	(x >= 11 && x <= 15) * 3 +
		//	(x >= 16 && x <=20) * 4;
		
		//cout << x << " ";


		int x = thrust::get<0>(param1);
		

		int bin = (x <= 5) * 1 +
			(x >= 6 && x <= 9) * 2 +
			(x >= 10 && x <= 14) * 3 +
			(x >= 15) * 4;
		
		return thrust::make_tuple(bin);

		
	}
	
};



struct IndexFinder
{
	//This kernel looks at a bin element and assigns the counting index if it is not 0 and assigns -1 if it is 0
	__host__ __device__ int operator()(const int & x, const int & y) const
	{
		//if (x != 0)
		//{
		//	return y;
		//}
		//else
		//{
		//	return -1;
		//}
		return (x != 0) * y + (x == 0) * -1;
		
		
		
	}
	
};


struct ZipComparator
{
	__host__ __device__
	inline bool operator() (const Int & a, const Int & b)
	{
		return thrust::get<0>(a) < thrust::get<0>(b);
	}
};


#endif