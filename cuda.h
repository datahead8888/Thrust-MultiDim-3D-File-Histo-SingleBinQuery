#ifndef CUDA_H
#define CUDA_H

#include <thrust/device_vector.h>
#include <vector>
#include <iostream>
#include <opencv2/opencv.hpp>

using namespace std;


thrust::host_vector<int> doHistogramGPU();
std::vector<int> doHistogramCPU();


//#define IS_LOGGING 1

typedef thrust::tuple<int, int, int> Int3;

struct BinFinder
{
	//This kernel assigns each element to a bin group
	__host__ __device__ Int3 operator()(const Int3 & param1, const Int3 & param2) const
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
		int y = thrust::get<1>(param1);
		int z = thrust::get<2>(param1);

		int bin1 = (x >= 0 && x <= 63) * 1 +
			(x >=64 && x <= 127) * 2 +
			(x >= 128 && x <= 191) * 3 +
			(x >= 192) * 4;
		int bin2 = (y >= 0 && y <= 63) * 1 +
			(y >=64 && y <= 127) * 2 +
			(y >= 128 && y <= 191) * 3 +
			(y >= 192) * 4;
		int bin3 = (z >= 0 && z <= 63) * 1 +
			(z >=64 && z <= 127) * 2 +
			(z >= 128 && z <= 191) * 3 +
			(z >= 192) * 4;

		return thrust::make_tuple(bin1, bin2, bin3);

		
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
	inline bool operator() (const Int3 & a, const Int3 & b)
	{
		return thrust::get<0>(a) < thrust::get<0>(b) ||
			(thrust::get<0>(a) == thrust::get<0>(b) && thrust::get<1>(a) < thrust::get<1>(b)) ||
			(thrust::get<0>(a) == thrust::get<0>(b) && thrust::get<1>(a) == thrust::get<1>(b) && thrust::get<2>(a) < thrust::get<2>(b));
	}
};


#endif