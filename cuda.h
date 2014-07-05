#ifndef CUDA_H
#define CUDA_H

#include <thrust/device_vector.h>
#include <vector>
#include <iostream>
//#include <opencv2/opencv.hpp>


using namespace std;

void printData(int rows, int cols, int printWidth, thrust::device_vector<int> & data);
void printData(int rows, int cols, int printWidth, thrust::host_vector<int> & data);
void printData(int rows, int cols, int printWidth, thrust::host_vector<float> & data);
bool generateRandomData(int rows, int cols, int max, thrust::host_vector<int> & data);
void loadTextFile(const string & fileName, int xSize, int ySize, int zSize, int numvars, thrust::host_vector<float> & h_data );
thrust::host_vector<int> doHistogramGPU(int xSize, int ySize, int zSize, int numvars, thrust::host_vector<float> & h_data);
std::vector<int> doHistogramCPU(int ROWS, int COLS, thrust::host_vector<int> & h_data);

#define PRINT_INPUT 1
#define IS_LOGGING 1
//#define PRINT_RESULT 1

typedef thrust::tuple<int, int, int> Int3;
typedef thrust::tuple<int, int> Int2;
typedef thrust::tuple<int> Int;

typedef thrust::tuple<float, float, float> Float3;
typedef thrust::tuple<float, float> Float2;
typedef thrust::tuple<float> Float;


struct BinFinder
{
	float * rawMinVector;
	float * rawMaxVector;
	int numVars;
	int numBins;

	
	BinFinder(float * rawMinVector, float * rawMaxVector, int numVars, int numBins)
	{
		this -> rawMinVector = rawMinVector;
		this -> rawMaxVector = rawMaxVector;
		this -> numVars = numVars;
		this -> numBins = numBins;
	}


	//This kernel assigns each element to a bin group
	__host__ __device__ Int operator()(const Float & param1, const int & param2) const
	{
		

		float value = thrust::get<0>(param1);
		int id = param2;

		float min = rawMinVector[id % numVars];
		float max = rawMaxVector[id % numVars];

		

		float percentage = (value - min) / float(max - min);


		int bin = percentage * numBins;

		if (bin == numBins)
		{
			bin--;
		}

		
		return thrust::make_tuple(bin);

		
	}
	
};


struct MultiToSingleDim
{
	int * rawVector;

	
	MultiToSingleDim(int * rawVector)
	{
		this -> rawVector = rawVector;
	}

	//This kernel assigns each element to a bin group
	template <typename Tuple>
	__device__ void operator()( Tuple param) 
	{
	

		//int data = thrust::get<0>(param);
		int singleDimIndex = thrust::get<0>(param);
		int cols = thrust::get<1>(param);
		
		int newValue = 0;
		int factor = 1;
		for (int j = cols - 1; j >= 0; j--)
		{
			newValue += (rawVector[singleDimIndex * cols + j]) * factor;

			factor *= 4;


		}


		thrust::get<2>(param) = newValue;
		
	}
	
};

//TO DO: Port this
struct SingleToMultiDim
{
	int * rawVector;

	
	SingleToMultiDim(int * rawVector)
	{
		this -> rawVector = rawVector;
	}

	template <typename Tuple>
	__device__ void operator()( Tuple param) 
	{
	

		int singleDimIndex = thrust::get<0>(param);
		int cols = thrust::get<1>(param);
		int dataValue = thrust::get<2>(param);
		
		for (int j = cols - 1; j >= 0; j--)
		{
			//newValue += (rawVector[singleDimIndex * cols + j] - 1) * factor;
			int moddedValue = dataValue % 4 + 1;
			rawVector[singleDimIndex * cols + j] = moddedValue;
			dataValue /= 4;


		}

		/*
		//Multidimensional representation reconstruction - CPU
	int i = 0;
	for (DVI it = d_single_data.begin(); it != endPosition.first; it++, i++)
	{
		int value = *it;

		for (int j = COLS - 1; j >= 0; j--)
		{
			int moddedValue = value % 4 + 1;
			final_data[i * COLS + j] = moddedValue;
			value /= 4;

		}
	}
	*/
		
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