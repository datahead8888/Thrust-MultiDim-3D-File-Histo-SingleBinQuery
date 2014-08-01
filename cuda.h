#ifndef CUDA_H
#define CUDA_H

//#define PRINT_INPUT 1		//If true prints the input used on the screen
//#define IS_LOGGING 1		//If true does logging for detailed statements
#define PRINT_RESULT 1		//If true will print final results
//#define DO_CPU_COMPUTATION  //If true will calculate the histogram on the CPU

#include <thrust/device_vector.h>
#include <vector>
#include <iostream>
#include "cudaTimer.h"
#include "windowsCpuTimer.h"


using namespace std;

typedef thrust::device_vector<int>::iterator DVI;
typedef thrust::device_vector<long long>::iterator DVL;

void printData(int rows, int printWidth, thrust::host_vector<int> & data);
void printDataNoZeroes(int rows, int printWidth, thrust::host_vector<int> & data);
void printData(int rows, int cols, int printWidth, thrust::device_vector<int> & data);
void printData(int rows, int cols, int printWidth, thrust::host_vector<int> & data);
void printData(int rows, int cols, int printWidth, thrust::host_vector<float> & data);
bool generateRandomData(int rows, int cols, int max, thrust::host_vector<int> & data);
bool loadTextFile(FILE *infile, int xSize, int ySize, int zSize, int numvars, int maxVars, thrust::host_vector<float> & h_data, int bufferSize, int & xPos, int & yPos, int & zPos);
void doHistogramGPU(int xSize, int ySize, int zSize, int numvars, thrust::host_vector<float> & h_buffer, thrust::host_vector<long long> & h_data, thrust::host_vector<long long> & h_data2, int numBins, CudaTimer & cudaTimer, WindowsCpuTimer & cpuTimer);
void histogramMapReduceGPU(thrust::host_vector<long long> & h_data, thrust::host_vector<long long> & h_data2, thrust::pair<DVL, DVL> & endPosition, int numVars, int numBins, CudaTimer & cudaTimer, WindowsCpuTimer & cpuTimer);
void doHistogramCPU(int xSize, int ySize, int zSize, int numVars, int numBins, thrust::host_vector<float> & h_data);
void printHistoData(int rows, int cols, int printWidth, thrust::host_vector<long long> & multiDimKeys, thrust::host_vector<long long> & counts);
int printMinMaxes(string & fileName, int numRecords, int numvars);
void doQuery(int xSize, int ySize, int zSize, int xMin, int xMax, int yMin, int yMax, int zMin, int zMax, thrust::host_vector<long long> & h_data, thrust::host_vector<long long> & h_data2);

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
	int numBins;

	
	MultiToSingleDim(int * rawVector, int numBins)
	{
		this -> rawVector = rawVector;
		this -> numBins = numBins;
	}

	//This kernel converts the multidimensional bin representation to a single dimensional representation
	template <typename Tuple>
	__device__ void operator()( Tuple param) 
	{
		int singleDimIndex = thrust::get<0>(param);
		int cols = thrust::get<1>(param);
		
		long long newValue = 0;
		long long factor = 1;
		for (int j = cols - 1; j >= 0; j--)
		{
			newValue += (rawVector[singleDimIndex * cols + j]) * factor;

			factor *= numBins;


		}


		thrust::get<2>(param) = newValue;
		
	}
	
};

struct SingleToMultiDim
{
	long long * rawVector;
	int numBins;

	
	SingleToMultiDim(long long * rawVector, int numBins)
	{
		this -> rawVector = rawVector;
		this -> numBins = numBins;
	}

	//This kernel converts a single dimensional bin representation back to a multidimensional representation
	template <typename Tuple>
	__device__ void operator()( Tuple param) 
	{
	

		int singleDimIndex = thrust::get<0>(param);
		int cols = thrust::get<1>(param);
		long long dataValue = thrust::get<2>(param);
		
		for (int j = cols - 1; j >= 0; j--)
		{
			int moddedValue = dataValue % numBins;
			rawVector[singleDimIndex * cols + j] = moddedValue;
			dataValue /= numBins;


		}

		
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

//Input: a device_vector of ids (first param in ())
//	Parameters: xSize, ySize, zSize - dimensions of the grid
//Output: a second device vector of values (0 for non inclusion, 1 for inclusion)
struct QueryRange
{
	int xSize, ySize, zSize;

	int xMin, yMin, zMin;
	int xMax, yMax, zMax;


	
	QueryRange(int xSize, int ySize, int zSize, int xMin, int xMax, int yMin, int yMax, int zMin, int zMax)
	{
		this -> xSize = xSize;
		this -> ySize = ySize;
		this -> zSize = zSize;

		this -> xMin = xMin;
		this -> xMax = xMax;
		this -> yMin = yMin;
		this -> yMax = yMax;
		this -> zMin = zMin;
		this -> zMax = zMax;
	}


	//This kernel does query work
	//0 means use the element.  1 means don't use it.
	__host__ __device__ long long operator()(const long long & param) const
	{
		

		int id = param;
		long long result = 0;

		//int bin = id; // tempedy temp tempest! --> TEMPORARY!!!!

		int xPos = id % xSize;
		int yPos = (id / xSize) % ySize;
		int zPos = id / xSize / ySize;

		if (xPos >= xMin && xPos <= xMax && yPos >= yMin && yPos <= yMax && zPos >= zMin && zPos <= zMax)
		{
			result = 1;
			
		}



		//1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16
		//0 0 1 1 0 0 1 1 0 0   1  1  0  0  1  1

		
		return result;
		//return zPos;

		
	}
	
};

struct StoreQuery
{
	long long * rawVector;
	//int numBins;

	
	StoreQuery(long long * rawVector)
	{
		this -> rawVector = rawVector;
		//this -> numBins = numBins;
	}

	//This kernel converts a single dimensional bin representation back to a multidimensional representation
	template <typename Tuple>
	__device__ void operator()( Tuple param) 
	{
	

		int singleDimIndex = thrust::get<0>(param);
		//int cols = thrust::get<1>(param);
		long long dataValue = thrust::get<1>(param);
		
		//for (int j = cols - 1; j >= 0; j--)
		//{
		//	int moddedValue = dataValue % numBins;
		//	rawVector[singleDimIndex * cols + j] = moddedValue;
		//	dataValue /= numBins;


		//}

		rawVector[singleDimIndex] = dataValue;

		
	}
	
};