#include "cuda.h"
#include "cudaTimer.h"
#include "windowsCpuTimer.h"

#include <iostream>
#include <iomanip>

#include <thrust/device_vector.h>
#include <thrust/transform.h>
#include <thrust/sequence.h>
#include <thrust/copy.h>
#include <thrust/fill.h>
#include <thrust/replace.h>
#include <thrust/functional.h>
#include <thrust/sort.h>
#include <thrust/adjacent_difference.h>
#include <thrust/iterator/constant_iterator.h>
#include <opencv2/opencv.hpp>

#include <Windows.h>

using namespace std;
using namespace cv;

bool loadImage(string fileName, Mat & image)
{
	image = imread(fileName);

	if (image.empty())
	{
		cerr << "Error in loading image" << endl;
		return false;
	}

	cout << "Image dimensions: " << image.cols << " X " << image.rows << endl;

	return true;

	
}

bool generateRandomData(int rows, int cols, int max, thrust::host_vector<int> & data)
{
	for (int i = 0; i < rows; i++)
	{
		for (int j = 0; j < cols; j++)
		{
			data[i * cols + j] = rand() % max + 1;

		}
	}

	return true;

}

void printData(int rows, int printWidth, thrust::host_vector<int> & data)
{
	for (int i = 0; i < rows; i++)
	{
		cout << setw(printWidth) << data[i] << endl;
	
	}

}

void printData(int rows, int printWidth, thrust::device_vector<int> & data)
{
	for (int i = 0; i < rows; i++)
	{
		cout << setw(printWidth) << data[i] << endl;
	
	}

}

void printData(int rows, int cols, int printWidth, thrust::host_vector<int> & data)
{
	for (int i = 0; i < rows; i++)
	{
		for (int j = 0; j < cols; j++)
		{
			cout << setw(printWidth) << data[i * cols + j];

		}
		cout << endl;
	}

}

void printData(int rows, int cols, int printWidth, thrust::device_vector<int> & data)
{
	for (int i = 0; i < rows; i++)
	{
		for (int j = 0; j < cols; j++)
		{
			cout << setw(printWidth) << data[i * cols + j];

		}
		cout << endl;
	}

}

void printHistoData(int rows, int cols, int printWidth, thrust::host_vector<int> & multiDimKeys, thrust::host_vector<int> & counts)
{
	for (int i = 0; i < rows; i++)
	{
		for (int j = 0; j < cols; j++)
		{
			cout << setw(printWidth) << multiDimKeys[i * cols + j];

		}

		cout << setw(printWidth) << "*" << counts[i];

		cout << endl;
	}

}

thrust::host_vector<int> doHistogramGPU(int ROWS, int COLS, int MAX)
{
	
	thrust::host_vector<int> h_data(COLS * ROWS);
	
	generateRandomData(ROWS, COLS, MAX, h_data);
	
	#ifdef IS_LOGGING	
	printData(ROWS, COLS, 5, h_data);
	#endif
	
	thrust::device_vector<int>d_data(h_data.begin(), h_data.end());

	//auto zipFirst = thrust::make_zip_iterator(thrust::make_tuple(d_red_vector.begin(), d_green_vector.begin(), d_blue_vector.begin()));
	//auto zipLast = thrust::make_zip_iterator(thrust::make_tuple(d_red_vector.end(), d_green_vector.end(), d_blue_vector.end()));
	auto zipFirst = thrust::make_zip_iterator(thrust::make_tuple(d_data.begin()));
	auto zipLast = thrust::make_zip_iterator(thrust::make_tuple(d_data.end()));

	
	CudaTimer cudaTimer;
	WindowsCpuTimer cpuTimer;
	
	//Reference: http://stackoverflow.com/questions/1739259/how-to-use-queryperformancecounter
	
	//Timing code start
	

	#ifdef IS_LOGGING
	cout << "Running multidimensional histogram GPU method..." << endl;
	cout << endl;
	#endif

	cudaTimer.startTimer();
	cpuTimer.startTimer();
	
	#ifdef IS_LOGGING
	cout << "Running transform:" << endl;
	#endif

	
	//Phase 1: Find the bins for each of the elements

    thrust::transform(zipFirst, zipLast, zipFirst, zipFirst, BinFinder());

	#ifdef IS_LOGGING
	cout << "Printing bin assignment" << endl;
	printData(ROWS, COLS, 5, d_data);
	#endif
	

	//Phase 2: Convert this effectively multi-dimensional vector into a one dimensional vector
	
	//TO DO: Parallelize this

	//////////////////////////////////////////
	h_data = d_data; //Copy from device_vector back to host_vector, since this code is currently executed on the CPU

	thrust::host_vector<int> h_single_data(ROWS);

	for (int i = 0; i < ROWS; i++)
	{
		h_single_data[i] = 0;
		int factor = 1;
		for (int j = COLS - 1; j >= 0; j--)
		{
			h_single_data[i] += (h_data[i * COLS + j] - 1) * factor;

			factor *= 4;

		}
	}

	#ifdef IS_LOGGING	
	cout << "Printing 1-D representation of data - from CPU" << endl;
	printData(ROWS, 5, h_single_data);
	#endif

	/////////////////////////////////////////////////////////////////////////////////

	thrust::device_vector<int> d_single_data(ROWS);

	thrust::constant_iterator<int> colCountIt(COLS);
	thrust::counting_iterator<int> counter(0);
	auto zipStart = thrust::make_zip_iterator(thrust::make_tuple(counter, colCountIt, d_single_data.begin()));
	auto zipEnd = thrust::make_zip_iterator(thrust::make_tuple(counter + d_single_data.size(), colCountIt + d_single_data.size(), d_single_data.end()));

	thrust::device_ptr<int> devPtr = &d_data[0];

	thrust::for_each(zipStart, zipEnd, MultiToSingleDim(thrust::raw_pointer_cast(devPtr)));

	#ifdef IS_LOGGING	
	cout << "Printing 1-D representation of data - from GPU - Prelim" << endl;
	printData(ROWS, 5, d_single_data);
	#endif

	cout << endl;
	
	////Step 2: Sort those bin ids
	thrust::sort(d_single_data.begin(), d_single_data.end());

	#ifdef IS_LOGGING	
	cout << "Printing SORTED 1-D representation of data" << endl;
	printData(ROWS, 5, d_single_data);
	#endif

	////Step 3: Use the reduce by key function to get a count of each bin type
	thrust::constant_iterator<int> cit(1);
	thrust::device_vector<int> d_counts(h_single_data.size());  //4 ^ 3

	typedef thrust::device_vector<int>::iterator DVI;

	thrust::pair<DVI, DVI> endPosition = thrust::reduce_by_key(d_single_data.begin(), d_single_data.end(), cit, d_single_data.begin(), d_counts.begin());

	#ifdef IS_LOGGING
	
	cout << "Results after reduce key: " << endl;

	cout << "Keys (the 1-d representation of data): " << endl;

	for (DVI it = d_single_data.begin(); it != endPosition.first; it++)
	{
		cout << setw(4) << *it << " ";
	}
		
	cout << endl << "Counts:" << endl;

	for (DVI it = d_counts.begin(); it != endPosition.second; it++)
	{
		cout << setw(4) << *it << " ";
	}

	cout << endl;
	cout << endl;
	#endif
	
	thrust::host_vector<int> final_data (d_single_data.size() * COLS);

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

	#ifdef IS_LOGGING
	cout << "Final multidimensional representation from CPU" << endl;
	printHistoData(i, COLS, 5, final_data, thrust::host_vector<int>(d_counts.begin(), d_counts.end()));
	#endif

	///////////////////////////////////////////////////////////////////////////////////////////////////////
	//Multidimensional representation construction - GPU - WIP...
	thrust::device_vector<int> d_final_data (d_single_data.size() * COLS);
	devPtr = &d_final_data[0];

	//auto zipStart = thrust::make_zip_iterator(thrust::make_tuple(counter, colCountIt, d_single_data.begin()));
	//auto zipEnd = thrust::make_zip_iterator(thrust::make_tuple(counter + d_single_data.size(), colCountIt + d_single_data.size(), d_single_data.end()));

	
	//Note: We can use the same zipStart and zipEnd iterators as before; we just use a different kernel and a different raw data pointer
	thrust::for_each(zipStart, zipEnd, SingleToMultiDim(thrust::raw_pointer_cast(devPtr)));

	#ifdef IS_LOGGING
	cout << "Final multidimensional representation from GPU" << endl;
	printHistoData(d_single_data.size(), COLS, 5, thrust::host_vector<int>(d_final_data.begin(), d_final_data.end()), thrust::host_vector<int>(d_counts.begin(), d_counts.end()));
	#endif

	
	cudaTimer.stopTimer();
	cpuTimer.stopTimer();

	cout << "GPU time elapsed for GPU method #2: " << cudaTimer.getTimeElapsed() << endl;
	
	

	cout << "CPU time elapsed for GPU method #2: " << cpuTimer.getTimeElapsed() << endl;
	

	return final_data;

	

}

std::vector<int> doHistogramCPU(int ROWS, int COLS, int MAX)
{
	thrust::host_vector<int> h_data(COLS * ROWS);
	
	generateRandomData(ROWS, COLS, MAX, h_data);
	
	#ifdef IS_LOGGING
	cout << "Random data:" << endl;
	printData(ROWS, COLS, 5, h_data);
	#endif
	
	//Reference: http://stackoverflow.com/questions/1739259/how-to-use-queryperformancecounter
	//Timing code start
	WindowsCpuTimer cpuTimer;
	cpuTimer.startTimer();
	
	
	#ifdef IS_LOGGING
	cout << "Running histogram CPU Method..." << endl;
	cout << endl;
	#endif

	//Calculate the number of elements belonging in each bin on the CPU using a for loop
	
	int numElements = 1;
	for (int i = 0; i < COLS; i++)
	{
		numElements *= 4;
	}


	std::vector<int> finalCounts(numElements);

	for (int i = 0; i < finalCounts.size(); i++)
	{
		finalCounts[i] = 0;
	}


	for (int i = 0; i < ROWS; i++)
	{
		int factor = 1;
		int sum = 0;
		for (int j = COLS - 1; j >= 0; j--)
		{
			//sum += (h_data[i * COLS + j] - 1) * factor;
			int value = h_data[i * COLS + j];

			int binValue = 0;

			if (value <= 5)
			{
				binValue = 0;
			}
			else if (value >= 6 && value <= 9)
			{
				binValue = 1;
			}
			else if (value >= 10 && value <= 14)
			{
				binValue = 2;
			}
			else
			{
				binValue = 3;
			}

			sum += binValue * factor;

			factor *= 4;

		}

		finalCounts[sum]++;
	}

	//Timing code end
	cpuTimer.stopTimer();

	#ifdef IS_LOGGING
	cout << "Generated histogram:" << endl;
	printData(finalCounts.size(), 5, thrust::host_vector<int>(finalCounts.begin(), finalCounts.end()));

	cout << endl;
	#endif
	


	cout << "CPU time elapsed for CPU method: " << cpuTimer.getTimeElapsed() << endl;

	return finalCounts;

}

