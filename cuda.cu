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
//#include <vtkExecutive.h>
//#include <vtkStructuredPointsReader.h>
//#include <vtkAlgorithm.h>

#include <Windows.h>

using namespace std;


bool loadTextFile(FILE *infile, int xSize, int ySize, int zSize, int numvars, int maxVars, thrust::host_vector<float> & h_data, int bufferSize, int & xPos, int & yPos, int & zPos )
{

	WindowsCpuTimer cpuTimer;

	cpuTimer.startTimer();

	
	
	
	//Data from http://sciviscontest.ieeevis.org/2008/data.html
	//fscanf code below is also partially borrowed from those pages

	float currentValue = 0;
	int recordsRead = 0;

	for (int z = zPos; z < zSize; z++)
	{
		for (int y = yPos; y < ySize; y++)
		{
			for (int x = xPos; x < xSize; x++)
			{
				bool hadEOF = false;
				for (int v = 0; v < numvars; v++)
				{

					fscanf(infile, "%f", &currentValue);

					if (feof(infile))
					{
						hadEOF = true;
						break;
					}

					#ifdef PRINT_INPUT
					cout << "x = " << x << " y = " << y << " z = " << z << " v = " << v << endl;
					//cout << "Density: " << density << " Temperature: " << temperature << " ab_H " << ab_H << " ab_HP " << ab_HP << " ab_He " << ab_He << " ab_HeP " << ab_HeP << " ab_HEPP " << ab_HePP << " ab_HM " << ab_HM << " ab_H2 "<< ab_H2 << " ab_H2P " << ab_H2P << endl;
					cout << "Value: " << currentValue << endl;
					#endif

					h_data[recordsRead * numvars + v] = currentValue;

				} //END: for (int v = 0; v < numvars && keepGoing; v++)

				
				//If less variables are requested than are in the file (currently 10), burn through variables until we get to the next record
				if (!hadEOF && numvars < maxVars)
				{
					for (int v = 0; v < maxVars - numvars; v++)
					{
						fscanf(infile, "%f", &currentValue);

						if (feof(infile))
						{
							hadEOF = true;
							break;
						}


					}
				} //END: if (!hadEOF && numvars < maxVars)

				recordsRead++;

				if (recordsRead == bufferSize || hadEOF)
				{
					cpuTimer.stopTimer();

					cout << "File load time: " << cpuTimer.getTimeElapsed() << endl;

					
					//Hacky code to store the proper x, y, and z values to pick up on the for loop next time
					x++;
					
					if (x >= xSize)
					{
						y++;
						x = 0;
					}
					
					if (y >= ySize)
					{
						z++;
						x = 0;
						y = 0;
					}
										

					
					xPos = x; yPos = y; zPos = z;

					if (x >= xSize && y >= ySize && z >= zSize)
					{
						//Would have exited loop if didn't have a file size that is a multiple of the buffer size.  Return false to end the loop in the main function
						return false;
					}
					else
					{
						//More data remains in the file.  Return true to keep that loop in the main function going.
						return true;
					}
				}

			}
		}
	}

	//If records were read, we will return true so that the loop that calls this can do one more iteration.
	//It will then try to call this function again.  We need to set the x, y, and z starting positions so that no records will be read next time.
	xPos = xSize;
	yPos = ySize;
	zPos = zSize;

	if (recordsRead < bufferSize)
	{
		h_data.resize(recordsRead * numvars);
	}

	cpuTimer.stopTimer();
	cout << "File load time: " << cpuTimer.getTimeElapsed() << endl;


	if (recordsRead == 0)
	{
		return false;
	}
	else
	{
		return true;
	}


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

void printDataNoZeroes(int rows, int printWidth, thrust::host_vector<int> & data)
{
	for (int i = 0; i < rows; i++)
	{
		if (data[i] != 0)
		{
			cout << "i = " << i << ":" << setw(printWidth) << data[i] << endl;
		}
	
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

void printData(int rows, int cols, int printWidth, thrust::device_vector<float> & data)
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

void printData(int rows, int cols, int printWidth, thrust::host_vector<float> & data)
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

void doHistogramGPU(int xSize, int ySize, int zSize, int numVars, thrust::host_vector<float> & h_buffer, thrust::host_vector<int> & h_data, thrust::host_vector<int> & h_data2, int numBins, CudaTimer & cudaTimer, WindowsCpuTimer & cpuTimer)
{
	
	thrust::device_vector<float>d_data(h_buffer.begin(), h_buffer.end());
	thrust::device_vector<int>d_bins(h_buffer.size());

	auto zipInFirst = thrust::make_zip_iterator(thrust::make_tuple(d_data.begin()));
	auto zipInLast = thrust::make_zip_iterator(thrust::make_tuple(d_data.end()));
	auto zipOutFirst = thrust::make_zip_iterator(thrust::make_tuple(d_bins.begin()));
	thrust::counting_iterator<int> counter(0);
	
	
	
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
	
	float minValues[] = {0, 0, 0, 0, 0, 0, 7.392e-039, 0, 0, 0};
	float maxValues[] = {1001, 19910, 0.7599, 0.7595, 0.24, 0.2397, 0.1623, 1.1e-007, 3.464e-006, 4.447e-008};

	thrust::device_vector<float> d_minValues(minValues, minValues+10);
	thrust::device_vector<float> d_maxValues(maxValues, maxValues+10);

	#ifdef IS_LOGGING
	cout << "Min values vector:" << endl;
	for (int i = 0; i < d_minValues.size(); i++)
	{
		cout << d_minValues[i] << " ";
	}
	cout << endl;

	cout << "Max values vector:" << endl;
	for (int i = 0; i < d_maxValues.size(); i++)
	{
		cout << d_maxValues[i] << " ";
	}
	cout << endl;

	
	#endif

	thrust::device_ptr<float> minDevPtr = &d_minValues[0];
	thrust::device_ptr<float> maxDevPtr = &d_maxValues[0];

    thrust::transform(zipInFirst, zipInLast, counter, zipOutFirst, BinFinder(thrust::raw_pointer_cast(minDevPtr), thrust::raw_pointer_cast(maxDevPtr), numVars, numBins));

	#ifdef IS_LOGGING
	cout << "Printing bin assignment" << endl;
	printData(h_buffer.size() / numVars, numVars, 10, d_bins);
	#endif

	cout << endl;
	

	////Phase 2: Convert this effectively multi-dimensional vector into a one dimensional vector

	thrust::device_vector<int> d_single_data(h_buffer.size() / numVars);

	thrust::constant_iterator<int> colCountIt(numVars);
	//thrust::counting_iterator<int> counter(0);
	auto zipStart = thrust::make_zip_iterator(thrust::make_tuple(counter, colCountIt, d_single_data.begin()));
	auto zipEnd = thrust::make_zip_iterator(thrust::make_tuple(counter + d_single_data.size(), colCountIt + d_single_data.size(), d_single_data.end()));

	thrust::device_ptr<int> devPtr = &d_bins[0];

	thrust::for_each(zipStart, zipEnd, MultiToSingleDim(thrust::raw_pointer_cast(devPtr), numBins));

	#ifdef IS_LOGGING	
	cout << "Printing 1-D representation of data - from GPU - Prelim" << endl;
	printData(h_buffer.size() / numVars, 10, d_single_data);
	#endif

	//cout << endl;
	//
	//////Step 2: Sort those bin ids
	thrust::sort(d_single_data.begin(), d_single_data.end());

	#ifdef IS_LOGGING	
	cout << "Printing SORTED 1-D representation of data" << endl;
	printData(h_buffer.size() / numVars, 10, d_single_data);
	#endif

	//////Step 3: Use the reduce by key function to get a count of each bin type
	thrust::constant_iterator<int> cit(1);
	thrust::device_vector<int> d_counts(d_single_data.size());  //4 ^ 3

	//typedef thrust::device_vector<int>::iterator DVI;

	thrust::pair<DVI, DVI> endPosition = thrust::reduce_by_key(d_single_data.begin(), d_single_data.end(), cit, d_single_data.begin(), d_counts.begin());

	int numElements = endPosition.first - d_single_data.begin();
	
	#ifdef IS_LOGGING

	cout << "Number of elements from reduce key: " << numElements << endl;
	
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
	
	h_data.insert(h_data.begin(), d_single_data.begin(), endPosition.first);
	h_data2.insert(h_data2.begin(), d_counts.begin(), endPosition.second);
	
	
	
	cudaTimer.stopTimer();
	cpuTimer.stopTimer();

	/*
	#ifdef IS_LOGGING
	cout << "Final multidimensional representation from GPU" << endl;
	printHistoData(h_buffer.size() / numVars, numVars, 10, thrust::host_vector<int>(d_final_data.begin(), d_final_data.end()), thrust::host_vector<int>(d_counts.begin(), d_counts.end()));
	#endif
	*/

	cout << "GPU time elapsed for GPU method: " << cudaTimer.getTimeElapsed() << endl;

	cout << "CPU time elapsed for GPU method: " << cpuTimer.getTimeElapsed() << endl;
	
	


	

}

//h_data - the keys
//h_data2 - the counts
void histogramMapReduceGPU(thrust::host_vector<int> & h_data, thrust::host_vector<int> & h_data2, thrust::pair<DVI, DVI> & endPosition, int numVars, int numBins, CudaTimer & cudaTimer, WindowsCpuTimer & cpuTimer)
{
	cudaTimer.startTimer();
	cpuTimer.startTimer();
	
	thrust::device_vector<int> d_data(h_data.begin(), h_data.end());
	thrust::device_vector<int> d_data2(h_data2.begin(), h_data2.end());

	
	thrust::sort_by_key(d_data.begin(), d_data.end(), d_data2.begin());

	endPosition = thrust::reduce_by_key(d_data.begin(), d_data.end(), d_data2.begin(), d_data.begin(), d_data2.begin());

	#ifdef IS_LOGGING

	cout << "Did final map reduce..." << endl;
	cout << "GPU Keys:" << endl;                               //The new "d_single_data"

	for (DVI it = d_data.begin(); it != endPosition.first; it++)
	{
		cout << setw(4) << *it << " ";
	}
		
	cout << endl << "Counts:" << endl;

	cout << "GPU Counts:" << endl;

	for (DVI it = d_data2.begin(); it != endPosition.second; it++)
	{
		cout << setw(4) << *it << " ";
	}

	cout << endl;
	#endif
	
	int d_data_size = endPosition.first - d_data.begin();

	/////////////////////////////////////////////////////////////////////////////////////////////////////////
	////Multidimensional representation construction - GPU...
	thrust::counting_iterator<int> counter(0);
	thrust::constant_iterator<int> colCountIt(numVars);
	
	auto zipStart = thrust::make_zip_iterator(thrust::make_tuple(counter, colCountIt, d_data.begin()));
	auto zipEnd = thrust::make_zip_iterator(thrust::make_tuple(counter + d_data_size, colCountIt + d_data_size, endPosition.first));


	thrust::device_vector<int> d_final_data (d_data_size * numVars);
	thrust::device_ptr<int> devPtr = &d_final_data[0];
	
	////Note: We can use the same zipStart and zipEnd iterators as before; we just use a different kernel and a different raw data pointer
	thrust::for_each(zipStart, zipEnd, SingleToMultiDim(thrust::raw_pointer_cast(devPtr), numBins));

	//WIP Section below
	h_data.clear();
	h_data2.clear();

	h_data.insert(h_data.end(), d_final_data.begin(), d_final_data.end());
	h_data2.insert(h_data2.end(), d_data2.begin(), endPosition.second);

	cudaTimer.stopTimer();
	cpuTimer.stopTimer();

	cout << "GPU time elapsed for GPU map reduce: " << cudaTimer.getTimeElapsed() << endl;

	cout << "CPU time elapsed for GPU map reduce: " << cpuTimer.getTimeElapsed() << endl;
}

std::vector<int> doHistogramCPU(int xSize, int ySize, int zSize, int numVars, thrust::host_vector<float> & h_data)
{		
	//Reference: http://stackoverflow.com/questions/1739259/how-to-use-queryperformancecounter
	//Timing code start

	int rows = xSize * ySize * zSize;

	WindowsCpuTimer cpuTimer;
	cpuTimer.startTimer();

	float minValues[] = {0, 0, 0, 0, 0, 0, 7.392e-039, 0, 0, 0};
	float maxValues[] = {1001, 19910, 0.7599, 0.7595, 0.24, 0.2397, 0.1623, 1.1e-007, 3.464e-006, 4.447e-008};
	
	
	#ifdef IS_LOGGING
	cout << "Running histogram CPU Method..." << endl;
	cout << endl;
	#endif

	//Calculate the number of elements belonging in each bin on the CPU using a for loop
	
	int numElements = 1;
	for (int i = 0; i < numVars; i++)
	{
		numElements *= 4; //numBins!
	}


	std::vector<int> finalCounts(numElements);

	for (int i = 0; i < finalCounts.size(); i++)
	{
		finalCounts[i] = 0;
	}


	for (int i = 0; i < rows; i++)
	{
		int factor = 1;
		int sum = 0;
		for (int j = numVars - 1; j >= 0; j--)
		{
			float value = h_data[i * numVars + j];

			float min = minValues[j];
			float max = maxValues[j];

			float percentage = (value - min) / float(max - min);


			int binValue = percentage * 4;

			if (binValue == 4) //numBins!
			{
				binValue--;
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
	//printData(finalCounts.size(), 10, thrust::host_vector<int>(finalCounts.begin(), finalCounts.end()));

	printDataNoZeroes(finalCounts.size(), 10, thrust::host_vector<int>(finalCounts.begin(), finalCounts.end()));


	cout << endl;
	#endif
	


	cout << "CPU time elapsed for CPU method: " << cpuTimer.getTimeElapsed() << endl;

	return finalCounts;

}

