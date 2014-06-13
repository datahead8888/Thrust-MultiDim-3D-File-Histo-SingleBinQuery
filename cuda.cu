#include "cuda.h"

#include <iostream>

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

#include <Windows.h>

using namespace std;






thrust::host_vector<int> doHistogramGPU(std::vector<int> numbers)
{
	//Reference: http://stackoverflow.com/questions/1739259/how-to-use-queryperformancecounter
	LARGE_INTEGER freqLi;
	QueryPerformanceFrequency(&freqLi);

	double pcFreq = double(freqLi.QuadPart)/1000.0;
	QueryPerformanceCounter(&freqLi);
	__int64 startTime = freqLi.QuadPart;


	thrust::device_vector<int> device_numbers(numbers.begin(), numbers.end());
	//thrust::device_vector<int> device_output(numbers.size());

	#ifdef IS_LOGGING
	cout << "Running transform:" << endl;
	#endif


    thrust::transform(device_numbers.begin(), device_numbers.end(), device_numbers.begin(), device_numbers.begin(), BinFinder());
	
	#ifdef IS_LOGGING
	cout << endl;
	cout << "Printing bins for input elements" << endl;
	for (int i = 0; i < device_numbers.size(); i++)
	{
		cout << device_numbers[i] << " ";
	}
	cout << endl;
	#endif

	thrust::sort(device_numbers.begin(), device_numbers.end());

	#ifdef IS_LOGGING
	cout << "Printing sorted bins for input elements" << endl;
	for (int i = 0; i < device_numbers.size(); i++)
	{
		cout << device_numbers[i] << " ";
	}
	cout << endl;
	#endif

	//thrust::device_vector<int> differences(numbers.size());
	thrust::adjacent_difference(device_numbers.begin(), device_numbers.end(), device_numbers.begin());

	#ifdef IS_LOGGING
	cout << "Printing adjacent differences" << endl;
	for (int i = 0; i < device_numbers.size(); i++)
	{
		cout << device_numbers[i] << " ";
	}
	cout << endl;
	#endif

	//thrust::device_vector<int> modifiedIndexes(numbers.size());
	thrust::counting_iterator<int>  indexes(0);
	
	thrust::transform(device_numbers.begin(), device_numbers.end(), indexes, device_numbers.begin(), IndexFinder());

	#ifdef IS_LOGGING
	cout << "Printing special indexes" << endl;
	for (int i = 0; i < device_numbers.size(); i++)
	{
		cout << device_numbers[i] << " ";
	}
	cout << endl;
	#endif

	//Reference: http://stackoverflow.com/questions/12269773/type-of-return-value-of-thrustremove-if
	typedef thrust::device_vector<int>::iterator Dvi;
	typedef thrust::tuple<Dvi, Dvi> DviTuple;
	typedef thrust::zip_iterator<DviTuple> ZipDviTuple;


	ZipDviTuple endZipped = thrust::remove(device_numbers.begin(), device_numbers.end(), -1);
	DviTuple endTuple = endZipped.get_iterator_tuple();
	Dvi end = thrust::get<0>(endTuple);

	
	#ifdef IS_LOGGING
	cout << "Printing special indexes after removals" << endl;
	for (Dvi it = device_numbers.begin(); it != end; it++)
	{
		int value = *it;
		cout << value << " ";
	}
	cout << endl;
	#endif

	thrust::device_vector<int> modifiedIndexes2(device_numbers.begin(), end);

	modifiedIndexes2.push_back(device_numbers.size());
	
	#ifdef IS_LOGGING
	cout << "Printing special indexes after size insertion" << endl;
	for (int i = 0; i < modifiedIndexes2.size(); i++)
	{
		cout << modifiedIndexes2[i] << " ";
	}
	cout << endl;
	#endif
	

	//thrust::device_vector<int> counts(modifiedIndexes2.size());
	thrust::adjacent_difference(modifiedIndexes2.begin(), modifiedIndexes2.end(), modifiedIndexes2.begin());

	#ifdef IS_LOGGING
	cout << "Printing (semi) final counts:" << endl;
	for (int i = 0; i < modifiedIndexes2.size(); i++)
	{
		cout << modifiedIndexes2[i] << " ";
	}
	cout << endl;
	#endif

	thrust::host_vector<int> finalCounts(modifiedIndexes2.begin() + 1, modifiedIndexes2.end());
	
	#ifdef IS_LOGGING
	cout << "Printing (semi) final counts:" << endl;
	for (int i = 0; i < finalCounts.size(); i++)
	{
		cout << finalCounts[i] << " ";
	}
	cout << endl;

	cout << endl;
	#endif

	QueryPerformanceCounter(&freqLi);
	double timePassed = double(freqLi.QuadPart-startTime) / pcFreq;

	cout << "CPU time elapsed for GPU method #1: " << timePassed << endl;

	return finalCounts;
	
}

thrust::host_vector<int> doHistogramGPUB(std::vector<int> numbers)
{
	//Reference: http://stackoverflow.com/questions/1739259/how-to-use-queryperformancecounter
	LARGE_INTEGER freqLi;
	QueryPerformanceFrequency(&freqLi);

	double pcFreq = double(freqLi.QuadPart)/1000.0;
	QueryPerformanceCounter(&freqLi);
	__int64 startTime = freqLi.QuadPart;


	thrust::device_vector<int> device_numbers(numbers.begin(), numbers.end());
	thrust::device_vector<int> device_output(numbers.size());

	#ifdef IS_LOGGING
	cout << "Running transform:" << endl;
	#endif


    thrust::transform(device_numbers.begin(), device_numbers.end(), device_numbers.begin(), device_numbers.begin(), BinFinder());

	#ifdef IS_LOGGING
	cout << endl;
	cout << "Printing bins for input elements" << endl;
	for (int i = 0; i < device_output.size(); i++)
	{
		cout << device_output[i] << " ";
	}
	cout << endl;
	#endif

	thrust::sort(device_numbers.begin(), device_numbers.end());

	#ifdef IS_LOGGING
	cout << "Printing sorted bins for input elements" << endl;
	for (int i = 0; i < device_output.size(); i++)
	{
		cout << device_output[i] << " ";
	}
	cout << endl;
	#endif

	
	thrust::constant_iterator<int> cit(1);
	thrust::device_vector<int> newKeys(4);
	thrust::device_vector<int> newValues(4);
	thrust::reduce_by_key(device_numbers.begin(), device_numbers.end(), cit, newKeys.begin(), device_numbers.begin());

	#ifdef IS_LOGGING
	cout << "Printing (semi?) final bins" << endl;
	for (int i = 0; i < device_numbers.size(); i++)
	{
		cout << device_numbers[i] << " ";
	}
	cout << endl;
	#endif

	QueryPerformanceCounter(&freqLi);
	double timePassed = double(freqLi.QuadPart-startTime) / pcFreq;

	cout << "CPU time elapsed for GPU method #2: " << timePassed << endl;
	

	return newValues;

}

std::vector<int> doHistogramCPU(std::vector<int> numbers)
{
	//Reference: http://stackoverflow.com/questions/1739259/how-to-use-queryperformancecounter
	LARGE_INTEGER freqLi;
	QueryPerformanceFrequency(&freqLi);

	double pcFreq = double(freqLi.QuadPart)/1000.0;
	QueryPerformanceCounter(&freqLi);
	__int64 startTime = freqLi.QuadPart;


	std::vector<int> workingVector(numbers.begin(), numbers.end());
	
	std::vector<int> finalCounts(4);
	finalCounts[0] = finalCounts[1] = finalCounts[2] = finalCounts[3] = 0;

	for (int i = 0; i < workingVector.size(); i++)
	{
		if (workingVector[i] >= 1 && workingVector[i] <= 5)
		{
			finalCounts[0]++;
		}
		else if (workingVector[i] >= 6 && workingVector[i] <= 10)
		{
			finalCounts[1]++;
		}
		else if (workingVector[i] >= 11 && workingVector[i] <= 15)
		{
			finalCounts[2]++;
		}
		else
		{
			finalCounts[3]++;
		}
	}

	QueryPerformanceCounter(&freqLi);
	double timePassed = double(freqLi.QuadPart-startTime) / pcFreq;

	cout << "CPU time elapsed for CPU method: " << timePassed << endl;

	return finalCounts;

}

