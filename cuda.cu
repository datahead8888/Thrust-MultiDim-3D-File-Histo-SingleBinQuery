#include "cuda.h"
#include "cudaTimer.h"

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

thrust::host_vector<int> doHistogramGPU()
{
	Mat image;

	if (!loadImage("colors.jpg", image))
	{
		cerr << "Error in loading image" << endl;
		return thrust::host_vector<int> ();

	}
	
	
	//Based on http://stackoverflow.com/questions/16473621/convert-opencv-matrix-into-vector
	//std::vector<thrust::tuple<int, int, int>> imageMatrix;
	//std::vector<Vec3i> imageMatrix;
	thrust::host_vector<int> h_blue_vector(image.rows * image.cols);
	thrust::host_vector<int> h_green_vector(image.rows * image.cols);
	thrust::host_vector<int> h_red_vector(image.rows * image.cols);

	//Phase 1
	//Separate the bgr data into separate arrays for each of the colors to allow coalesced memory access by thrust
	//Based on: http://stackoverflow.com/questions/7899108/opencv-get-pixel-information-from-mat-image
	for (int y = 0; y < image.rows; y++)
	{
		
		for (int x = 0; x < image.cols; x++)
		{
			//Vec3i pixelEntry = image.at<Vec3i>(y,x);
			Vec3b pixelEntry = image.at<Vec3b>(y,x);
			//thrust::tuple<int, int, int> tuple = thrust::tuple(pixelEntry[0], pixelEntry[1], pixelEntry[2]);
			//auto tuple = thrust::make_tuple(pixelEntry[0], pixelEntry[1], pixelEntry[2]);
			h_blue_vector[y * image.cols + x] = static_cast<int>(pixelEntry[0]);

			//cout << (int)pixelEntry[0] << endl;
			//cout << h_blue_vector[y * image.cols + x] << endl;

			h_green_vector[y * image.cols + x] = static_cast<int>(pixelEntry[1]);
			h_red_vector[y * image.cols + x] = static_cast<int>(pixelEntry[2]);


		}

	}


	/////////////
	//special testing code
	//h_blue_vector.clear();
	//h_green_vector.clear();
	//h_red_vector.clear();
	//////////
	
	
	for (int i = 0; i < 4; i++)
	{
		for (int j = 0; j < 4; j++)
		{
			for (int k = 0; k < 4; k++)
			{
				h_blue_vector.push_back(i * 64 + 1);
				h_green_vector.push_back(j * 64 + 1);
				h_red_vector.push_back(k * 64 + 1);
			}
		}
	}
	

	/////special testing code
	//h_blue_vector.push_back(1);		h_blue_vector.push_back(1);		h_blue_vector.push_back(240);	h_blue_vector.push_back(240);	h_blue_vector.push_back(1);		h_blue_vector.push_back(1);		h_blue_vector.push_back(130);		h_blue_vector.push_back(1);
	//h_green_vector.push_back(100);	h_green_vector.push_back(131);	h_green_vector.push_back(67);	h_green_vector.push_back(67);	h_green_vector.push_back(131);	h_green_vector.push_back(100);	h_green_vector.push_back(244);		h_green_vector.push_back(100);
	//h_red_vector.push_back(66);		h_red_vector.push_back(254);	h_red_vector.push_back(1);		h_red_vector.push_back(1);		h_red_vector.push_back(66);		h_red_vector.push_back(66);		h_red_vector.push_back(50);			h_red_vector.push_back(66);

	#ifdef IS_LOGGING

	cout << "Printing color vectors" << endl;

	cout << "Blue:" << endl;
	
	for (int i = 0; i < h_blue_vector.size(); i++)
	{
		cout << h_blue_vector[i] << " ";
	}
	cout << endl;

	cout << "Green:" << endl;

	for (int i = 0; i < h_green_vector.size(); i++)
	{
		cout << h_green_vector[i] << " ";
	}
	cout << endl;

	cout << "Red:" << endl;

	for (int i = 0; i < h_red_vector.size(); i++)
	{
		cout << h_red_vector[i] << " ";
	}
	cout << endl;



	#endif

	/////////////

	thrust::device_vector<int> d_blue_vector(h_blue_vector.begin(), h_blue_vector.end());
	thrust::device_vector<int> d_green_vector(h_green_vector.begin(), h_green_vector.end());
	thrust::device_vector<int> d_red_vector(h_red_vector.begin(), h_red_vector.end());

	//auto zipFirst = thrust::make_zip_iterator(thrust::make_tuple(d_red_vector.begin(), d_green_vector.begin(), d_blue_vector.begin()));
	//auto zipLast = thrust::make_zip_iterator(thrust::make_tuple(d_red_vector.end(), d_green_vector.end(), d_blue_vector.end()));
	auto zipFirst = thrust::make_zip_iterator(thrust::make_tuple(d_blue_vector.begin(), d_green_vector.begin(), d_red_vector.begin()));
	auto zipLast = thrust::make_zip_iterator(thrust::make_tuple(d_blue_vector.end(), d_green_vector.end(), d_red_vector.end()));
	


	//Phase 2
	
	CudaTimer cudaTimer;
	
	//Reference: http://stackoverflow.com/questions/1739259/how-to-use-queryperformancecounter
	
	//Timing code start
	LARGE_INTEGER freqLi;
	QueryPerformanceFrequency(&freqLi);

	double pcFreq = double(freqLi.QuadPart)/1000.0;
	QueryPerformanceCounter(&freqLi);
	__int64 startTime = freqLi.QuadPart;

	#ifdef IS_LOGGING
	cout << "Running histogram GPU method #2..." << endl;
	cout << endl;
	#endif

	cudaTimer.startTimer();

	
	//Set up device vector
	//thrust::device_vector<int> device_numbers(numbers.begin(), numbers.end());
	////////thrust::device_vector<Vec3f> device_numbers(imageMatrix.begin(), imageMatrix.end());
	//thrust::device_vector<Vec3f> device_numbers;

	
	
	#ifdef IS_LOGGING
	cout << "Running transform:" << endl;
	#endif

	
	//Phase 2: Find the bins for each of the elements

    thrust::transform(zipFirst, zipLast, zipFirst, zipFirst, BinFinder());

	#ifdef IS_LOGGING

	cout << "Printing identified color bins" << endl;

	cout << "Blue:" << endl;
	
	for (int i = 0; i < d_blue_vector.size(); i++)
	{
		cout << d_blue_vector[i] << " ";
	}
	cout << endl;

	cout << "Green:" << endl;

	for (int i = 0; i < d_green_vector.size(); i++)
	{
		cout << d_green_vector[i] << " ";
	}
	cout << endl;

	cout << "Red:" << endl;

	for (int i = 0; i < d_red_vector.size(); i++)
	{
		cout << d_red_vector[i] << " ";
	}
	cout << endl;


	#endif
	

	

	
	//Step 2: Sort those bin ids
	thrust::sort(zipFirst, zipLast, ZipComparator());

	#ifdef IS_LOGGING
	
	cout << "Printing sorted color bins" << endl;

	cout << "Blue:" << endl;
	
	for (int i = 0; i < d_blue_vector.size(); i++)
	{
		cout << d_blue_vector[i] << " ";
	}
	cout << endl;

	cout << "Green:" << endl;

	for (int i = 0; i < d_green_vector.size(); i++)
	{
		cout << d_green_vector[i] << " ";
	}
	cout << endl;

	cout << "Red:" << endl;

	for (int i = 0; i < d_red_vector.size(); i++)
	{
		cout << d_red_vector[i] << " ";
	}
	cout << endl;


	#endif


	

	//Step 3: Use the reduce by key function to get a count of each bin type
	thrust::constant_iterator<int> cit(1);
	thrust::device_vector<int> counts(64);  //4 ^ 3

	thrust::reduce_by_key(zipFirst, zipLast, cit, zipFirst, counts.begin());

	#ifdef IS_LOGGING
	cout << "Printing counts (each one high)" << endl;
	for (int i = 0; i < counts.size(); i++)
	{
		cout << counts[i] << " ";
	}

	cout << endl;
	cout << endl;
	#endif
	


	thrust::constant_iterator<int> one(1);
	thrust::transform(counts.begin(), counts.end(), one, counts.begin(), thrust::minus<int>());

	thrust::host_vector<int> finalCounts(counts.begin(), counts.begin() + 64);  //device_numbers will have extra junk elements that we don't want any more

	#ifdef IS_LOGGING
	cout << "Printing final counts" << endl;
	for (int i = 0; i < finalCounts.size(); i++)
	{
		cout << finalCounts[i] << " ";
	}

	cout << endl;
	cout << endl;
	#endif
	

	cudaTimer.stopTimer();

	cout << "GPU time elapsed for GPU method #2: " << cudaTimer.getTimeElapsed() << endl;
	
	//Timing code end
	QueryPerformanceCounter(&freqLi);
	double timePassed = double(freqLi.QuadPart-startTime) / pcFreq;

	cout << "CPU time elapsed for GPU method #2: " << timePassed << endl;
	

	return finalCounts;
	//return h_blue_vector;

}

std::vector<int> doHistogramCPU()
{
	Mat image;

	if (!loadImage("colors.jpg", image))
	{
		cerr << "Error in loading image" << endl;
		return std::vector<int> ();
	}

	
	//Based on http://stackoverflow.com/questions/16473621/convert-opencv-matrix-into-vector
	//std::vector<thrust::tuple<int, int, int>> imageMatrix;
	//std::vector<Vec3i> imageMatrix;
	thrust::host_vector<int> h_blue_vector(image.rows * image.cols);
	thrust::host_vector<int> h_green_vector(image.rows * image.cols);
	thrust::host_vector<int> h_red_vector(image.rows * image.cols);

	//Phase 1
	//Separate the bgr data into separate arrays for each of the colors to allow coalesced memory access by thrust
	//Based on: http://stackoverflow.com/questions/7899108/opencv-get-pixel-information-from-mat-image
	for (int y = 0; y < image.rows; y++)
	{
		
		for (int x = 0; x < image.cols; x++)
		{
			//Vec3i pixelEntry = image.at<Vec3i>(y,x);
			Vec3b pixelEntry = image.at<Vec3b>(y,x);
			//thrust::tuple<int, int, int> tuple = thrust::tuple(pixelEntry[0], pixelEntry[1], pixelEntry[2]);
			//auto tuple = thrust::make_tuple(pixelEntry[0], pixelEntry[1], pixelEntry[2]);
			h_blue_vector[y * image.cols + x] = static_cast<int>(pixelEntry[0]);

			//cout << (int)pixelEntry[0] << endl;
			//cout << h_blue_vector[y * image.cols + x] << endl;

			h_green_vector[y * image.cols + x] = static_cast<int>(pixelEntry[1]);
			h_red_vector[y * image.cols + x] = static_cast<int>(pixelEntry[2]);


		}

	}

	//Reference: http://stackoverflow.com/questions/1739259/how-to-use-queryperformancecounter

	//Timing code start
	LARGE_INTEGER freqLi;
	QueryPerformanceFrequency(&freqLi);

	double pcFreq = double(freqLi.QuadPart)/1000.0;
	QueryPerformanceCounter(&freqLi);
	__int64 startTime = freqLi.QuadPart;

	#ifdef IS_LOGGING
	cout << "Running histogram CPU Method..." << endl;
	cout << endl;
	#endif

	//Calculate the number of elements belonging in each bin on the CPU using a for loop
	
	std::vector<int> finalCounts(64);
	for (int i = 0; i < finalCounts.size(); i++)
	{
		finalCounts[i] = 0;
	}

	for (int i = 0; i < h_blue_vector.size(); i++)
	{
		int blueBin = 0, greenBin = 0, redBin = 0;
		if (h_blue_vector[i] >= 0 && h_blue_vector[i] <= 63)
		{
			blueBin = 0;
		}
		else if (h_blue_vector[i] >= 64 && h_blue_vector[i] <= 127)
		{
			blueBin = 1;
		}
		else if (h_blue_vector[i] >= 128 && h_blue_vector[i] <= 191)
		{
			blueBin = 2;
		}
		else
		{
			blueBin = 3;
		}

		
		if (h_green_vector[i] >= 0 && h_green_vector[i] <= 63)
		{
			greenBin = 0;
		}
		else if (h_green_vector[i] >= 64 && h_green_vector[i] <= 127)
		{
			greenBin = 1;
		}
		else if (h_green_vector[i] >= 128 && h_green_vector[i] <= 191)
		{
			greenBin = 2;
		}
		else
		{
			greenBin = 3;
		}


		if (h_red_vector[i] >= 0 && h_red_vector[i] <= 63)
		{
			redBin = 0;
		}
		else if (h_red_vector[i] >= 64 && h_red_vector[i] <= 127)
		{
			redBin = 1;
		}
		else if (h_red_vector[i] >= 128 && h_red_vector[i] <= 191)
		{
			redBin = 2;
		}
		else
		{
			redBin = 3;
		}

		finalCounts[blueBin * 16 + greenBin * 4 + redBin]++;
	}

	

	//Timing code end
	QueryPerformanceCounter(&freqLi);
	double timePassed = double(freqLi.QuadPart-startTime) / pcFreq;

	cout << "CPU time elapsed for CPU method: " << timePassed << endl;

	return finalCounts;

}

