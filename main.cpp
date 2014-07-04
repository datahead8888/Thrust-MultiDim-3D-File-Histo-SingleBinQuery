#include <thrust/device_vector.h>

#include "cuda.h"
#include <time.h>
#include <cstdlib>
#include <vector>
#include <iomanip>
#include <iostream>
#include <fstream>



using namespace std;
int main(int argc, char *argv[])
{
	
	srand(time(0));
	
	//const int ROWS = 20;
	//const int COLS = 2;
	
	const int XSIZE = 600;
	const int YSIZE = 248;
	const int ZSIZE = 248;

	const int NUMVARS = 10;
	//const int STEP = 4;
	const int STEP = 30000000 / 8;  //Cut the data size WAY down while still using the input file

	thrust::host_vector<float> h_data(XSIZE * YSIZE * ZSIZE * NUMVARS / STEP);

	cout << h_data.size() << endl;
	 
	cout << endl;
	//generateRandomData(ROWS, COLS, MAX, h_data);

	

	//thrust::host_vector<int> b_data = h_data;

	try
	{
		loadTextFile("multifield.0001.txt", XSIZE, YSIZE, ZSIZE, NUMVARS, STEP, h_data);
	}
	catch (ifstream::failure e)
	{
		cerr << "Problems in loading 'multifield.0001.txt'" << endl;
		cerr << e.what() << endl;
		exit(1);
	}

	#ifdef IS_LOGGING
	cout << "Input data:" << endl;
	printData(XSIZE * YSIZE * ZSIZE / STEP, NUMVARS, 10, h_data);
	#endif


	


	thrust::host_vector<int> resultVector1 = doHistogramGPU(XSIZE, YSIZE, ZSIZE, NUMVARS, STEP, h_data);

	/*
	//generateRandomData(ROWS, COLS, MAX, h_data);

	//#ifdef IS_LOGGING
	//cout << "Random data:" << endl;
	//printData(ROWS, COLS, 5, h_data);
	//#endif

	std::vector<int> resultVector2 = doHistogramCPU(ROWS, COLS, b_data);

	#ifdef PRINT_RESULT
	cout << "Histogram from GPU:" << endl;
	for (int i = 0; i < resultVector1.size(); i++)
	{
		cout << resultVector1[i] << " ";
	}
	cout << endl;

	
	cout << "Histogram from CPU:" << endl;
	for (int i = 0; i < resultVector2.size(); i++)
	{
		cout << resultVector2[i] << " ";
	}
	cout << endl;
	#endif
	*/
	
	
	

	return 0;
	
        

}




	

