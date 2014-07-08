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
	
	/*
	const int XSIZE = 10;
	const int YSIZE = 1;
	const int ZSIZE = 1;
	*/

	const int NUMVARS = 10;
	int rowCount = XSIZE * YSIZE * ZSIZE;
	int bufferSize = 10;

	thrust::host_vector<float> h_data(bufferSize * NUMVARS);
	 
	FILE *inFile;
	string fileName = "multifield.0001.txt";

	if ( (inFile = fopen(fileName.c_str(), "r")) == NULL) 
	{
		fprintf(stderr,"Could not open %s for reading\n", fileName.c_str());
		return -2;
	}

	int xPos = 0, yPos = 0, zPos = 0;


	while (loadTextFile(inFile, XSIZE, YSIZE, ZSIZE, NUMVARS, h_data, bufferSize, xPos, yPos, zPos))
	{
	


		#ifdef IS_LOGGING
		cout << "Input data:" << endl;
		printData(h_data.size() / NUMVARS, NUMVARS, 10, h_data);
		#endif


	


		thrust::host_vector<int> resultVector1 = doHistogramGPU(XSIZE, YSIZE, ZSIZE, NUMVARS, h_data);

	
		//generateRandomData(ROWS, COLS, MAX, h_data);

		//#ifdef IS_LOGGING
		//cout << "Random data:" << endl;
		//printData(ROWS, COLS, 5, h_data);
		//#endif

		//////std::vector<int> resultVector2 = doHistogramCPU(XSIZE, YSIZE, ZSIZE, NUMVARS, b_data);

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

	}
	
	
	
	fclose(inFile);


	return 0;
	
        

}




	

