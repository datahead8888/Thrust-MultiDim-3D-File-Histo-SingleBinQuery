#include <thrust/device_vector.h>

#include "cuda.h"
#include <time.h>
#include <cstdlib>
#include <vector>
#include <iomanip>

//Civil war data from http://www.stat.ufl.edu/~winner/data/civwar2.dat

using namespace std;
int main(int argc, char *argv[])
{
	
	srand(time(0));
	
	const int ROWS = 20;
	const int COLS = 2;
	const int MAX = 20;


	thrust::host_vector<int> resultVector1 = doHistogramGPU(ROWS, COLS, MAX);



	std::vector<int> resultVector2 = doHistogramCPU(ROWS, COLS, MAX);

	
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

	
	
	

	return 0;
	
        

}




	

