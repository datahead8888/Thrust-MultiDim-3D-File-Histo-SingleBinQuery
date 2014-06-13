#include <thrust/device_vector.h>

#include "cuda.h"
#include <time.h>
#include <cstdlib>
#include <vector>



using namespace std;
int main(int argc, char *argv[])
{
	srand(time(0));
	//const int HISTO_SIZE = 10000;
	const int HISTO_SIZE = 90000000;
	const int MAX = 20;

	vector<int> numbers(HISTO_SIZE);

	for (int i = 0; i < HISTO_SIZE; i++)
	{
		numbers[i] = rand() % MAX + 1;
	
	}

	#ifdef IS_LOGGING
	cout << "These are the numbers to form a histogram from:" << endl;
	for (int i = 0; i < HISTO_SIZE; i++)
	{
		cout << numbers[i] << " ";
	}
	cout << endl;
	#endif


	thrust::host_vector<int> resultVector1 = doHistogramGPU(numbers);

	std::vector<int> resultVector2 = doHistogramCPU(numbers);

	thrust::host_vector<int> resultVector3 = doHistogramGPUB(numbers);

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
	

	cout << "Histogram from GPU:" << endl;
	for (int i = 0; i < resultVector3.size(); i++)
	{
		cout << resultVector3[i] << " ";
	}
	cout << endl;
	

	return 0;
	
        

}




	

