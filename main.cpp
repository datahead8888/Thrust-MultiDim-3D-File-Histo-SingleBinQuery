#include <thrust/device_vector.h>

#include "cuda.h"
#include <time.h>
#include <cstdlib>
#include <vector>
#include <iomanip>



using namespace std;
int main(int argc, char *argv[])
{
	/*
	srand(time(0));
	const int HISTO_SIZE = 20;
	//const int HISTO_SIZE = 90000000;
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
	*/

	thrust::host_vector<int> resultVector1 = doHistogramGPU();

	std::vector<int> resultVector2 = doHistogramCPU();

	
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

	const int WIDTH = 6;
	cout << "Histogram in tabular form (printed from bins from GPU result):" << endl;
	cout << endl;
	cout << "Bin explanations:" << endl;
	cout << "Bin 1: Values 0-63" << endl;
	cout << "Bin 2: Values 64-127" << endl;
	cout << "Bin 3: Values 128-191" << endl;
	cout << "Bin 4: Values 192-255" << endl;
	cout << endl;
	for (int i = 0; i < 4; i++) //i = blue
	{
		cout << "B = " << i + 1 << "..." << endl;

		cout << setw(WIDTH) << "R=>";
		for (int k = 0; k < 4; k++)
		{
			cout << setw(WIDTH) << k + 1;
		}
		cout << endl;

		cout << setw(WIDTH) << "G" << endl;
		cout << setw(WIDTH) << "|" << endl;
		cout << setw(WIDTH) << "V" << endl << endl;

		for (int j = 0; j < 4; j++) //j = green
		{
			cout << setw(WIDTH) << j + 1;
			

			for (int k = 0; k < 4; k++) //k = red
			{
				cout << setw(WIDTH) << resultVector1[i * 16 + j * 4 + k];

			}

			cout << endl << endl;
		}
		cout << endl << endl;
	}
	
	

	return 0;
	
        

}




	

