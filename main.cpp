#include <thrust/device_vector.h>

#include "cudaTimer.h"
#include "windowsCpuTimer.h"
#include "cuda.h"
#include <time.h>
#include <cstdlib>
#include <vector>
#include <iomanip>
#include <iostream>
#include <fstream>

//#include <vtkExecutive.h>
#include <vtkStructuredPointsReader.h>
#include <vtkSmartPointer.h>
#include <vtkStreamingDemandDrivenPipeline.h>
#include <vtkStructuredPoints.h>


#include <vtkImageDataGeometryFilter.h>
#include <vtkPolyDataMapper.h>
#include <vtkActor.h>
#include <vtkRenderWindow.h>
#include <vtkRenderWindowInteractor.h>
#include <vtkRenderer.h>
#include <vtkFloatArray.h>
#include <vtkImageData.h>
#include <vtkPointData.h>
#include <vtkChartHistogram2D.h>
//#include <vtkRenderer.h>
//#include <vtkRenderWindow.h>
//#include <vtkRenderWindowInteractor.h>
#include <vtkContextView.h>
#include <vtkContextScene.h>
#include <vtkColorTransferFunction.h>
#include <vtkAxis.h>
#include <vtkInformation.h>


using namespace std;

//bool loadTextFile(FILE *infile, int xSize, int ySize, int zSize, int numvars, thrust::host_vector<float> & h_data, int bufferSize, int & xPos, int & yPos, int & zPos );

///bool loadTextFile(FILE *infile, int xSize, int ySize, int zSize, int numvars, thrust::host_vector<float> & h_data, int bufferSize, int & xPos, int & yPos, int & zPos )

int main(int argc, char *argv[])
{
	

	const int NUM_BINS = 10;
	
	
	
	const int XSIZE = 600;
	const int YSIZE = 248;
	const int ZSIZE = 248;
	
	//const int XSIZE = 10;
	//const int XSIZE = 4;
	//const int XSIZE = 600 * 248 * 248;
	//const int YSIZE = 1;
	//const int ZSIZE = 1;

	const int NUMVARS = 10;
	//const int NUMVARS = 10;
	int rowCount = XSIZE * YSIZE * ZSIZE;
	int bufferSize = rowCount / 10;
	//int bufferSize = 5;
	int var1 = 1;  //1st column (var) on which to build a VTK histogram
	//int var2 = 5;  //2nd column (var) on which to build a VTK histogram
	int var2 = 2;
	string fileName = "multifield.0001.txt";

	//printMinMaxes(fileName, XSIZE * YSIZE * ZSIZE, 10);


	thrust::host_vector<float> h_buffer(bufferSize * NUMVARS);
	thrust::host_vector<long long> h_data;
	thrust::host_vector<long long> h_data2;
	 
	FILE *inFile;

	if ( (inFile = fopen(fileName.c_str(), "r")) == NULL) 
	{
		fprintf(stderr,"Could not open %s for reading\n", fileName.c_str());
		return -2;
	}

	int xPos = 0, yPos = 0, zPos = 0;

	CudaTimer cudaTimer;
	WindowsCpuTimer cpuTimer;

	//for (int i = 0; i < 9; i++)
	//{
	//	loadTextFile(inFile, XSIZE, YSIZE, ZSIZE, NUMVARS, 10, h_buffer, bufferSize, xPos, yPos, zPos);
	//}

	while (loadTextFile(inFile, XSIZE * YSIZE * ZSIZE, 1, 1, NUMVARS, 10, h_buffer, bufferSize, xPos, yPos, zPos))
	{
		#ifdef IS_LOGGING
		cout << "Input data:" << endl;
		printData(h_buffer.size() / NUMVARS, NUMVARS, 10, h_buffer);
		#endif

		//thrust::host_vector<int> resultVector1 = doHistogramGPU(XSIZE, YSIZE, ZSIZE, NUMVARS, h_buffer);
		doHistogramGPU(XSIZE, YSIZE, ZSIZE, NUMVARS, h_buffer, h_data, h_data2, NUM_BINS, cudaTimer, cpuTimer);
	}

	#ifdef IS_LOGGING
	cout << "Did initial reductions (in buffered segments)..." << endl;
	cout << "GPU Keys:" << endl;
	for (int i = 0; i < h_data.size(); i++)
	{
		cout << h_data[i] << " ";
	}
	cout << endl;

	
	//cout << "GPU Counts:" << endl;
	//for (int i = 0; i < h_data2.size(); i++)
	//{
	//	cout << h_data2[i] << " ";
	//}
	//cout << endl;
	#endif


	//Do the query
	//At this point, we have a single dimensional representation with 1 bin per element, allowing the query with the current specifications
	
	//TEST CODE!!!!!!!!!!!!!!
	//for (int i = 0; i < 64; i++)
	//{
	//	h_data[i] = i + 1;
	//}


	doQuery(XSIZE, YSIZE, ZSIZE, 0, 3, 0, 3, 0, 3, h_data, h_data2);


	thrust::pair<DVL, DVL> endPosition;
	histogramMapReduceGPU(h_data, h_data2, endPosition, NUMVARS, NUM_BINS, cudaTimer, cpuTimer);

	int numRows = h_data.size() / NUMVARS;

	#ifdef PRINT_RESULT
	cout << "Final multidimensional representation from GPU" << endl;
	printHistoData(numRows, NUMVARS, 10, h_data, h_data2);
	#endif
	
	fclose(inFile);

	//////CPU Baseline/////////////////////////////////////////
	//NOTE: Only works for lower number of vars and smaller file record counts - will run out of memory otherwise!
	///////////////////////////////////////////////////////////
	#ifdef DO_CPU_COMPUTATION

	FILE *cpuFile;
	string cpuFileName = "multifield.0001.txt";
	xPos = yPos = zPos = 0;

	if ( (cpuFile = fopen(cpuFileName.c_str(), "r")) == NULL) 
	{
		fprintf(stderr,"Could not open %s for reading\n", cpuFileName.c_str());
		return -2;
	}

	h_buffer.resize(XSIZE * YSIZE * ZSIZE * NUMVARS);
	loadTextFile(cpuFile, XSIZE, YSIZE, ZSIZE, NUMVARS, 10, h_buffer, XSIZE * YSIZE * ZSIZE, xPos, yPos, zPos);
	doHistogramCPU(XSIZE, YSIZE, ZSIZE, NUMVARS, NUM_BINS, h_buffer);

	fclose(cpuFile);

	#endif

	
	//////Render a histogram in VTK///////////////////////////////
	int histoSize = NUM_BINS;

	//Reference: http://www.vtk.org/pipermail/vtkusers/2002-June/011682.html
	vtkImageData * imageData = vtkImageData::New();
	imageData -> SetExtent(0, histoSize - 1, 0, histoSize - 1, 0, 0);
	//imageData -> GetPointData() -> SetScalars(floatArray);
	vtkInformation * vtkInfo = vtkInformation::New();
	imageData ->AllocateScalars(vtkInfo);

	double * renderedHistogram = (double*) imageData -> GetScalarPointer(0, 0, 0);



	//std::vector<int> renderedHistogram(histoSize * histoSize);
	for (int i = 0; i < histoSize * histoSize; i++)
	{
		renderedHistogram[i] = 0;
	}

	double * dataPtr = (double *) imageData ->GetScalarPointer(0,0,0);
	

	for (int i = 0; i < numRows; i++)
	{
		//h_data - CPU calculated histogram - access element # i * NUMVARS + j
		//h_data2 - CPU calculated count for a row - access element # i
		//renderedHistogram - what we're building to give to VTK - access which element: row # value in col 0; col# value in col 1 - for this row
		renderedHistogram[h_data[i * NUMVARS + var1] * histoSize + h_data[i * NUMVARS + var2]] += h_data2[i];
		
	}

	#ifdef PRINT_RESULT
	cout << "'Full' histogram for VTK to render (flipped to correspond to VTK):" << endl;
	for (int i = histoSize - 1; i >= 0; i--)
	{
		for (int j = 0; j < histoSize; j++)
		{
			cout << setw(10) << renderedHistogram[i * histoSize + j] << " ";
		}
		cout << endl;
	}
	#endif

	
	vtkChartHistogram2D * chart = vtkChartHistogram2D::New();
	chart ->SetInputData(imageData);
	//chart ->SetRenderEmpty(true);

	//chart ->SetAutoAxes(false);
	//chart ->GetAxis(vtkAxis::BOTTOM) -> SetRange(0, 4);
	//chart ->GetAxis(vtkAxis::BOTTOM) -> SetTitle("Bottom");
	//chart ->GetAxis(vtkAxis::LEFT) -> SetRange(1, 10);
	//chart ->GetAxis(vtkAxis::LEFT) -> SetTitle("Left");
	//chart ->GetAxis(vtkAxis::FIXED) -> SetRange(1, 10);


	
	
	//Based on: https://github.com/qsnake/vtk/blob/master/Charts/Testing/Cxx/TestHistogram2D.cxx
	vtkColorTransferFunction * colorFunction = vtkColorTransferFunction::New();

	colorFunction -> AddRGBSegment(0.0f, 0.0, 1.0, 0.0, 824.0, 0.0f, 0.0f, 1.0f);

	colorFunction -> Build();

	chart ->SetTransferFunction(colorFunction);
	
	vtkContextView * view = vtkContextView::New();
	view ->GetScene() ->AddItem(chart);


	view ->GetInteractor() -> Initialize();
	view ->GetInteractor() -> Start();
	
	
	


	return 0;
	
        

}






	

