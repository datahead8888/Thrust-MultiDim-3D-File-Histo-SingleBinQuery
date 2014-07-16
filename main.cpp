#include <thrust/device_vector.h>

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


using namespace std;

//bool loadTextFile(FILE *infile, int xSize, int ySize, int zSize, int numvars, thrust::host_vector<float> & h_data, int bufferSize, int & xPos, int & yPos, int & zPos );

///bool loadTextFile(FILE *infile, int xSize, int ySize, int zSize, int numvars, thrust::host_vector<float> & h_data, int bufferSize, int & xPos, int & yPos, int & zPos )

int main(int argc, char *argv[])
{
	




	srand(time(0));
	
	
	/*
	const int XSIZE = 600;
	const int YSIZE = 248;
	const int ZSIZE = 248;
	*/
	
	const int XSIZE = 8;
	const int YSIZE = 1;
	const int ZSIZE = 1;
	
	const int NUMVARS = 10;
	int rowCount = XSIZE * YSIZE * ZSIZE;
	int bufferSize = 2;

	thrust::host_vector<float> h_buffer(bufferSize * NUMVARS);
	thrust::host_vector<int> h_data;
	thrust::host_vector<int> h_data2;
	 
	FILE *inFile;
	string fileName = "multifield.0001.txt";

	if ( (inFile = fopen(fileName.c_str(), "r")) == NULL) 
	{
		fprintf(stderr,"Could not open %s for reading\n", fileName.c_str());
		return -2;
	}

	int xPos = 0, yPos = 0, zPos = 0;


	while (loadTextFile(inFile, XSIZE, YSIZE, ZSIZE, NUMVARS, h_buffer, bufferSize, xPos, yPos, zPos))
	{
	


		#ifdef IS_LOGGING
		cout << "Input data:" << endl;
		printData(h_buffer.size() / NUMVARS, NUMVARS, 10, h_buffer);
		#endif


	


		//thrust::host_vector<int> resultVector1 = doHistogramGPU(XSIZE, YSIZE, ZSIZE, NUMVARS, h_buffer);
		doHistogramGPU(XSIZE, YSIZE, ZSIZE, NUMVARS, h_buffer, h_data, h_data2);


		

		//////std::vector<int> resultVector2 = doHistogramCPU(XSIZE, YSIZE, ZSIZE, NUMVARS, b_data);

		

	}

	#ifdef PRINT_RESULT
	cout << "Did initial reductions (in buffered segments)..." << endl;
	cout << "GPU Keys:" << endl;
	for (int i = 0; i < h_data.size(); i++)
	{
		cout << h_data[i] << " ";
	}
	cout << endl;

	
	cout << "GPU Counts:" << endl;
	for (int i = 0; i < h_data2.size(); i++)
	{
		cout << h_data2[i] << " ";
	}
	cout << endl;
	#endif


	thrust::pair<DVI, DVI> endPosition;
	histogramMapReduceGPU(h_data, h_data2, endPosition, NUMVARS);

	#ifdef IS_LOGGING
	cout << "Final multidimensional representation from GPU" << endl;
	printHistoData(h_data.size() / NUMVARS, NUMVARS, 10, h_data, h_data2);
	#endif
	
	
	//////Render a histogram in VTK
	vtkFloatArray * floatArray = vtkFloatArray::New();
	floatArray ->SetNumberOfComponents(3);
	for (int i = 0; i < h_data.size() / NUMVARS; i++)
	{
		floatArray -> InsertNextTuple3(h_data[i * NUMVARS], h_data[i * NUMVARS + 1], h_data[i * NUMVARS + 2]);
	}

	//Reference: http://www.vtk.org/pipermail/vtkusers/2002-June/011682.html
	vtkImageData * imageData = vtkImageData::New();
	imageData -> SetDimensions(h_data.size() / NUMVARS, 1, 1);
	imageData -> GetPointData() -> SetScalars(floatArray);

	vtkChartHistogram2D * chart = vtkChartHistogram2D::New();
	chart ->SetInputData(imageData);
	chart ->SetRenderEmpty(true);

	//vtkRenderer * renderer = vtkRenderer::New();
	//renderer ->AddActor(chart);

	vtkContextView * view = vtkContextView::New();
	view ->GetScene() ->AddItem(chart);

	view ->GetInteractor() -> Initialize();
	view ->GetInteractor() -> Start();

	
	fclose(inFile);


	return 0;
	
        

}






	

