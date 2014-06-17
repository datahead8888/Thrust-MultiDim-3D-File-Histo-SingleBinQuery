#include "cudaTimer.h"

CudaTimer::CudaTimer()
{
	reinit();
}

void CudaTimer::reinit()
{
	timeElapsed = 0.0f;
	
}

void CudaTimer::startTimer()
{
	(cudaEventCreate(&startEvent));
	(cudaEventCreate(&stopEvent));
	cudaEventRecord(startEvent, 0);
	
}

void CudaTimer::stopTimer()
{
	float stopWatchTime = 0.0f;
	cudaEventRecord(stopEvent, 0);
	cudaEventSynchronize(stopEvent);
	cudaEventElapsedTime(&stopWatchTime, startEvent, stopEvent);

	timeElapsed += stopWatchTime;
}

int CudaTimer::getTimeElapsed()
{
	return timeElapsed;
}
