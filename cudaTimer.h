#pragma once

class CudaTimer
{
private:
	float timeElapsed;
	cudaEvent_t startEvent;
	cudaEvent_t stopEvent;
public:
	CudaTimer();
	void reinit();
	void startTimer();
	void stopTimer();
	int getTimeElapsed();
};