#pragma once

#include <Windows.h>

class WindowsCpuTimer
{
private:
	float timeElapsed;
	LARGE_INTEGER freqLi;
	double pcFreq;
	__int64 startTime;
	
public:
	WindowsCpuTimer();
	void reinit();
	void startTimer();
	void stopTimer();
	int getTimeElapsed();
};