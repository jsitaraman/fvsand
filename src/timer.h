#pragma once
#include <chrono> 
#include <stack>
#include <cstdio>
#include <ctime>
#include <string>
#if FVSAND_HAS_GPU
#include "cuda_runtime.h"
#endif

class Timer {

#if FVSAND_HAS_GPU
   cudaEvent_t cuda_t0, cuda_t1;
#else
   std::chrono::time_point<std::chrono::high_resolution_clock> t0;
#endif

   double counter;

public:
   Timer();
   ~Timer(){};
   void tick();
   double tock();
   std::string timestring();
   double elapsed();
};
