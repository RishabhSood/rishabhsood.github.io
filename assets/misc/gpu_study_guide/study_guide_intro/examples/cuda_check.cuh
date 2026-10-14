#pragma once

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

// Ordinary C++ function: receive one CUDA status and stop on failure.
inline void cudaCheckImplementation(cudaError_t status,
                                    const char* operation,
                                    const char* file,
                                    int line) {
    if (status != cudaSuccess) {
        std::fprintf(stderr, "CUDA error from %s at %s:%d: %s\n",
                     operation, file, line, cudaGetErrorString(status));
        std::exit(EXIT_FAILURE);
    }
}

// Small preprocessor wrapper that automatically passes:
//   #call     -> the CUDA call converted to readable text
//   __FILE__  -> this source filename
//   __LINE__  -> this source line number
// The CUDA expression is evaluated exactly once, inside the function call.
#define CUDA_CHECK(call) \
    cudaCheckImplementation((call), #call, __FILE__, __LINE__)

// A kernel launch is asynchronous. Check cudaGetLastError() immediately after
// the launch, then synchronize (while learning/debugging) to reveal errors that
// happened while GPU instructions were executing.
