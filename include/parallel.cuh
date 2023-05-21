#ifndef PARALLEL_H
#define PARALLEL_H

#include "vector"
#include "cuda.h"
#include "cuda_runtime.h"

namespace parallel {
    template <typename T, typename O, typename F>
    __global__ void map_kernel(T* input, O* output, int size, F func);

    template <typename T, typename O, typename F>
    __global__ void reduce_kernel(T* input, O* output, int size, F func);

    template <typename T1, typename T2, typename O, typename F>
    __global__ void zip_kernel(T1* input1, T2* input2, O* output, int size, F func);

    template<typename T, typename F>
    void map(std::vector <T>& input, std::vector <T>& output, F func, int numThreads);

    template<typename T, typename F>
    void reduce(std::vector <T>& input, T& output, F func, int numThreads);

    template<typename T1, typename T2, typename T3, typename F>
    void zip(std::vector <T1>& input1, std::vector <T2>& input2, std::vector <T3>& output, F func, int numThreads);

    #include "../src/parallel.cu"
}

#endif