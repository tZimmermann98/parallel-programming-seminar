#include "vector"
#include "algorithm"
#include "cuda.h"
#include "cuda_runtime.h"
#include <cuda_device_runtime_api.h>
#include <iostream>
#include <functional>
#include <chrono>

struct Add {
    __device__ int operator()(int a, int b) const { 
        return a + b; 
    }
};

template <unsigned int blockSize, typename F>
__device__ void warpReduce(volatile int *sdata, unsigned int tid, F func) {
    if (blockSize >= 64) sdata[tid] = func(sdata[tid], sdata[tid + 32]);
    if (blockSize >= 32) sdata[tid] = func(sdata[tid], sdata[tid + 16]);
    if (blockSize >= 16) sdata[tid] = func(sdata[tid], sdata[tid + 8]);
    if (blockSize >= 8) sdata[tid] = func(sdata[tid], sdata[tid + 4]);
    if (blockSize >= 4) sdata[tid] = func(sdata[tid], sdata[tid + 2]);
    if (blockSize >= 2) sdata[tid] = func(sdata[tid], sdata[tid + 1]);
}

template <typename T, typename F, unsigned int blockSize>
__global__ void reduce_kernel(T *input, T *output, int n, F func){
    extern __shared__ T sdata[];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * (blockSize * 2) + tid;
    unsigned int gridSize = blockSize * 2 * gridDim.x;
    sdata[tid] = 0;

    while (i < n) {
        if (i + blockSize < n) {
            sdata[tid] = func(input[i], input[i + blockSize]);
        } else {
            sdata[tid] = func(input[i], 0);
        }
        i += gridSize;
    }
    __syncthreads();

    if (blockSize >= 1024) {
        if (tid < 512) {
            sdata[tid] = func(sdata[tid], sdata[tid + 512]);
        }
        __syncthreads();
    }
    if (blockSize >= 512) {
        if (tid < 256) {
            sdata[tid] = func(sdata[tid], sdata[tid + 256]);
        }
        __syncthreads();
    }
    if (blockSize >= 256) {
        if (tid < 128) {
            sdata[tid] = func(sdata[tid], sdata[tid + 128]);
        }
        __syncthreads();
    }
    if (blockSize >= 128) {
        if (tid < 64) {
            sdata[tid] = func(sdata[tid], sdata[tid + 64]);
        }
        __syncthreads();
    }

    if (tid < 32) warpReduce<blockSize>(sdata, tid, func);

    if (tid == 0){
        output[blockIdx.x] = sdata[0];
    }
    
}


template<typename T, typename F>
void reduce(std::vector <T>& input, T& output, F func, int numThreads, float& reduce_copy_device, float& reduce_kernel_time, float& reduce_copy_host, float& reduce_total){
    cudaEvent_t start_all, stop_all;
    cudaEventCreate(&start_all);
    cudaEventCreate(&stop_all);
    cudaEventRecord(start_all);

    unsigned int size = input.size();

    T* d_input;
    T* d_output;
    T* d_final_output;

    cudaMalloc(&d_input, size * sizeof(T));
    cudaMalloc(&d_output, sizeof(T) * ((size + 1023) / 1024));
    cudaMalloc(&d_final_output, sizeof(T));

    cudaEvent_t start_copy_device, stop_copy_device;
    cudaEventCreate(&start_copy_device);
    cudaEventCreate(&stop_copy_device);
    cudaEventRecord(start_copy_device);

    cudaMemcpy(d_input, input.data(), size * sizeof(T), cudaMemcpyHostToDevice);

    cudaEventRecord(stop_copy_device);
    cudaEventSynchronize(stop_copy_device);
    cudaEventElapsedTime(&reduce_copy_device, start_copy_device, stop_copy_device);
    cudaEventDestroy(start_copy_device);
    cudaEventDestroy(stop_copy_device);

    dim3 dimBlock(numThreads);
    dim3 dimGrid((size + dimBlock.x - 1) / dimBlock.x);

    cudaEvent_t start_kernel, stop_kernel;
    cudaEventCreate(&start_kernel);
    cudaEventCreate(&stop_kernel);
    cudaEventRecord(start_kernel);

    switch (numThreads) {
        case 1024:
            reduce_kernel<T, F, 1024><<<dimGrid, dimBlock, dimBlock.x * sizeof(T)>>>(d_input, d_output, size, func);
            break;
        case 512:
            reduce_kernel<T, F, 512><<<dimGrid, dimBlock, dimBlock.x * sizeof(T)>>>(d_input, d_output, size, func);
            break;
        case 256:
            reduce_kernel<T, F, 256><<<dimGrid, dimBlock, dimBlock.x * sizeof(T)>>>(d_input, d_output, size, func);
            break;
        case 128:
            reduce_kernel<T, F, 128><<<dimGrid, dimBlock, dimBlock.x * sizeof(T)>>>(d_input, d_output, size, func);
            break;
        case 64:
            reduce_kernel<T, F, 64><<<dimGrid, dimBlock, dimBlock.x * sizeof(T)>>>(d_input, d_output, size, func);
            break;
        case 32:
            reduce_kernel<T, F, 32><<<dimGrid, dimBlock, dimBlock.x * sizeof(T)>>>(d_input, d_output, size, func);
            break;
        case 16:
            reduce_kernel<T, F, 16><<<dimGrid, dimBlock, dimBlock.x * sizeof(T)>>>(d_input, d_output, size, func);
            break;
        case 8:
            reduce_kernel<T, F, 8><<<dimGrid, dimBlock, dimBlock.x * sizeof(T)>>>(d_input, d_output, size, func);
            break;
        case 4:
            reduce_kernel<T, F, 4><<<dimGrid, dimBlock, dimBlock.x * sizeof(T)>>>(d_input, d_output, size, func);
            break;
        case 2:
            reduce_kernel<T, F, 2><<<dimGrid, dimBlock, dimBlock.x * sizeof(T)>>>(d_input, d_output, size, func);
            break;
        case 1:
            reduce_kernel<T, F, 1><<<dimGrid, dimBlock, dimBlock.x * sizeof(T)>>>(d_input, d_output, size, func);
            break;
    }

    cudaDeviceSynchronize();

    reduce_kernel<T, F, 1024><<<1, dimBlock, dimBlock.x * sizeof(T)>>>(d_output, d_final_output, dimGrid.x, func);

    cudaEventRecord(stop_kernel);
    cudaEventSynchronize(stop_kernel);
    cudaEventElapsedTime(&reduce_kernel_time, start_kernel, stop_kernel);
    cudaEventDestroy(start_kernel);
    cudaEventDestroy(stop_kernel);

    cudaEvent_t start_copy_host, stop_copy_host;
    cudaEventCreate(&start_copy_host);
    cudaEventCreate(&stop_copy_host);
    cudaEventRecord(start_copy_host);

    cudaMemcpy(&output, d_final_output, sizeof(T), cudaMemcpyDeviceToHost);

    cudaEventRecord(stop_copy_host);
    cudaEventSynchronize(stop_copy_host);
    cudaEventElapsedTime(&reduce_copy_host, start_copy_host, stop_copy_host);
    cudaEventDestroy(start_copy_host);
    cudaEventDestroy(stop_copy_host);

    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_final_output);

    cudaEventRecord(stop_all);
    cudaEventSynchronize(stop_all);
    cudaEventElapsedTime(&reduce_total, start_all, stop_all);
    cudaEventDestroy(start_all);
    cudaEventDestroy(stop_all);
}

int getNumThreads(int size) {
    // get device properties
    int device;
    cudaGetDevice(&device);
    cudaDeviceProp props;
    cudaGetDeviceProperties(&props, device);

    // get max threads per block and warp size
    int maxThreadsPerBlock = props.maxThreadsPerBlock;
    int warpSize = props.warpSize;

    // calculate number of threads to use
    int numThreads = (size + warpSize - 1) / warpSize * warpSize;
    numThreads = std::min(numThreads, maxThreadsPerBlock);
    numThreads = std::min(numThreads, size); 

    return numThreads;
}

int main(int argc, char** argv) {
    int size = 1000000; // default size is 1000000

    if (argc > 1) {
        // if user provides a size, use that size instead
        size = std::atoi(argv[1]);
    }

    // calcualte number of threads to use
    int numThreads = getNumThreads(size);

    // initialize input and output vectors
    std::vector <int> input(size);
    int output_reduce = 0;

    // fill input vectors with values
    for (int i = 0; i < size; i++){
        input[i] = i + 1;
    }

    // define device lambda functions for kernels
    Add add;

    // initialize timeings
    float reduce_copy_device, reduce_kernel_time, reduce_copy_host, reduce_total;

    // run parallel optimized reduce functions
    auto reduce_start = std::chrono::high_resolution_clock::now();
    reduce(input, output_reduce, add, numThreads, reduce_copy_device, reduce_kernel_time, reduce_copy_host, reduce_total);
    auto reduce_stop = std::chrono::high_resolution_clock::now();
    auto duration_reduce = reduce_stop - reduce_start;
    auto reduce_duration = std::chrono::duration<float, std::milli>(duration_reduce).count();

    // print size, threads and timings in csv format
    std::cout << size << "," << numThreads << ",";
    std::cout << reduce_copy_device << "," << reduce_kernel_time << "," << reduce_copy_host << "," << reduce_total << "," << reduce_duration << std::endl;

    // print output
    // std::cout << "Reduce output" << std::endl;
    // std::cout << output_reduce << std::endl;

    return 0;
}