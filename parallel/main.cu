#include <iostream>
#include <functional>
#include <chrono>

#include "include/parallel.cuh"

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

    return numThreads;
}

int main(int argc, char** argv) {
    int size = 1000000; // default size is 10

    if (argc > 1) {
        // if user provides a size, use that size instead
        size = std::atoi(argv[1]);
    }

    // calcualte number of threads to use
    int numThreads = getNumThreads(size);

    // initialize input and output vectors
    std::vector <int> input(size);
    std::vector <int> input2(size);
    std::vector <int> output_map(size);
    int output_reduce = 0;
    std::vector <int> output_zip(size);

    // fill input vectors with values
    for (int i = 0; i < size; i++){
        input[i] = i + 1;
        input2[i] = (i + 1) * 2;
    }

    // define device lambda functions for kernels
    auto square = [] __device__ (int x) { return x * x; }; 
    auto add = [] __device__ (int x, int y) { return x + y; };
    auto mult = [] __device__ (int x, int y) { return x * y; };

    // initialize timeings
    float map_copy_device, map_kernel, map_copy_host, map_total;
    float reduce_copy_device, reduce_kernel, reduce_copy_host, reduce_total;
    float zip_copy_device, zip_kernel, zip_copy_host, zip_total;

    // run parallel map, reduce, and zip functions
    parallel::map(input, output_map, square, numThreads, map_copy_device, map_kernel, map_copy_host, map_total);

    parallel::reduce(input, output_reduce, add, numThreads, reduce_copy_device, reduce_kernel, reduce_copy_host, reduce_total);
    
    parallel::zip(input, input2, output_zip, mult, numThreads, zip_copy_device, zip_kernel, zip_copy_host, zip_total);

    // print size, threads and timings in csv format
    std::cout << size << "," << numThreads << ",";
    std::cout << map_copy_device << "," << map_kernel << "," << map_copy_host << "," << map_total << ","; 
    std::cout << reduce_copy_device << "," << reduce_kernel << "," << reduce_copy_host << "," << reduce_total << ",";
    std::cout << zip_copy_device << "," << zip_kernel << "," << zip_copy_host << "," << zip_total << std::endl;

    return 0;
}