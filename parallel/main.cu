#include <iostream>
#include <functional>
#include <chrono>

#include "include/parallel.cuh"

struct my_pair {
    int first;
    int second;
};

int getNumThreads(int size) {
    int device;
    cudaGetDevice(&device);
    cudaDeviceProp props;
    cudaGetDeviceProperties(&props, device);

    int maxThreadsPerBlock = props.maxThreadsPerBlock;
    int warpSize = props.warpSize;

    int numThreads = (size + warpSize - 1) / warpSize * warpSize;
    numThreads = std::min(numThreads, maxThreadsPerBlock);

    return numThreads;
}

int main(int argc, char** argv) {
    int size = 10; // default size is 10

    if (argc > 1) {
        // if user provides a size, use that size instead
        size = std::atoi(argv[1]);
    }

    // calcualte number of threads to use
    int numThreads = getNumThreads(size);

    std::cout << "Number of Threads: " << numThreads << std::endl;

    std::vector <int> input(size);
    std::vector <int> input2(size);
    std::vector <int> output_map(size);
    int output_reduce = 0;
    std::vector <my_pair> output_zip(size);

    for (int i = 0; i < size; i++){
        input[i] = i + 1;
        input2[i] = (i + 1) * 2;
    }

    auto square = [] __device__ (int x) { return x * x; }; 

    parallel::map(input, output_map, square, numThreads);

    std::cout << "Parallel Map:" << std::endl;
    
    for (int i = 0; i < size; i++){
        std::cout << output_map[i] << " ";
    }
    std::cout << std::endl;

    auto add = [] __device__ (int x, int y) { return x + y; };

    parallel::reduce(input, output_reduce, add, numThreads);

    std::cout << "Parallel Reduce:" << std::endl;

    std::cout << output_reduce << std::endl;
    
    auto create_pair_parallel = [] __device__ (int x, int y) {
    my_pair pair;
    pair.first = x;
    pair.second = y;
    return pair;
    };

    parallel::zip(input, input2, output_zip, create_pair_parallel, numThreads);

    std::cout << "Parallel Zip:" << std::endl;

    for (int i = 0; i < size; i++){
        std::cout << output_zip[i].first << "," << output_zip[i].second << " ";
    }

    return 0;
}