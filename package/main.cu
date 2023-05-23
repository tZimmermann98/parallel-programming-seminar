#include <iostream>
#include <functional>

#include "include/sequential.h"
#include "include/parallel.cuh"

std::pair<int, int> create_pair(int x, int y){
    return std::make_pair(x, y);
}

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
    int size = 10;

    if (argc > 1) {
        size = std::atoi(argv[1]);
    }

    int numThreads = getNumThreads(size);

    std::cout << "Number of Threads: " << numThreads << std::endl;

    std::vector <int> input(size);
    std::vector <int> output(input.size());
    std::vector <int> input2(size);
    std::vector <std::pair<int, int>> output3(input.size());
    std::vector <my_pair> output3_parallel(input.size());

    for (int i = 0; i < size; i++){
        input[i] = i + 1;
        input2[i] = (i + 1) * 2;
    }

    sequential::map(input, output, [](int x){return x * x;});

    std::cout << "Sequential Map:" << std::endl;


    for (int i = 0; i < size; i++){
        std::cout << output[i] << " ";
    }
    std::cout << std::endl;
    std::vector <int> output_parallel(input.size());

    auto square = [] __device__ (int x) { return x * x; }; 

    parallel::map(input, output_parallel, square, numThreads);

    std::cout << "Parallel Map:" << std::endl;
    
    for (int i = 0; i < size; i++){
        std::cout << output_parallel[i] << " ";
    }
    std::cout << std::endl;
    int output2;
    sequential::reduce(input, output2, [](int x, int y){return x + y;});

    std::cout << "Sequential Reduce:" << std::endl;

    std::cout << output2 << std::endl;
    int output2_parallel = 0;

    auto add = [] __device__ (int x, int y) { return x + y; };

    parallel::reduce(input, output2_parallel, add, numThreads);

    std::cout << "Parallel Reduce:" << std::endl;

    std::cout << output2_parallel << std::endl;

    sequential::zip(input, input2, output3, create_pair);

    std::cout << "Sequential Zip:" << std::endl;

    for (int i = 0; i < size; i++){
        std::cout << output3[i].first << "," << output3[i].second << " ";
    }
    std::cout << std::endl;
    
    auto create_pair_parallel = [] __device__ (int x, int y) {
    my_pair pair;
    pair.first = x;
    pair.second = y;
    return pair;
    };

    parallel::zip(input, input2, output3_parallel, create_pair_parallel, numThreads);

    std::cout << "Parallel Zip:" << std::endl;

    for (int i = 0; i < size; i++){
        std::cout << output3_parallel[i].first << "," << output3_parallel[i].second << " ";
    }

    return 0;
}