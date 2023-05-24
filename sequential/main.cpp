#include <iostream>
#include <functional>
#include <chrono>

#include "include/sequential.h"

int main(int argc, char** argv) {
    int size = 1000000; // default size is 1 Mio.

    // get size from command line if given
    if (argc > 1) {
        size = std::atoi(argv[1]);
    }

    // initialize input and output vectors
    std::vector <int> input(size);
    std::vector <int> input2(size);
    std::vector <int> output_map(size);
    int output_reduce = 0;
    std::vector <int> output_zip(size);

    for (int i = 0; i < size; i++){
        input[i] = i + 1;
        input2[i] = (i + 1) * 2;
    }

    // time measurement Sequential Map
    auto map_start = std::chrono::high_resolution_clock::now();
    sequential::map(input, output_map, [](int x){return x * x;});
    auto map_stop = std::chrono::high_resolution_clock::now();
    auto duration_map = map_stop - map_start;
    auto map_duration = std::chrono::duration<float, std::milli>(duration_map).count();

    // time measurement Sequential Reduce
    auto reduce_start = std::chrono::high_resolution_clock::now();
    sequential::reduce(input, output_reduce, [](int x, int y){return x + y;});
    auto reduce_stop = std::chrono::high_resolution_clock::now();
    auto duration_reduce = reduce_stop - reduce_start;
    auto reduce_duration = std::chrono::duration<float, std::milli>(duration_reduce).count();

    // time measurement Sequential Zip
    auto zip_start = std::chrono::high_resolution_clock::now();
    sequential::zip(input, input2, output_zip, [](int x, int y){return x * y;});
    auto zip_stop = std::chrono::high_resolution_clock::now();
    auto duration_zip = zip_stop - zip_start;
    auto zip_duration = std::chrono::duration<float, std::milli>(duration_zip).count();

    // print  size and timings in csv format with semicolon as separator
    std::cout << size << ",";
    std::cout << map_duration << ",";
    std::cout << reduce_duration << ",";
    std::cout << zip_duration << std::endl;

    return 0;
}