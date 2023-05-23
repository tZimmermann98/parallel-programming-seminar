#include <iostream>
#include <functional>

#include "include/sequential.h"

int main(int argc, char** argv) {
    int size = 10;

    if (argc > 1) {
        size = std::atoi(argv[1]);
    }

    std::vector <int> input(size);
    std::vector <int> input2(size);
    std::vector <int> output_map(size);
    int output_reduce = 0;
    std::vector <std::pair<int, int>> output_zip(size);

    for (int i = 0; i < size; i++){
        input[i] = i + 1;
        input2[i] = (i + 1) * 2;
    }

    sequential::map(input, output_map, [](int x){return x * x;});

    std::cout << "Sequential Map:" << std::endl;

    for (int i = 0; i < size; i++){
        std::cout << output_map[i] << " ";
    }
    std::cout << std::endl;

    sequential::reduce(input, output_reduce, [](int x, int y){return x + y;});

    std::cout << "Sequential Reduce:" << std::endl;

    std::cout << output_reduce << std::endl;

    sequential::zip(input, input2, output_zip, [](int x, int y){return std::make_pair(x, y);});

    std::cout << "Sequential Zip:" << std::endl;

    for (int i = 0; i < size; i++){
        std::cout << output_zip[i].first << "," << output_zip[i].second << " ";
    }
    std::cout << std::endl;

    return 0;
}