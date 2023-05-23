#ifndef SEQUENTIAL_H
#define SEQUENTIAL_H

#include <vector>

namespace sequential
{
    // sequeantial Map Implementation
    template<typename T, typename F>
    void map(std::vector <T>& input, std::vector <T>& output, F func);

    // sequential Reduce Implementation
    template<typename T, typename F>
    void reduce(std::vector <T>& input, T& output, F func);

    // sequential Zip Implementation
    template<typename T1, typename T2, typename T3, typename F>
    void zip(std::vector <T1>& input1, std::vector <T2>& input2, std::vector <T3>& output, F func);

    #include "../src/sequential.cpp"
}

#endif