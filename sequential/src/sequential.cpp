// sequeantial Map Implementation
template<typename T, typename F>
void map(std::vector <T>& input, std::vector <T>& output, F func){
    int n  = input.size();
    for (int i = 0; i < n; i++){
        // for each element in the input vector, apply the function and save the result in the output vector
        output[i] = func(input[i]);
    }
}

// sequential Reduce Implementation
template<typename T, typename F>
void reduce(std::vector <T>& input, T& output, F func){
    int n = input.size();
    output = input[0];
    for (int i = 1; i < n; i++){
        // for each element in the input vector, apply the function and overwrite output
        output = func(output, input[i]);
    }
}

// sequential Zip Implementation
template<typename T1, typename T2, typename T3, typename F>
void zip(std::vector <T1>& input1, std::vector <T2>& input2, std::vector <T3>& output, F func){
    int n = std::min(input1.size(), input2.size());
    for (int i = 0; i < n; i++){
        // for each element in the input vectors, create a pair and save it in the output vector
        output[i] = func(input1[i], input2[i]);
    }
}