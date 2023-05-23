#define cudaCheckError() { \
    cudaError_t e=cudaGetLastError(); \
    if(e!=cudaSuccess) { \
        printf("Cuda failure %s:%d: '%s'\n",__FILE__,__LINE__,cudaGetErrorString(e)); \
        exit(EXIT_FAILURE); \
    } \
}

template <typename T, typename O, typename F>
__global__ void map_kernel(T* input, O* output, int size, F func){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size){
        output[idx] = func(input[idx]);
    }
}

template <typename T, typename O, typename F>
__global__ void reduce_kernel(T* input, O* output, int size, F func){
    __shared__ O sdata[1024];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < size){
        sdata[tid] = input[i];
    }
    else {
        sdata[tid] = 0;
    }
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1){
        if (tid < s){
            sdata[tid] = func(sdata[tid], sdata[tid + s]);
        }
        __syncthreads();
    }

    if (tid == 0){
        output[blockIdx.x] = sdata[0];
    }
    
}

template <typename T1, typename T2, typename O, typename F>
__global__ void zip_kernel(T1* input1, T2* input2, O* output, int size, F func){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size){
        output[idx] = func(input1[idx], input2[idx]);
    }
}

template<typename T, typename F>
void map(std::vector <T>& input, std::vector <T>& output, F func, int numThreads){
    int size = input.size();

    T* d_input;
    T* d_output;

    cudaMalloc(&d_input, size * sizeof(T));
    cudaCheckError();
    cudaMalloc(&d_output, size * sizeof(T));
    cudaCheckError();
    
    cudaMemcpy(d_input, input.data(), size * sizeof(T), cudaMemcpyHostToDevice);
    cudaCheckError();

    dim3 dimBlock(numThreads);
    dim3 dimGrid((size + dimBlock.x - 1) / dimBlock.x);

    map_kernel<<<dimGrid, dimBlock>>>(d_input, d_output, size, func);
    cudaCheckError();

    cudaMemcpy(output.data(), d_output, size * sizeof(T), cudaMemcpyDeviceToHost);
    cudaCheckError();

    cudaFree(d_input);
    cudaCheckError();
    cudaFree(d_output);
    cudaCheckError();
}

template<typename T, typename F>
void reduce(std::vector <T>& input, T& output, F func, int numThreads){
    int size = input.size();

    T* d_input;
    T* d_output;
    T* d_final_output;

    cudaMalloc(&d_input, size * sizeof(T));
    cudaMalloc(&d_output, sizeof(T) * ((size + 1023) / 1024));
    cudaMalloc(&d_final_output, sizeof(T));

    cudaMemcpy(d_input, input.data(), size * sizeof(T), cudaMemcpyHostToDevice);

    dim3 dimBlock(numThreads);
    dim3 dimGrid((size + dimBlock.x - 1) / dimBlock.x);

    reduce_kernel<<<dimGrid, dimBlock>>>(d_input, d_output, size, func);

    reduce_kernel<<<1, dimGrid.x>>>(d_output, d_final_output, dimGrid.x, func);

    cudaMemcpy(&output, d_final_output, sizeof(T), cudaMemcpyDeviceToHost);


    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_final_output);
}

template<typename T1, typename T2, typename T3, typename F>
void zip(std::vector <T1>& input1, std::vector <T2>& input2, std::vector <T3>& output, F func, int numThreads){
    int size = input1.size();

    T1* d_input1;
    T2* d_input2;
    T3* d_output;

    cudaMalloc(&d_input1, size * sizeof(T1));
    cudaMalloc(&d_input2, size * sizeof(T2));
    cudaMalloc(&d_output, size * sizeof(T3));

    cudaMemcpy(d_input1, input1.data(), size * sizeof(T1), cudaMemcpyHostToDevice);
    cudaMemcpy(d_input2, input2.data(), size * sizeof(T2), cudaMemcpyHostToDevice);

    dim3 dimBlock(numThreads);
    dim3 dimGrid((size + dimBlock.x - 1) / dimBlock.x);

    zip_kernel<<<dimGrid, dimBlock>>>(d_input1, d_input2, d_output, size, func);

    cudaMemcpy(output.data(), d_output, size * sizeof(T3), cudaMemcpyDeviceToHost);

    cudaFree(d_input1);
    cudaFree(d_input2);
    cudaFree(d_output);
}