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
void map(std::vector <T>& input, std::vector <T>& output, F func, int numThreads, float& map_copy_device, float& map_kernel, float& map_copy_host, float& map_total){
    cudaEvent_t start_all, stop_all;
    cudaEventCreate(&start_all);
    cudaEventCreate(&stop_all);
    cudaEventRecord(start_all);

    int size = input.size();
    T* d_input;
    T* d_output;

    cudaMalloc(&d_input, size * sizeof(T));
    cudaMalloc(&d_output, size * sizeof(T));
    
    cudaEvent_t start_copy_device, stop_copy_device;
    cudaEventCreate(&start_copy_device);
    cudaEventCreate(&stop_copy_device);
    cudaEventRecord(start_copy_device);

    cudaMemcpy(d_input, input.data(), size * sizeof(T), cudaMemcpyHostToDevice);

    cudaEventRecord(stop_copy_device);
    cudaEventSynchronize(stop_copy_device);
    cudaEventElapsedTime(&map_copy_device, start_copy_device, stop_copy_device);
    cudaEventDestroy(start_copy_device);
    cudaEventDestroy(stop_copy_device);

    dim3 dimBlock(numThreads);
    dim3 dimGrid((size + dimBlock.x - 1) / dimBlock.x);

    cudaEvent_t start_kernel, stop_kernel;
    cudaEventCreate(&start_kernel);
    cudaEventCreate(&stop_kernel);
    cudaEventRecord(start_kernel);

    map_kernel<<<dimGrid, dimBlock>>>(d_input, d_output, size, func);

    cudaEventRecord(stop_kernel);
    cudaEventSynchronize(stop_kernel);
    cudaEventElapsedTime(&map_kernel, start_kernel, stop_kernel);
    cudaEventDestroy(start_kernel);
    cudaEventDestroy(stop_kernel);

    cudaEvent_t start_copy_host, stop_copy_host;
    cudaEventCreate(&start_copy_host);
    cudaEventCreate(&stop_copy_host);
    cudaEventRecord(start_copy_host);

    cudaMemcpy(output.data(), d_output, size * sizeof(T), cudaMemcpyDeviceToHost);

    cudaEventRecord(stop_copy_host);
    cudaEventSynchronize(stop_copy_host);
    cudaEventElapsedTime(&map_copy_host, start_copy_host, stop_copy_host);
    cudaEventDestroy(start_copy_host);
    cudaEventDestroy(stop_copy_host);

    cudaFree(d_input);
    cudaFree(d_output);

    cudaEventRecord(stop_all);
    cudaEventSynchronize(stop_all);
    cudaEventElapsedTime(&map_total, start_all, stop_all);
    cudaEventDestroy(start_all);
    cudaEventDestroy(stop_all);
}

template<typename T, typename F>
void reduce(std::vector <T>& input, T& output, F func, int numThreads, float& reduce_copy_device, float& reduce_kernel, float& reduce_copy_host, float& reduce_total){
    cudaEvent_t start_all, stop_all;
    cudaEventCreate(&start_all);
    cudaEventCreate(&stop_all);
    cudaEventRecord(start_all);

    int size = input.size();

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

    reduce_kernel<<<dimGrid, dimBlock>>>(d_input, d_output, size, func);

    reduce_kernel<<<1, dimGrid.x>>>(d_output, d_final_output, dimGrid.x, func);

    cudaEventRecord(stop_kernel);
    cudaEventSynchronize(stop_kernel);
    cudaEventElapsedTime(&reduce_kernel, start_kernel, stop_kernel);
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

template<typename T1, typename T2, typename T3, typename F>
void zip(std::vector <T1>& input1, std::vector <T2>& input2, std::vector <T3>& output, F func, int numThreads, float& zip_copy_device, float& zip_kernel, float& zip_copy_host, float& zip_total){
    cudaEvent_t start_all, stop_all;
    cudaEventCreate(&start_all);
    cudaEventCreate(&stop_all);
    cudaEventRecord(start_all);

    int size = input1.size();

    T1* d_input1;
    T2* d_input2;
    T3* d_output;

    cudaMalloc(&d_input1, size * sizeof(T1));
    cudaMalloc(&d_input2, size * sizeof(T2));
    cudaMalloc(&d_output, size * sizeof(T3));

    cudaEvent_t start_copy_device, stop_copy_device;
    cudaEventCreate(&start_copy_device);
    cudaEventCreate(&stop_copy_device);
    cudaEventRecord(start_copy_device);

    cudaMemcpy(d_input1, input1.data(), size * sizeof(T1), cudaMemcpyHostToDevice);
    cudaMemcpy(d_input2, input2.data(), size * sizeof(T2), cudaMemcpyHostToDevice);

    cudaEventRecord(stop_copy_device);
    cudaEventSynchronize(stop_copy_device);
    cudaEventElapsedTime(&zip_copy_device, start_copy_device, stop_copy_device);
    cudaEventDestroy(start_copy_device);
    cudaEventDestroy(stop_copy_device);

    dim3 dimBlock(numThreads);
    dim3 dimGrid((size + dimBlock.x - 1) / dimBlock.x);

    cudaEvent_t start_kernel, stop_kernel;
    cudaEventCreate(&start_kernel);
    cudaEventCreate(&stop_kernel);
    cudaEventRecord(start_kernel);

    zip_kernel<<<dimGrid, dimBlock>>>(d_input1, d_input2, d_output, size, func);

    cudaEventRecord(stop_kernel);
    cudaEventSynchronize(stop_kernel);
    cudaEventElapsedTime(&zip_kernel, start_kernel, stop_kernel);
    cudaEventDestroy(start_kernel);
    cudaEventDestroy(stop_kernel);

    cudaEvent_t start_copy_host, stop_copy_host;
    cudaEventCreate(&start_copy_host);
    cudaEventCreate(&stop_copy_host);
    cudaEventRecord(start_copy_host);

    cudaMemcpy(output.data(), d_output, size * sizeof(T3), cudaMemcpyDeviceToHost);

    cudaEventRecord(stop_copy_host);
    cudaEventSynchronize(stop_copy_host);
    cudaEventElapsedTime(&zip_copy_host, start_copy_host, stop_copy_host);
    cudaEventDestroy(start_copy_host);
    cudaEventDestroy(stop_copy_host);

    cudaFree(d_input1);
    cudaFree(d_input2);
    cudaFree(d_output);

    cudaEventRecord(stop_all);
    cudaEventSynchronize(stop_all);
    cudaEventElapsedTime(&zip_total, start_all, stop_all);
    cudaEventDestroy(start_all);
    cudaEventDestroy(stop_all);
}