template <typename T, typename O, typename F>
__global__ void map_kernel(T* input, O* output, int size, F func){
    // Calculate the global index for the current thread across all blocks
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    // if index is less than the size of the input vector, apply the function to the input and store the result in the output
    if (idx < size){
        output[idx] = func(input[idx]);
    }
}

template <typename T, typename O, typename F>
__global__ void reduce_kernel(T* input, O* output, int size, F func){
    // Shared memory for each block
    __shared__ O sdata[1024];
    // Get the thread ID within the current block
    unsigned int tid = threadIdx.x;
    // Calculate the global index for the current thread across all blocks
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size){
        // if index is less than the size of the input vector, store the input in the shared memory
        sdata[tid] = input[idx];
    }
    else {
        // else store 0 in the shared memory
        sdata[tid] = 0;
    }
    // Wait for all threads to finish writing to shared memory
    __syncthreads();

    // Reduction in shared memory
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1){
        // if the thread ID is less than the current shared memory size
        if (tid < s){
            // apply the function to the value at current index and at index + the current shared memory size
            sdata[tid] = func(sdata[tid], sdata[tid + s]);
        }
        // Wait for all threads to finish writing to shared memory
        __syncthreads();
    }

    if (tid == 0){
        // Store the result of the reduction in the output vector
        output[blockIdx.x] = sdata[0];
    }
    
}

template <typename T1, typename T2, typename O, typename F>
__global__ void zip_kernel(T1* input1, T2* input2, O* output, int size, F func){
    // Calculate the global index for the current thread across all blocks
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size){
        // if index is less than the size of the input vectors, apply the function to the inputs and store the result in the output
        output[idx] = func(input1[idx], input2[idx]);
    }
}

template<typename T, typename F>
void map(std::vector <T>& input, std::vector <T>& output, F func, int numThreads, float& map_copy_device, float& map_kernel_time, float& map_copy_host, float& map_total){
    // Start overall timer
    cudaEvent_t start_all, stop_all;
    cudaEventCreate(&start_all);
    cudaEventCreate(&stop_all);
    cudaEventRecord(start_all);

    // Get the size of the input vector and initialize device pointers
    int size = input.size();
    T* d_input;
    T* d_output;

    // Allocate memory on the device
    cudaMalloc(&d_input, size * sizeof(T));
    cudaMalloc(&d_output, size * sizeof(T));
    
    // Start timer for copying input vector to device
    cudaEvent_t start_copy_device, stop_copy_device;
    cudaEventCreate(&start_copy_device);
    cudaEventCreate(&stop_copy_device);
    cudaEventRecord(start_copy_device);

    // Copy input vector to device
    cudaMemcpy(d_input, input.data(), size * sizeof(T), cudaMemcpyHostToDevice);

    // Stop timer for copying input vector to device
    cudaEventRecord(stop_copy_device);
    cudaEventSynchronize(stop_copy_device);
    cudaEventElapsedTime(&map_copy_device, start_copy_device, stop_copy_device);
    cudaEventDestroy(start_copy_device);
    cudaEventDestroy(stop_copy_device);

    // inititalize block and grid dimensions
    dim3 dimBlock(numThreads);
    dim3 dimGrid((size + dimBlock.x - 1) / dimBlock.x);

    // Start timer for kernel
    cudaEvent_t start_kernel, stop_kernel;
    cudaEventCreate(&start_kernel);
    cudaEventCreate(&stop_kernel);
    cudaEventRecord(start_kernel);

    // Call the map kernel
    map_kernel<<<dimGrid, dimBlock>>>(d_input, d_output, size, func);

    // Stop timer for kernel
    cudaEventRecord(stop_kernel);
    cudaEventSynchronize(stop_kernel);
    cudaEventElapsedTime(&map_kernel_time, start_kernel, stop_kernel);
    cudaEventDestroy(start_kernel);
    cudaEventDestroy(stop_kernel);

    // Start timer for copying output vector to host
    cudaEvent_t start_copy_host, stop_copy_host;
    cudaEventCreate(&start_copy_host);
    cudaEventCreate(&stop_copy_host);
    cudaEventRecord(start_copy_host);

    // Copy output vector to host
    cudaMemcpy(output.data(), d_output, size * sizeof(T), cudaMemcpyDeviceToHost);

    // Stop timer for copying output vector to host
    cudaEventRecord(stop_copy_host);
    cudaEventSynchronize(stop_copy_host);
    cudaEventElapsedTime(&map_copy_host, start_copy_host, stop_copy_host);
    cudaEventDestroy(start_copy_host);
    cudaEventDestroy(stop_copy_host);

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_output);

    // Stop overall timer
    cudaEventRecord(stop_all);
    cudaEventSynchronize(stop_all);
    cudaEventElapsedTime(&map_total, start_all, stop_all);
    cudaEventDestroy(start_all);
    cudaEventDestroy(stop_all);
}

template<typename T, typename F>
void reduce(std::vector <T>& input, T& output, F func, int numThreads, float& reduce_copy_device, float& reduce_kernel_time, float& reduce_copy_host, float& reduce_total){
    // Start overall timer
    cudaEvent_t start_all, stop_all;
    cudaEventCreate(&start_all);
    cudaEventCreate(&stop_all);
    cudaEventRecord(start_all);

    // Get the size of the input vector and initialize device pointers
    int size = input.size();
    T* d_input;
    T* d_output;
    T* d_final_output;

    // Allocate memory on the device
    cudaMalloc(&d_input, size * sizeof(T));
    cudaMalloc(&d_output, sizeof(T) * ((size + 1023) / 1024));
    cudaMalloc(&d_final_output, sizeof(T));

    // Start timer for copying input vector to device
    cudaEvent_t start_copy_device, stop_copy_device;
    cudaEventCreate(&start_copy_device);
    cudaEventCreate(&stop_copy_device);
    cudaEventRecord(start_copy_device);

    // Copy input vector to device
    cudaMemcpy(d_input, input.data(), size * sizeof(T), cudaMemcpyHostToDevice);

    // Stop timer for copying input vector to device
    cudaEventRecord(stop_copy_device);
    cudaEventSynchronize(stop_copy_device);
    cudaEventElapsedTime(&reduce_copy_device, start_copy_device, stop_copy_device);
    cudaEventDestroy(start_copy_device);
    cudaEventDestroy(stop_copy_device);

    // inititalize block and grid dimensions
    dim3 dimBlock(numThreads);
    dim3 dimGrid((size + dimBlock.x - 1) / dimBlock.x);

    // Start timer for kernel
    cudaEvent_t start_kernel, stop_kernel;
    cudaEventCreate(&start_kernel);
    cudaEventCreate(&stop_kernel);
    cudaEventRecord(start_kernel);

    // Call the reduce kernel for intermediate results
    reduce_kernel<<<dimGrid, dimBlock>>>(d_input, d_output, size, func);

    // Call the reduce kernel for final result
    reduce_kernel<<<1, dimGrid.x>>>(d_output, d_final_output, dimGrid.x, func);

    // Stop timer for kernel
    cudaEventRecord(stop_kernel);
    cudaEventSynchronize(stop_kernel);
    cudaEventElapsedTime(&reduce_kernel_time, start_kernel, stop_kernel);
    cudaEventDestroy(start_kernel);
    cudaEventDestroy(stop_kernel);

    // Start timer for copying output vector to host
    cudaEvent_t start_copy_host, stop_copy_host;
    cudaEventCreate(&start_copy_host);
    cudaEventCreate(&stop_copy_host);
    cudaEventRecord(start_copy_host);

    // Copy output vector to host
    cudaMemcpy(&output, d_final_output, sizeof(T), cudaMemcpyDeviceToHost);

    // Stop timer for copying output vector to host
    cudaEventRecord(stop_copy_host);
    cudaEventSynchronize(stop_copy_host);
    cudaEventElapsedTime(&reduce_copy_host, start_copy_host, stop_copy_host);
    cudaEventDestroy(start_copy_host);
    cudaEventDestroy(stop_copy_host);

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_final_output);

    // Stop overall timer
    cudaEventRecord(stop_all);
    cudaEventSynchronize(stop_all);
    cudaEventElapsedTime(&reduce_total, start_all, stop_all);
    cudaEventDestroy(start_all);
    cudaEventDestroy(stop_all);
}

template<typename T1, typename T2, typename T3, typename F>
void zip(std::vector <T1>& input1, std::vector <T2>& input2, std::vector <T3>& output, F func, int numThreads, float& zip_copy_device, float& zip_kernel_time, float& zip_copy_host, float& zip_total){
    // Start overall timer
    cudaEvent_t start_all, stop_all;
    cudaEventCreate(&start_all);
    cudaEventCreate(&stop_all);
    cudaEventRecord(start_all);

    // Get the size of the input vector and initialize device pointers
    int size = std::min(input1.size(), input2.size());
    T1* d_input1;
    T2* d_input2;
    T3* d_output;

    // Allocate memory on the device
    cudaMalloc(&d_input1, size * sizeof(T1));
    cudaMalloc(&d_input2, size * sizeof(T2));
    cudaMalloc(&d_output, size * sizeof(T3));

    // Start timer for copying input vectors to device
    cudaEvent_t start_copy_device, stop_copy_device;
    cudaEventCreate(&start_copy_device);
    cudaEventCreate(&stop_copy_device);
    cudaEventRecord(start_copy_device);

    // Copy input vectors to device
    cudaMemcpy(d_input1, input1.data(), size * sizeof(T1), cudaMemcpyHostToDevice);
    cudaMemcpy(d_input2, input2.data(), size * sizeof(T2), cudaMemcpyHostToDevice);

    // Stop timer for copying input vectors to device
    cudaEventRecord(stop_copy_device);
    cudaEventSynchronize(stop_copy_device);
    cudaEventElapsedTime(&zip_copy_device, start_copy_device, stop_copy_device);
    cudaEventDestroy(start_copy_device);
    cudaEventDestroy(stop_copy_device);

    // inititalize block and grid dimensions
    dim3 dimBlock(numThreads);
    dim3 dimGrid((size + dimBlock.x - 1) / dimBlock.x);

    // Start timer for kernel
    cudaEvent_t start_kernel, stop_kernel;
    cudaEventCreate(&start_kernel);
    cudaEventCreate(&stop_kernel);
    cudaEventRecord(start_kernel);

    // Call the zip kernel
    zip_kernel<<<dimGrid, dimBlock>>>(d_input1, d_input2, d_output, size, func);

    // Stop timer for kernel
    cudaEventRecord(stop_kernel);
    cudaEventSynchronize(stop_kernel);
    cudaEventElapsedTime(&zip_kernel_time, start_kernel, stop_kernel);
    cudaEventDestroy(start_kernel);
    cudaEventDestroy(stop_kernel);

    // Start timer for copying output vector to host
    cudaEvent_t start_copy_host, stop_copy_host;
    cudaEventCreate(&start_copy_host);
    cudaEventCreate(&stop_copy_host);
    cudaEventRecord(start_copy_host);

    // Copy output vector to host
    cudaMemcpy(output.data(), d_output, size * sizeof(T3), cudaMemcpyDeviceToHost);

    // Stop timer for copying output vector to host
    cudaEventRecord(stop_copy_host);
    cudaEventSynchronize(stop_copy_host);
    cudaEventElapsedTime(&zip_copy_host, start_copy_host, stop_copy_host);
    cudaEventDestroy(start_copy_host);
    cudaEventDestroy(stop_copy_host);

    // Free device memory
    cudaFree(d_input1);
    cudaFree(d_input2);
    cudaFree(d_output);

    // Stop overall timer
    cudaEventRecord(stop_all);
    cudaEventSynchronize(stop_all);
    cudaEventElapsedTime(&zip_total, start_all, stop_all);
    cudaEventDestroy(start_all);
    cudaEventDestroy(stop_all);
}