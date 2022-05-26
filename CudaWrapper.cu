#include "CudaWrapper.h"
#include <iostream>
#include <stdio.h>

#include <cuda_runtime.h>

using namespace std;

void CudaWrapper::handleCUDAError(int errorId)
{
    if(errorId==0){
        return;
    }
    cout << "error in CUDA memCpy: ";
    if(errorId==1)
        cout << "cudaErrorInvalidValue" << endl;
    else if(errorId==2)
        cout << "cudaErrorMemoryAllocation" << endl;
    else if(errorId==3)
        cout << "cudaErrorInitializationError" << endl;
    else if(errorId==4)
        cout << "cudaErrorCudartUnloading" << endl;
    else if(errorId==5)
        cout << "cudaErrorProfilerDisabled" << endl;
    else
        cout << "code[" << errorId << "]" << endl;
}

float* CudaWrapper::load2DArrayOnGPU(float** h_array, int rows, int columns)
{
    size_t bytes = rows * columns * sizeof(float);
    float* d_array;
    float* h_flatInput = flattenArray(h_array, rows, columns);
    cudaMalloc(&d_array, bytes);
    handleCUDAError(cudaMemcpy(d_array, h_flatInput, bytes, cudaMemcpyHostToDevice));
    delete[] h_flatInput;
    return d_array;
}

float* CudaWrapper::loadArrayOnGPU(float* h_array, int size)
{
    size_t bytes = size * sizeof(float);
    float* d_array;
    cudaMalloc(&d_array, bytes);
    handleCUDAError(cudaMemcpy(d_array, h_array, bytes, cudaMemcpyHostToDevice));
    return d_array;
}

float** CudaWrapper::unload2DArrayFromGPU(float* d_array, int rows, int columns)
{
    float* flatArray = unloadArrayFromGPU(d_array, rows*columns);
    float** h_array = unflattenArray(flatArray, rows, columns);
    delete[] flatArray;
    return h_array;
}

float* CudaWrapper::unloadArrayFromGPU(float* d_array, int size)
{
    size_t bytes = size * sizeof(float);
    int arraySize = bytes / sizeof(float);
    float* h_array = new float[arraySize];
    handleCUDAError(cudaMemcpy(h_array, d_array, bytes, cudaMemcpyDeviceToHost));
    cudaFree(d_array);
    return h_array;
}

float* CudaWrapper::flattenArray(float** array, int rows, int columns)
{
    float* flatArray = new float[rows*columns];
    int k=0;
    for(int i=0; i<rows; i++){
        for(int j=0; j<columns; j++)
        {
            flatArray[k] = array[i][j];
            k++;
        }
    }
    return flatArray;
}

float** CudaWrapper::unflattenArray(float* array, int rows, int columns)
{
    float** unflattened = new float*[rows];
    int k=0;
    for(int i=0; i<rows; i++)
    {
        unflattened[i] = new float[columns];
        for(int j=0; j<columns; j++)
        {
            unflattened[i][j] = array[k];
            k++;
        }
    }
    return unflattened;
}


void __global__ dotProductMultiply(float* d_multiplier, float* d_array, float* d_copy, int size, int numCopies)
{
    int index = blockIdx.x*blockDim.x + threadIdx.x;
    int t_index = index%size;
    if(index<size*numCopies)
        d_copy[index] = d_array[t_index] * d_multiplier[index];
}

void __global__ crossProductMultiply(float* d_multiplier, float* d_array, float* d_copy, int currentSize, int previousSize)
{
    int index = blockIdx.x*blockDim.x + threadIdx.x;
    int i = index%currentSize;
    int j = index/currentSize;
    int w_index = i*previousSize + j;
    if(index<currentSize*previousSize)
        d_copy[index] = d_array[i] * d_multiplier[w_index];
}

void __global__ sequentialCombine(float *data, int size, int cellSize)
{
    int tid = threadIdx.x;
    int index = blockIdx.x*blockDim.x + tid;
    if(index>=size/cellSize)
        return;
    int global_start = index*cellSize;
    int global_end = global_start + cellSize;
    int next_block = ((global_start+1024)/1024)*1024;
    for(int j=0; j<(cellSize/1024)+1; j++)
    {
        if(next_block < global_end){
            data[global_start]+=data[next_block];
            data[next_block] = 0;
        }
        next_block+=1024;
    }
}

void __global__ dotProductReduce(float *data, int size, int cellSize)
{
    int tid = threadIdx.x;
    int index = (blockIdx.x*blockDim.x+tid);
    __shared__ float sblock[1024];
    if(index<size){
        sblock[tid] = data[index];
        __syncthreads();
        int global_start = (index/cellSize)*cellSize;
        int local_start = (index/1024)*1024;
        int offset = global_start - local_start;

        int local_index = index % cellSize;

        int start = offset;
        int local_end = offset + cellSize;
        int stride = 1;
        if(offset<0){
            start = 0;
            local_index = tid;
        }
        int t_1 = start+local_index*2*stride;
        int t_2 = t_1 + stride;
        if(t_1%2==start%2){
            while(stride<513){
                if(t_2 < local_end && t_2 < 1024){ 
                    sblock[t_1] = sblock[t_1] + sblock[t_2];
                }else{
                    break;
                }
                __syncthreads();
                stride*=2;
                t_1 = start+local_index*2*stride;
                t_2 = t_1 + stride;
            }
            if(index%1024==0 || index%cellSize==0)
                data[index] = sblock[tid];
        }
    }
}

void __global__ remap(float* dummy, float* finalArray, int size, int finalSize)
{
    int tid = threadIdx.x;
    int index = (blockIdx.x*blockDim.x+tid);
    int cellSize = size/finalSize;
    if(index*cellSize<size)
        finalArray[index] = dummy[index*cellSize];
}

void __global__ printArr(float* anArray, int size)
{
    for(int i=0; i<size; i++)
    {
        printf("[%d] %f\n", i, anArray[i]);
    }
}


void CudaWrapper::dotProduct(float* d_dummy, float* d_multiplier, float* d_multiplicand, float* d_product, int size1, int size2)
{
    int size = size1*size2;
    int cellSize = size2;
    int threadsPerBlock = 1024;
    int blocks = ((size-1) + threadsPerBlock) / threadsPerBlock;
    int blocks2 = (((size/cellSize) - 1) + threadsPerBlock) / threadsPerBlock;
    dotProductMultiply<<<blocks, threadsPerBlock>>>(d_multiplier, d_multiplicand, d_dummy, size2, size1);
    dotProductReduce<<<blocks, threadsPerBlock>>>(d_dummy, size, cellSize);
    sequentialCombine<<<blocks2, threadsPerBlock>>>(d_dummy, size, cellSize);
    remap<<<blocks2, threadsPerBlock>>>(d_dummy, d_product, size, size/cellSize);
}

void CudaWrapper::crossProduct(float* d_dummy, float* d_multiplier, float* d_multiplicand, float* d_product, int size1, int size2)
{
    int size = size1*size2;
    int cellSize = size2;
    int threadsPerBlock = 1024;
    int blocks = ((size-1) + threadsPerBlock) / threadsPerBlock;
    int blocks2 = (((size/cellSize) - 1) + threadsPerBlock) / threadsPerBlock;
    int blocks3 = ((size2 - 1) + threadsPerBlock) / threadsPerBlock;
    crossProductMultiply<<<blocks, threadsPerBlock>>>(d_multiplier, d_multiplicand, d_dummy, size1, size2);
    dotProductReduce<<<blocks, threadsPerBlock>>>(d_dummy, size, size1);
    sequentialCombine<<<blocks2, threadsPerBlock>>>(d_dummy, size, size1);
    remap<<<blocks2, threadsPerBlock>>>(d_dummy, d_product, size, size2);
}

void CudaWrapper::setDevice(int deviceNum)
{
    int nDevices;
    cudaGetDeviceCount(&nDevices);
    if(deviceNum>=0 && deviceNum<nDevices)
    {
        cudaSetDevice(deviceNum);
        cout << "Running program on device " << deviceNum << endl;
    }else{
        cout << "[ERROR]: Unable to set device to " << deviceNum << ": device not found" << endl;
    }
}
void CudaWrapper::profileDevices()
{
    //*
    int nDevices;

  cudaGetDeviceCount(&nDevices);
  for (int i = 0; i < nDevices; i++) {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, i);
    printf("Device Number: %d\n", i);
    printf("  Device name: %s\n", prop.name);
    printf("  Compute mode: %x\n", prop.computeMode);
    printf("  Memory Clock Rate (KHz): %d\n",
           prop.memoryClockRate);
    printf("  Memory Bus Width (bits): %d\n",
           prop.memoryBusWidth);
    printf("  Peak Memory Bandwidth (GB/s): %f\n",
           2.0*prop.memoryClockRate*(prop.memoryBusWidth/8)/1.0e6);
    printf("  Maximum Threads Per Block: %d\n",
           prop.maxThreadsPerBlock);
    printf("  Global Memory: %f (GB)\n", (((float) prop.totalGlobalMem/ (float) 1024)/ (float) 1024)/ (float) 1024);
    printf("  Shared Memory Per Block: %zu (KB)\n\n", prop.sharedMemPerBlock/1024);
  }
  //*/
}

void CudaWrapper::test()
{

}