#include "MeanSquaredError.h"
#include <iostream>
#include <math.h>

using namespace std;

void __global__ gpuCost(float* d_returnCost, float* d_output, float* d_correctOutput, int size)
{
    int x = blockIdx.x*blockDim.x + threadIdx.x;
    if(x<size)
        d_returnCost[x] = (d_correctOutput[x] - d_output[x]) * (d_correctOutput[x] - d_output[x]) * 0.5;
}

void __global__ gpuError(float* d_returnError, float* d_output, float* d_correctOutput, int size)
{
    int x = blockIdx.x*blockDim.x + threadIdx.x;
    if(x<size)
        d_returnError[x] = d_correctOutput[x] - d_output[x];
}

MeanSquaredError::MeanSquaredError()
{
    this->gpuLoaded = false;
}

MeanSquaredError::~MeanSquaredError()
{

}

void MeanSquaredError::cost(float* returnCost, float* output, float* correctOutput, int aSize)
{
    if(!gpuLoaded){
        for(int i=0; i<aSize; i++)
        {
            returnCost[i] = (correctOutput[i] - output[i]) * (correctOutput[i] - output[i]) * 0.5;
        }
    }else{
        int threadsPerBlock = 1024;
        int blocks = (aSize + threadsPerBlock) / threadsPerBlock;
        gpuCost<<<blocks,threadsPerBlock>>>(returnCost, output, correctOutput, aSize);
    }
}

void MeanSquaredError::error(float* returnError, float* output, float* correctOutput, int aSize)
{
    if(!gpuLoaded){
        for(int i=0; i<aSize; i++)
        {
            returnError[i] = correctOutput[i] - output[i];
        }
    }else{
        int threadsPerBlock = 1024;
        int blocks = (aSize + threadsPerBlock) / threadsPerBlock;
        gpuError<<<blocks,threadsPerBlock>>>(returnError, output, correctOutput, aSize);
    }
}

void MeanSquaredError::loadGPU()
{
    this->gpuLoaded = true;
}

void MeanSquaredError::unloadGPU()
{
    this->gpuLoaded = false;
}

string MeanSquaredError::toString()
{
    return "Mean Squared Error";
}