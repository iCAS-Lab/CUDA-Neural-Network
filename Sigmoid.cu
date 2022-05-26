#include "Sigmoid.h"
#include <iostream>
#include <math.h>

using namespace std;

void __global__ gpuSigmoid(float* d_layer, int d_size)
{
    int x = blockIdx.x*blockDim.x + threadIdx.x;
    if(x<d_size)
        d_layer[x] = (1 / (1 + exp(-1*d_layer[x])));
}

void __global__ gpuSigmoidError(float* d_layer, int d_size)
{
    int x = blockIdx.x*blockDim.x + threadIdx.x;
    if(x<d_size)
        d_layer[x] = (d_layer[x] * (1 - d_layer[x]));
}

Sigmoid::Sigmoid()
{
    this->gpuLoaded = false;
}

Sigmoid::~Sigmoid()
{

}

float Sigmoid::activate(float aValue)
{
    return (1 / (1 + exp(-1*aValue)));
}

float Sigmoid::activationDerivative(float aValue)
{
    return (aValue * (1 - aValue));
}

float* Sigmoid::activate(float* anArray, int layerSize)
{
    if(!this->gpuLoaded){
        for(int i=0; i<layerSize; i++)
        {
            anArray[i] = activate(anArray[i]);
        }
    }else{
        int threadsPerBlock = 1024;
        int blocks = (layerSize + threadsPerBlock) / threadsPerBlock;
        gpuSigmoid<<<blocks,threadsPerBlock>>>(anArray, layerSize);
    }
    return anArray;
}

float* Sigmoid::activationDerivative(float* anArray, int layerSize)
{
    if(!this->gpuLoaded){
        for(int i=0; i<layerSize; i++)
        {
            anArray[i] = activationDerivative(anArray[i]);
        }
    }else{
        int threadsPerBlock = 1024;
        int blocks = (layerSize + threadsPerBlock) / threadsPerBlock;
        gpuSigmoidError<<<blocks,threadsPerBlock>>>(anArray, layerSize);
    }
    return anArray;
}

void Sigmoid::loadGPU()
{
    if(this->gpuLoaded)
    {
        cout << "[ERROR] Unable to load sigmoid on GPU: already loaded on gpu" << endl;
        return;
    }
    this->gpuLoaded = true;
}

void Sigmoid::unloadGPU()
{
    if(!this->gpuLoaded)
    {
        cout << "[ERROR] Unable to unload sigmoid from GPU: already unloaded from gpu" << endl;
        return;
    }
    this->gpuLoaded = false;
}