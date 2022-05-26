#include "InputLayer.h"
#include "CudaWrapper.h"
#include <iostream>

using namespace std;

void __global__ deviceSetNeurons(float* d_a, float* d_b, int d_size)
{
    int x = blockIdx.x*blockDim.x + threadIdx.x;
    if(x<d_size);
        d_a[x] = d_b[x];
    __syncthreads();
}

InputLayer::InputLayer()
{
    cout << "[FATAL] Default constructor called for InputLayer\n\tUsage: InputLayer(int size)" << endl;
}

InputLayer::InputLayer(int aSize)
{
    this->size = aSize;
    this->neurons = new float[aSize];
    for(int i=0; i<aSize; i++)
        neurons[i] = 0;
    this->gpuLoaded = false;
}

void InputLayer::feedForward(float* aLayer)
{
    cout << "[WARNING] feedForward called for InputLayer" << endl;
}

float* InputLayer::propogateBack(float* aLayer)
{
    cout << "[WARNING] propogateBack called for InputLayer" << endl;
    return nullptr;
}

void InputLayer::loadGPU()
{
    if(gpuLoaded){
        cout << "[ERROR] Unable to load input layer on gpu: already loaded" << endl;
        return;
    }
    gpuLoaded = true;
    this->d_neurons = CudaWrapper::loadArrayOnGPU(this->neurons, this->size);
}

void InputLayer::unloadGPU()
{
    if(!gpuLoaded){
        cout << "[ERROR] Unable to unload input layer from gpu: already unloaded" << endl;
        return;
    }
    gpuLoaded = false;
    this->neurons = CudaWrapper::unloadArrayFromGPU(this->d_neurons, this->size);
}

float* InputLayer::getNeurons()
{
    if(gpuLoaded)
        return this->d_neurons;
    return this->neurons;
}

void InputLayer::setNeurons(float* aNeurons)
{
    if(!gpuLoaded){
        for(int i=0; i<size; i++)
        {
            this->neurons[i] = aNeurons[i];
        }
    }else{
        int threadsPerBlock = 1024;
        int blocks = (this->size + (threadsPerBlock-1)) / threadsPerBlock;
        deviceSetNeurons<<<blocks, threadsPerBlock>>>(this->d_neurons, aNeurons, this->size);
    }
}
void InputLayer::setPreviousLayer(Layer* aLayer)
{
    cout << "[WARNING] setPreviousLayer called for InputLayer" << endl;
}
void InputLayer::setLearningRate(float aLearningRate)
{
    cout << "[WARNING] setLearningRate called for InputLayer" << endl;
}
void InputLayer::setBest()
{
    cout << "[WARNING] setBest called for InputLayer" << endl;
}
void InputLayer::saveWeightsToFile(string aFileName)
{
    cout << "[WARNING] Save weights called for InputLayer" << endl;
}
void InputLayer::saveBiasToFile(string aFileName)
{
    cout << "[WARNING] Save bias called for InputLayer" << endl;
}
void InputLayer::saveLayerToFile(string aFileName)
{   
    //TODO: Add save for layer
}
int InputLayer::getSize()
{
    return this->size;
}
string InputLayer::toString()
{
    return "";
}

InputLayer::~InputLayer()
{
    delete[] this->neurons;
}