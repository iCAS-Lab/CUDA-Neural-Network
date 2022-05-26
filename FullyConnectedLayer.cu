#include "FullyConnectedLayer.h"
#include <iostream>
#include "NeuralNetwork.h"
#include "CudaWrapper.h"
#include "ArrayMath.h"
#include "Matrix.h"
#include "Sigmoid.h"

using namespace std;

/*
----------------------------------------------------------------
    CUDA KERNELS
----------------------------------------------------------------
*/

// Feed forward kernels

void __global__ dotProduct(float* weights, float* inputNeurons, float* neurons, int size, int previousSize)
{
    int x = blockIdx.x*blockDim.x + threadIdx.x;
    if(x<size){
        neurons[x] = 0;
        int k=x*previousSize;
        for(int y=0; y<previousSize; y++)
        {
            neurons[x] += weights[k] * inputNeurons[y];
            k++;
        }
    }
    __syncthreads();
}

void __global__ sum(float* d_a, float* d_b, int size)
{
    int x = blockIdx.x*blockDim.x + threadIdx.x;
    if(x<size)
        d_a[x] = d_a[x] + d_b[x];
    __syncthreads();
}
void __global__ multiply(float* d_a, float* d_b, int size)
{
    int x = blockIdx.x*blockDim.x + threadIdx.x;
    if(x<size)
        d_a[x] = d_a[x]*d_b[x];
    __syncthreads();
}

// Back propogation kernels

void __global__ backPropogate(float* d_weights, float* d_bias, float* d_previousNeurons, float* d_activationError, float* d_layerError, float* d_currentError,
    int size, int previousSize, float learningRate)
{
    int k=0;
    for(int i=0; i<previousSize; i++)
        d_layerError[i] = 0;
    for(int i=0; i<size; i++)
    {
        float neuronError = d_activationError[i] * d_currentError[i];
        for(int j=0; j<previousSize; j++)
        {
            d_layerError[j] += d_weights[k] * neuronError;
            d_weights[k] += neuronError * d_previousNeurons[j] * learningRate;
            k++;
        }
        d_bias[i]+=neuronError * learningRate;
    }
    __syncthreads();
}

void __global__ updateWeights(float* d_weights, float* d_previousNeurons, float* d_currentError, float learningRate, int currentSize, int previousSize)
{
    int x = blockIdx.x*blockDim.x + threadIdx.x;
    if(x<currentSize*previousSize){
        int y = x/previousSize;
        int z = x%previousSize;
        d_weights[x] = d_weights[x] + d_currentError[y] * d_previousNeurons[z] * learningRate;
    }
}

void __global__ updateBias(float* d_bias, float* d_currentError, float learningRate, int size)
{
    int x = blockIdx.x*blockDim.x + threadIdx.x;
    if(x<size)
    {
        d_bias[x] = d_bias[x] + d_currentError[x]*learningRate;
    }
}

void __global__ clear(float* array, int size)
{
    int x = blockIdx.x*blockDim.x + threadIdx.x;
    if(x<size)
        array[x] = 0;
    __syncthreads();
}


/*
----------------------------------------------------------------
    CONSTRUCTORS
----------------------------------------------------------------
*/

FullyConnectedLayer::FullyConnectedLayer()
{
    cout << "[FATAL] Default constructor called for FullyConnectedLayer\n\tUsage: FullyConnectedLayer(int size, string activationFunctionName)" << endl;
}
FullyConnectedLayer::FullyConnectedLayer(int aSize, string activationName)
{
    this->size = aSize;
    this->neurons = new float[aSize];
    for(int i=0; i<aSize; i++)
        neurons[i] = 0;
    this->learningRate = 0.1f;
    this->bias = NeuralNetwork::GenerateBias(aSize);
    this->bestBias = new float[aSize];
    //TODO Get activation function from string
    this->activationFunction = new Sigmoid(); // TEMPORARY
    this->gpuLoaded=this->didLoadGPU=this->loaded=false;
}

/*
----------------------------------------------------------------
    PRIVATE METHODS
----------------------------------------------------------------
*/

void FullyConnectedLayer::feedForwardCPU(float* aLayer)
{
    ArrayMath::DotProduct(aLayer, this->neurons, this->weights, size, previousSize); // Get dot product of current layer and weights and store in next layer
    ArrayMath::Add(this->neurons, this->bias, this->size);
    this->neurons = activationFunction->activate(this->neurons, this->size);
}

void FullyConnectedLayer::feedForwardGPU(float* aLayer)
{
    int threadsPerBlock = 1024;
    int blocks = (this->size + threadsPerBlock) / threadsPerBlock;
    CudaWrapper::dotProduct(this->d_dummy, this->d_weights, aLayer, this->d_neurons, this->size, this->previousSize);
    sum<<<blocks,threadsPerBlock>>>(this->d_neurons, this->d_bias, this->size);
    this->activationFunction->activate(this->d_neurons, this->size);
}

float* FullyConnectedLayer::propogateBackCPU(float* currentError)
{
    float* activationError = activationFunction->activationDerivative(this->neurons, this->size);
    for(int i=0; i<previousSize; i++)
        layerError[i] = 0;
    for(int i=0; i<this->size; i++)
    {
        float neuronError = activationError[i] * currentError[i];
        for(int j=0; j<previousSize; j++)
        {
            layerError[j] += weights[i][j] * neuronError;
            this->weights[i][j] += neuronError * previousLayer->getNeurons()[j] * this->learningRate;
        }
        this->bias[i]+=neuronError * this->learningRate;
    }
    return layerError;
}

float* FullyConnectedLayer::propogateBackGPU(float* currentError)
{
    int threadsPerBlock = 1024;
    int blocks = (this->size + threadsPerBlock) / threadsPerBlock;
    int blocks2 = (this->size*this->previousSize + threadsPerBlock) / threadsPerBlock;
    float* activationError = activationFunction->activationDerivative(this->d_neurons, this->size);
    multiply<<<blocks, threadsPerBlock>>>(currentError, activationError, this->size);
    CudaWrapper::crossProduct(this->d_dummy, this->d_weights, currentError, this->d_layerError, this->size, this->previousSize);
    updateWeights<<<blocks2, threadsPerBlock>>>(this->d_weights, this->previousLayer->getNeurons(), currentError, this->learningRate, this->size, this->previousSize);
    updateBias<<<blocks, threadsPerBlock>>>(this->d_bias, currentError, this->learningRate, this->size);

    return d_layerError;
}

/*
----------------------------------------------------------------
    PUBLIC METHODS
----------------------------------------------------------------
*/

void FullyConnectedLayer::feedForward(float* aLayer)
{
    if(!gpuLoaded){
        feedForwardCPU(aLayer);
    }else{
        feedForwardGPU(aLayer);
    }
}

float* FullyConnectedLayer::propogateBack(float* currentError)
{
    if(!gpuLoaded)
        return propogateBackCPU(currentError);
    return propogateBackGPU(currentError);
}

void FullyConnectedLayer::loadGPU()
{
    if(gpuLoaded){
        cout << "[ERROR] Unable to load layer on gpu: already loaded" << endl;
        return;
    }
    gpuLoaded = true;
    this->activationFunction->loadGPU();
    this->d_weights = CudaWrapper::load2DArrayOnGPU(this->weights, this->size, this->previousSize);
    this->d_dummy = CudaWrapper::load2DArrayOnGPU(this->weights, this->size, this->previousSize);
    this->d_bias = CudaWrapper::loadArrayOnGPU(this->bias, this->size);
    this->d_neurons = CudaWrapper::loadArrayOnGPU(this->neurons, this->size);
    if(!didLoadGPU)
    {
        didLoadGPU = true;
        this->d_layerError = CudaWrapper::loadArrayOnGPU(this->layerError, this->previousSize);
    }
}

void FullyConnectedLayer::unloadGPU()
{
    if(!gpuLoaded){
        cout << "[ERROR] Unable to unload layer from gpu: already unloaded" << endl;
        return;
    }
    gpuLoaded = false;
    this->activationFunction->unloadGPU();
    this->weights = CudaWrapper::unload2DArrayFromGPU(this->d_weights, this->size, this->previousSize);
    this->bias = CudaWrapper::unloadArrayFromGPU(this->d_bias, this->size);
    this->neurons = CudaWrapper::unloadArrayFromGPU(this->d_neurons, this->size);
}

float* FullyConnectedLayer::getNeurons()
{
    if(gpuLoaded)
        return this->d_neurons;
    return this->neurons;
}

void FullyConnectedLayer::setNeurons(float* aNeurons)
{
    for(int i=0; i<size; i++)
        this->neurons[i] = aNeurons[i];
}
void FullyConnectedLayer::setPreviousLayer(Layer* aLayer)
{
    if(aLayer==nullptr){
        cout << "[FATAL] unable to set previous. Value is null" << endl;
        return;
    }
    this->previousLayer = aLayer;
    this->previousSize = aLayer->getSize();
    this->weights = NeuralNetwork::GenerateWeights(previousSize, size);
    this->bestWeights = new float*[size];
    this->layerError = new float[previousSize];
    for(int i=0; i<size; i++)
    {
        this->bestWeights[i] = new float[previousSize];
        for(int j=0; j<previousSize; j++)
        {
            bestWeights[i][j] = 0;
        }
    }
    this->loaded = true;
    setBest();
}
void FullyConnectedLayer::setLearningRate(float aLearningRate)
{
    this->learningRate = aLearningRate;
}
void FullyConnectedLayer::setBest()
{
    for(int i=0; i<this->size; i++)
    {
        this->bestBias[i] = bias[i];
        for(int j=0; j<this->previousSize; j++)
        {
            this->bestWeights[i][j] = this->weights[i][j];
        }
    }
}
void FullyConnectedLayer::saveWeightsToFile(string aFileName)
{
    MatrixUtils::Matrix tempWeights(this->weights, this->size, this->previousSize);
    tempWeights.saveToFile(aFileName + ".txt");
}
void FullyConnectedLayer::saveBiasToFile(string aFileName)
{
    MatrixUtils::Matrix tempBias(this->bias, this->size);
    tempBias.saveToFile(aFileName + ".txt");
}
void FullyConnectedLayer::saveLayerToFile(string aFileName)
{

}
int FullyConnectedLayer::getSize()
{
    return this->size;
}
string FullyConnectedLayer::toString()
{
    return "Test";
}

/*
----------------------------------------------------------------
    DESTRUCTOR
----------------------------------------------------------------
*/

FullyConnectedLayer::~FullyConnectedLayer()
{
    if(didLoadGPU){
        cudaFree(d_layerError);
        cudaFree(d_dummy);
    }
    if(loaded){
        for(int i=0; i<this->size; i++)
        {
            delete[] weights[i];
            delete[] bestWeights[i];
        }
        delete[] weights;
        delete[] bestWeights;
    }
    delete[] neurons;
    delete[] bias;
    delete[] bestBias;
    delete activationFunction;
}