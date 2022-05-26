#include "NeuralNetwork.h"
#include "CudaWrapper.h"
#include "MeanSquaredError.h"
#include <iostream>
#include <stdio.h>
#include <time.h>
#include "ArrayMath.h"

#include <cuda_runtime.h>

using namespace std;

NeuralNetwork::NeuralNetwork()
{
    this->size = 0;
    this->layers = new Layer*[0];
    this->costFunction = new MeanSquaredError(); // Default Cost Function
    this->testLabels=this->trainLabels=this->inputTestData=this->inputTrainData=nullptr;
    this->outputError=new float[0];
    this->outputCost=new float[0];
    this->testSamples=this->trainingSamples=0;
    this->didTestWarn=this->didTrainNotify=this->gpuLoaded=this->didLoadGPU=false;
}

/*--------------------------------------------------------------------
    Private Methods
----------------------------------------------------------------------*/

__global__ void resetValue(float* a)
{
    a[0] = 0;
}

__global__ void addTo(float* a, float* b)
{
    a[0]+=b[0];
}

__global__ void divideValueInArray(float* array, int index, int amount)
{
    array[index]/=amount;
}

__global__ void printValueInArray(float* array, int index)
{
    printf("Cost: %f\n", array[index]);
    __syncthreads();
}

__global__ void blockWiseSummation(float *data, int size, int preStride, int layerSize)
{
    int tid = threadIdx.x;
    int index = (blockIdx.x*blockDim.x+tid)*preStride;
    __shared__ float sblock[1024];
    sblock[tid]=0;
    if(index<size){
        sblock[tid] = data[index];
        __syncthreads();
        int stride=1;
        while(stride<513 && stride <= layerSize){
            int t_1 = tid*stride*2;
            int t_2 = tid*stride*2 + stride;
            if(t_2<1024){
                sblock[t_1]+=sblock[t_2];
            }else{
                return;
            }
            __syncthreads();
            stride*=2;
        }
        if(tid==0)
            data[index]=sblock[tid];
        __syncthreads();
    }
}

void __global__ printDummy()
{
    printf("Test\n");
}

void NeuralNetwork::trainCPU()
{
    float totalCost = 0;
    for(int i=0; i<trainingSamples; i++)
    {
        ArrayMath::displayProgress(i, trainingSamples);
        float* input = inputTrainData[i];
        float* correctOutput = trainLabels[i];
        setInput(input);
        feedForward();
        float* output = outputLayer->getNeurons();
        costFunction->cost(this->outputCost, output, correctOutput, this->outputLayer->getSize());
        for(int j=0; j<outputLayer->getSize(); j++)
        {
            totalCost+=this->outputCost[j];
        }
        propogateBack(correctOutput);
    }
    cout << "Cost: " << totalCost / (float) trainingSamples << endl;
}

void NeuralNetwork::trainGPU()
{
    resetValue<<<1,1>>>(this->d_totalEpochCost);
    for(int i=0; i<trainingSamples; i++)
    {
        ArrayMath::displayProgress(i, trainingSamples);
        float* input = d_inputTrainData[i];
        float* correctOutput = d_trainLabels[i];
        setInput(input);
        cudaDeviceSynchronize();
        feedForward();
        float* output = outputLayer->getNeurons();
        costFunction->cost(this->d_outputCost, output, correctOutput, this->outputLayer->getSize());
        int preStride=1;
        int tempSize = this->outputLayer->getSize();
        
        while(tempSize>0){
            int threadsPerBlock = 1024;
            int blocks = (((tempSize-1)/preStride) + threadsPerBlock) / threadsPerBlock;
            blockWiseSummation<<<blocks,threadsPerBlock>>>(this->d_outputCost, this->outputLayer->getSize(), preStride, tempSize);
            tempSize=(tempSize-1) / threadsPerBlock;
            preStride*=1024;
        }
        addTo<<<1,1>>>(this->d_totalEpochCost, this->d_outputCost);
        propogateBack(correctOutput);
        cudaDeviceSynchronize();
    }
    divideValueInArray<<<1,1>>>(this->d_totalEpochCost, 0, trainingSamples);
    printValueInArray<<<1,1>>>(this->d_totalEpochCost, 0);
}

/*--------------------------------------------------------------------
    Public Methods
----------------------------------------------------------------------*/

void NeuralNetwork::train()
{
    if(this->size<=1)
    {
        cout << "[FATAL] Unable to train network: not enough layers" << endl;
        return;
    }
    if(trainingSamples<=0)
    {
        cout << "[FATAL] Unable to train network: training data not set" << endl;
        return;
    }else if(!didTestWarn && testSamples<=0)
    {
        this->didTestWarn=true;
        cout << "[WARNING] There is no test (validation) dataset for the network" << endl;
    }
    if(!didTrainNotify)
    {
        this->didTrainNotify=true;
        cout << "Training..." << endl;
    }
    if(!gpuLoaded)
        trainCPU();
    else
        trainGPU();
}

void NeuralNetwork::test(int size, int cellSize)
{
    /*
    float** Multiplier2D = new float*[size/cellSize];
    float* multiplicand = new float[cellSize];
    float* dummy = new float[size];
    float* product = new float[size/cellSize];
    srand(clock());
    for(int i=0; i<size/cellSize; i++)
    {
        Multiplier2D[i] = new float[cellSize];
        for(int j=0; j<cellSize; j++)
        {
            float r = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
            Multiplier2D[i][j] = r*100 - 50;
        }
    }
    for(int i=0; i<cellSize; i++)
    {
        float r = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
        multiplicand[i] = r*100 - 50;
    }

    float* validationProduct = new float[size/cellSize];
    ArrayMath::DotProduct(multiplicand, validationProduct, Multiplier2D, size/cellSize, cellSize);

    for(int i=0; i<size/cellSize; i++)
    {
        cout << validationProduct[i] << ",";
    }
    cout << endl;

    float* multiplier = CudaWrapper::flattenArray(Multiplier2D, size/cellSize, cellSize);

    float* d_multiplier = CudaWrapper::loadArrayOnGPU(multiplier, size);
    float* d_multiplicand = CudaWrapper::loadArrayOnGPU(multiplicand, cellSize);
    float* d_dummy = CudaWrapper::loadArrayOnGPU(dummy, size);
    float* d_product = CudaWrapper::loadArrayOnGPU(product, size/cellSize);

    
    delete[] multiplier;
    delete[] multiplicand;
    delete[] dummy;
    delete[] product;

    CudaWrapper::dotProduct(d_dummy, d_multiplier, d_multiplicand, d_product, size/cellSize, cellSize);

    multiplier = CudaWrapper::unloadArrayFromGPU(d_multiplier, size);
    multiplicand = CudaWrapper::unloadArrayFromGPU(d_multiplicand, cellSize);
    dummy = CudaWrapper::unloadArrayFromGPU(d_dummy, size);
    product = CudaWrapper::unloadArrayFromGPU(d_product, size/cellSize);
    /*
    for(int i=0; i<size; i++)
    {
        cout << multiplier[i] << ",";
    }
    cout << endl;
    for(int i=0; i<cellSize; i++)
    {
        cout << multiplicand[i] << ",";
    }
    cout << endl;
    for(int i=0; i<size; i++)
    {
        cout << dummy[i] << ",";
    }
    cout << endl;
    */
}

void __global__ sequentialSum(float* d_array, int size)
{
    for(int i=1; i<size; i++)
    {
        d_array[0]+=d_array[i];
    }
}

void __global__ flagCuda()
{

}

int NeuralNetwork::test2(int size)
{
    float* testValues = new float[size];
    srand(clock());
    for(int i=0; i<size; i++){
        float r = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
        testValues[i] = r*100-50;
    }
    float* d_testValues = CudaWrapper::loadArrayOnGPU(testValues, size);
    float* d_testValues2 = CudaWrapper::loadArrayOnGPU(testValues, size);
    flagCuda<<<1,1>>>();
    long firstTime = clock();
    //*
    for(int i=1; i<size; i++)
    {
        testValues[0]+=testValues[i];
    }
    //*/
    float correctValue = testValues[0];
    long secondTime = clock();
    flagCuda<<<1,1>>>();
    float timeElapsed = (secondTime - firstTime);
    cout << "[0] Time for " << size << ": " << timeElapsed << endl;
    delete[] testValues;
    firstTime = clock();
    //sequentialSum<<<1,1>>>(d_testValues, size);
    secondTime = clock();
    timeElapsed = (secondTime - firstTime);// / ((float) CLOCKS_PER_SEC);
    cout << "[1] Time for " << size << ": " << timeElapsed << endl;
    firstTime = clock();
    int preStride=1;
    int tempSize = size;
    while(tempSize>0){
        int threadsPerBlock = 1024;
        int blocks = (((size-1)/preStride) + threadsPerBlock) / threadsPerBlock;
        blockWiseSummation<<<blocks,threadsPerBlock>>>(d_testValues2, size, preStride, tempSize);
        tempSize=(tempSize-1) / threadsPerBlock;
        preStride*=1024;
    }
    secondTime = clock();
    timeElapsed = (secondTime - firstTime);// / ((float) CLOCKS_PER_SEC);
    cout << "[2] Time for " << size << ": " << timeElapsed << endl;
    /*
    cudaFree(d_testValues);
    cudaFree(d_testValues2);
    //*/

    //*
    testValues = CudaWrapper::unloadArrayFromGPU(d_testValues, size);
    float* testValues2 = CudaWrapper::unloadArrayFromGPU(d_testValues2, size);
    cout << "Final Value: " << testValues2[0] << endl;
    //cout << "Corre Value: " << testValues[0] << endl;
    cout << "Corr2 Value: " << correctValue << endl;
    /*
    for(int i=0; i<size; i++)
        cout << testValues[i] << ",";
    cout << endl;
    ///*
    for(int i=0; i<size; i++)
        cout << (int) testValues2[i] << ",";
    cout << endl;
    //*/
    /*
    if(abs(testValues2[0]-correctValue) > 0.1*correctValue){
        cout << abs(testValues2[0]-correctValue) << ", " << 0.01*size << endl;
        return 0;
    }
    //*/
    return 1;
}

void NeuralNetwork::addLayer(Layer* aLayer)
{
    if(this->size==0)
    {
        this->inputLayer = aLayer;
    }else{
        aLayer->setPreviousLayer(outputLayer);
    }
    Layer** temp = new Layer*[size+1];
    for(int i=0; i<size; i++)
        temp[i] = this->layers[i];
    temp[this->size] = aLayer;
    this->size++;
    this->outputLayer = aLayer;
    delete[] layers;
    delete[] outputError;
    delete[] outputCost;
    this->layers = temp;
    this->outputError = new float[this->outputLayer->getSize()];
    this->outputCost = new float[this->outputLayer->getSize()];
}

void NeuralNetwork::setCostFunction(CostFunction* aCostFunction)
{
    delete this->costFunction;
    this->costFunction = aCostFunction;
}

void NeuralNetwork::setLearningRate(float aRate)
{
    for(int i=1; i<this->size; i++)
    {
        layers[i]->setLearningRate(aRate);
    }
}

void NeuralNetwork::setTrainingData(float** input, float** output, int numSamples)
{
    this->inputTrainData = input;
    this->trainLabels = output;
    this->trainingSamples = numSamples;
    this->d_inputTrainData = new float*[numSamples];
    this->d_trainLabels = new float*[numSamples];
}

void NeuralNetwork::setTestData(float** input, float** output, int numSamples)
{
    this->inputTestData = input;
    this->testLabels = output;
    this->testSamples = numSamples;
}

void NeuralNetwork::setInput(float* anInput)
{
    inputLayer->setNeurons(anInput);
}

float* NeuralNetwork::getOutput()
{
    return this->outputLayer->getNeurons();
}

void NeuralNetwork::feedForward()
{
    float* current = inputLayer->getNeurons();
    for(int i=1; i<size; i++)
    {
        layers[i]->feedForward(current);
        current = layers[i]->getNeurons();
    }
}

void NeuralNetwork::propogateBack(float* correctOutput)
{
    float* currentError = this->outputError;
    if(gpuLoaded)
        currentError = this->d_outputError;
    costFunction->error(currentError, outputLayer->getNeurons(), correctOutput, outputLayer->getSize());
    for(int i=size-1; i>0; i--)
    {
        currentError = layers[i]->propogateBack(currentError);
    }
}

void NeuralNetwork::printTopology()
{
    cout << "Topology: \n";
    for(int i=0; i<size; i++)
    {
        cout << "   " << layers[i]->getSize() << endl;
    }
}

/*--------------------------------------------------------------------
    CUDA Kernals
----------------------------------------------------------------------*/

void __global__ aTest(float* aValue)
{
    for(int i=0; i<3*8; i++)
    {
        printf("I: %f\n", aValue[i]);
    }
    __syncthreads();
}

void __global__ arraySum(float* d_array, float* d_sum, int size, int finalSize)
{
    printf("Hello world!");
}

/*--------------------------------------------------------------------
    GPU Methods
----------------------------------------------------------------------*/

void NeuralNetwork::loadGPU()
{
    if(gpuLoaded){
        cout << "[WARNING] Unable to load GPU: GPU already loaded" << endl;
        return;
    }else if(trainingSamples<=0){
        cout << "[ERROR] Unable to load GPU: no training data" << endl;
        return;
    }else if(this->size <= 1){
        cout << "[ERROR] Unable to load GPU: not enough layers in network" << endl;
        return;
    }
    gpuLoaded = true;
    cout << "Loading GPU..." << endl;
    for(int i=0; i<size; i++)
    {
        layers[i]->loadGPU();
    }
    this->costFunction->loadGPU();
    if(!didLoadGPU){
        didLoadGPU = true;
        this->d_outputError = CudaWrapper::loadArrayOnGPU(this->outputError, this->outputLayer->getSize());
        this->d_outputCost = CudaWrapper::loadArrayOnGPU(this->outputCost, this->outputLayer->getSize());
        float* temp = new float[1];
        this->d_totalEpochCost = CudaWrapper::loadArrayOnGPU(temp, 1);
        delete[] temp;
        for(int i=0; i<this->trainingSamples; i++)
        {
            d_inputTrainData[i] = CudaWrapper::loadArrayOnGPU(this->inputTrainData[i], this->inputLayer->getSize());
            d_trainLabels[i] = CudaWrapper::loadArrayOnGPU(this->trainLabels[i], this->outputLayer->getSize());
        }
        cout << "Done loading GPU..." << endl;
    }
    cudaDeviceSynchronize();
}

void NeuralNetwork::unloadGPU()
{
    if(!gpuLoaded){
        cout << "[WARNING] Unable to unload GPU: GPU already unloaded" << endl;
        return;
    }
    gpuLoaded = false;
    this->costFunction->unloadGPU();
    for(int i=0; i<size; i++)
    {
        layers[i]->unloadGPU();
    }
}

/*--------------------------------------------------------------------
    Static Methods
----------------------------------------------------------------------*/

float** NeuralNetwork::GenerateWeights(int dimX, int dimY)
{
    float** weights = new float*[dimY];
    for(int i=0; i<dimY; i++)
    {
        weights[i] = new float[dimX];
        for(int j=0; j<dimX; j++)
        {
            float r = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
            r=r*2-1;
            weights[i][j] = r;
        }
    }
    return weights;
}

float* NeuralNetwork::GenerateBias(int size)
{
    float* bias = new float[size];
    for(int i=0; i<size; i++)
    {
        float r = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
        r=r*2-1;
        bias[i] = r;
    }
    return bias;
}

/*--------------------------------------------------------------------
    Destructor
----------------------------------------------------------------------*/

NeuralNetwork::~NeuralNetwork()
{
    for(int i=0; i<size; i++)
    {
        delete layers[i];
    }
    if(didLoadGPU)
    {
        for(int i=0; i<trainingSamples; i++){
            cudaFree(this->d_inputTrainData[i]);
            cudaFree(this->d_trainLabels[i]);
        }
        cudaFree(d_outputError);
        cudaFree(d_outputCost);
    }
    if(this->trainingSamples>0){
        delete[] d_inputTrainData;
        delete[] d_trainLabels;
    }
    if(this->size>0)
        delete[] outputError;
    delete costFunction;
    delete[] layers;
    cudaDeviceReset();
}