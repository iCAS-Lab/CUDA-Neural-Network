#include <iostream>
#include "Layer.h"
#include "FullyConnectedLayer.h"
#include "CudaWrapper.h"
#include "Sigmoid.h"
#include "InputLayer.h"
#include "NeuralNetwork.h"
#include "Matrix.h"
#include <time.h>
#include <string>

using namespace std;

void blockAdd(int* array, int size, int cellSize, int maxSize)
{
    for(int j=0; j<((size/maxSize)/maxSize) + 1; j++){
        int* sBlock = new int[maxSize];
        for(int i=0; i<maxSize; i++)
        {
            int tid = i;
            int blockid = j;
            int index = (blockid*maxSize + tid)*maxSize;
            if(index<size){
                sBlock[i] = array[index];
                cout << sBlock[i] << ", ";
            }
        }
        cout << endl;
        for(int i=0; i<maxSize; i++)
        {
            int tid = i;
            int blockid = j;
            int index = (blockid*maxSize + tid)*maxSize;
            int global_start = ((index)/cellSize)*cellSize;
            int global_end = global_start + cellSize;
            int block_start = ((index/maxSize/maxSize))*maxSize*maxSize; // Global Index block starts at
            int first_elem_index = (index - block_start)/(cellSize);
            cout << first_elem_index << endl;
            if(global_start - block_start < 0)
            {

            }

            

        }
        for(int i=0; i<maxSize; i++)
        {
            int tid = i;
            int blockid = j;
            int index = (blockid*maxSize + tid)*maxSize;
            if(index<size)
                array[index] = sBlock[i];
        }
        delete[] sBlock;
    }
}

void nuclearAdd(int* array, int size, int cellSize, int maxSize, int stride)
{
    if(stride>cellSize)
        return;
    int k=0;
    for(int j=0; j<(size/maxSize) + 1; j++){
        int* sBlock = new int[maxSize];
        for(int i=0; i<maxSize; i++)
        {
            int index = j*maxSize + i;
            if(index<size)
                sBlock[i] = array[index];
        }
        for(int i=0; i<maxSize; i++)
        {
            int tid = i;
            int blockid = j;
            int index = blockid*maxSize + tid;

            int global_start = ((index)/cellSize)*cellSize;
            int local_start = (index / maxSize)*maxSize;
            int offset = global_start - local_start;

            int local_index = index % cellSize;

            int start = offset;
            if(offset<0)
                start = 0;
            int local_end = offset + cellSize;
            if(index < size){
                int t_1 = offset+local_index*2*stride;
                if(offset<0)
                    t_1 = tid*2*stride;
                int t_2 = t_1 + stride;
                if(t_2 < local_end && t_1%2==start%2 && t_2 < maxSize){ 
                    sBlock[t_1] = sBlock[t_1] + sBlock[t_2];
                    sBlock[t_2] = 0;
                }
            }
        }
        for(int i=0; i<maxSize; i++)
        {
            int index = j*maxSize + i;
            if(index<size)
                array[index] = sBlock[i];
        }
        delete[] sBlock;
    }
}
void sequentialBlockAdd(int* array, int size, int cellSize, int maxSize)
{
    for(int i=0; i<(size/cellSize); i++)
    {
        int tid = i;
        int global_start = tid*cellSize;
        int global_end = global_start + cellSize;
        int next_block = ((global_start+maxSize)/maxSize)*maxSize;
        for(int j=0; j<cellSize/maxSize; j++)
        {
            if(next_block < global_end){
                array[global_start]+=array[next_block];
                array[next_block] = 0;
            }
            next_block+=maxSize;
        }
    }
}
void sequentialBlockAdd2(int* array, int size, int cellSize, int maxSize)
{
    for(int i=0; i<(size/maxSize)+1; i++)
    {
        int index = i*maxSize;
        int global_start = ((index)/cellSize)*cellSize;
        if(index!=global_start){
            array[global_start]+=array[index];
            array[index] = 0;
        }
    }
}
void nuclearPrint(int* array, int size, int cellSize, int maxSize)
{
    for(int i=0; i<size; i++)
    {
        if(i%cellSize==0)
        {
            cout << "[" << array[i] << "],";
        }else if(i%maxSize==0){
            cout << "(" << array[i] << "),";
        }else{
            cout << array[i] << ",";
        }
    }
    cout << endl;
}

int main(int argc, char **argv)
{
    int device = 0;
    if(argc>1)
        device = atoi(argv[1]);
    CudaWrapper::setDevice(device);
    CudaWrapper::profileDevices();
    //*
    MatrixUtils::Matrix inputTrainData("TrainingData/CIFAR10_TrainInputs.csv");
    //MatrixUtils::Matrix inputTestData("TrainingData/MNIST_TestInputs.csv");
    MatrixUtils::Matrix trainLabels("TrainingData/CIFAR10_TrainOutputs.csv");
    //MatrixUtils::Matrix testLabels("TrainingData/MNIST_TestOutputs.csv");

    /*
    MatrixUtils::Matrix inputTrainData("TrainingData/simpleInputs.csv");
    MatrixUtils::Matrix trainLabels("TrainingData/simpleOutputs.csv");
    /*/
    //*

    int inputSize = inputTrainData.getColumns();
    int outputSize = trainLabels.getColumns();
    int trainNum = inputTrainData.getRows();

    cout << "Train Num " << trainNum << endl;

    float* d_inputDataSample = CudaWrapper::loadArrayOnGPU(inputTrainData.getArray()[0], inputSize);
    float* d_outputSample = CudaWrapper::loadArrayOnGPU(trainLabels.getArray()[0], outputSize);
//*/
    /*
    int inputSize = 8;
    int outputSize = 11;
    float* testInput = new float[inputSize];
    float* correctOutput = new float[outputSize];
    for(int i=0; i<inputSize; i++)
        testInput[i] = 1;
    for(int i=0; i<outputSize; i++)
        correctOutput[i] = 1;

    float* d_testInput = CudaWrapper::loadArrayOnGPU(testInput, inputSize);
    float* d_correctOutput = CudaWrapper::loadArrayOnGPU(correctOutput, outputSize);

    Layer* input = new InputLayer(inputSize);
    Layer* fullyConnected = new FullyConnectedLayer(outputSize, "");
    fullyConnected->setPreviousLayer(input);
    input->setNeurons(testInput);
    fullyConnected->feedForward(testInput);
    /*
    float* returnVal = fullyConnected->propogateBack(correctOutput);
    for(int i=0; i<inputSize; i++)
        cout << returnVal[i] << ",";
    cout << endl;
    */
    //float* returnVal;
    /*
    input->loadGPU();
    fullyConnected->loadGPU();
    input->setNeurons(d_testInput);
    fullyConnected->feedForward(d_testInput);

    returnVal = fullyConnected->propogateBack(d_correctOutput);
    float* currentOut = CudaWrapper::unloadArrayFromGPU(returnVal, inputSize);
    for(int i=0; i<inputSize; i++)
        cout << currentOut[i] << ",";
    cout << endl;

    //*/
   //*
    NeuralNetwork network;
    //network.setTestData(inputTestData.getArray(), testLabels.getArray(), 10000);
    network.addLayer(new InputLayer(inputSize));
    network.addLayer(new FullyConnectedLayer(10000, "test"));
    network.addLayer(new FullyConnectedLayer(outputSize, "test"));
    network.setTrainingData(inputTrainData.getArray(), trainLabels.getArray(), 50000);
    network.setLearningRate(0.0005);
    network.printTopology();
    network.loadGPU();
    for(int i=0; i<50; i++){
        network.train();
    }
    //network.unloadGPU();
    //*
    //network.train();
    /*
    network.setInput(inputTrainData.getArray()[0]);
    network.feedForward();
    float* out = network.getOutput();
    for(int i=0; i<10; i++)
    {
        //cout << out[i] << ",";
    }
    network.setInput(inputTrainData.getArray()[1]);
    //network.feedForward();
    cout << endl;
    network.loadGPU();
    network.setInput(d_inputDataSample);
    network.feedForward();
    for(int i=0; i<5; i++){
        //network.train();
    }
    cout << "Unloading" << endl;
    network.unloadGPU();
    float* out2 = network.getOutput();
    for(int i=0; i<10; i++)
    {
        //cout << out2[i] << ",";
    }
    cout << endl;
    //*/
    //network.train();
    //network.printTopology();
    //network.test(200*784, 200);
    /*
    long time = clock();
    float avgTime = 0;
    int iterations = 6;
    cout << "Iterations: " << iterations << endl;
    //*
    for(int i=0; i<iterations; i++){
        time = clock();
        network.setInput( inputTrainData.getArray()[0]);
        network.feedForward();
        network.propogateBack(trainLabels.getArray()[0]);
        long time2 = clock();
        float timeElapsed = (time2 - time) / ((float) CLOCKS_PER_SEC);
        avgTime+=timeElapsed;
    }
    cout << "CPU: " << endl;
    cout << "   Time Elapsed: " << avgTime*2 << endl;
    avgTime/=iterations;
    cout << "   Avg Time: " << avgTime << endl;
    network.loadGPU();
    avgTime = 0;
    for(int i=0; i<iterations; i++){
        time = clock();
        network.setInput(d_inputDataSample);
        network.feedForward();
        network.propogateBack(d_outputSample);
        long time2 = clock();
        float timeElapsed = (time2 - time) / ((float) CLOCKS_PER_SEC);
        avgTime+=timeElapsed;
    }
    cout << "GPU:" << endl;
    cout << "   Time Elapsed: " << avgTime*2 << endl;
    avgTime/=iterations;
    cout << "   Avg Time: " << avgTime << endl;
    network.unloadGPU();
    cout << "Done" << endl;
    //*/
    /*
    for(int i=0; i<1000; i++)
        network.train();
    for(int i=0; i<trainNum; i++)
    {
        network.setInput(inputTrainData.getArray()[i]);
        network.loadGPU();
        network.feedForward();
        network.unloadGPU();
        float* out = network.getOutput();
        for(int j=0; j<outputSize; j++)
        {
            cout << (int) (out[j]+0.5) << ",";
        }
        cout << endl;
    }
    //*/
    return 0;
}


/*
void nuclearAddWorking(int* array, int size, int cellSize, int maxSize, int stride)
{
    if(stride>cellSize)
        return;
    int k=0;
    for(int j=0; j<(size/maxSize) + 1; j++){
        int* sBlock = new int[maxSize];
        for(int i=0; i<maxSize; i++)
        {
            int index = j*maxSize + i;
            if(index<size)
                sBlock[i] = array[index];
        }
        for(int i=0; i<maxSize; i++)
        {
            int tid = i;
            int blockid = j;
            int index = blockid*maxSize + tid;

            int global_start = ((index)/cellSize)*cellSize;
            int local_start = (index / maxSize)*maxSize;
            int offset = global_start - local_start;

            int localIndex = index % cellSize;

            int start = offset;
            if(offset<0)
                start = 0;
            int local_end = offset + cellSize;
            int k = localIndex;

            if(index < size){
                //cout << "[" << tid - start << "]" << start << "_" << local_end << ",";
                int t_1 = tid;
                int t_2 = t_1 + stride;
                //cout << "[" << index << "]" << t_1 << ", " << t_2 << " : " << local_end;
                cout << start;
                if(t_2 < local_end && t_2 < maxSize && t_1%2==start%2){
                    cout << "[PASS]";
                    //cout << "[" << index << "]" << t_1 << ", " << t_2 << " : " << local_end << endl;
                    sBlock[t_1] = sBlock[t_1] + sBlock[t_2];
                    sBlock[t_2] = 0;
                }
                cout << endl;
            }

        }
        for(int i=0; i<maxSize; i++)
        {
            int index = j*maxSize + i;
            if(index<size)
                array[index] = sBlock[i];
        }
        delete[] sBlock;
    }
    cout << endl;
}
*/