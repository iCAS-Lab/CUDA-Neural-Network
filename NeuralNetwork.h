#ifndef NEURALNETWORK_H
#define NEURALNETWORK_H
#include <string>
#include "Layer.h"
#include "CostFunction.h"

namespace std
{
    class NeuralNetwork
    {
    private:
        Layer** layers;
        Layer* inputLayer;
        Layer* outputLayer;

        CostFunction* costFunction;

        int size;
        bool didTestWarn, didTrainNotify, gpuLoaded, didLoadGPU;

        // CUDA VARIABLES
        float** d_inputTrainData;
        float** d_inputTestData;
        float** d_trainLabels;
        float** d_testLabels;
        float* d_outputError;
        float* d_outputCost;
        float* d_totalEpochCost;

        // PRIVATE METHODS
        void trainCPU();
        void trainGPU();
    public:
        // PUBLIC MEMBER VARIABLES
        int trainingSamples, testSamples;
        float** inputTrainData;
        float** inputTestData;
        float** trainLabels;
        float** testLabels;
        float* outputError;
        float* outputCost;

        // CONSTRUCTORS & DESTRUCTORS
        NeuralNetwork();
        ~NeuralNetwork();

        // PUBLIC METHODS
        void train();
        void test(int size, int numCopies);
        int test2(int size);

        void addLayer(Layer* aLayer);
        void setCostFunction(CostFunction* aCostFunction);
        void setTrainingData(float** input, float** output, int numberSamples);
        void setTestData(float** input, float** output, int numberSamples);
        void setLearningRate(float aRate);
        void setInput(float* anInput);
        float* getOutput();
        void feedForward();
        void propogateBack(float* correctOutput);

        void printTopology();

        // CUDA METHODS
        void loadGPU();
        void unloadGPU();

        // STATIC METHODS
        static float** GenerateWeights(int dimX, int dimY);
        static float* GenerateBias(int size);
    };
}
#endif