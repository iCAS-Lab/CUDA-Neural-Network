#ifndef FULLYCONNECTEDLAYER_H
#define FULLYCONNECTEDLAYER_H
#include "Layer.h"
#include "ActivationFunction.h"
#include <string>

namespace std
{
    class FullyConnectedLayer : public Layer
    {
    private:
        // Private member variables
        Layer* previousLayer;
        ActivationFunction* activationFunction;

        int size, previousSize;
        float learningRate;
        float* neurons;
        float* bias;
        float** weights;

        float** bestWeights;
        float* bestBias;

        float* layerError;

        bool loaded, gpuLoaded, didLoadGPU;

        // GPU variables
        float* d_neurons;
        float* d_bias;
        float* d_weights;
        float* d_dummy; // Place holder for dot product

        float* d_layerError;

        // Private methods
        void feedForwardCPU(float* currentLayer);
        void feedForwardGPU(float* currentLayer);

        float* propogateBackCPU(float* currentError);
        float* propogateBackGPU(float* currentError);
    public:
        // Constructors & destructor
        FullyConnectedLayer();
        FullyConnectedLayer(int size, string activationName);
        ~FullyConnectedLayer();

        // Public methods
        void feedForward(float* currentLayer);
        float* propogateBack(float* previousLayer);

        float* getNeurons();
        int getSize();
        string toString();

        void setNeurons(float* aNeurons);
        void setPreviousLayer(Layer* aLayer);
        void setLearningRate(float aLearningRate);
        void setBest();
        
        void saveWeightsToFile(string aFileName);
        void saveBiasToFile(string aFileName);
        void saveLayerToFile(string aFileName);

        void loadGPU();
        void unloadGPU();
    };
}
#endif