#ifndef INPUTLAYER_H
#define INPUTLAYER_H
#include "Layer.h"
#include <string>

namespace std
{
    class InputLayer : public Layer
    {
    private:
        int size;
        float* neurons;

        // GPU Variables
        float* d_neurons;
        bool gpuLoaded;
    public:
        InputLayer();
        InputLayer(int asize);
        ~InputLayer();
        void feedForward(float* currentLayer);
        float* propogateBack(float* previousLayer);
        float* getNeurons();
        void loadGPU();
        void unloadGPU();
        void setNeurons(float* aNeurons);
        void setPreviousLayer(Layer* aLayer);
        void setLearningRate(float aLearningRate);
        void setBest();
        void saveWeightsToFile(string aFileName);
        void saveBiasToFile(string aFileName);
        void saveLayerToFile(string aFileName);
        int getSize();
        string toString();
    };
}
#endif