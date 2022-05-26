#ifndef LAYER_H
#define LAYER_H
#include <string>

namespace std
{
    class Layer
    {
    private:
        
    public:
        Layer(){}
        virtual ~Layer(){}
        virtual void feedForward(float* currentLayer) = 0;
        virtual float* propogateBack(float* previousLayer) = 0;
        virtual float* getNeurons() = 0;
        virtual void loadGPU() = 0;
        virtual void unloadGPU() = 0;
        virtual void setNeurons(float* aNeurons) = 0;
        virtual void setPreviousLayer(Layer* aLayer) = 0;
        virtual void setLearningRate(float aLearningRate) = 0;
        virtual void setBest() = 0;
        virtual void saveWeightsToFile(string aFileName) = 0;
        virtual void saveBiasToFile(string aFileName) = 0;
        virtual void saveLayerToFile(string aFileName) = 0;
        virtual int getSize() = 0;
        virtual string toString() = 0;
    };
}
#endif