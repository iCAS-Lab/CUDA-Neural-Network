#ifndef SIGMOID_H
#define SIGMOID_H
#include "ActivationFunction.h"

namespace std
{
    class Sigmoid : public ActivationFunction
    {
    private:
        bool gpuLoaded;
    public:
        Sigmoid();
        ~Sigmoid();
        float activate(float aValue);
        float activationDerivative(float aValue);
        float* activate(float* anArray, int layerSize);
        float* activationDerivative(float* anArray, int layerSize);
        void loadGPU();
        void unloadGPU();
    };
}

#endif