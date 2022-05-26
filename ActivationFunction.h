#ifndef ACTIVATIONFUNCTION_H
#define ACTIVATIONFUNCTION_H

namespace std
{
    class ActivationFunction
    {
    public:
        ActivationFunction(){}
        virtual ~ActivationFunction(){}
        virtual float* activate(float* anArray, int aSize) = 0;
        virtual float* activationDerivative(float* anArray, int aSize) = 0;
        virtual void loadGPU() = 0;
        virtual void unloadGPU() = 0;
    };
}
#endif