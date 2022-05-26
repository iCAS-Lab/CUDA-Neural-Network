#ifndef MEANSQUAREDERROR_H
#define MEANSQUAREDERROR_H
#include "CostFunction.h"
#include <string>

namespace std
{
    class MeanSquaredError : public CostFunction
    {
    private:
        bool gpuLoaded;
    public:
        MeanSquaredError();
        ~MeanSquaredError();
        void cost(float* returnError, float* output, float* correctOutput, int aSize);
        void error(float* returnCost, float* output, float* correctOutput, int aSize);
        string toString();
        void loadGPU();
        void unloadGPU();
    };
}

#endif