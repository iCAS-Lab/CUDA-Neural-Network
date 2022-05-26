#ifndef COSTFUNCTION_H
#define COSTFUNCTION_H
#include <string>

namespace std
{
    class CostFunction
    {
    public:
        CostFunction(){}
        virtual ~CostFunction(){}
        virtual void cost(float* returnCost, float* output, float* correctOutput, int aSize) = 0;
        virtual void error(float* returnError, float* output, float* correctOutput, int aSize) = 0;
        virtual string toString() = 0;
        virtual void loadGPU() = 0;
        virtual void unloadGPU() = 0;
    };
}
#endif