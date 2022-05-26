#ifndef CUDAWRAPPER_H
#define CUDAWRAPPER_H
#include <stdlib.h>

namespace std
{
    class CudaWrapper
    {
    private:
        
    public:
        static void handleCUDAError(int errorId);

        static float* load2DArrayOnGPU(float** h_array, int rows, int columns);
        static float* loadArrayOnGPU(float* h_array, int size);

        static float** unload2DArrayFromGPU(float* d_array, int rows, int columns);
        static float* unloadArrayFromGPU(float* d_array, int size);

        static float* flattenArray(float** array, int rows, int columns);
        static float** unflattenArray(float* array, int rows, int columns);

        static void dotProduct(float* dummy, float* d_multiplier, float* d_multiplicand, float* d_product, int size1, int size2);
        static void crossProduct(float* dummy, float* d_multiplier, float* d_multiplicand, float* d_product, int size1, int size2);

        static void setDevice(int deviceNum);
        static void profileDevices();
        static void test();
    };
}

#endif