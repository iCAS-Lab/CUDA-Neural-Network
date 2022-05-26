#ifndef ARRAYMATH_H
#define ARRAYMATH_H

namespace std
{
    class ArrayMath
    {
    public:
        static void DotProduct(float* multiplier, float* product, float** multiplicand, int sizeX, int sizeY);
        static void Multiply(float* multiplicand, int length, float multiplier);
        static void Add(float* a, float* b, int length);
        static void Sub(float* a, float* b, int length);
        static void CrossProduct(float** multiplicand, int sizeX, int sizeY, float* multiplier, int length);
        static void Multiply(float** multiplicand, int sizeX, int sizeY, float multiplier);
        static void Add(float** a, float** b, int sizeX, int sizeY);
        static float** Copy(float** toCopy, int sizeX, int sizeY);
        static float* Copy(float* toCopy, int length);

        static void print(float** array, int sizeX, int sizeY);
        static void print(float* array, int length);
        static void displayProgress(int progress, int maxProgress);

        static void deleteArray(float** array, int sizeX);
        static void deleteArray(float* array);
    };
}
#endif