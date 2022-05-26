#include "ArrayMath.h"
#include <iostream>

using namespace std;

void ArrayMath::DotProduct(float* multiplier, float* product, float** multiplicand, int sizeX, int sizeY)
{
    int newSize = sizeX;
    int length = sizeY;
    for(int x=0; x<sizeX; x++)
        product[x]=0;
    for(int x=0; x<sizeX; x++)
    {
        for(int y=0; y<sizeY; y++)
        {
            if(sizeX==newSize)
            {
                product[x] += multiplicand[x][y] * multiplier[y];
            }else if(sizeY==newSize)
            {
                cout << "Error" << endl;
            }
        }
    }
}

void ArrayMath::Multiply(float* multiplicand, int length, float multiplier)
{
    for(int i=0; i<length; i++)
    {
        multiplicand[i] = multiplicand[i]*multiplier;
    }
}

void ArrayMath::Add(float* a, float* b, int length)
{
    for(int i=0; i<length; i++)
    {
        a[i]+=b[i];
    }
}

void ArrayMath::Sub(float* a, float* b, int length)
{
    for(int i=0; i<length; i++)
    {
        a[i]-=b[i];
    }
}

void ArrayMath::CrossProduct(float** multiplicand, int sizeX, int sizeY, float* multiplier, int length)
{
    if(sizeX!=length && sizeY!=length)
    {
        cout << "INCOMPATIBLE SIZES: " << length << " crossProduct(" << sizeX << "x" << sizeY << endl;
        return;
    }
    for(int x=0; x<sizeX; x++)
    {
        for(int y=0; y<sizeY; y++)
        {
            if(sizeX==length)
            {
                multiplicand[x][y] = multiplicand[x][y] * multiplier[x];
            }else if(sizeY==length)
            {
                multiplicand[x][y] = multiplicand[x][y] * multiplier[y];
            }
        }
    }
}

void ArrayMath::Multiply(float** multiplicand, int sizeX, int sizeY, float multiplier)
{
    for(int x=0; x<sizeX; x++)
    {
        for(int y=0; y<sizeY; y++)
        {
            multiplicand[x][y] = multiplicand[x][y] * multiplier;
        }
    }
}

void ArrayMath::Add(float** a, float** b, int sizeX, int sizeY)
{
    for(int x=0; x<sizeX; x++)
    {
        for(int y=0; y<sizeY; y++)
        {
            a[x][y]+=b[x][y];
        }
    }
}

float** ArrayMath::Copy(float** toCopy, int sizeX, int sizeY)
{
    float** copy = new float*[sizeX];
    for(int x=0; x<sizeX; x++)
    {
        copy[x] = new float[sizeY];
        for(int y=0; y<sizeY; y++)
        {
            copy[x][y] = toCopy[x][y];
        }
    }
    return copy;
}

float* ArrayMath::Copy(float* toCopy, int length)
{
    float* copy = new float[length];
    for(int i=0; i<length; i++)
    {
        copy[i] = toCopy[i];
    }
    return copy;
}

void ArrayMath::displayProgress(int aProgress, int maxProgress)
{
    int barWidth = 70;
    float progress = ((float) aProgress+1) / (float) maxProgress;
    cout << "[";
    int pos = barWidth * progress;
    for (int i = 0; i < barWidth; ++i) {
        if (i < pos) cout << "=";
        else if (i == pos) cout << ">";
        else cout << " ";
    }
    cout << "] " << int(progress * 100.0) << " %\r";
    cout.flush();
}


void ArrayMath::print(float** array, int sizeX, int sizeY)
{
    for(int x=0; x<sizeX; x++)
    {
        print(array[x], sizeY);
    }
}

void ArrayMath::print(float* array, int length)
{
    for(int i=0; i<length; i++)
    {
        cout << array[i] << ",";
    }
    cout << endl;
}

void ArrayMath::deleteArray(float** array, int sizeX)
{
    for(int i=0; i<sizeX; i++)
    {
        deleteArray(array[i]);
    }
    delete[] array;
}
void ArrayMath::deleteArray(float* array)
{
    delete[] array;
}