#ifndef MATRIX_H
#define MATRIX_H
#include <string>

namespace MatrixUtils
{
    class Matrix
    {
    private:
        int rows, columns;
        float** array;
    public:
        Matrix(std::string aFileName);
        Matrix(int rows, int col);
        Matrix(int rows, int col, int randRange);
        Matrix(float* anArray, int rows);
        Matrix(float** anArray, int rows, int col);
        ~Matrix();

        int getRows();
        int getColumns();
        float** getArray();
        float* getArrayAt(int i);
        float getValueAt(int x, int y);
        void setValueAt(int x, int y, float value);
        void setRow(int i, float* row);
        void setColumn(int j, float* column);
        void transpose();
        void maxSort();
        void minSort();
        

        float max();
        float min();
        float sum();
        float mean();
        float median();

        void print();

        void saveToFile(std::string aFileName);
        void replaceMatrix(float** newMatrix);
        void deleteMatrix();
    };
}
#endif