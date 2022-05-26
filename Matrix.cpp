#include "Matrix.h"
#include <iostream>
#include <string>
#include <fstream>
#include <sstream>
#include <istream>
#include <vector>
#include <list>

using namespace MatrixUtils;

Matrix::Matrix(std::string aFileName)
{
    std::ifstream f(aFileName);
    if(!f.good()){
        std::cout << "File does not exist! (" << aFileName << ")! [Matrix(string fileName)]" << std::endl;
    }
    std::cout << "Loading file: " << aFileName << std::endl;
    std::string DELIM = ",";
    std::ifstream infile(aFileName);
    std::string line;
    this->rows = 0;
    while (std::getline(infile, line))
        rows++;
    float** temp = new float*[rows];
    infile.clear();
    infile.seekg(0, std::ios::beg);
    int currentLine = 0;
    int barWidth = 50;
    while (std::getline(infile, line))
    {
        currentLine++;
        float progress = ((float) currentLine)/((float) rows);
        std::cout << "[";
        int pos = barWidth * progress;
        for (int i = 0; i < barWidth; ++i) {
            if (i < pos) std::cout << "=";
            else if (i == pos) std::cout << ">";
            else std::cout << " ";
        }
        std::cout << "] " << int(progress * 100.0) << " %\r";
        std::cout.flush();
        std::istringstream iss(line);
        std::vector<float> values;
        int firstComma = 0;
        int secondComma = 0;
        int elementNum = 0;
        for(int i=0; i<=line.size(); i++)
        {
            std::string sub = line.substr(i, 1);
            if(sub==DELIM)
            {
                secondComma=i;
                int subSize = secondComma - firstComma;
                std::string stringFloat = line.substr(firstComma, subSize);
                values.push_back(stof(stringFloat));
                firstComma = secondComma+1;
                elementNum++;
            }
        }
        float* newValues = new float[elementNum];
        for(int i=0; i<values.size(); i++)
        {
            newValues[i] = values[i];
        }
        temp[currentLine-1] = newValues;
        this->columns = elementNum;
    }
    std::cout << std::endl;
    infile.close();
    this->array = temp;
}

Matrix::Matrix(int rows, int col)
{
    srand (static_cast <unsigned> (time(0)));
    this->rows = rows;
    this->columns = col;
    this->array = new float*[rows];
    for(int i=0; i<rows; i++)
    {
        this->array[i] = new float[col];
        for(int j=0; j<col; j++)
        {
            this->array[i][j] = 0;
        }
    }
}

Matrix::Matrix(int rows, int col, int randRange)
{
    srand (static_cast <unsigned> (time(0)));
    this->rows = rows;
    this->columns = col;
    this->array = new float*[rows];
    for(int i=0; i<rows; i++)
    {
        this->array[i] = new float[col];
        for(int j=0; j<col; j++)
        {
            float r = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
            int rand = r*randRange*2 - randRange;
            this->array[i][j] = rand;
        }
    }
}


Matrix::Matrix(float** anArray, int rows, int col)
{
    this->rows = rows;
    this->columns = col;
    this->array = new float*[rows];
    for(int i=0; i<rows; i++)
    {
        this->array[i] = new float[col];
        for(int j=0; j<columns; j++)
        {
            this->array[i][j] = anArray[i][j];
        }
    }
}

Matrix::Matrix(float* anArray, int rows)
{
    this->rows = rows;
    this->columns = 1;
    this->array = new float*[1];
    for(int i=0; i<1; i++)
    {
        this->array[i] = new float[rows];
        for(int j=0; j<rows; j++)
        {
            this->array[i][j] = anArray[j];
        }
    }
}

int Matrix::getRows()
{
    return rows;
}

int Matrix::getColumns()
{
    return columns;
}

float** Matrix::getArray()
{
    return this->array;
}

float* Matrix::getArrayAt(int i)
{
    if(i<0 || i>=rows)
    {
        std::cout << "Error: Index out of bounds [Matrix:getArrayAt] rows (" << i << "," << rows << ")" << std::endl;
        return nullptr;
    }
    return this->array[i];
}

float Matrix::getValueAt(int x, int y)
{
    float* currentRow = getArrayAt(x);
    if(y<0 || y>=columns)
    {
        std::cout << "Error: Index out of bounds [Matrix:getValueAt] columns (" << y << "," << columns << ")" << std::endl;
        return -1;
    }
    return currentRow[y];
}

void Matrix::setValueAt(int x, int y, float value)
{
    float* currentRow = getArrayAt(x);
    if(y<0 || y>=columns)
    {
        std::cout << "Error: Index out of bounds [Matrix:setValueAt] columns (" << y << "," << columns << ")" << std::endl;
        return;
    }
    currentRow[y] = value;
}

void Matrix::setRow(int index, float* row)
{
    for(int i=0; i<columns; i++)
    {
        setValueAt(index, i, row[i]);
    }
}
void Matrix::setColumn(int index, float* column)
{
    for(int i=0; i<rows; i++)
    {
        setValueAt(i, index, column[i]);
    }
}

void Matrix::transpose()
{
    float** temp = new float*[this->columns];
    for(int i=0; i<columns; i++)
        temp[i] = new float[rows];
    for(int i=0; i<rows; i++)
    {
        for(int j=0; j<columns; j++)
        {
            temp[j][i] = this->array[i][j];
        }
    }
    replaceMatrix(temp);
    int placeHolder = this->rows;
    this->rows = this->columns;
    this->columns = placeHolder;
}

void Matrix::minSort()
{
    for(int i=0; i<rows*columns; i++)
    {
        for(int j=0; j<rows*columns; j++)
        {
            int y = j % columns;
            int x = (j-y) / columns;
            int y2 = ((j+1) % columns);
            int x2 = (j+1-y2) / columns;
            if(x2<rows && y2<columns)
            {
                if(getValueAt(x,y) > getValueAt(x2,y2))
                {
                    float temp = getValueAt(x,y);
                    setValueAt(x,y, getValueAt(x2,y2));
                    setValueAt(x2,y2, temp);
                }
            }
        }
    }
}

void Matrix::maxSort()
{
    for(int i=0; i<rows*columns; i++)
    {
        for(int j=0; j<rows*columns; j++)
        {
            int y = j % columns;
            int x = (j-y) / columns;
            int y2 = ((j+1) % columns);
            int x2 = (j+1-y2) / columns;
            if(x2<rows && y2<columns)
            {
                if(getValueAt(x,y) < getValueAt(x2,y2))
                {
                    float temp = getValueAt(x,y);
                    setValueAt(x,y, getValueAt(x2,y2));
                    setValueAt(x2,y2, temp);
                }
            }
        }
    }
}

float Matrix::max()
{
    float highestValue=getValueAt(0,0);
    int highestX,highestY;
    highestX=highestY=0;
    for(int i=0; i<rows; i++)
    {
        for(int j=0; j<columns; j++)
        {
            if(getValueAt(i,j)>highestValue)
            {
                highestValue = getValueAt(i,j);
                highestX = i;
                highestY = j;
            }
        }
    }
    return highestValue;
}
float Matrix::min()
{
    float smallestValue=getValueAt(0,0);
    int smallestX,smallestY;
    smallestX=smallestY=0;
    for(int i=0; i<rows; i++)
    {
        for(int j=0; j<columns; j++)
        {
            if(getValueAt(i,j)<smallestValue)
            {
                smallestValue = getValueAt(i,j);
                smallestX = i;
                smallestY = j;
            }
        }
    }
    return smallestValue;
}
float Matrix::sum()
{
    float currentSum = 0;
    for(int i=0; i<rows; i++)
    {
        for(int j=0; j<columns; j++)
        {
            currentSum+=getValueAt(i,j);
        }
    }
    return currentSum;
}

float Matrix::mean()
{
    return sum() / (float) (rows*columns);
}

float Matrix::median()
{
    //TODO: Add median
    return getValueAt(0,0);
}

void Matrix::print()
{
    for(int i=0; i<rows; i++)
    {
        for(int j=0; j<columns; j++)
        {
            std::cout << this->array[i][j] << ",";
        }
        std::cout << std::endl;
    }
}

void Matrix::replaceMatrix(float** newMatrix)
{
    deleteMatrix();
    this->array = newMatrix;
}

void Matrix::deleteMatrix()
{
    for(int i=0; i<rows; i++)
        delete[] this->array[i];
    delete[] this->array;
}

void Matrix::saveToFile(std::string aFileName)
{
    std::ofstream outFile;
    outFile.open(aFileName);
    std::string currentLine;
    for(int i=0; i<rows; i++)
    {
        currentLine="";
        for(int j=0; j<columns; j++)
        {
            currentLine += std::to_string(array[i][j]);
            currentLine += ",";
        }
        currentLine += "\n";
        outFile << currentLine;
    }
    outFile.close();
}

Matrix::~Matrix()
{
    deleteMatrix();
}