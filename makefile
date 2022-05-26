all: NeuralNetwork

NeuralNetwork: main.o FullyConnectedLayer.o InputLayer.o Matrix.o MeanSquaredError.o NeuralNetwork.o CudaWrapper.o Sigmoid.o ArrayMath.o
	g++ -o NeuralNetwork main.o FullyConnectedLayer.o InputLayer.o Matrix.o MeanSquaredError.o Sigmoid.o ArrayMath.o NeuralNetwork.o CudaWrapper.o -L/usr/local/cuda/lib64 -lcudart

main: main.cpp
	g++ -c main.cpp

FullyConnectedLayer.o: FullyConnectedLayer.cu
	nvcc -arch=sm_35 -c FullyConnectedLayer.cu

InputLayer.o: InputLayer.cu
	nvcc -arch=sm_35 -c InputLayer.cu

Matrix.o: Matrix.cpp
	g++ -c Matrix.cpp

MeanSquaredError.o: MeanSquaredError.cu
	nvcc -arch=sm_35 -c MeanSquaredError.cu

NeuralNetwork.o: NeuralNetwork.cu
	nvcc -arch=sm_35 -c NeuralNetwork.cu

CudaWrapper.o: CudaWrapper.cu
	nvcc -arch=sm_35 -c CudaWrapper.cu

Sigmoid.o: Sigmoid.cu
	nvcc -arch=sm_35 -c Sigmoid.cu

ArrayMath.o: ArrayMath.cpp
	g++ -c ArrayMath.cpp

clean:
	rm -rf *o NeuralNetwork
