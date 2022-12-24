## CUDA-Neural-Network
A CUDA/C++ implementation of the [code](https://github.com/BrendanCReidy/Java-ML-Framework/blob/master) used in our [paper](https://ieeexplore.ieee.org/abstract/document/9516756). The program uses *vanilla* CUDA/C++ (no additional libraries outside of minimum required to communicate with CUDA from C++). This program adds support for full GPU utilization using specialized CUDA Kernels to perform DNN inference and back propogations. 

### Initialize GPU
```
int device = 0;
CudaWrapper::setDevice(device);
CudaWrapper::profileDevices();
```

### Loading Data
```
MatrixUtils::Matrix inputTrainData("TrainingData/CIFAR10_TrainInputs.csv");
MatrixUtils::Matrix trainLabels("TrainingData/CIFAR10_TrainOutputs.csv");
```

### Defining Network Topology
```
NeuralNetwork network;
network.addLayer(new InputLayer(inputSize));
network.addLayer(new FullyConnectedLayer(10000, "test"));
network.addLayer(new FullyConnectedLayer(outputSize, "test"));
```

### Setting Network Hyper-Parameters
```
network.setTrainingData(inputTrainData.getArray(), trainLabels.getArray(), 50000);
network.setLearningRate(0.0005);
network.printTopology();
```

### Loading Network and Training
```
network.loadGPU();
for(int i=0; i<50; i++){
  network.train();
}
```
