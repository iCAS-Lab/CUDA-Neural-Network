## CUDA-Neural-Network
A CUDA based neural network application that trains neural networks using only CUDA/C++ with no external libraries

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
