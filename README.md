## CUDA-Neural-Network

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
