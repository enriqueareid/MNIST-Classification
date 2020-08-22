# MNIST Model | GPU (Compatible with CPU)
## Train Accuracy: 99.96%
## Cross-Validation Accuracy: 99.55%

# Introduction
This repository includes a Jupyter Notebook, *MNIST.ipynb*, which can be executed in order to train the model again. In order to run the notebook, [tqdm](https://pypi.org/project/tqdm/), [PyTorch](https://pytorch.org/get-started/locally/) and [Python 3](https://www.python.org) must be installed on your machine. The Jupyter Notebook is compatible with both GPUs (recommended) and CPUs. The pretrained network's saved parameters can be found at *MNIST.prm*.  
Assuming that [this](https://github.com/erarvin2007/MNIST-Classification/#Architecture) is the *model*'s architecture, parameters can be loaded using the following code:   
```python
model.load_state_dict(torch.load('MNIST.prm'))
```

### Dataset
Train Length: 60000  
CV Length: 10000  
Original Input Dimensions: (Examples) x 28 x 28  
Reshaped Input Dimensions: (Examples) x 1 x 32 x 32

### Architecture
```
ConvNet5(  
  (conv1): Conv2d(1, 8, kernel_size=(5, 5), stride=(1, 1))  
  (conv1_bn): BatchNorm2d(8, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)  
  (conv2): Conv2d(8, 32, kernel_size=(5, 5), stride=(1, 1))  
  (conv2_dp): Dropout2d(p=0.4, inplace=False)  
  (fc1): Linear(in_features=800, out_features=256, bias=True)  
  (fc1_dp): Dropout(p=0.4, inplace=False)  
  (fc2): Linear(in_features=256, out_features=84, bias=True)  
  (fc2_bn): BatchNorm1d(84, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)  
  (fc3): Linear(in_features=84, out_features=10, bias=True)  
)
```
### Hyperparameters
Epochs: 70  
Batch Size: 512  
Optimizer: ADAM  
Learning Rate: 0.0033  
Learning Rate Decay: StepLR(step_size=2, gamma=0.79433)  
Loss Function: (Log-Softmax) Cross-Entropy Loss
