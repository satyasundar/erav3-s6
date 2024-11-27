[![Build and Test](https://github.com/satyasundar/erav3-s6/actions/workflows/ml-pipeline.yml/badge.svg)](https://github.com/satyasundar/erav3-s6/actions/workflows/ml-pipeline.yml)

# Optimized MNIST Model

This repository contains a PyTorch model for classifying MNIST digits. The model is optimized for both speed and accuracy.

## Model Details

- It got more than **99.4% test accuracy** in 17th and 19th epoch as shown below logs section
- It uses batch normalization and dropout
- It uses fully connected layer instead of global average pooling
- It uses **15,522 parameters**
- It trains under 20 epochs
- It uses Adam optimizer with learning rate 0.01
- It uses random rotation of images to augment the data
- It uses 128 batch size

## CI/CD Pipeline

- The CI/CD pipeline is configured in [Github Actions](https://github.com/satyasundar/erav3-s6/actions)

## Local Run

```
To run the model locally, run the following command:
$ python main.py

To test the model, run the following command:
$ python -m unittest test_model.py -v
```

## Jupyter Notebook File
[mnist_model_notebook.ipynb](mnist_model_notebook.ipynb)

## Test Model File
[test_model.py](test_model.py)

## Training Logs With 99.4% Test Accuracy, Under 20,000 Parameters, Under 20 Epochs

```
$ python main.py
Model Architecture:
Network(
  (conv1): Conv2d(1, 8, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (bn1): BatchNorm2d(8, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (conv2): Conv2d(8, 12, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (bn2): BatchNorm2d(12, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (pool1): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  (conv3): Conv2d(12, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (bn3): BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (conv4): Conv2d(16, 20, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (bn4): BatchNorm2d(20, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (pool2): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  (relu): ReLU()
  (dropout): Dropout(p=0.1, inplace=False)
  (fc1): Linear(in_features=980, out_features=10, bias=True)
)

Total Parameters: 15522

Starting training...
Epoch 1 Loss=0.1365: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 469/469 [00:09<00:00, 49.81it/s]

Test set: Average loss: 0.0559, Accuracy: 9820/10000 (98.20%)

Epoch 2 Loss=0.0306: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 469/469 [00:08<00:00, 53.31it/s]

Test set: Average loss: 0.0367, Accuracy: 9873/10000 (98.73%)

Epoch 3 Loss=0.1496: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 469/469 [00:08<00:00, 53.63it/s]

Test set: Average loss: 0.0331, Accuracy: 9901/10000 (99.01%)

Epoch 4 Loss=0.0053: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 469/469 [00:08<00:00, 53.70it/s]

Test set: Average loss: 0.0273, Accuracy: 9911/10000 (99.11%)

Epoch 5 Loss=0.0192: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 469/469 [00:08<00:00, 53.70it/s]

Test set: Average loss: 0.0352, Accuracy: 9883/10000 (98.83%)

Epoch 6 Loss=0.0704: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 469/469 [00:08<00:00, 53.41it/s]

Test set: Average loss: 0.0342, Accuracy: 9880/10000 (98.80%)

Epoch 7 Loss=0.0516: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 469/469 [00:08<00:00, 53.70it/s]

Test set: Average loss: 0.0263, Accuracy: 9919/10000 (99.19%)

Epoch 8 Loss=0.0128: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 469/469 [00:08<00:00, 53.76it/s]

Test set: Average loss: 0.0240, Accuracy: 9923/10000 (99.23%)

Epoch 9 Loss=0.0198: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 469/469 [00:08<00:00, 53.47it/s]

Test set: Average loss: 0.0302, Accuracy: 9899/10000 (98.99%)

Epoch 10 Loss=0.0589: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████| 469/469 [00:08<00:00, 53.51it/s]

Test set: Average loss: 0.0293, Accuracy: 9904/10000 (99.04%)

Epoch 11 Loss=0.0199: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████| 469/469 [00:08<00:00, 53.66it/s]

Test set: Average loss: 0.0230, Accuracy: 9928/10000 (99.28%)

Epoch 12 Loss=0.0051: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████| 469/469 [00:08<00:00, 53.54it/s]

Test set: Average loss: 0.0248, Accuracy: 9921/10000 (99.21%)

Epoch 13 Loss=0.0045: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████| 469/469 [00:08<00:00, 53.20it/s]

Test set: Average loss: 0.0249, Accuracy: 9922/10000 (99.22%)

Epoch 14 Loss=0.0287: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████| 469/469 [00:08<00:00, 53.29it/s]

Test set: Average loss: 0.0301, Accuracy: 9907/10000 (99.07%)

Epoch 15 Loss=0.0409: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████| 469/469 [00:08<00:00, 53.70it/s]

Test set: Average loss: 0.0226, Accuracy: 9935/10000 (99.35%)

Epoch 16 Loss=0.0844: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████| 469/469 [00:08<00:00, 53.33it/s]

Test set: Average loss: 0.0225, Accuracy: 9931/10000 (99.31%)

Epoch 17 Loss=0.0249: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████| 469/469 [00:08<00:00, 53.71it/s]

Test set: Average loss: 0.0183, Accuracy: 9941/10000 (99.41%)

Epoch 18 Loss=0.0369: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████| 469/469 [00:08<00:00, 53.69it/s]

Test set: Average loss: 0.0208, Accuracy: 9937/10000 (99.37%)

Epoch 19 Loss=0.0248: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████| 469/469 [00:08<00:00, 53.77it/s]

Test set: Average loss: 0.0184, Accuracy: 9942/10000 (99.42%)

Epoch 20 Loss=0.0663: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████| 469/469 [00:08<00:00, 53.43it/s]

Test set: Average loss: 0.0215, Accuracy: 9934/10000 (99.34%)
```