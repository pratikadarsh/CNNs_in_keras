# CNNs Implementation in Keras


This repo primarily consists of implementation of various CNN backbones. Usage is shown for classification tasks, but these can be re-used in other tasks as well such as object detection, text detection, image segmentation etc. 

## Requirements

Needs Python3.5+. For other package requirements, check [requirements.txt]().

## Models 

  - AlexNet
  - InceptionNet
  - ResNet
  - XceptionNet
 
### Training

Current implementation performs training on the CIFAR10 dataset that comes with the Keras library. User has the option of specifying different CNN architectures for training. 

```python
python train.py -model alexnet -epoch 100 -batch_size 8
```

### Demo

```python
python predict.py -model alexnet -weight ./weights/alexnet.hdf5 -image ./sample_images/1.png
```

### Evaluation

Evaluation uses the test subset of CIFAR10 dataset from Keras.

```python
python evaluate.py -model resnet -weight ./weights/resnet.hdf5
```
