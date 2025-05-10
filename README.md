

# DLIENet

## Requirements

```bash
OS：Ubuntu 20.04.6
nvdia 11.6
python 3.7.1  
pytorch 1.13.0  
numpy==1.21.0  
opencv-python==4.5.5.64  
scikit-image==0.18.3  
```


## TRAINING

Modify and set training parameters.

### Train the teacher network:

```bash
python train.py
```
### Train DLIENet:

```bash
python train_student.py
```


## Pretrained Models

[pretrained_model](https://pan.baidu.com/s/1hrLHEqCxJE3PSUEg01nf8g?pwd=ccpg).


## TEST

Modify and adjust testing parameters.

```bash
python test.py
```



