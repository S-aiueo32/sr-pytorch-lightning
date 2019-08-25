# sr-pytorch-lightning
## Introduction
Here provides Super-Resolution with pytorch-lightning.

## Requirements
- Python3.7
- CUDA9.2+ (It seems to be the most important.)

Python packages: see [Pipfile](Pipfile).


## Usage
### Training
```shell
pipenv run python train.py --model srgan
```
You can set the other options, see [train.py](train.py) and [models/srgan_model.py](models/srgan_model.py)

### Testing
```
pipenv run python test.py --model srgan --ckpt ./logs/srgan/default/version_0/media/_ckpt_epoch_xx.ckpt
```
