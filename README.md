# sr-pytorch-lightning
## Introduction
Here provides Super-Resolution with [PyTorch Lightning](https://github.com/williamFalcon/pytorch-lightning).

## Requirements
Here is written in Python 3.7 and following packages are required(All of them are latest versions as of Aug. 29, 2019).
```
torch>=1.2
torchvision>=0.4
pytorch-lightning>=0.4.6
kornia>=0.1.3.post2
```
You can create virtual environment easily using pipenv like below:
```
pipenv install
```

[IMPORTANT] `whl` file of `torch` via `pip` requires CUDA9.2+.


## Usage
### Training
To train SRGAN, run the following command.
The other models will be added.
```
pipenv run python train.py --model srgan
```
You can set the other options, see [train.py](train.py)(common training settings) and [models/srgan_model.py](models/srgan_model.py)(model specific settings).

### Testing
```
pipenv run python test.py --model srgan --ckpt ./logs/srgan/default/version_foo/media/_ckpt_epoch_bar.ckpt
```
