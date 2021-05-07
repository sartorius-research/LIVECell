# Anchor based model
Each folder contains model and the config file.
The anchor based models used in this benchmark is based on the detectron2-ResNeSt architecture which is built upon the detectron2 library.
## Requirements:
```sh
- Linux or macOS with Python ≥ 3.6
- PyTorch ≥ 1.3
- torchvision that matches the PyTorch installation. You can install them together at pytorch.org to make sure of this.
- OpenCV, optional, needed by demo and visualization
- pycocotools: pip install cython; pip install -U 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'
```
### detectron2
```sh
- python -m pip install 'git+https://github.com/zhanghang1989/detectron2-ResNeSt.git'
```
Or, to install it from a local clone:
```sh
git clone https://github.com/zhanghang1989/detectron2-ResNeSt.git
cd detectron2 && python -m pip install -e .
```
Or if you are on macOS
```sh
CC=clang CXX=clang++ python -m pip install -e 
```
To install a pre-built detectron for different torch and cuda versions and further information, see the detectron2 install document 
https://github.com/facebookresearch/detectron2/blob/master/INSTALL.md

## detectron2-ResNeSt
Retrive the detectron2-ResNeSt code:
```sh
git clone https://github.com/zhanghang1989/detectron2-ResNeSt.git
```
For further information, on installation and usage, see the detectron2-ResNeSt documentation at https://github.com/chongruo/detectron2-ResNeSt

### Common Installation Issues
- Not compiled with GPU support" or "Detectron2 CUDA Compiler: not available".
```sh
CUDA is not found when building detectron2. You should make sure
python -c 'import torch; from torch.utils.cpp_extension import CUDA_HOME; print(torch.cuda.is_available(), CUDA_HOME)'
print valid outputs at the time you build detectron2.
```
# Training and evaluation

## Register LIVECell dataset

Using a custom dataset such as LIVECell together with the detectron2 code base is done by first registering the dataset via the detectron2 python API. In practice this can be done adding the following code to the train_net.py file in the cloned centermask2 repo:
https://detectron2.readthedocs.io/en/latest/tutorials/datasets.html
```sh
from detectron2.data.datasets import register_coco_instances
register_coco_instances([dataset_name], {}, [/path/coco/annotations.json], [path/to/image/dir])
```
Where dataset_name will be the name of your dataset and will be how you decide what dataset to use in your config file. Per default, the config file will point to *TRAIN* and *TEST*, so registering a test dataset as *TEST* will work directly with the provided config files, for other names, make sure to update your config file accordingly.

- In the config file change the dataset entries with the name used to register the dataset.
- Set the output directory in the config file to save the models and results.

## Train a model

To train a model, change the OUTPUT directory in the config file to where the models and checkpoints should be saved. Make sure you follow the previous step and register a TRAIN and TEST dataset and run the following code:
```sh
python tools/train_net.py  --num-gpus 8 --config-file your_config.yaml
```
To train a model on the dataset defined in *your_config.yaml* with 8 gpus.

## Evaluate a model
To evaluate a model, make sure to register a TEST dataset and point to it in your config file and then run the following code
```sh
python train_net.py  --config-file your_config.yaml --eval-only MODEL.WEIGHTS /path/to/checkpoint_file.pth
```
This will evaluate a model defined in *your_config.yaml* with the weights saved in */path/to/checkpoint_file.pth*

For further details on training, testing and inference, visit: https://github.com/chongruo/detectron2-ResNeSt/blob/resnest/GETTING_STARTED.md
