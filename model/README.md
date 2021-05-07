# Usage

The anchor free models used in this benchmark is based on the [centermask2](https://github.com/youngwanLEE/centermask2#evaluation) architecture and the anchor based models are 
based on the [detectron2-ResNeSt](https://github.com/chongruo/detectron2-ResNeSt/blob/resnest/GETTING_STARTED.md) architectures, 
which is both built upon the [detectron2](https://github.com/facebookresearch/detectron2) library.

The models in the LIVECell paper was trained in on 8 Nvidia V100 GPUS.
To help others reproduce our results and use the models for further research, we provide pre-trained models and config files.

| Architecture    | Dataset | box AP | mask AP | download | 
| ------------ |  ------  |  ----  | ---- | ----- | 
<td rowspan="9">Anchor free</td>  | LIVECell | 48.45 | 47.78 | [config](https://github.com/sartorius-research/LIVECell/blob/main/model/anchor_free/livecell_config.yaml) / [model]() |
|         | A172 | 31.49 | 34.57 | [config](https://github.com/sartorius-research/LIVECell/blob/main/model/anchor_free/a172_config.yaml) / [model]() |  
|         | BT-474 | 42.12 | 42.60 | [config](https://github.com/sartorius-research/LIVECell/blob/main/model/anchor_free/bt474_config.yaml) / [model]() |  
|         | BV-2 | 42.62 | 45.69 | [config](https://github.com/sartorius-research/LIVECell/blob/main/model/anchor_free/bv2_config.yaml) / [model]() |  
|         | Huh7 | 42.44 | 45.85 | [config](https://github.com/sartorius-research/LIVECell/blob/main/model/anchor_free/huh7_config.yaml) / [model]() |  
|          | MCF7 | 36.53 | 37.30 | [config](https://github.com/sartorius-research/LIVECell/blob/main/model/anchor_free/mcf7_config.yaml) / [model]() |  
|          | SH-SY5Y | 25.20 | 23.91 | [config](https://github.com/sartorius-research/LIVECell/blob/main/model/anchor_free/shsy5y_config.yaml) / [model]() |  
|          | SkBr3 | 64.35 | 65.85 | [config](https://github.com/sartorius-research/LIVECell/blob/main/model/anchor_free/skbr3_config.yaml) / [model]() |  
|          | SK-OV-3 | 46.43 | 49.39 | [config](https://github.com/sartorius-research/LIVECell/blob/main/model/anchor_free/livecell_config.yaml) / [model]() |  
<td rowspan="9">Anchor based</td>| LIVECell | 48.43 | 47.89 | [config](https://github.com/sartorius-research/LIVECell/blob/main/model/anchor_based/livecell_config.yaml) / [model]() |  
|          | A172 | 36.37 | 38.02 | [config](https://github.com/sartorius-research/LIVECell/blob/main/model/anchor_based/a172_config.yaml) / [model]() |  
|          | BT-474 | 43.25 | 43.00 | [config](https://github.com/sartorius-research/LIVECell/blob/main/model/anchor_based/bt474_config.yaml) / [model]() |  
|          | BV-2 | 54.36 | 52.60 | [config](https://github.com/sartorius-research/LIVECell/blob/main/model/anchor_based/bv2_config.yaml) / [model]() |  
|          | Huh7 | 52.79 | 51.83 | [config](https://github.com/sartorius-research/LIVECell/blob/main/model/anchor_based/huh7_config.yaml) / [model]() |  
|          | MCF7 | 37.53 | 37.94 | [config](https://github.com/sartorius-research/LIVECell/blob/main/model/anchor_based/mcf7_config.yaml) / [model]() |  
|          | SH-SY5Y | 27.87 | 24.92 | [config](https://github.com/sartorius-research/LIVECell/blob/main/model/anchor_based/shsy5y_config.yaml) / [model]() |  
|          | SkBr3 | 64.41 | 65.39 | [config](https://github.com/sartorius-research/LIVECell/blob/main/model/anchor_based/skbr3_config.yaml) / [model]() |  
|          | SK-OV-3 | 53.29 | 54.12 | [config](https://github.com/sartorius-research/LIVECell/blob/main/model/anchor_based/skov3_config.yaml) / [model]() |  

<table class="tg">
  <tr>
    <th class="tg-0pky">Architecture</th>
    <th class="tg-0pky">Dataset</th>
    <th class="tg-0pky">Box mAP%</th>
    <th class="tg-0pky">MAsk mAP%</th>
    <th class="tg-0pky">download</th>
  </tr>
  <tr>
    <td rowspan="9" class="tg-0pky">Anchor free</td>
    <td class="tg-0pky">LIVECell</td>
    <td class="tg-0pky">48.45</td>
    <td class="tg-0pky">47.78</td>
    <td class="tg-0lax"><a href="https://github.com/sartorius-research/LIVECell/blob/main/model/anchor_free/livecell_config.yaml">config</a> | <a href="https://s3.us-west-1.wasabisys.com/resnest/detectron/faster_rcnn_R_50_FPN_syncbn_range-scale_1x-fde56e2b.pth ">model </a> 
  </tr>
    <tr>
    <td class="tg-0pky">LIVECell</td>
    <td class="tg-0pky">48.45</td>
    <td class="tg-0pky">47.78</td>
    <td class="tg-0lax"><a href="https://github.com/sartorius-research/LIVECell/blob/main/model/anchor_free/livecell_config.yaml">config</a> | <a href="https://s3.us-west-1.wasabisys.com/resnest/detectron/faster_rcnn_R_50_FPN_syncbn_range-scale_1x-fde56e2b.pth ">model </a> 
  </tr>
    <tr>
    <td class="tg-0pky">LIVECell</td>
    <td class="tg-0pky">48.45</td>
    <td class="tg-0pky">47.78</td>
    <td class="tg-0lax"><a href="https://github.com/sartorius-research/LIVECell/blob/main/model/anchor_free/livecell_config.yaml">config</a> | <a href="https://s3.us-west-1.wasabisys.com/resnest/detectron/faster_rcnn_R_50_FPN_syncbn_range-scale_1x-fde56e2b.pth ">model </a> 
  </tr>
    <tr>
    <td class="tg-0pky">LIVECell</td>
    <td class="tg-0pky">48.45</td>
    <td class="tg-0pky">47.78</td>
    <td class="tg-0lax"><a href="https://github.com/sartorius-research/LIVECell/blob/main/model/anchor_free/livecell_config.yaml">config</a> | <a href="https://s3.us-west-1.wasabisys.com/resnest/detectron/faster_rcnn_R_50_FPN_syncbn_range-scale_1x-fde56e2b.pth ">model </a> 
  </tr>
    <tr>
    <td class="tg-0pky">LIVECell</td>
    <td class="tg-0pky">48.45</td>
    <td class="tg-0pky">47.78</td>
    <td class="tg-0lax"><a href="https://github.com/sartorius-research/LIVECell/blob/main/model/anchor_free/livecell_config.yaml">config</a> | <a href="https://s3.us-west-1.wasabisys.com/resnest/detectron/faster_rcnn_R_50_FPN_syncbn_range-scale_1x-fde56e2b.pth ">model </a> 
  </tr>
    <tr>
    <td class="tg-0pky">LIVECell</td>
    <td class="tg-0pky">48.45</td>
    <td class="tg-0pky">47.78</td>
    <td class="tg-0lax"><a href="https://github.com/sartorius-research/LIVECell/blob/main/model/anchor_free/livecell_config.yaml">config</a> | <a href="https://s3.us-west-1.wasabisys.com/resnest/detectron/faster_rcnn_R_50_FPN_syncbn_range-scale_1x-fde56e2b.pth ">model </a> 
  </tr>
    <tr>
    <td class="tg-0pky">LIVECell</td>
    <td class="tg-0pky">48.45</td>
    <td class="tg-0pky">47.78</td>
    <td class="tg-0lax"><a href="https://github.com/sartorius-research/LIVECell/blob/main/model/anchor_free/livecell_config.yaml">config</a> | <a href="https://s3.us-west-1.wasabisys.com/resnest/detectron/faster_rcnn_R_50_FPN_syncbn_range-scale_1x-fde56e2b.pth ">model </a> 
  </tr>
    <tr>
    <td class="tg-0pky">LIVECell</td>
    <td class="tg-0pky">48.45</td>
    <td class="tg-0pky">47.78</td>
    <td class="tg-0lax"><a href="https://github.com/sartorius-research/LIVECell/blob/main/model/anchor_free/livecell_config.yaml">config</a> | <a href="https://s3.us-west-1.wasabisys.com/resnest/detectron/faster_rcnn_R_50_FPN_syncbn_range-scale_1x-fde56e2b.pth ">model </a> 
  </tr>
  
   <tr>
    <td rowspan="9" class="tg-0pky">Anchor based</td>
    <td class="tg-0pky">LIVECell</td>
    <td class="tg-0pky">48.45</td>
    <td class="tg-0pky">47.78</td>
    <td class="tg-0lax"><a href="https://github.com/sartorius-research/LIVECell/blob/main/model/anchor_free/livecell_config.yaml">config</a> | <a href="https://s3.us-west-1.wasabisys.com/resnest/detectron/faster_rcnn_R_50_FPN_syncbn_range-scale_1x-fde56e2b.pth ">model </a> 
  </tr>
    <tr>
    <td class="tg-0pky">LIVECell</td>
    <td class="tg-0pky">48.45</td>
    <td class="tg-0pky">47.78</td>
    <td class="tg-0lax"><a href="https://github.com/sartorius-research/LIVECell/blob/main/model/anchor_free/livecell_config.yaml">config</a> | <a href="https://s3.us-west-1.wasabisys.com/resnest/detectron/faster_rcnn_R_50_FPN_syncbn_range-scale_1x-fde56e2b.pth ">model </a> 
  </tr>
    <tr>
    <td class="tg-0pky">LIVECell</td>
    <td class="tg-0pky">48.45</td>
    <td class="tg-0pky">47.78</td>
    <td class="tg-0lax"><a href="https://github.com/sartorius-research/LIVECell/blob/main/model/anchor_free/livecell_config.yaml">config</a> | <a href="https://s3.us-west-1.wasabisys.com/resnest/detectron/faster_rcnn_R_50_FPN_syncbn_range-scale_1x-fde56e2b.pth ">model </a> 
  </tr>
    <tr>
    <td class="tg-0pky">LIVECell</td>
    <td class="tg-0pky">48.45</td>
    <td class="tg-0pky">47.78</td>
    <td class="tg-0lax"><a href="https://github.com/sartorius-research/LIVECell/blob/main/model/anchor_free/livecell_config.yaml">config</a> | <a href="https://s3.us-west-1.wasabisys.com/resnest/detectron/faster_rcnn_R_50_FPN_syncbn_range-scale_1x-fde56e2b.pth ">model </a> 
  </tr>
    <tr>
    <td class="tg-0pky">LIVECell</td>
    <td class="tg-0pky">48.45</td>
    <td class="tg-0pky">47.78</td>
    <td class="tg-0lax"><a href="https://github.com/sartorius-research/LIVECell/blob/main/model/anchor_free/livecell_config.yaml">config</a> | <a href="https://s3.us-west-1.wasabisys.com/resnest/detectron/faster_rcnn_R_50_FPN_syncbn_range-scale_1x-fde56e2b.pth ">model </a> 
  </tr>
    <tr>
    <td class="tg-0pky">LIVECell</td>
    <td class="tg-0pky">48.45</td>
    <td class="tg-0pky">47.78</td>
    <td class="tg-0lax"><a href="https://github.com/sartorius-research/LIVECell/blob/main/model/anchor_free/livecell_config.yaml">config</a> | <a href="https://s3.us-west-1.wasabisys.com/resnest/detectron/faster_rcnn_R_50_FPN_syncbn_range-scale_1x-fde56e2b.pth ">model </a> 
  </tr>
    <tr>
    <td class="tg-0pky">LIVECell</td>
    <td class="tg-0pky">48.45</td>
    <td class="tg-0pky">47.78</td>
    <td class="tg-0lax"><a href="https://github.com/sartorius-research/LIVECell/blob/main/model/anchor_free/livecell_config.yaml">config</a> | <a href="https://s3.us-west-1.wasabisys.com/resnest/detectron/faster_rcnn_R_50_FPN_syncbn_range-scale_1x-fde56e2b.pth ">model </a> 
  </tr>
    <tr>
    <td class="tg-0pky">LIVECell</td>
    <td class="tg-0pky">48.45</td>
    <td class="tg-0pky">47.78</td>
    <td class="tg-0lax"><a href="https://github.com/sartorius-research/LIVECell/blob/main/model/anchor_free/livecell_config.yaml">config</a> | <a href="https://s3.us-west-1.wasabisys.com/resnest/detectron/faster_rcnn_R_50_FPN_syncbn_range-scale_1x-fde56e2b.pth ">model </a> 
  </tr>


</table>

The box and mask AP presented here is derived by training on either the whole LIVECell dataset or a cell 
cell specific subset, and then evaluated on the corresponding test dataset.

To use our fully trained models download them from our S3 bucket, and use it togheter with appropriate config file as 
described below in the [traing and evaluation section](#Training and evaluation)



# Installation

## Requirements:

- Linux or macOS with Python ≥ 3.6
- PyTorch ≥ 1.3
- torchvision that matches the PyTorch installation. You can install them together at pytorch.org to make sure of this.
- OpenCV, optional, needed by demo and visualization
- pycocotools: pip install cython; pip install -U 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'
## detectron2

Build from source
````python
python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'
````

Or, to install it from a local clone:
````python
git clone https://github.com/facebookresearch/detectron2.git
python -m pip install -e detectron2
````


Or if you are on macOS
````python
CC=clang CXX=clang++ python -m pip install -e detectron2
````


To install a pre-built detectron for different torch and cuda versions and further information, 
see the detectron2 [install document](https://github.com/facebookresearch/detectron2/blob/master/INSTALL.md)

## Model specific instructions

### Anchor-free (centermask2)
Retrive the centermask2 code:
````python
git clone https://github.com/youngwanLEE/centermask2.git
````

For further information, on installation and usage, see the [centermask2 documentation](https://github.com/youngwanLEE/centermask2#evaluation)

### Anchor-based (detectron2-ResNeSt)
Retrive the detectron2-ResNeSt code:
```sh
git clone https://github.com/zhanghang1989/detectron2-ResNeSt.git
```
For further information, on installation and usage, see the [detectron2-ResNeSt documentation](https://github.com/chongruo/detectron2-ResNeSt/blob/resnest/GETTING_STARTED.md)


### Common Installation Issues
- Not compiled with GPU support" or "Detectron2 CUDA Compiler: not available".
```sh
CUDA is not found when building detectron2. You should make sure
python -c 'import torch; from torch.utils.cpp_extension import CUDA_HOME; print(torch.cuda.is_available(), CUDA_HOME)'
print valid outputs at the time you build detectron2.
```

# Training and evaluation
### Register LIVECell dataset
Using a custom dataset such as LIVECell together with the detectron2 code base is done by first registering the dataset
via the detectron2 python API. In practice this can be done adding the following code to the train_net.py file in the cloned
centermask2 repo:

https://detectron2.readthedocs.io/en/latest/tutorials/datasets.html
````python
from detectron2.data.datasets import register_coco_instances
register_coco_instances([dataset_name], {}, [/path/coco/annotations.json], [path/to/image/dir])
````

Were dataset_name will be the name of your dataset and will be how you decide what dataset to use in your config file.
Per default, the config file will point to *TRAIN* and *TEST*, so registering a test dataset as *TEST* will work directly with the
provided config files, for other names, make sure to update your config file accordingly.

- In the config file change the dataset entries with the name used to register the dataset.
- Set the output directory in the config file to save the models and results.

### Train
To train a model, change the OUTPUT directory in the config file to where the models and checkpoints should be saved.
Make sure you follow the previous step and register a TRAIN and TEST dataset, cd into 
the cloned directory (centermask2 or detectron2-ResNeSt), and run the following code:

````python
python tools/train_net.py --num-gpus 8  --config-file your_config.yaml
````
To train a model on the dataset defined in *you_config.yaml* with 8 gpus.
 
 ### Evaluate
 To evaluate a model, make sure to register a TEST dataset and point to it in your config file and cd into 
the cloned directory (centermask2 or detectron2-ResNeSt), 
then run the following code
 ````python
python train_net.py  --config-file your_config.yaml --eval-only MODEL.WEIGHTS /path/to/checkpoint_file.pth
````
This will evaluate a model defined in *your_config.yaml* with the weights saved in */path/to/checkpoint_file.pth*
 
For further details on training, testing and inference, 
visit the [centermask2](https://github.com/youngwanLEE/centermask2#evaluation) or 
[detectron2-ResNeSt](https://github.com/chongruo/detectron2-ResNeSt/blob/resnest/GETTING_STARTED.md) docs
