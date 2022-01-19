# Usage

The anchor free models used in this benchmark is based on the [centermask2](https://github.com/youngwanLEE/centermask2#evaluation) architecture and the anchor based models are 
based on the [detectron2-ResNeSt](https://github.com/chongruo/detectron2-ResNeSt/blob/resnest/GETTING_STARTED.md) architectures, 
which is both built upon the [detectron2](https://github.com/facebookresearch/detectron2) library.

The models in the LIVECell paper was trained in on 8 Nvidia V100 GPUS.
To help others reproduce our results and use the models for further research, we provide pre-trained models and config files.

<table class="tg">
  <tr>
    <th class="tg-0pky">Architecture</th>
    <th class="tg-0pky">Dataset</th>
    <th class="tg-0pky">Box mAP%</th>
    <th class="tg-0pky">Mask mAP%</th>
    <th class="tg-0pky">download</th>
  </tr>
  <tr>
    <td rowspan="9" class="tg-0pky">Anchor free</td>
    <td class="tg-0pky">LIVECell</td>
    <td class="tg-0pky">48.45</td>
    <td class="tg-0pky">47.78</td>
    <td class="tg-0lax"><a href="https://github.com/sartorius-research/LIVECell/blob/main/model/anchor_free/livecell_config.yaml">config</a> | <a href="http://livecell-dataset.s3.eu-central-1.amazonaws.com/LIVECell_dataset_2021/models/Anchor_free/ALL/LIVECell_anchor_free_model.pth">model </a> 
  </tr>
  <tr>
    <td class="tg-0pky">A172</td>
    <td class="tg-0pky">31.49</td>
    <td class="tg-0pky">34.57</td>
    <td class="tg-0lax"><a href="https://github.com/sartorius-research/LIVECell/blob/main/model/anchor_free/a172_config.yaml">config</a> | <a href="https://livecell-dataset.s3.eu-central-1.amazonaws.com/LIVECell_dataset_2021/models/Anchor_free/A172/LIVECell_anchor_free_a172_model.pth">model </a> 
  </tr>
   <tr>
    <td class="tg-0pky">BT-474</td>
    <td class="tg-0pky">42.12</td>
    <td class="tg-0pky">42.60</td>
    <td class="tg-0lax"><a href="https://github.com/sartorius-research/LIVECell/blob/main/model/anchor_free/bt474_config.yaml">config</a> | <a href="https://livecell-dataset.s3.eu-central-1.amazonaws.com/LIVECell_dataset_2021/models/Anchor_free/BT474/LIVECell_anchor_free_bt474_model.pth ">model </a> 
  </tr>
  <tr>
    <td class="tg-0pky">BV-2</td>
    <td class="tg-0pky">42.62</td>
    <td class="tg-0pky">45.69</td>
    <td class="tg-0lax"><a href="https://github.com/sartorius-research/LIVECell/blob/main/model/anchor_free/bv2_config.yaml">config</a> | <a href="https://livecell-dataset.s3.eu-central-1.amazonaws.com/LIVECell_dataset_2021/models/Anchor_free/BV2/LIVECell_anchor_free_bv2_model.pth">model </a> 
  </tr>
   <tr>
    <td class="tg-0pky">Huh7</td>
    <td class="tg-0pky">42.44</td>
    <td class="tg-0pky">45.85</td>
    <td class="tg-0lax"><a href="https://github.com/sartorius-research/LIVECell/blob/main/model/anchor_free/huh7_config.yaml">config</a> | <a href="https://livecell-dataset.s3.eu-central-1.amazonaws.com/LIVECell_dataset_2021/models/Anchor_free/HUH7/LIVECell_anchor_free_huh7_model.pth">model </a> 
  </tr>
  <tr>
    <td class="tg-0pky">MCF7</td>
    <td class="tg-0pky">36.53</td>
    <td class="tg-0pky">37.30 </td>
    <td class="tg-0lax"><a href="https://github.com/sartorius-research/LIVECell/blob/main/model/anchor_free/mcf7_config.yaml">config</a> | <a href="https://livecell-dataset.s3.eu-central-1.amazonaws.com/LIVECell_dataset_2021/models/Anchor_free/MCF7/LIVECell_anchor_free_mcf7_model.pth">model </a> 
  </tr>
  <tr>
    <td class="tg-0pky">SH-SY5Y</td>
    <td class="tg-0pky">25.20</td>
    <td class="tg-0pky">23.91</td>
    <td class="tg-0lax"><a href="https://github.com/sartorius-research/LIVECell/blob/main/model/anchor_free/shsy5y_config.yaml">config</a> | <a href="https://livecell-dataset.s3.eu-central-1.amazonaws.com/LIVECell_dataset_2021/models/Anchor_free/SHSY5Y/LIVECell_anchor_free_shsy5y_model.pth">model </a>
  </tr>
  <tr>
    <td class="tg-0pky">SkBr3</td>
    <td class="tg-0pky">64.35</td>
    <td class="tg-0pky">65.85</td>
    <td class="tg-0lax"><a href="https://github.com/sartorius-research/LIVECell/blob/main/model/anchor_free/skbr3_config.yaml">config</a> | <a href="https://livecell-dataset.s3.eu-central-1.amazonaws.com/LIVECell_dataset_2021/models/Anchor_free/SKBR3/LIVECell_anchor_free_skbr3_model.pth">model </a>
  </tr>
  <tr>
    <td class="tg-0pky">SK-OV-3</td>
    <td class="tg-0pky">46.43</td>
    <td class="tg-0pky">49.39</td>
    <td class="tg-0lax"><a href="https://github.com/sartorius-research/LIVECell/blob/main/model/anchor_free/skov3_config.yaml">config</a> | <a href="https://livecell-dataset.s3.eu-central-1.amazonaws.com/LIVECell_dataset_2021/models/Anchor_free/SKOV3/LIVECell_anchor_free_skov3_model.pth">model </a>
  </tr>
  
   <tr>
    <td rowspan="9" class="tg-0pky">Anchor based</td>
    <td class="tg-0pky">LIVECell</td>
    <td class="tg-0pky">48.43</td>
    <td class="tg-0pky">47.89</td>
    <td class="tg-0lax"><a href="https://github.com/sartorius-research/LIVECell/blob/main/model/anchor_based/livecell_config.yaml">config</a> | <a href="https://livecell-dataset.s3.eu-central-1.amazonaws.com/LIVECell_dataset_2021/models/Anchor_based/ALL/LIVECell_anchor_based_model.pth">model </a>
  </tr>
  <tr>
    <td class="tg-0pky">A172</td>
    <td class="tg-0pky">36.37</td>
    <td class="tg-0pky">38.02</td>
    <td class="tg-0lax"><a href="https://github.com/sartorius-research/LIVECell/blob/main/model/anchor_based/a172_config.yaml">config</a> | <a href="https://livecell-dataset.s3.eu-central-1.amazonaws.com/LIVECell_dataset_2021/models/Anchor_based/A172/LIVECell_anchor_based_a172_model.pth">model </a> 
  </tr>
   <tr>
    <td class="tg-0pky">BT-474</td>
    <td class="tg-0pky">43.25</td>
    <td class="tg-0pky">43.00</td>
    <td class="tg-0lax"><a href="https://github.com/sartorius-research/LIVECell/blob/main/model/anchor_based/bt474_config.yaml">config</a> | <a href="https://livecell-dataset.s3.eu-central-1.amazonaws.com/LIVECell_dataset_2021/models/Anchor_based/BT474/LIVECell_anchor_based_bt474_model.pth">model </a> 
  </tr>
  <tr>
    <td class="tg-0pky">BV-2</td>
    <td class="tg-0pky">54.36</td>
    <td class="tg-0pky">52.60</td>
    <td class="tg-0lax"><a href="https://github.com/sartorius-research/LIVECell/blob/main/model/anchor_based/bv2_config.yaml">config</a> | <a href="https://livecell-dataset.s3.eu-central-1.amazonaws.com/LIVECell_dataset_2021/models/Anchor_based/BV2/LIVECell_anchor_based_bv2_model.pth">model </a> 
  </tr>
   <tr>
    <td class="tg-0pky">Huh7</td>
    <td class="tg-0pky">52.79</td>
    <td class="tg-0pky">51.83</td>
    <td class="tg-0lax"><a href="https://github.com/sartorius-research/LIVECell/blob/main/model/anchor_based/huh7_config.yaml">config</a> | <a href="https://livecell-dataset.s3.eu-central-1.amazonaws.com/LIVECell_dataset_2021/models/Anchor_based/HUH7/LIVECell_anchor_based_huh7_model.pth">model </a> 
  </tr>
  <tr>
    <td class="tg-0pky">MCF7</td>
    <td class="tg-0pky">37.53</td>
    <td class="tg-0pky">37.94 </td>
    <td class="tg-0lax"><a href="https://github.com/sartorius-research/LIVECell/blob/main/model/anchor_based/mcf7_config.yaml">config</a> | <a href="https://livecell-dataset.s3.eu-central-1.amazonaws.com/LIVECell_dataset_2021/models/Anchor_based/MCF7/LIVECell_anchor_based_mcf7_model.pth">model </a> 
  </tr>
  <tr>
    <td class="tg-0pky">SH-SY5Y</td>
    <td class="tg-0pky">27.87</td>
    <td class="tg-0pky">24.92</td>
    <td class="tg-0lax"><a href="https://github.com/sartorius-research/LIVECell/blob/main/model/anchor_based/shsy5y_config.yaml">config</a> | <a href="https://livecell-dataset.s3.eu-central-1.amazonaws.com/LIVECell_dataset_2021/models/Anchor_based/SHSY5Y/LIVECell_anchor_based_shsy5y_model.pth">model </a> 
  </tr>
  <tr>
    <td class="tg-0pky">SkBr3</td>
    <td class="tg-0pky">64.41</td>
    <td class="tg-0pky">65.39</td>
    <td class="tg-0lax"><a href="https://github.com/sartorius-research/LIVECell/blob/main/model/anchor_based/skbr3_config.yaml">config</a> | <a href="https://livecell-dataset.s3.eu-central-1.amazonaws.com/LIVECell_dataset_2021/models/Anchor_based/SKBR3/LIVECell_anchor_based_skbr3_model.pth">model </a> 
  </tr>
  <tr>
    <td class="tg-0pky">SK-OV-3</td>
    <td class="tg-0pky">53.29</td>
    <td class="tg-0pky">54.12</td>
    <td class="tg-0lax"><a href="https://github.com/sartorius-research/LIVECell/blob/main/model/anchor_based/skov3_config.yaml">config</a> | <a href="https://livecell-dataset.s3.eu-central-1.amazonaws.com/LIVECell_dataset_2021/models/Anchor_based/SKOV3/LIVECell_anchor_based_skov3_model.pth">model </a> 
  </tr>
</table>

The box and mask AP presented here is derived by training on either the whole LIVECell dataset or a cell 
cell specific subset, and then evaluated on the corresponding test dataset.

To use our fully trained models download them from our S3 bucket, and use it together with appropriate config file as 
described below in the [traing and evaluation section](#Training and evaluation)



# Installation

The installation takes approximately 30 minutes 
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
git clone https://github.com/chongruo/detectron2-ResNeSt
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

To fine-tune a model on your own dataset, set MODEL.WEIGTS in the config file to point at one of our weight files,
if you want to finetune our centermask2 model for instance.
````python
MODEL:
  WEIGHTS: "http://livecell-dataset.s3.eu-central-1.amazonaws.com/LIVECell_dataset_2021/models/Anchor_free/ALL/LIVECell_anchor_free_model.pth"
````
 
 ### Evaluate
To evaluate a model, make sure to register a TEST dataset and point to it in your config file and cd into 
the cloned directory (centermask2 or detectron2-ResNeSt), then run the following code
 ````python
python train_net.py  --config-file <your_config.yaml> --eval-only MODEL.WEIGHTS </path/to/checkpoint_file.pth>
````

This will evaluate a model defined in `your_config.yaml` with the weights saved in `/path/to/checkpoint_file.pth`

To evaluate one of our models, like the centermask2 (anchor-free), you can point directly at the URI link for the weight 
file.


 ````python
python train_net.py  --config-file livecell_config.yaml --eval-only MODEL.WEIGHTS http://livecell-dataset.s3.eu-central-1.amazonaws.com/LIVECell_dataset_2021/models/Anchor_free/ALL/LIVECell_anchor_free_model.pth
````

#### Evaluation script
The original evaluation script available in the centermask and detectron2 repo is based on there being no more than 100
detections in an image. In our case we can have thousands of annotations and thus the AP evaluation will be off. We 
therefore provide `coco_evaluation.py` evaluation script in the [code](../code) folder. \

To use this script, go into the `train_net.py` file and remove (or comment out) the current import of `COCOEvaluator`.
Then import `COCOEvaluator` for from the provided `coco_evaluator.py` file instead. This will result in AP evaluation
supporting for up to 2000 instances in one image.
 
The evaluation script will take approximately 30 minutes to run on our test dataset with a tesla V100 GPU.
The output of the evaluation will appear in the terminal, begining with information about the environment, data and 
architecture used. Then it will start evaluating all the images and summerize the results in the following manner:
 
````python
.
.
.
[32m[11/18 17:19:06 d2.evaluation.evaluator]: [0mInference done 1557/1564. 0.1733 s / img. ETA=0:00:06
[32m[11/18 17:19:11 d2.evaluation.evaluator]: [0mInference done 1561/1564. 0.1734 s / img. ETA=0:00:02
[32m[11/18 17:19:14 d2.evaluation.evaluator]: [0mTotal inference time: 0:22:23.437057 (0.861730 s / img per device, on 1 devices)
[32m[11/18 17:19:14 d2.evaluation.evaluator]: [0mTotal inference pure compute time: 0:04:30 (0.173426 s / img per device, on 1 devices)
Loading and preparing results...
DONE (t=1.12s)
creating index...
index created!
Size parameters: [[0, 10000000000.0], [0, 324], [324, 961], [961, 10000000000.0]]
Running per image evaluation...
Evaluate annotation type *bbox*
COCOeval_opt.evaluate() finished in 119.67 seconds.
Accumulating evaluation results...
COCOeval_opt.accumulate() finished in 5.86 seconds.
In method
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=2000 ] = 0.485
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=2000 ] = 0.830
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=2000 ] = 0.504
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=2000 ] = 0.483
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=2000 ] = 0.494
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=2000 ] = 0.507
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.212
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=500 ] = 0.480
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=2000 ] = 0.569
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=2000 ] = 0.531
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=2000 ] = 0.602
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=2000 ] = 0.672
Loading and preparing results...
DONE (t=11.04s)
creating index...
index created!
Size parameters: [[0, 10000000000.0], [0, 324], [324, 961], [961, 10000000000.0]]
Running per image evaluation...
Evaluate annotation type *segm*
COCOeval_opt.evaluate() finished in 135.80 seconds.
Accumulating evaluation results...
COCOeval_opt.accumulate() finished in 5.78 seconds.
In method
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=2000 ] = 0.478
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=2000 ] = 0.816
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=2000 ] = 0.509
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=2000 ] = 0.451
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=2000 ] = 0.491
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=2000 ] = 0.570
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.210
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=500 ] = 0.470
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=2000 ] = 0.547
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=2000 ] = 0.516
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=2000 ] = 0.565
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=2000 ] = 0.649
[32m[11/18 17:25:07 d2.engine.defaults]: [0mEvaluation results for cell_phase_test in csv format:
[32m[11/18 17:25:07 d2.evaluation.testing]: [0mcopypaste: Task: bbox
[32m[11/18 17:25:07 d2.evaluation.testing]: [0mcopypaste: AP,AP50,AP75,APs,APm,APl
[32m[11/18 17:25:07 d2.evaluation.testing]: [0mcopypaste: 48.4529,82.9806,50.4426,48.3240,49.4476,50.7434
[32m[11/18 17:25:07 d2.evaluation.testing]: [0mcopypaste: Task: segm
[32m[11/18 17:25:07 d2.evaluation.testing]: [0mcopypaste: AP,AP50,AP75,APs,APm,APl
[32m[11/18 17:25:07 d2.evaluation.testing]: [0mcopypaste: 47.7810,81.6260,50.8958,45.1110,49.0684,56.9874

````
 
For further details on training, testing and inference, 
visit the [centermask2](https://github.com/youngwanLEE/centermask2#evaluation) or 
[detectron2-ResNeSt](https://github.com/chongruo/detectron2-ResNeSt/blob/resnest/GETTING_STARTED.md) docs

## One-shot usage
For LIVECell experiments with zero-shot learning of EVICAN and Cellpose the input images was preprocessed using the 
preprocessing-script preprocessing.py found under the [code folder](../code).
