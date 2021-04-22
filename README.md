# LIVECell dataset

This document contains instructions of how to access the data associated with the submitted
manuscript "LIVECell - A large-scale dataset for label-free live cell segmentation" by Edlund et. al. 2021.

## Background
Light microscopy is a cheap, accessible, non-invasive modality that when combined with well-established
protocols of two-dimensional cell culture facilitates high-throughput quantitative imaging to study biological
phenomena. Accurate segmentation of individual cells enables exploration of complex biological questions, but
this requires sophisticated imaging processing pipelines due to the low contrast and high object density.
Deep learning-based methods are considered state-of-the-art for most computer vision problems but require vast
amounts of annotated data, for which there is no suitable resource available in the field of label-free cellular
imaging. To address this gap we present LIVECell, a high-quality, manually annotated and expert-validated dataset
that is the largest of its kind to date, consisting of over 1.6 million cells from a diverse set of cell morphologies
and culture densities. To further demonstrate its utility, we provide convolutional neural network-based models
trained and evaluated on LIVECell.

## How to access LIVECell

All images in LIVECell are available following [this link](http://livecell-dataset.s3.eu-central-1.amazonaws.com/LIVECell_dataset_2021/images.zip)  (requires 1.3 GB). Annotations for the different experiments are linked below.

### LIVECell-wide train and evaluate

| Annotation set             | URL           |
| -------------------------- |:-------------:|
| Training set    | [link](http://livecell-dataset.s3.eu-central-1.amazonaws.com/LIVECell_dataset_2021/annotations/LIVECell/livecell_coco_train.json)   |
| Validation set  | [link](http://livecell-dataset.s3.eu-central-1.amazonaws.com/LIVECell_dataset_2021/annotations/LIVECell/livecell_coco_val.json) |
| Test set        | [link](http://livecell-dataset.s3.eu-central-1.amazonaws.com/LIVECell_dataset_2021/annotations/LIVECell/livecell_coco_test.json) |

### Single cell-type experiments


| Cell Type      | Training set  | Validation set | Test set |
| ---------------|:-------------:|:--------------:|:--------:|
| A172           | [link](//livecell-dataset.s3.eu-central-1.amazonaws.com/LIVECell_dataset_2021/annotations/LIVECell_single_cells/a172/train.json) | [link](//livecell-dataset.s3.eu-central-1.amazonaws.com/LIVECell_dataset_2021/annotations/LIVECell_single_cells/a172/val.json) | [link](//livecell-dataset.s3.eu-central-1.amazonaws.com/LIVECell_dataset_2021/annotations/LIVECell_single_cells/a172/test.json) |
| BT474          | [link](//livecell-dataset.s3.eu-central-1.amazonaws.com/LIVECell_dataset_2021/annotations/LIVECell_single_cells/bt474/train.json) | [link](//livecell-dataset.s3.eu-central-1.amazonaws.com/LIVECell_dataset_2021/annotations/LIVECell_single_cells/bt474/val.json) | [link](//livecell-dataset.s3.eu-central-1.amazonaws.com/LIVECell_dataset_2021/annotations/LIVECell_single_cells/bt474/test.json) |
| BV-2           | [link](//livecell-dataset.s3.eu-central-1.amazonaws.com/LIVECell_dataset_2021/annotations/LIVECell_single_cells/bv2/train.json) | [link](//livecell-dataset.s3.eu-central-1.amazonaws.com/LIVECell_dataset_2021/annotations/LIVECell_single_cells/bv2/val.json) | [link](//livecell-dataset.s3.eu-central-1.amazonaws.com/LIVECell_dataset_2021/annotations/LIVECell_single_cells/bv2/test.json) |
| Huh7           | [link](//livecell-dataset.s3.eu-central-1.amazonaws.com/LIVECell_dataset_2021/annotations/LIVECell_single_cells/huh7/train.json) | [link](//livecell-dataset.s3.eu-central-1.amazonaws.com/LIVECell_dataset_2021/annotations/LIVECell_single_cells/huh7/val.json) | [link](//livecell-dataset.s3.eu-central-1.amazonaws.com/LIVECell_dataset_2021/annotations/LIVECell_single_cells/huh7/test.json) |
| MCF7           | [link](//livecell-dataset.s3.eu-central-1.amazonaws.com/LIVECell_dataset_2021/annotations/LIVECell_single_cells/mcf7/train.json) | [link](//livecell-dataset.s3.eu-central-1.amazonaws.com/LIVECell_dataset_2021/annotations/LIVECell_single_cells/mcf7/val.json) | [link](//livecell-dataset.s3.eu-central-1.amazonaws.com/LIVECell_dataset_2021/annotations/LIVECell_single_cells/mcf7/test.json) |
| SH-SHY5Y       | [link](//livecell-dataset.s3.eu-central-1.amazonaws.com/LIVECell_dataset_2021/annotations/LIVECell_single_cells/shsy5y/train.json) | [link](//livecell-dataset.s3.eu-central-1.amazonaws.com/LIVECell_dataset_2021/annotations/LIVECell_single_cells/shsy5y/val.json) | [link](//livecell-dataset.s3.eu-central-1.amazonaws.com/LIVECell_dataset_2021/annotations/LIVECell_single_cells/shsy5y/test.json) |
| SkBr3          | [link](//livecell-dataset.s3.eu-central-1.amazonaws.com/LIVECell_dataset_2021/annotations/LIVECell_single_cells/skbr3/train.json) | [link](//livecell-dataset.s3.eu-central-1.amazonaws.com/LIVECell_dataset_2021/annotations/LIVECell_single_cells/skbr3/val.json) | [link](//livecell-dataset.s3.eu-central-1.amazonaws.com/LIVECell_dataset_2021/annotations/LIVECell_single_cells/skbr3/test.json) |
| SK-OV-3        | [link](//livecell-dataset.s3.eu-central-1.amazonaws.com/LIVECell_dataset_2021/annotations/LIVECell_single_cells/skov3/train.json) | [link](//livecell-dataset.s3.eu-central-1.amazonaws.com/LIVECell_dataset_2021/annotations/LIVECell_single_cells/skov3/val.json) | [link](//livecell-dataset.s3.eu-central-1.amazonaws.com/LIVECell_dataset_2021/annotations/LIVECell_single_cells/skov3/test.json) |


### Dataset size experiments

| Split      | URL   |
| ---------- |:-----:|
| 2 \%       | [link](http://livecell-dataset.s3.eu-central-1.amazonaws.com/LIVECell_dataset_2021/annotations/LIVECell_dataset_size_split/0_train2percent.json) |
| 4 \%       | [link](http://livecell-dataset.s3.eu-central-1.amazonaws.com/LIVECell_dataset_2021/annotations/LIVECell_dataset_size_split/1_train4percent.json)|
| 5 \%       | [link](http://livecell-dataset.s3.eu-central-1.amazonaws.com/LIVECell_dataset_2021/annotations/LIVECell_dataset_size_split/2_train5percent.json)|
| 25 \%      | [link](http://livecell-dataset.s3.eu-central-1.amazonaws.com/LIVECell_dataset_2021/annotations/LIVECell_dataset_size_split/3_train25percent.json)|
| 50 \%      | [link](http://livecell-dataset.s3.eu-central-1.amazonaws.com/LIVECell_dataset_2021/annotations/LIVECell_dataset_size_split/4_train50percent.json)|


### Comparison to fluorescence-based object counts
The images and corresponding json-file with object count per image is available together with the raw fluorescent 
images the counts is based on.

| Cell Type    | Images | Counts | Fluorescent images
| ------------ |:------:|:----------:| :-----: |
| A549         | [link](http://livecell-dataset.s3.eu-central-1.amazonaws.com/LIVECell_dataset_2021/nuclear_count_benchmark/A549.zip) | [link](http://livecell-dataset.s3.eu-central-1.amazonaws.com/LIVECell_dataset_2021/nuclear_count_benchmark/A549_counts.json) | [link](http://livecell-dataset.s3.eu-central-1.amazonaws.com/LIVECell_dataset_2021/nuclear_count_benchmark/A549_fluorescent_images.zip.zip) 
| A172         | [link](http://livecell-dataset.s3.eu-central-1.amazonaws.com/LIVECell_dataset_2021/nuclear_count_benchmark/A172.zip) | [link](http://livecell-dataset.s3.eu-central-1.amazonaws.com/LIVECell_dataset_2021/nuclear_count_benchmark/A172_counts.json) | [link](http://livecell-dataset.s3.eu-central-1.amazonaws.com/LIVECell_dataset_2021/nuclear_count_benchmark/A172_fluorescent_images.zip.zip) 


### Download all of LIVECell

The LIVECell-dataset and trained models is stored in an Amazon Web Services (AWS) S3-bucket. It is easiest to
download the dataset if you have an AWS IAM-user using the
[AWS-CLI](https://docs.aws.amazon.com/cli/latest/userguide/cli-chap-install.html) in the folder
you would like to download the dataset to by simply:
```
aws s3 sync s3://livecell-dataset .
```

If you do not have an AWS IAM-user, the procedure is a little bit more involved. We can use `curl` to make an
HTTP-request to get the S3 XML-response and save to `files.xml`:

```
curl -H "GET /?list-type=2 HTTP/1.1" \
     -H "Host: livecell-dataset.s3.eu-central-1.amazonaws.com" \
     -H "Date: 20161025T124500Z" \
     -H "Content-Type: text/plain" http://livecell-dataset.s3.eu-central-1.amazonaws.com/ > files.xml
```

We then get the urls from files using `grep`:

```
grep -oPm1 "(?<=<Key>)[^<]+" files.xml | sed -e 's/^/http:\/\/livecell-dataset.s3.eu-central-1.amazonaws.com\//' > urls.txt
```

Then download the files you like using `wget`.

## File structure
The top-level structure of the files is arranged like:
```
/livecell-dataset/
    ├── LIVECell_dataset_2021  
    |       ├── annotations/
    |       ├── models/
    |       ├── nuclear_count_benchmark/	
    |       └── images.zip  
    ├── README.md  
    └── LICENSE
```

### LIVECell_dataset_2021/images
The images of the LIVECell-dataset are stored in `/livecell-dataset/LIVECell_dataset_2021/images.zip` along with
their annotations in `/livecell-dataset/LIVECell_dataset_2021/annotations/`.

Within `images.zip` are the training/validation-set and test-set images are completely separate to
facilitate fair comparison between studies. The images require 1.3 GB disk space unzipped and are arranged like:
```
images/
    ├── livecell_test_images
    |       └── <Cell Type>
    |               └── <Cell Type>_Phase_<Well>_<Location>_<Timestamp>_<Crop>.tif
    └── livecell_train_val_images
            └── <Cell Type>
```
Where `<Cell Type>` is each of the eight cell-types in LIVECell (A172, BT474, BV2, Huh7, MCF7, SHSY5Y, SkBr3, SKOV3).
Wells `<Well>` are the location in the 96-well plate used to culture cells, `<Location>` indicates location in the well
where the image was acquired, `<Timestamp>` the time passed since the beginning of the experiment to image acquisition
and `<Crop>` index of the crop of the original larger image. An example image name is `A172_Phase_C7_1_02d16h00m_2.tif`,
which is an image of A172-cells, grown in well C7 where the image is acquired in position 1 two days and 16 hours after
experiment start (crop position 2).

### LIVECell_dataset_2021/annotations/
The annotations of LIVECell are prepared for all tasks along with the training/validation/test splits used for all
experiments in the paper. The annotations require 2.1 GB of disk space and are arranged like:

```
annotations/
    ├── LIVECell
    |       └── livecell_coco_<train/val/test>.json
    ├── LIVECell_single_cells
    |       └── <Cell Type>
    |               └── <train/val/test>.json
    └── LIVECell_dataset_size_split
            └── <Split>_train<Percentage>percent.json
```

*  `annotations/LIVECell` contains the annotations used for the LIVECell-wide train and evaluate task.
*  `annotations/LIVECell_single_cells` contains the annotations used for Single cell type train and evaluate as well
   as the Single cell type transferability tasks.
*  `annotations/LIVECell_dataset_size_split` contains the annotations used to investigate the impact of training set
   scale.

All annotations are in [Microsoft COCO Object Detection-format](https://cocodataset.org/#format-data), and can for
instance be parsed by the Python package [`pycocotools`](https://pypi.org/project/pycocotools/).

### models/
ALL models trained and evaluated for tasks associated with LIVECell are made available for wider use. The models
are trained using [detectron2](https://github.com/facebookresearch/detectron2), Facebook's framework for
object detection and instance segmentation. The models require 15 GB of disk space and are arranged like:

```
models/
   └── Anchor_<free/based>
            ├── ALL/
            |    ├── <Model>.pth
            |    └── config.yaml       
            ├── <Cell Type>/
            |    ├── <Model>.pth
            |    └── config.yaml
            ├── <Base-Config>.yaml
            └── README.md            
```
Where each `<Model>.pth` is a binary file containing the model weights, `<Base-Config>.yaml` the basic Detectron2-config
each model training's `config.yaml` inherits and a README explaining how to set up and use  the model.

### nuclear_count_benchmark/
The images and fluorescence-based object counts are stored as the label-free images in a zip-archive
and the corresponding counts in a json as below:

```
nuclear_count_benchmark/
    ├── A172.zip
    ├── A172_counts.json
    ├── A172_fluorescent_images.zip
    ├── A549.zip
    ├── A549_counts.json 
    └── A549_fluorescent_images.zip
      
```

The json files are on the following format:

```
{
    "<filename>": "<count>"
}
```
Where `<filename>` points to one of the images in the zip-archive, and `<count>` refers to the object count
according fluorescent nuclear labels.

## LICENSE
All images, annotations and models associated with LIVECell are published under
Attribution-NonCommercial 4.0 International ([CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/)) license.