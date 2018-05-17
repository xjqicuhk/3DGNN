# 3DGNN
This is a Caffe implementation of 3DGNN 

<img src="./overallpipeline.png"/>

## Setup

### Requirement
Required CUDA (7.0) + Ubuntu14.04.

### Installation

For installation, please follow the instructions of [Caffe](https://github.com/BVLC/caffe) and [DeepLab v2](https://bitbucket.org/aquariusjay/deeplab-public-ver2).

### Usage

1. Clone the repository:

2. Build Caffe and matcaffe:

   ```shell
   cd caffe_code
   vim Makefile.config
   make -j8 && make matcaffe
   ```

3. Inference:

   - Evaluation code is in folder 'matlabscript'. 
   - Download trained models and unzip it. Pretrained model is saved in folder "model/nyu_40/". 
   ```shell
   cd matlabscript
    run nyu_crop_data_mask_msc.m
   ```

4. Training
```shell
   cd model
   sh run_train.sh
   ```
## Citation
If you use our code for research, please cite our paper:

Xiaojuan Qi, Renjie Liao, Jiaya Jia, Sanja Fidler and Raquel Urtasun. 3D Graph Neural Network for RGBD Semantic Segmentation. In ICCV 2017.

## Question
If you have any question or request about the code and data, please email me at qxj0125@gmail.com . If you need more information for other datasets plesase send email. 

## License
MIT License
