# 3DGNN
This is the Caffe implementation of [3D Graph Neural Networks for RGBD Semantic Segmentation](http://openaccess.thecvf.com/content_ICCV_2017/papers/Qi_3D_Graph_Neural_ICCV_2017_paper.pdf): 

<img src="./overallpipeline.png"/>

## Setup

### Requirement
Required CUDA (7.0) + Ubuntu14.04.

### Installation

For installation, please follow the instructions of [Caffe](https://github.com/BVLC/caffe) and [DeepLab v2](https://bitbucket.org/aquariusjay/deeplab-public-ver2).

### Data Preparation
1. Download the trained model (https://mycuhk-my.sharepoint.com/:u:/g/personal/1155051740_link_cuhk_edu_hk/ETsf3ekhGbxOp1xYKJxv2hQB8I5OCCui86QLvWvK65_5sw?e=KThQe9).
2. Download the prepared training data (prepared hdf5 data) (https://mycuhk-my.sharepoint.com/:u:/g/personal/1155051740_link_cuhk_edu_hk/EVGJ_xXvtNVCh7spid94AmQB_byhW49i-VH_vqx8oZbrZQ?e=COhKwr).
3. Download the testing data  (https://mycuhk-my.sharepoint.com/:u:/g/personal/1155051740_link_cuhk_edu_hk/EVdjeNQqnINOj359HN8WXDgBsouAqSoZC1lRgkSbPNo2hA?e=e0w2sO).
4. Download the original provided data (https://mycuhk-my.sharepoint.com/:u:/g/personal/1155051740_link_cuhk_edu_hk/EZuJHYVcULRNkQ3qm34ugIoBg-69Vprq2POiaat4u5ZLXQ?e=QmWXec).

### Usage

1. Clone the repository.

2. Build Caffe and matcaffe:

   ```shell
   cd caffe_code
   make -j8 && make matcaffe
   ```

3. Inference:

   - Evaluation code is in folder 'matlabscript'. 
   - Download trained models and unzip it. Pretrained model is saved in folder "model/nyu_40/". 
   ```shell
   cd matlabscript
   run nyu_crop_data_mask_msc.m
   ```
   - The result is saved in folder "../result/nyu_40_msc/"
4. Training:

   - Training data preparation
   ```shell
       cd matlabscript
       run generatedata (setting training = true)
       cd ..
       cd train_data_hdf5_file_generate
       python generate_hdf5
       cd ..
      ```
      We have also provided the training data in folder "traindata/"
   - Run caffe training


   
## Citation
If you use our code for research, please cite our paper:

```
@inproceedings{qi20173d,
  title={3D Graph Neural Networks for RGBD Semantic Segmentation},
  author={Qi, Xiaojuan and Liao, Renjie and Jia, Jiaya and Fidler, Sanja and Urtasun, Raquel},
  booktitle={ICCV},
  year={2017}
}
```

## Question
If you have any question or request about the code and data, please email me at qxj0125@gmail.com . If you need more information for other datasets plesase send email. 

## License
MIT License
