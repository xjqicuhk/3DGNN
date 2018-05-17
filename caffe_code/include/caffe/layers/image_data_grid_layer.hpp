#ifndef CAFFE_IMAGE_SEG_DATA_LAYER_HPP_
#define CAFFE_IMAGE_SEG_DATA_LAYER_HPP_

#include <string>
#include <utility>
#include <vector>
#include <queue>

#include "caffe/blob.hpp"
#include "caffe/data_transformer.hpp"
#include "caffe/internal_thread.hpp"
#include "caffe/layer.hpp"
#include "caffe/layers/base_data_layer.hpp"
#include "caffe/proto/caffe.pb.h"


namespace caffe {

// Add 04282016
template<typename Dtype>
class ImageDataGridLayer : public ImageDimPrefetchingDataLayer<Dtype> {
public:
   explicit ImageDataGridLayer(const LayerParameter& param)
       : ImageDimPrefetchingDataLayer<Dtype>(param){}
   virtual ~ImageDataGridLayer();
   virtual void DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
       const vector<Blob<Dtype>*>& top);
  
   virtual inline const char* type() const {return "ImageDataGrid";}
   virtual inline int ExactNumBottomBlobs() const {return 0;}
   virtual inline int MaxTopBlobs() const { return 3;}
   virtual inline int MinTopBlobs() const { return 2;}
   virtual inline bool AutoTopBlobs() const {return true;}
protected:
   virtual void InternalThreadEntry();
   virtual void PrepareSubimgData();

   vector<std::string> lines_img_;
   vector<std::string> lines_label_;
   bool has_label_;
   int lines_id_;
   //Blob<Dtype> prefetch_data_;
   //Blob<Dtype> prefetch_label_;
   //Blob<Dtype> prefetch_dim_;
   std::queue< pair<cv::Mat, vector<int> > > data_grids_;
   std::queue<cv::Mat> scaleMat_;
   int height_, width_, stride_, crop_size_;
   int h_grid_, w_grid_, subimg_num_, subimg_alliter_, subimg_iter_, subimg_rem_;
   Dtype stride_rate_;
   // support mutiscale test
   int test_scale_num_, test_scale_iter_;
   cv::Mat cv_img_, cv_seg_, new_cv_img_;
   bool isFirstReadImage_, isFirstScaleImage_;
   int new_height_, new_width_;
};

}  // namespace caffe

#endif  // CAFFE_IMAGE_SEG_DATA_LAYER_HPP_
