// Copyright 2016 Max Planck Society
// Distributed under the BSD-3 Software license,
// (See accompanying file LICENSE.txt or copy at
// https://opensource.org/licenses/BSD-3-Clause)

#ifndef SPIXEL_FEATURE_LAYER_HPP_
#define SPIXEL_FEATURE_LAYER_HPP_

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

  template <typename Dtype>
  class SpixelFeatureLayer : public Layer<Dtype> {
   public:
    explicit SpixelFeatureLayer(const LayerParameter& param)
        : Layer<Dtype>(param) {}
    virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
    virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
        const vector<Blob<Dtype>*>& top);
  
    virtual inline const char* type() const { return "SpixelFeature"; }
    virtual inline int ExactNumBottomBlobs() const { return 2; }
  
   protected:
    virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
        const vector<Blob<Dtype>*>& top);
    virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
        const vector<Blob<Dtype>*>& top);
    virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
        const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
    virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
        const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  
    int num_;
    int in_channels_;
    int height_;
    int width_;
    int out_channels_;
    int max_spixels_;
  
    float rgbxy_rgb_scale_;
    float rgbxy_xy_scale_;
    float xy_scale_;
    float rgb_scale_;
    float ignore_idx_value_;
    float ignore_feature_value_;
  
    Blob<Dtype> spixel_counts_;
  };
  
} //namespace caffe
  
#endif  // SPIXEL_FEATURE_LAYER_HPP_
