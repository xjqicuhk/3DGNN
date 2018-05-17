#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/interp.hpp"
#include "caffe/layers/interp_layer.hpp"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/highgui/highgui_c.h>
#include <opencv2/imgproc/imgproc.hpp>

namespace caffe {

template <typename Dtype>
void InterpLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  InterpParameter interp_param = this->layer_param_.interp_param();
  if(~interp_param.resize_flag()){
  pad_beg_ = interp_param.pad_beg();
  pad_end_ = interp_param.pad_end();
  CHECK_LE(pad_beg_, 0) << "Only supports non-pos padding (cropping) for now";
  CHECK_LE(pad_end_, 0) << "Only supports non-pos padding (cropping) for now";
  }
}

template <typename Dtype>
void InterpLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  InterpParameter interp_param = this->layer_param_.interp_param();
  num_ = bottom[0]->num();
  channels_ = bottom[0]->channels();
  height_in_ = bottom[0]->height();
  width_in_ = bottom[0]->width();
  //LOG(INFO) << interp_param.resize_flag();
  if(!interp_param.resize_flag()){
  num_ = bottom[0]->num();
  channels_ = bottom[0]->channels();
  height_in_ = bottom[0]->height();
  width_in_ = bottom[0]->width();
  height_in_eff_ = height_in_ + pad_beg_ + pad_end_;
  width_in_eff_ = width_in_ + pad_beg_ + pad_end_;
  InterpParameter interp_param = this->layer_param_.interp_param();
  if (interp_param.has_shrink_factor() &&
      !interp_param.has_zoom_factor()) {
    const int shrink_factor = interp_param.shrink_factor();
    CHECK_GE(shrink_factor, 1) << "Shrink factor must be positive";
    height_out_ = (height_in_eff_ - 1) / shrink_factor + 1;
    width_out_ = (width_in_eff_ - 1) / shrink_factor + 1;
  } else if (interp_param.has_zoom_factor() &&
             !interp_param.has_shrink_factor()) {
    const int zoom_factor = interp_param.zoom_factor();
    CHECK_GE(zoom_factor, 1) << "Zoom factor must be positive";
    height_out_ = height_in_eff_ + (height_in_eff_ - 1) * (zoom_factor - 1);
    width_out_ = width_in_eff_ + (width_in_eff_ - 1) * (zoom_factor - 1);
  } else if (interp_param.has_height() && interp_param.has_width()) {
    height_out_  = interp_param.height();
    width_out_  = interp_param.width();
  } else if (interp_param.has_shrink_factor() &&
             interp_param.has_zoom_factor()) {
    const int shrink_factor = interp_param.shrink_factor();
    const int zoom_factor = interp_param.zoom_factor();
    CHECK_GE(shrink_factor, 1) << "Shrink factor must be positive";
    CHECK_GE(zoom_factor, 1) << "Zoom factor must be positive";
    height_out_ = (height_in_eff_ - 1) / shrink_factor + 1;
    width_out_ = (width_in_eff_ - 1) / shrink_factor + 1;
    height_out_ = height_out_ + (height_out_ - 1) * (zoom_factor - 1);
    width_out_ = width_out_ + (width_out_ - 1) * (zoom_factor - 1);
  } else {
    LOG(FATAL);
  }
  CHECK_GT(height_in_eff_, 0) << "height should be positive";
  CHECK_GT(width_in_eff_, 0) << "width should be positive";
  CHECK_GT(height_out_, 0) << "height should be positive";
  CHECK_GT(width_out_, 0) << "width should be positive";
  top[0]->Reshape(num_, channels_, height_out_, width_out_);}
  else {
   // height_in_ = bottom[0]->height();
   // width_in_ = bottom[0]->width();
    height_out_ = Dtype(height_in_)*interp_param.resize_ratio();
    width_out_ = Dtype(width_in_)*interp_param.resize_ratio();
    //LOG(INFO)<< height_out_ << width_out_;
    top[0]->Reshape(num_, channels_, height_out_, width_out_);
  }
}

template <typename Dtype>
void InterpLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  InterpParameter interp_param = this->layer_param_.interp_param();
  if(!interp_param.resize_flag())
  //const Dtype *label = bottom[0]->cpu_data();
  //for (int i = 0; i<bottom[0]->count();++i)
   // LOG(INFO)<<label[i];
  caffe_cpu_interp2<Dtype,false>(num_ * channels_,
    bottom[0]->cpu_data(), - pad_beg_, - pad_beg_, height_in_eff_, width_in_eff_, height_in_, width_in_,
    top[0]->mutable_cpu_data(), 0, 0, height_out_, width_out_, height_out_, width_out_);
 else 
 {    
      cv::Mat temp_mat(height_in_,width_in_,CV_32FC1);
      cv::Mat temp_out(height_out_,width_out_,CV_32FC1);
      const Dtype *data_in = bottom[0]->cpu_data();
      Dtype *data_out = top[0]->mutable_cpu_data();
      for (int n = 0; n< num_;++n)
      {
        for (int c = 0; c < channels_;++c)
        {
           for(int w = 0 ; w <width_in_; ++w)
           {
            for(int h = 0; h< height_in_; ++h)
            {
              temp_mat.at<float>(h,w) = data_in[((n * channels_+c) * height_in_+ h) * width_in_+ w];
            }
           }
            if(interp_param.resize_method() == 0)
            {cv::resize(temp_mat, temp_out, cv::Size(height_out_, width_out_),0,0,cv::INTER_NEAREST);
             // LOG(INFO)<<"NEAREST";
            }
            else cv::resize(temp_mat, temp_out, cv::Size(height_out_, width_out_));
            for(int w_out = 0; w_out < height_out_; ++w_out)
            {
              for(int h_out = 0; h_out<width_out_; ++h_out)
              {
                data_out[((n * channels_+c) * height_out_+ h_out) * width_out_+ w_out] =  
                        temp_out.at<float>(h_out,w_out);
               }
              }
            }
        }
      }

}

template <typename Dtype>
void InterpLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  InterpParameter interp_param = this->layer_param_.interp_param();
  if(interp_param.resize_flag()) {return; }
  if (!propagate_down[0]) { return; }
  caffe_set(bottom[0]->count(), Dtype(0), bottom[0]->mutable_cpu_diff());
  caffe_cpu_interp2_backward<Dtype,false>(num_ * channels_,
    bottom[0]->mutable_cpu_diff(), - pad_beg_, - pad_beg_, height_in_eff_, width_in_eff_, height_in_, width_in_,
    top[0]->cpu_diff(), 0, 0, height_out_, width_out_, height_out_, width_out_);
}

#ifndef CPU_ONLY
template <typename Dtype>
void InterpLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  //const Dtype *label = bottom[0]->cpu_data();
  //for (int i = 0; i<bottom[0]->count();++i)
   // LOG(INFO)<<label[i];
   InterpParameter interp_param = this->layer_param_.interp_param();
  if(interp_param.resize_flag()) {Forward_cpu(bottom, top); return;}
  caffe_gpu_interp2<Dtype,false>(num_ * channels_,
    bottom[0]->gpu_data(), - pad_beg_, - pad_beg_, height_in_eff_, width_in_eff_, height_in_, width_in_,
    top[0]->mutable_gpu_data(), 0, 0, height_out_, width_out_, height_out_, width_out_);
 // const Dtype *label = top[0]->cpu_data();
  //for (int i = 0; i<top[0]->count();++i)
  //  LOG(INFO)<<label[i];

}

template <typename Dtype>
void InterpLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
   InterpParameter interp_param = this->layer_param_.interp_param();
  if(interp_param.resize_flag()) { return;}
  if (!propagate_down[0]) { return; }
  caffe_gpu_set(bottom[0]->count(), Dtype(0), bottom[0]->mutable_gpu_diff());
  caffe_gpu_interp2_backward<Dtype,false>(num_ * channels_,
    bottom[0]->mutable_gpu_diff(), - pad_beg_, - pad_beg_, height_in_eff_, width_in_eff_, height_in_, width_in_,
    top[0]->gpu_diff(), 0, 0, height_out_, width_out_, height_out_, width_out_);
}
#endif

#ifdef CPU_ONLY
STUB_GPU(InterpLayer);
#endif

INSTANTIATE_CLASS(InterpLayer);
REGISTER_LAYER_CLASS(Interp);

}  // namespace caffe
