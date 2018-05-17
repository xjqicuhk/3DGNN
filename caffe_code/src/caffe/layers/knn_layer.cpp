#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv/ml.h>
#include <opencv/highgui.h>
#endif  // USE_OPENCV

#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layers/knn_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

using std::min;
using std::max;

template <typename Dtype>
void KnnLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  KnnParameter knn_param = this->layer_param_.knn_param();
  K_ = knn_param.k();
 
}

template <typename Dtype>
void KnnLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {

  channels_ = bottom[0]->channels();
  height_ = bottom[0]->height();
  width_ = bottom[0]->width();
  num_ = bottom[0]->num();
  
  top[0]->Reshape(bottom[0]->num(), K_, height_,
      width_);
  top[1]->Reshape(bottom[0]->num(), 1, height_,
      width_);
  
}

// TODO(Yangqing): Is there a faster way to do pooling in the channel-first
// case?
template <typename Dtype>
void KnnLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  Dtype* top_data_top = top[1]->mutable_cpu_data();
  //LOG(INFO) << "KnnLayer Forward";
  for(int n = 0; n < num_; ++n)
  {
    //LOG(INFO) << num_;
  	// cv::Mat cv_feature(height_ * width_, channels_, CV_32FC1);
  	CvMat* cv_feature = cvCreateMat(height_ * width_, channels_, CV_32FC1);
  	CvMat* cv_class = cvCreateMat(height_ * width_, 1, CV_32FC1);
  	for(int h = 0; h < height_; ++h)
  	{
  		for(int w = 0; w < width_; ++w)
  		{
  			for(int c = 0; c < channels_; ++c)
  			{   
  				CV_MAT_ELEM( *cv_feature, float, h * width_ + w, c ) = 
  				  bottom_data[n * height_ * width_ * channels_ + c * height_ * width_ + h * width_ + w];
  			}
  			CV_MAT_ELEM( *cv_class, float, h * width_ + w, 0 ) = float(h * width_ + w);
  		}
  	}
   // LOG(INFO) << "start knn";
  	CvKNearest knn(cv_feature, cv_class, 0, false, K_);
  	//cv::Mat cv_neighbour(height_ * width_ , K_, CV_32FC1);
  	CvMat* cv_neighbour = cvCreateMat( height_ * width_, K_, CV_32FC1);
  	//float response;
   // LOG(INFO) << "find knn";
    knn.find_nearest(cv_feature, K_, 0, 0, cv_neighbour, 0);
   // LOG(INFO) << "end knn";
   for(int h = 0; h < height_; ++h)
   {
   	for(int w = 0; w < width_; ++w)
   	{
   		for(int c = 0; c < K_; ++c)
   		{
   			top_data[n * height_ * width_ * K_ + c * height_ * width_ + h * width_ + w] = 
   			    CV_MAT_ELEM( *cv_neighbour, float, h*width_ + w,  c );
   			 Dtype check_value = CV_MAT_ELEM( *cv_neighbour, float, h*width_ + w,  c );
   			 if(std::abs(check_value - static_cast<Dtype>(static_cast<int>(check_value))) > 0.0001)
   			 	LOG(INFO) << "Inavlid neighbpurhood";
   			     //cv_neighbour.at<float>(h*width_ + w, c);
   		}
      top_data_top[n * height_ * width_  + 0 * height_ * width_ + h * width_ + w] = 
      CV_MAT_ELEM( *cv_class, float, h*width_ + w,  0 );
   	}
   }
  cvReleaseMat( &cv_feature );
  cvReleaseMat( &cv_class );
  cvReleaseMat( &cv_neighbour );
  }
   
   
}

template <typename Dtype>
void KnnLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (!propagate_down[0]) {
    return;
  }
  caffe_set(bottom[0]->count(), Dtype(0.0), bottom[0]->mutable_cpu_diff());
}


#ifdef CPU_ONLY
STUB_GPU(KnnLayer);
#endif

INSTANTIATE_CLASS(KnnLayer);
REGISTER_LAYER_CLASS(Knn);

}  // namespace caffe
