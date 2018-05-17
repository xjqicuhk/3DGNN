#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layers/pooling_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
__global__ void MaxPoolForward(const int nthreads,
    const Dtype* const bottom_data, const int num, const int channels,
    const int height, const int width, const int pooled_height,
    const int pooled_width, const int kernel_h, const int kernel_w,
    const int stride_h, const int stride_w, const int pad_h, const int pad_w,
    Dtype* const top_data, int* mask, Dtype* top_mask) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    const int pw = index % pooled_width;
    const int ph = (index / pooled_width) % pooled_height;
    const int c = (index / pooled_width / pooled_height) % channels;
    const int n = index / pooled_width / pooled_height / channels;
    int hstart = ph * stride_h - pad_h;
    int wstart = pw * stride_w - pad_w;
    const int hend = min(hstart + kernel_h, height);
    const int wend = min(wstart + kernel_w, width);
    hstart = max(hstart, 0);
    wstart = max(wstart, 0);
    Dtype maxval = -FLT_MAX;
    int maxidx = -1;
    const Dtype* const bottom_slice =
        bottom_data + (n * channels + c) * height * width;
    for (int h = hstart; h < hend; ++h) {
      for (int w = wstart; w < wend; ++w) {
        if (bottom_slice[h * width + w] > maxval) {
          maxidx = h * width + w;
          maxval = bottom_slice[maxidx];
        }
      }
    }
    top_data[index] = maxval;
    if (mask) {
      mask[index] = maxidx;
    } else {
      top_mask[index] = maxidx;
    }
  }
}



template <typename Dtype>
__global__ void AvePoolForward(const int nthreads,
    const Dtype* const bottom_data, const int num, const int channels,
    const int height, const int width, const int pooled_height,
    const int pooled_width, const int kernel_h, const int kernel_w,
    const int stride_h, const int stride_w, const int pad_h, const int pad_w,
    Dtype* const top_data) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    const int pw = index % pooled_width;
    const int ph = (index / pooled_width) % pooled_height;
    const int c = (index / pooled_width / pooled_height) % channels;
    const int n = index / pooled_width / pooled_height / channels;
    int hstart = ph * stride_h - pad_h;
    int wstart = pw * stride_w - pad_w;
    int hend = min(hstart + kernel_h, height + pad_h);
    int wend = min(wstart + kernel_w, width + pad_w);
    const int pool_size = (hend - hstart) * (wend - wstart);
    hstart = max(hstart, 0);
    wstart = max(wstart, 0);
    hend = min(hend, height);
    wend = min(wend, width);
    Dtype aveval = 0;
    const Dtype* const bottom_slice =
        bottom_data + (n * channels + c) * height * width;
    for (int h = hstart; h < hend; ++h) {
      for (int w = wstart; w < wend; ++w) {
        aveval += bottom_slice[h * width + w];
      }
    }
    top_data[index] = aveval / pool_size;
  }
}



template <typename Dtype>
__global__ void WeightAvePoolForward(const int nthreads,
    const Dtype* const bottom_data, const Dtype* const bottom_feature, const int num, const int channels,
    const int height, const int width, const int pooled_height,
    const int pooled_width, const int kernel_h, const int kernel_w,
    const int stride_h, const int stride_w, const int pad_h, const int pad_w, const int feature_length,
    Dtype* const top_data, Dtype* top_sum) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    const int pw = index % pooled_width;
    const int ph = (index / pooled_width) % pooled_height;
    const int c = (index / pooled_width / pooled_height) % channels;
    const int n = index / pooled_width / pooled_height / channels;
    int hstart = ph * stride_h - pad_h;
    int wstart = pw * stride_w - pad_w;
    int hend = min(hstart + kernel_h, height + pad_h);
    int wend = min(wstart + kernel_w, width + pad_w);
    //const int pool_size = (hend - hstart) * (wend - wstart);
    hstart = max(hstart, 0);
    wstart = max(wstart, 0);
    hend = min(hend, height);
    wend = min(wend, width);
    Dtype aveval = 0;
    Dtype similarity = 0.0;
    int n_index = 0;
    Dtype sum_value = 0.0;

    
    const Dtype* const bottom_slice =
        bottom_data + (n * channels + c) * height * width;
    for (int h = hstart; h < hend; ++h) {
      for (int w = wstart; w < wend; ++w) {
        // calculate simlarity
         similarity = 0 ;
        for(int l = 0 ; l < feature_length; ++ l)
        {   
            n_index = n * feature_length * height * width + l * height * width;
            similarity += fabs(bottom_feature[n_index + ph * width + pw] - bottom_feature[n_index + h * height + w]);
        }

        aveval += bottom_slice[h * width + w] * expf(-1.0 * similarity);
        sum_value = sum_value + expf(-1.0*similarity);
      }
    }
    top_data[index] = aveval / sum_value;
    top_sum[n* height * width + ph * width + pw] = sum_value;
  }
}



template <typename Dtype>
__global__ void MaskAvePoolForward(const int nthreads,
    const Dtype* const bottom_data, const Dtype* const bottom_mask, const int num, const int channels,
    const int height, const int width, const int pooled_height,
    const int pooled_width, const int kernel_h, const int kernel_w,
    const int stride_h, const int stride_w, const int pad_h, const int pad_w,
    Dtype* const top_data, Dtype* const top_sum) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    const int pw = index % pooled_width;
    const int ph = (index / pooled_width) % pooled_height;
    const int c = (index / pooled_width / pooled_height) % channels;
    const int n = index / pooled_width / pooled_height / channels;

    int hstart = ph * stride_h - pad_h;
    int wstart = pw * stride_w - pad_w;
   
    int hend = min(hstart + kernel_h, height + pad_h);
    int wend = min(wstart + kernel_w, width + pad_w);
    //const int pool_size = (hend - hstart) * (wend - wstart);
    Dtype cache_value = 0 ;
    hstart = max(hstart, 0);
    wstart = max(wstart, 0);
    hend = min(hend, height);
    wend = min(wend, width);
    Dtype aveval = 0;
    int w_index = index % width;
    int h_index = (index / width) % height;
    const Dtype* const bottom_slice =
        bottom_data + (n * channels + c) * height * width;
    const Dtype*  const mask_slice = bottom_mask + n * height * width * kernel_h * kernel_w + h_index * width + w_index;
    int center_h = (kernel_h - 1) / 2;
    int center_w = (kernel_w - 1) / 2;
    for (int h = hstart; h < hend; ++h) {
      for (int w = wstart; w < wend; ++w) {
      	int channel_relative = (h- h_index + center_h) * kernel_w + (w - w_index + center_w);
        Dtype tmp = mask_slice[ channel_relative * (height * width)];
        //Dtype tmp = 0.0;
        aveval += bottom_slice[h * width + w] * tmp;
        cache_value += tmp;
      }
    }
    //top_data[index] = aveval / cache_value;
    //if(cache_value < 1e-3) cache_value += 1e-3;
    top_data[index] = aveval / cache_value;
    //top_sum[n * height * width + index % (height * width)] = cache_value;
    top_sum[n * height * width + index % (height * width)] = cache_value;
    
  }
}


template <typename Dtype>
__global__ void KNNPoolForward(const int nthreads,
    const Dtype* const bottom_data, const Dtype* const bottom_mask, const int num, const int channels,
    const int height, const int width, const int pooled_height,
    const int pooled_width, const int neighbour_num,
    Dtype* const top_data) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    const int pw = index % pooled_width;
    const int ph = (index / pooled_width) % pooled_height;
    const int c = (index / pooled_width / pooled_height) % channels;
    const int n = index / pooled_width / pooled_height / channels;


    Dtype cache_value = 0 ;
    Dtype aveval = 0;
    const Dtype* const bottom_slice =
        bottom_data + (n * channels + c) * height * width;
    const Dtype*  const mask_slice = bottom_mask + n * height * width * neighbour_num;
    for (int k = 0; k < neighbour_num; ++k) {
        int tmp = static_cast<int>(mask_slice[k * height * width + ph * width + pw]);
        aveval += bottom_slice[tmp];
        cache_value += 1;   
    }
    
    top_data[index] = aveval / cache_value;
  }
}
template <typename Dtype>
__global__ void KNNsimForward(const int nthreads,
    const Dtype* const bottom_mask, const Dtype* const bottom_feature, 
    const int num, const int channels,
    const int height, const int width, const int pooled_height,
    const int pooled_width, const int neighbour_num, const int feature_num, const int weight_type,
    Dtype* const top_bottom_sim) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    const int pw = index % pooled_width;
    const int ph = (index / pooled_width) % pooled_height;
    const int c = (index / pooled_width / pooled_height) % neighbour_num;
    const int n = index / pooled_width / pooled_height / neighbour_num;


    //Dtype cache_value = 0 ;
    Dtype aveval = 0;
    const Dtype* const feature_slice1 =
        bottom_feature + (n * feature_num) * height * width + ph * width + pw;
    const Dtype * const feature_slice2 = bottom_feature +  n * feature_num * height * width;
    const Dtype*  const mask_slice = bottom_mask + n * height * width * neighbour_num;
    for (int k = 0; k < feature_num; ++k) {
        int tmp = static_cast<int>(mask_slice[c * height * width + ph * width + pw]);
        Dtype difference = fabsf(feature_slice1[k * height * width] - feature_slice2[ k * height * width + tmp]);
        if(!weight_type)
              aveval +=  difference ;
        else aveval +=  difference * difference;
        //cache_value += 1;   
    }
    
    top_bottom_sim[index] = max(exp(-1.0 * aveval),1e-5);
  }
}
template <typename Dtype>
__global__ void KNNsimPoolForward(const int nthreads,
    const Dtype* const bottom_data, const Dtype* const bottom_mask, const Dtype* const top_bottom_sim,const int num, const int channels,
    const int height, const int width, const int pooled_height,
    const int pooled_width, const int neighbour_num,
    Dtype* const top_data, Dtype* const top_sum) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    const int pw = index % pooled_width;
    const int ph = (index / pooled_width) % pooled_height;
    const int c = (index / pooled_width / pooled_height) % channels;
    const int n = index / pooled_width / pooled_height / channels;


    Dtype cache_value = 0 ;
    Dtype aveval = 0;
    const Dtype* const bottom_slice =
        bottom_data + (n * channels + c) * height * width;
    const Dtype*  const mask_slice = bottom_mask + n * height * width * neighbour_num;
    const Dtype*  const top_bottom_sim_slice = top_bottom_sim + n * height * width * neighbour_num;
    for (int k = 0; k < neighbour_num; ++k) {
        int tmp = static_cast<int>(mask_slice[k * height * width + ph * width + pw]);
        if(tmp == (ph * width + pw)) continue;
        Dtype sim_value = top_bottom_sim_slice[k* height * width + ph * width + pw];
        aveval += bottom_slice[tmp] * sim_value;
        cache_value += sim_value;   
    }
    
    top_data[index] = aveval / cache_value;
    top_sum[n * height * width + ph * width + pw] = cache_value;
  }
}

template <typename Dtype>
__global__ void StoPoolForwardTrain(const int nthreads,
    const Dtype* const bottom_data,
    const int num, const int channels, const int height,
    const int width, const int pooled_height, const int pooled_width,
    const int kernel_h, const int kernel_w, const int stride_h,
    const int stride_w, Dtype* const rand_idx, Dtype* const top_data) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    const int pw = index % pooled_width;
    const int ph = (index / pooled_width) % pooled_height;
    const int c = (index / pooled_width / pooled_height) % channels;
    const int n = index / pooled_width / pooled_height / channels;
    const int hstart = ph * stride_h;
    const int hend = min(hstart + kernel_h, height);
    const int wstart = pw * stride_w;
    const int wend = min(wstart + kernel_w, width);
    Dtype cumsum = 0.;
    const Dtype* const bottom_slice =
        bottom_data + (n * channels + c) * height * width;
    // First pass: get sum
    for (int h = hstart; h < hend; ++h) {
      for (int w = wstart; w < wend; ++w) {
        cumsum += bottom_slice[h * width + w];
      }
    }
    const float thres = rand_idx[index] * cumsum;
    // Second pass: get value, and set index.
    cumsum = 0;
    for (int h = hstart; h < hend; ++h) {
      for (int w = wstart; w < wend; ++w) {
        cumsum += bottom_slice[h * width + w];
        if (cumsum >= thres) {
          rand_idx[index] = ((n * channels + c) * height + h) * width + w;
          top_data[index] = bottom_slice[h * width + w];
          return;
        }
      }
    }
  }
}


template <typename Dtype>
__global__ void StoPoolForwardTest(const int nthreads,
    const Dtype* const bottom_data,
    const int num, const int channels, const int height,
    const int width, const int pooled_height, const int pooled_width,
    const int kernel_h, const int kernel_w, const int stride_h,
    const int stride_w, Dtype* const top_data) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    const int pw = index % pooled_width;
    const int ph = (index / pooled_width) % pooled_height;
    const int c = (index / pooled_width / pooled_height) % channels;
    const int n = index / pooled_width / pooled_height / channels;
    const int hstart = ph * stride_h;
    const int hend = min(hstart + kernel_h, height);
    const int wstart = pw * stride_w;
    const int wend = min(wstart + kernel_w, width);
    // We set cumsum to be 0 to avoid divide-by-zero problems
    Dtype cumsum = FLT_MIN;
    Dtype cumvalues = 0.;
    const Dtype* const bottom_slice =
        bottom_data + (n * channels + c) * height * width;
    // First pass: get sum
    for (int h = hstart; h < hend; ++h) {
      for (int w = wstart; w < wend; ++w) {
        cumsum += bottom_slice[h * width + w];
        cumvalues += bottom_slice[h * width + w] * bottom_slice[h * width + w];
      }
    }
    top_data[index] = cumvalues / cumsum;
  }
}


template <typename Dtype>
void PoolingLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();

  Dtype* top_data = top[0]->mutable_gpu_data();
  int count = top[0]->count();
  int neighbour_num = 0;
  // We'll output the mask to top[1] if it's of size >1.
  const bool use_top_mask = top.size() > 1;
  int* mask = NULL;
  Dtype* top_mask = NULL;

  const Dtype *bottom_mask = NULL;
  Dtype* top_sum = NULL;
  const Dtype* bottom_feature = NULL;
   Dtype* top_bottom_sim_data = NULL;
   int weight_type = this->layer_param_.pooling_param().weight_type();
  // const Dtype* bottom_sim_data = NULL;
  
  switch (this->layer_param_.pooling_param().pool()) {
  case PoolingParameter_PoolMethod_MAX:
    if (use_top_mask) {
      top_mask = top[1]->mutable_gpu_data();
    } else {
      mask = max_idx_.mutable_gpu_data();
    }
  //  LOG(INFO) << "pooling layer forward";
   // LOG(INFO) << bottom[0]->num() << ' '<< bottom[0]->channels() << bottom[0]->height() << bottom[0]->width();
    // NOLINT_NEXT_LINE(whitespace/operators)
    MaxPoolForward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
        count, bottom_data, bottom[0]->num(), channels_,
        height_, width_, pooled_height_, pooled_width_, kernel_h_,
        kernel_w_, stride_h_, stride_w_, pad_h_, pad_w_, top_data,
        mask, top_mask);
    break;
  case PoolingParameter_PoolMethod_AVE:
    // NOLINT_NEXT_LINE(whitespace/operators)
    AvePoolForward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
        count, bottom_data, bottom[0]->num(), channels_,
        height_, width_, pooled_height_, pooled_width_, kernel_h_,
        kernel_w_, stride_h_, stride_w_, pad_h_, pad_w_, top_data);
    break;
  case PoolingParameter_PoolMethod_STOCHASTIC:
    if (this->phase_ == TRAIN) {
      // We need to create the random index as well.
      caffe_gpu_rng_uniform(count, Dtype(0), Dtype(1),
                            rand_idx_.mutable_gpu_data());
      // NOLINT_NEXT_LINE(whitespace/operators)
      StoPoolForwardTrain<Dtype><<<CAFFE_GET_BLOCKS(count),
                                   CAFFE_CUDA_NUM_THREADS>>>(
          count, bottom_data, bottom[0]->num(), channels_,
          height_, width_, pooled_height_, pooled_width_, kernel_h_,
          kernel_w_, stride_h_, stride_w_,
          rand_idx_.mutable_gpu_data(), top_data);
    } else {
      // NOLINT_NEXT_LINE(whitespace/operators)
      StoPoolForwardTest<Dtype><<<CAFFE_GET_BLOCKS(count),
                                  CAFFE_CUDA_NUM_THREADS>>>(
          count, bottom_data, bottom[0]->num(), channels_,
          height_, width_, pooled_height_, pooled_width_, kernel_h_,
          kernel_w_, stride_h_, stride_w_, top_data);
    }
    break;
    case PoolingParameter_PoolMethod_MASKAVE:
    bottom_mask = bottom[1]->gpu_data();
    top_sum = top_sum_.mutable_gpu_data();
    caffe_gpu_set(top_sum_.count(), Dtype(0.0), top_sum_.mutable_gpu_data());
    MaskAvePoolForward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
        count, bottom_data, bottom_mask, bottom[0]->num(), channels_,
        height_, width_, pooled_height_, pooled_width_, kernel_h_,
        kernel_w_, stride_h_, stride_w_, pad_h_, pad_w_, top_data, top_sum);
    break;
    case PoolingParameter_PoolMethod_KNNPOOL:
    bottom_mask = bottom[1]->gpu_data();
    neighbour_num = bottom[1]->channels();
    KNNPoolForward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
        count, bottom_data, bottom_mask, bottom[0]->num(), channels_,
        height_, width_, pooled_height_, pooled_width_, neighbour_num,
        top_data);
      //Forward_cpu(bottom, top);
    break;
    case PoolingParameter_PoolMethod_KNNWEIGHTEDPOOL:
    bottom_mask = bottom[1]->gpu_data();
    bottom_feature = bottom[2]->gpu_data();
    top_bottom_sim_data = top_bottom_sim_.mutable_gpu_data();
    neighbour_num = bottom[1]->channels();
    KNNsimForward<Dtype><<<CAFFE_GET_BLOCKS(bottom[1]->count()), CAFFE_CUDA_NUM_THREADS>>>(
        bottom[1]->count(), bottom_mask, bottom_feature, bottom[0]->num(), channels_,
        height_, width_, pooled_height_, pooled_width_, neighbour_num, bottom[2]->channels(), weight_type,
        top_bottom_sim_data);
    //bottom_sim_data = top_bottom_sim_.cpu_data();
   // for(int k = 0; k < top_bottom_sim_.count(); ++k) LOG(INFO) << bottom_sim_data[k];
    KNNsimPoolForward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
        count, bottom_data, bottom_mask, top_bottom_sim_.gpu_data(), bottom[0]->num(), channels_,
        height_, width_, pooled_height_, pooled_width_, neighbour_num,
        top_data, top_sum_.mutable_gpu_data());
    break;
   case PoolingParameter_PoolMethod_WEIGHTEDAVE:
    //bottom_mask = bottom[1]->gpu_data();
    top_sum = top_sum_.mutable_gpu_data();
    bottom_feature = bottom[1]->gpu_data();
    caffe_gpu_set(top_sum_.count(), Dtype(0.0), top_sum_.mutable_gpu_data());
    WeightAvePoolForward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
        count, bottom_data, bottom_feature, bottom[0]->num(), channels_,
        height_, width_, pooled_height_, pooled_width_, kernel_h_,
        kernel_w_, stride_h_, stride_w_, pad_h_, pad_w_, bottom[1]->channels(), top_data, top_sum);
    break;
  default:
    LOG(FATAL) << "Unknown pooling method.";
  }
  CUDA_POST_KERNEL_CHECK;
}


template <typename Dtype>
__global__ void MaxPoolBackward(const int nthreads, const Dtype* const top_diff,
    const int* const mask, const Dtype* const top_mask, const int num,
    const int channels, const int height, const int width,
    const int pooled_height, const int pooled_width, const int kernel_h,
    const int kernel_w, const int stride_h, const int stride_w, const int pad_h,
    const int pad_w, Dtype* const bottom_diff) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    // find out the local index
    // find out the local offset
    const int w = index % width;
    const int h = (index / width) % height;
    const int c = (index / width / height) % channels;
    const int n = index / width / height / channels;
    const int phstart =
         (h + pad_h < kernel_h) ? 0 : (h + pad_h - kernel_h) / stride_h + 1;
    const int phend = min((h + pad_h) / stride_h + 1, pooled_height);
    const int pwstart =
         (w + pad_w < kernel_w) ? 0 : (w + pad_w - kernel_w) / stride_w + 1;
    const int pwend = min((w + pad_w) / stride_w + 1, pooled_width);
    Dtype gradient = 0;
    const int offset = (n * channels + c) * pooled_height * pooled_width;
    const Dtype* const top_diff_slice = top_diff + offset;
    if (mask) {
      const int* const mask_slice = mask + offset;
      for (int ph = phstart; ph < phend; ++ph) {
        for (int pw = pwstart; pw < pwend; ++pw) {
          if (mask_slice[ph * pooled_width + pw] == h * width + w) {
            gradient += top_diff_slice[ph * pooled_width + pw];
          }
        }
      }
    } else {
      const Dtype* const top_mask_slice = top_mask + offset;
      for (int ph = phstart; ph < phend; ++ph) {
        for (int pw = pwstart; pw < pwend; ++pw) {
          if (top_mask_slice[ph * pooled_width + pw] == h * width + w) {
            gradient += top_diff_slice[ph * pooled_width + pw];
          }
        }
      }
    }
    bottom_diff[index] = gradient;
  }
}

template <typename Dtype>
__global__ void AvePoolBackward(const int nthreads, const Dtype* const top_diff,
    const int num, const int channels, const int height,
    const int width, const int pooled_height, const int pooled_width,
    const int kernel_h, const int kernel_w, const int stride_h,
    const int stride_w, const int pad_h, const int pad_w,
    Dtype* const bottom_diff) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    // find out the local index
    // find out the local offset
    const int w = index % width + pad_w;
    const int h = (index / width) % height + pad_h;
    const int c = (index / width / height) % channels;
    const int n = index / width / height / channels;
    const int phstart = (h < kernel_h) ? 0 : (h - kernel_h) / stride_h + 1;
    const int phend = min(h / stride_h + 1, pooled_height);
    const int pwstart = (w < kernel_w) ? 0 : (w - kernel_w) / stride_w + 1;
    const int pwend = min(w / stride_w + 1, pooled_width);
    Dtype gradient = 0;
    const Dtype* const top_diff_slice =
        top_diff + (n * channels + c) * pooled_height * pooled_width;
    for (int ph = phstart; ph < phend; ++ph) {
      for (int pw = pwstart; pw < pwend; ++pw) {
        // figure out the pooling size
        int hstart = ph * stride_h - pad_h;
        int wstart = pw * stride_w - pad_w;
        int hend = min(hstart + kernel_h, height + pad_h);
        int wend = min(wstart + kernel_w, width + pad_w);
        int pool_size = (hend - hstart) * (wend - wstart);
        gradient += top_diff_slice[ph * pooled_width + pw] / pool_size;
      }
    }
    bottom_diff[index] = gradient;
  }
}

template <typename Dtype>
__global__ void WeightAvePoolBackward(const int nthreads, const Dtype* const bottom_feature, const Dtype* const top_sum, 
    const Dtype* const top_diff, 
    const int num, const int channels, const int height,
    const int width, const int pooled_height, const int pooled_width,
    const int kernel_h, const int kernel_w, const int stride_h,
    const int stride_w, const int pad_h, const int pad_w, const int feature_length,
    Dtype* const bottom_diff) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    // find out the local index
    // find out the local offset
    const int w = index % width + pad_w;
    const int h = (index / width) % height + pad_h;
    const int c = (index / width / height) % channels;
    const int n = index / width / height / channels;
    const int phstart = (h < kernel_h) ? 0 : (h - kernel_h) / stride_h + 1;
    const int phend = min(h / stride_h + 1, pooled_height);
    const int pwstart = (w < kernel_w) ? 0 : (w - kernel_w) / stride_w + 1;
    const int pwend = min(w / stride_w + 1, pooled_width);
    Dtype gradient = 0;
    Dtype similarity = 0.0;
    const Dtype* const top_diff_slice =
        top_diff + (n * channels + c) * pooled_height * pooled_width;
    for (int ph = phstart; ph < phend; ++ph) {
      for (int pw = pwstart; pw < pwend; ++pw) {
        // figure out the pooling size
        int hstart = ph * stride_h - pad_h;
        int wstart = pw * stride_w - pad_w;
        int hend = min(hstart + kernel_h, height + pad_h);
        int wend = min(wstart + kernel_w, width + pad_w);
       // int pool_size = (hend - hstart) * (wend - wstart);
        //calculate similarity
        similarity = 0.0;
        for (int l = 0; l < feature_length; ++l)
        {
          int n_index = n * feature_length * height * width + l * height * width;
          similarity += fabs(bottom_feature[n_index + ((index/width)%height) * width + index % width] - bottom_feature[n_index + ph * pooled_width + pw]);
        }
        gradient += top_diff_slice[ph * pooled_width + pw] / top_sum[n * height * width + ph * width + pw];
      }
    }
    bottom_diff[index] = gradient;
  }
}


template <typename Dtype>
__global__ void MaskAvePoolBackward(const int nthreads, const Dtype* const bottom_data,
   const Dtype* const bottom_mask, const Dtype* const top_sum, const Dtype* const top_diff,
    const int num, const int channels, const int height,
    const int width, const int pooled_height, const int pooled_width,
    const int kernel_h, const int kernel_w, const int stride_h,
    const int stride_w, const int pad_h, const int pad_w,
    Dtype* const bottom_diff) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    // find out the local index
    // find out the local offset
    const int w = index % width + pad_w;
    const int h = (index / width) % height + pad_h;
    const int c = (index / width / height) % channels;
    const int n = index / width / height / channels;
    const int phstart = (h < kernel_h) ? 0 : (h - kernel_h) / stride_h + 1;
    const int phend = min(h / stride_h + 1, pooled_height);
    const int pwstart = (w < kernel_w) ? 0 : (w - kernel_w) / stride_w + 1;
    const int pwend = min(w / stride_w + 1, pooled_width);
    Dtype gradient = 0;
    //Dtype gradient_mask = 0;

    const Dtype* const top_diff_slice =
        top_diff + (n * channels + c) * pooled_height * pooled_width;
    const Dtype* const top_sum_slice = top_sum + n * pooled_height * pooled_width;
    const Dtype* const mask_slice = bottom_mask + n * height * width * kernel_h * kernel_w;

   // Dtype * const bottom_mask_diff_slice = bottom_mask_diff + n * height * width * kernel_h * kernel_w;
   
    int center_w = (kernel_w - 1)/ 2;
    int center_h = (kernel_h - 1)/ 2;
    int w_bottom = index % width;
    int h_bottom = (index / width) % height;
    for (int ph = phstart; ph < phend; ++ph) {
      for (int pw = pwstart; pw < pwend; ++pw) {
        int channel_relative = (ph - h_bottom + center_h) * kernel_w + (pw - w_bottom + center_w);
        gradient += top_diff_slice[ph * pooled_width + pw] 
                 * mask_slice[index % (height * width) + channel_relative * (height * width)] / top_sum_slice[ph * pooled_width + pw];
       
      }
    }
    bottom_diff[index] = gradient;
  }
}

template <typename Dtype>
__global__ void MaskAveMaskBackward(const int nthreads, const Dtype* const bottom_data,
   const Dtype* const bottom_mask,const Dtype* const top_sum, const Dtype* const top_diff,
    const int num, const int channels, const int height,
    const int width, const int pooled_height, const int pooled_width,
    const int kernel_h, const int kernel_w, const int stride_h,
    const int stride_w, const int pad_h, const int pad_w,
    Dtype* const bottom_diff_mask) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    // find out the local index
    // find out the local offset
    const int w = index % pooled_width;
    const int h = (index / pooled_width) % pooled_height;
    const int c = (index / (pooled_width * pooled_height)) % (kernel_h * kernel_w);
    const int n = index / pooled_width / pooled_height / (kernel_h * kernel_w);
    
    Dtype gradient = 0;
    //Dtype gradient_mask = 0;

    const Dtype* const top_diff_slice =
        top_diff + n * channels * pooled_height * pooled_width;
    //const Dtype* const top_sum_slice = top_sum + n * pooled_height * pooled_width;
    const Dtype* const bottom_data_slice = 
        bottom_data + n * channels * height * width;
    const Dtype* const top_sum_slice = 
       top_sum + n* pooled_height * pooled_width;

    
    int center_w = (kernel_w - 1)/ 2;
    int center_h = (kernel_h - 1)/ 2;
    
     int relative_w = c % kernel_w - center_w;
     int relative_h = (c / kernel_w) % kernel_h - center_h;
     int bottom_w = w + relative_w;
     int bottom_h = h + relative_h;
     if(bottom_w < 0 || bottom_h < 0 || bottom_w >= width || bottom_h >= height)
       {
       	 bottom_diff_mask[index] = Dtype(0.0);
         return;
       }


    for(int ch = 0; ch < channels; ++ch)
    {
    	gradient += top_diff_slice[ch * pooled_height * pooled_width + h * pooled_width + w] 
    	         * bottom_data_slice[ch * height * width + bottom_h * width + bottom_w]/ top_sum_slice[h * pooled_width + w] ;
    }
    bottom_diff_mask[index] = gradient;
  }
}

template <typename Dtype>
__global__ void KNNPoolBackward(const int nthreads, const Dtype* const bottom_data,
    const Dtype* const top_to_bottom_corr,const Dtype* const top_diff,
    const int num, const int channels, const int height,
    const int width, const int neighbour_num, Dtype* const bottom_diff) {
  CUDA_KERNEL_LOOP(index, nthreads) {
  	
    // find out the local index
    // find out the local offset
    const int w = index % width;
    const int h = (index / width) % height;
    const int c = (index / width / height) % channels;
    const int n = index / width / height / channels;
    
    Dtype gradient = 0;
    //Dtype aval = 0;
    int index_w = 0;
    int index_h = 0;
    int index_0 = 0;
    //Dtype gradient_mask = 0;

    const Dtype* const top_diff_slice =
        top_diff + (n * channels + c) * height * width;
    const Dtype* const top_to_bottom_slice = top_to_bottom_corr + n * height * width * 3 * neighbour_num;
      for (int k  = 0; k < 3 * neighbour_num; ++k) {
        if(top_to_bottom_slice[k * height * width + h * width + w] < 0)
        	break;
        index_0 = static_cast<int>(top_to_bottom_slice[k * height * width + h * width + w]);
        index_w = index_0 % width;
        index_h = (index_0 / width)% height;
        gradient += (top_diff_slice[index_h * width + index_w]/static_cast<Dtype>(neighbour_num)); 
        //aval = aval + 1;
       
      }
    bottom_diff[index] = gradient;
  }
}
template <typename Dtype>
__global__ void KNNsimPoolBackward(const int nthreads, const Dtype* const bottom_data,
    const Dtype* const top_to_bottom_corr,const Dtype* const top_diff, const Dtype* const bottom_top_sim,
    const Dtype* const top_sum, const int num, const int channels, const int height,
    const int width, const int neighbour_num, Dtype* const bottom_diff) {
  CUDA_KERNEL_LOOP(index, nthreads) {
  	
    // find out the local index
    // find out the local offset
    const int w = index % width;
    const int h = (index / width) % height;
    const int c = (index / width / height) % channels;
    const int n = index / width / height / channels;
    
    Dtype gradient = 0;
    //Dtype aval = 0;
    int index_w = 0;
    int index_h = 0;
    int index_0 = 0;
    //Dtype gradient_mask = 0;

    const Dtype* const top_diff_slice =
        top_diff + (n * channels + c) * height * width;
    const Dtype* const top_to_bottom_slice = top_to_bottom_corr + n * height * width * 3 * neighbour_num + h * width + w;
    const Dtype* const bottom_top_sim_slice = bottom_top_sim + n * 3 * neighbour_num * height * width + h * width + w;
    const Dtype* const top_sum_slice = top_sum + n * height * width;
      for (int k  = 0; k < 3 * neighbour_num; ++k) {
        if(top_to_bottom_slice[k * height * width] < 0)
        	break;
        index_0 = static_cast<int>(top_to_bottom_slice[k * height * width]);
        index_w = index_0 % width;
        index_h = (index_0 / width)% height;
        if(index_0 == (h * width + w)) continue;
        gradient += top_diff_slice[index_h * width + index_w] * bottom_top_sim_slice[k * height * width]
                   / top_sum_slice[index_h * width + index_w]; 
        //aval = aval + 1; 
      }
    bottom_diff[index] = gradient;
  }
}

template <typename Dtype>
__global__ void StoPoolBackward(const int nthreads,
    const Dtype* const rand_idx, const Dtype* const top_diff,
    const int num, const int channels, const int height,
    const int width, const int pooled_height, const int pooled_width,
    const int kernel_h, const int kernel_w, const int stride_h,
    const int stride_w, Dtype* const bottom_diff) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    // find out the local index
    // find out the local offset
    const int w = index % width;
    const int h = (index / width) % height;
    const int c = (index / width / height) % channels;
    const int n = index / width / height / channels;
    const int phstart = (h < kernel_h) ? 0 : (h - kernel_h) / stride_h + 1;
    const int phend = min(h / stride_h + 1, pooled_height);
    const int pwstart = (w < kernel_w) ? 0 : (w - kernel_w) / stride_w + 1;
    const int pwend = min(w / stride_w + 1, pooled_width);
    Dtype gradient = 0;
    const Dtype* const rand_idx_slice =
        rand_idx + (n * channels + c) * pooled_height * pooled_width;
    const Dtype* const top_diff_slice =
        top_diff + (n * channels + c) * pooled_height * pooled_width;
    for (int ph = phstart; ph < phend; ++ph) {
      for (int pw = pwstart; pw < pwend; ++pw) {
        gradient += top_diff_slice[ph * pooled_width + pw] *
            (index == static_cast<int>(rand_idx_slice[ph * pooled_width + pw]));
      }
    }
    bottom_diff[index] = gradient;
  }
}


template <typename Dtype>
void PoolingLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (!propagate_down[0]) {
    return;
  }
  const Dtype* top_diff = top[0]->gpu_diff();
  Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
  const int count = bottom[0]->count();
  int neighbour_num = 0;
 // Blob<Dtype> top_to_bottom;
  int index = 0;
  int index_h = 0;
  int index_w = 0;
  Dtype top_index = 0;
 // Blob<Dtype> top_to_bottom_count;
  Dtype* top_to_bottom_count_data = NULL;
  int next_position = 0;

  
  // We'll output the mask to top[1] if it's of size >1.
  Dtype* top_to_bottom_data = NULL;
  Dtype* bottom_top_sim_data = NULL;
  const Dtype *top_bottom_sim_data = NULL;
  const bool use_top_mask = top.size() > 1;
  const int* mask = NULL;
  const Dtype* top_mask = NULL;
  const Dtype *bottom_data = bottom[0]->gpu_data();
  const Dtype *bottom_feature = NULL;
  //const Dtype *bottom_mask = bottom[1]->gpu_data();
  const Dtype *bottom_mask = NULL;
  //Dtype *bottom_mask_diff = NULL;
  const Dtype* top_sum = top_sum_.gpu_data();
 // int mask_count = 0;
  caffe_gpu_set(count, Dtype(0.0), bottom_diff);

  switch (this->layer_param_.pooling_param().pool()) {
  case PoolingParameter_PoolMethod_MAX:
    if (use_top_mask) {
      top_mask = top[1]->gpu_data();
    } else {
      mask = max_idx_.gpu_data();
    }
    // NOLINT_NEXT_LINE(whitespace/operators)
    MaxPoolBackward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
        count, top_diff, mask, top_mask, top[0]->num(), channels_,
        height_, width_, pooled_height_, pooled_width_,
        kernel_h_, kernel_w_, stride_h_, stride_w_, pad_h_, pad_w_,
        bottom_diff);
    break;
  case PoolingParameter_PoolMethod_AVE:
    // NOLINT_NEXT_LINE(whitespace/operators)
    AvePoolBackward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
        count, top_diff, top[0]->num(), channels_,
        height_, width_, pooled_height_, pooled_width_, kernel_h_,
        kernel_w_, stride_h_, stride_w_, pad_h_, pad_w_, bottom_diff);
    break;
  case PoolingParameter_PoolMethod_STOCHASTIC:
    // NOLINT_NEXT_LINE(whitespace/operators)
    StoPoolBackward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
        count, rand_idx_.gpu_data(), top_diff,
        top[0]->num(), channels_, height_, width_, pooled_height_,
        pooled_width_, kernel_h_, kernel_w_, stride_h_, stride_w_,
        bottom_diff);
    break;
   case PoolingParameter_PoolMethod_MASKAVE:
   // mask_count = bottom[1]->num() * bottom[1]->channels() * bottom[1]->height() * bottom[1]->width();
    caffe_gpu_set(bottom[0]->count(), Dtype(0.0), bottom[0]->mutable_gpu_diff());
    caffe_gpu_set(bottom[1]->count(), Dtype(0.0), bottom[1]->mutable_gpu_diff());
    bottom_mask = bottom[1]->gpu_data();
    //bottom_mask_diff = bottom[1]->mutable_gpu_diff();
   // LOG(INFO) << "backward data average";
    // NOLINT_NEXT_LINE(whitespace/operators)
  //  LOG(INFO) << "backward mask average backward";
    MaskAvePoolBackward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
        count, bottom_data, bottom_mask, top_sum, top_diff, top[0]->num(), channels_,
        height_, width_, pooled_height_, pooled_width_, kernel_h_,
        kernel_w_, stride_h_, stride_w_, pad_h_, pad_w_, bottom_diff);
    //LOG(INFO) << "backward mask average";
    // NOLINT_NEXT_LINE(whitespace/operators)
    /*MaskAveMaskBackward<Dtype><<<CAFFE_GET_BLOCKS(mask_count), CAFFE_CUDA_NUM_THREADS>>>(
        mask_count, bottom_data, bottom_mask, top_sum, top_diff, top[0]->num(), channels_,
        height_, width_, pooled_height_, pooled_width_, kernel_h_,
        kernel_w_, stride_h_, stride_w_, pad_h_, pad_w_, bottom_mask_diff);
    caffe_gpu_set(bottom[1]->count(), Dtype(0.0), bottom[1]->mutable_gpu_diff());*/
   // LOG(INFO) << "backward mask average end";
    break;
    case PoolingParameter_PoolMethod_KNNPOOL:
    //Backward_cpu(top, propagate_down,bottom);
    neighbour_num = bottom[1]->channels();
    caffe_gpu_set(bottom[0]->count(), Dtype(0.0), bottom[0]->mutable_gpu_diff());
    //top_to_bottom.Reshape(num_, 3 * neighbour_num, height_, width_);
    caffe_gpu_set(top_to_bottom_.count(), Dtype(-1.0), top_to_bottom_.mutable_gpu_data());
    bottom_mask = bottom[1]->cpu_data();
    top_to_bottom_data = top_to_bottom_.mutable_cpu_data();
    //top_to_bottom_count.Reshape(num_, 1, height_, width_);
    caffe_gpu_set(top_to_bottom_count_.count(), Dtype(0.0), top_to_bottom_count_.mutable_gpu_data());
    top_to_bottom_count_data = top_to_bottom_count_.mutable_cpu_data();
    //LOG(INFO) << top_to_bottom.num()<< top_to_bottom.channels()<<top_to_bottom.height()<<top_to_bottom.width(); 
    for(int h = 0; h< height_; ++h)
    {
       for(int w = 0; w < width_; ++w)
    	 { 
    	  	top_index = static_cast<Dtype>(h * width_ + w);
    	  	for (int n = 0; n < num_; ++n)
           {  
    	    for(int k = 0; k < neighbour_num; ++k)
    	     {
    	     	//LOG(INFO) << k;
    	     	//LOG(INFO) << neighbour_num;
              index = static_cast<int>(bottom_mask[((n * neighbour_num + k) * height_ + h) * width_ + w ]);
            //  LOG(INFO)<< index;
              index_w = index % width_;
            //  LOG(INFO)<< index_w;
              index_h = (index/width_)%height_;
              //LOG(INFO)<< index_h;
              next_position = static_cast<int>(top_to_bottom_count_data[n * height_ * width_ + index_h * width_ + index_w]);
              
              //LOG(INFO) << next_position;
              if(next_position < 3 * neighbour_num)
               {
               	// LOG(INFO) << next_position;
               	 top_to_bottom_data[((n * 3 * neighbour_num + next_position) * height_ + index_h) * width_ + index_w] = top_index;
                // LOG(INFO) << "top_to_bottom_data";
                 top_to_bottom_count_data[n * height_ * width_ + index_h * width_ + index_w] += 1.0;
                // LOG(INFO) << "top_to_bottom_count";
                }
              else 
             	   {
                   LOG(INFO) << next_position;
                   LOG(INFO) << "number exceed 2 * num_neighbourhood";
                 }
    	  	}
    	  }
    	}
    }
    KNNPoolBackward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
        count, bottom_data, top_to_bottom_.gpu_data(), top_diff, top[0]->num(),
        channels_, height_, width_, neighbour_num , bottom_diff);
    break;
   case PoolingParameter_PoolMethod_KNNWEIGHTEDPOOL:
   //Backward_cpu(top, propagate_down,bottom);
    neighbour_num = bottom[1]->channels();
    caffe_gpu_set(bottom[0]->count(), Dtype(0.0), bottom[0]->mutable_gpu_diff());
    //top_to_bottom.Reshape(num_, 3 * neighbour_num, height_, width_);
    caffe_gpu_set(top_to_bottom_.count(), Dtype(-1.0), top_to_bottom_.mutable_gpu_data());
    bottom_mask = bottom[1]->cpu_data();
    top_to_bottom_data = top_to_bottom_.mutable_cpu_data();
    //top_to_bottom_count.Reshape(num_, 1, height_, width_);
    caffe_gpu_set(top_to_bottom_count_.count(), Dtype(0.0), top_to_bottom_count_.mutable_gpu_data());
    top_to_bottom_count_data = top_to_bottom_count_.mutable_cpu_data();

    caffe_gpu_set(bottom_top_sim_.count(), Dtype(0.0), bottom_top_sim_.mutable_gpu_data());
    bottom_top_sim_data = bottom_top_sim_.mutable_cpu_data();
    top_bottom_sim_data = top_bottom_sim_.cpu_data();
    //LOG(INFO) << top_to_bottom.num()<< top_to_bottom.channels()<<top_to_bottom.height()<<top_to_bottom.width(); 
    for(int h = 0; h< height_; ++h)
    {
       for(int w = 0; w < width_; ++w)
    	 { 
    	  	top_index = static_cast<Dtype>(h * width_ + w);
    	  	for (int n = 0; n < num_; ++n)
           {  
    	    for(int k = 0; k < neighbour_num; ++k)
    	     {
    	      index = static_cast<int>(bottom_mask[((n * neighbour_num + k) * height_ + h) * width_ + w ]);
              index_w = index % width_;
              index_h = (index/width_)%height_;
              next_position = static_cast<int>(top_to_bottom_count_data[n * height_ * width_ + index_h * width_ + index_w]);
              if(next_position < 3 * neighbour_num)
               {
                 top_to_bottom_data[((n * 3 * neighbour_num + next_position) * height_ + index_h) * width_ + index_w] = top_index;
                 bottom_top_sim_data[((n * 3 * neighbour_num + next_position) * height_ + index_h) * width_ + index_w] = 
                 top_bottom_sim_data[n * neighbour_num * height_ * width_ + k * height_ * width_ + h * width_ + w];
                 top_to_bottom_count_data[n * height_ * width_ + index_h * width_ + index_w] += 1.0;
               
                }
              else 
             	   {
                   LOG(INFO) << next_position;
                   LOG(INFO) << "number exceed 2 * num_neighbourhood";
                 }
    	  	}
    	  }
    	}
    }
    KNNsimPoolBackward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
        count, bottom_data, top_to_bottom_.gpu_data(), top_diff, bottom_top_sim_.gpu_data(),top_sum_.gpu_data(),
        top[0]->num(), channels_, height_, width_, neighbour_num , bottom_diff);
   break;
   case PoolingParameter_PoolMethod_WEIGHTEDAVE:
  
    caffe_gpu_set(bottom[0]->count(), Dtype(0.0), bottom[0]->mutable_gpu_diff());
    caffe_gpu_set(bottom[1]->count(), Dtype(0.0), bottom[1]->mutable_gpu_diff());
    bottom_feature = bottom[1]->gpu_data();
    
    WeightAvePoolBackward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
        count, bottom_feature, top_sum, top_diff, top[0]->num(), channels_,
        height_, width_, pooled_height_, pooled_width_, kernel_h_,
        kernel_w_, stride_h_, stride_w_, pad_h_, pad_w_, bottom[1]->channels(), bottom_diff);
    
    break;
  default:
    LOG(FATAL) << "Unknown pooling method.";
  }
  CUDA_POST_KERNEL_CHECK;
}


INSTANTIATE_LAYER_GPU_FUNCS(PoolingLayer);


}  // namespace caffe
