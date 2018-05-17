#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#endif  // USE_OPENCV

#include <string>
#include <vector>

#include "caffe/data_transformer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"

namespace caffe {

template<typename Dtype>
DataTransformer<Dtype>::DataTransformer(const TransformationParameter& param,
    Phase phase)
    : param_(param), phase_(phase) {
  // check if we want to use mean_file
  if (param_.has_mean_file()) {
    CHECK_EQ(param_.mean_value_size(), 0) <<
      "Cannot specify mean_file and mean_value at the same time";
    const string& mean_file = param.mean_file();
    if (Caffe::root_solver()) {
      LOG(INFO) << "Loading mean file from: " << mean_file;
    }
    BlobProto blob_proto;
    ReadProtoFromBinaryFileOrDie(mean_file.c_str(), &blob_proto);
    data_mean_.FromProto(blob_proto);
  }
  // check if we want to use mean_value
  if (param_.mean_value_size() > 0) {
    CHECK(param_.has_mean_file() == false) <<
      "Cannot specify mean_file and mean_value at the same time";
    for (int c = 0; c < param_.mean_value_size(); ++c) {
      mean_values_.push_back(param_.mean_value(c));
    }
  }
  // check if we want to do random scaling
  if (param_.scale_factors_size() > 0) {
    for (int i = 0; i < param_.scale_factors_size(); ++i) {
      scale_factors_.push_back(param_.scale_factors(i));
    }
  }  
}

template<typename Dtype>
void DataTransformer<Dtype>::Transform(const Datum& datum,
                                       Dtype* transformed_data) {
  const string& data = datum.data();
  const int datum_channels = datum.channels();
  const int datum_height = datum.height();
  const int datum_width = datum.width();

  const int crop_size = param_.crop_size();
  const Dtype scale = param_.scale();
  const bool do_mirror = param_.mirror() && Rand(2);
  const bool has_mean_file = param_.has_mean_file();
  const bool has_uint8 = data.size() > 0;
  const bool has_mean_values = mean_values_.size() > 0;

  CHECK_GT(datum_channels, 0);
  CHECK_GE(datum_height, crop_size);
  CHECK_GE(datum_width, crop_size);

  Dtype* mean = NULL;
  if (has_mean_file) {
    CHECK_EQ(datum_channels, data_mean_.channels());
    CHECK_EQ(datum_height, data_mean_.height());
    CHECK_EQ(datum_width, data_mean_.width());
    mean = data_mean_.mutable_cpu_data();
  }
  if (has_mean_values) {
    CHECK(mean_values_.size() == 1 || mean_values_.size() == datum_channels) <<
     "Specify either 1 mean_value or as many as channels: " << datum_channels;
    if (datum_channels > 1 && mean_values_.size() == 1) {
      // Replicate the mean_value for simplicity
      for (int c = 1; c < datum_channels; ++c) {
        mean_values_.push_back(mean_values_[0]);
      }
    }
  }

  int height = datum_height;
  int width = datum_width;

  int h_off = 0;
  int w_off = 0;
  if (crop_size) {
    height = crop_size;
    width = crop_size;
    // We only do random crop when we do training.
    if (phase_ == TRAIN) {
      h_off = Rand(datum_height - crop_size + 1);
      w_off = Rand(datum_width - crop_size + 1);
    } else {
      h_off = (datum_height - crop_size) / 2;
      w_off = (datum_width - crop_size) / 2;
    }
  }

  Dtype datum_element;
  int top_index, data_index;
  for (int c = 0; c < datum_channels; ++c) {
    for (int h = 0; h < height; ++h) {
      for (int w = 0; w < width; ++w) {
        data_index = (c * datum_height + h_off + h) * datum_width + w_off + w;
        if (do_mirror) {
          top_index = (c * height + h) * width + (width - 1 - w);
        } else {
          top_index = (c * height + h) * width + w;
        }
        if (has_uint8) {
          datum_element =
            static_cast<Dtype>(static_cast<uint8_t>(data[data_index]));
        } else {
          datum_element = datum.float_data(data_index);
        }
        if (has_mean_file) {
          transformed_data[top_index] =
            (datum_element - mean[data_index]) * scale;
        } else {
          if (has_mean_values) {
            transformed_data[top_index] =
              (datum_element - mean_values_[c]) * scale;
          } else {
            transformed_data[top_index] = datum_element * scale;
          }
        }
      }
    }
  }
}


template<typename Dtype>
void DataTransformer<Dtype>::Transform(const Datum& datum,
                                       Blob<Dtype>* transformed_blob) {
  // If datum is encoded, decoded and transform the cv::image.
  if (datum.encoded()) {
#ifdef USE_OPENCV
    CHECK(!(param_.force_color() && param_.force_gray()))
        << "cannot set both force_color and force_gray";
    cv::Mat cv_img;
    if (param_.force_color() || param_.force_gray()) {
    // If force_color then decode in color otherwise decode in gray.
      cv_img = DecodeDatumToCVMat(datum, param_.force_color());
    } else {
      cv_img = DecodeDatumToCVMatNative(datum);
    }
    // Transform the cv::image into blob.
    return Transform(cv_img, transformed_blob);
#else
    LOG(FATAL) << "Encoded datum requires OpenCV; compile with USE_OPENCV.";
#endif  // USE_OPENCV
  } else {
    if (param_.force_color() || param_.force_gray()) {
      LOG(ERROR) << "force_color and force_gray only for encoded datum";
    }
  }

  const int crop_size = param_.crop_size();
  const int datum_channels = datum.channels();
  const int datum_height = datum.height();
  const int datum_width = datum.width();

  // Check dimensions.
  const int channels = transformed_blob->channels();
  const int height = transformed_blob->height();
  const int width = transformed_blob->width();
  const int num = transformed_blob->num();

  CHECK_EQ(channels, datum_channels);
  CHECK_LE(height, datum_height);
  CHECK_LE(width, datum_width);
  CHECK_GE(num, 1);

  if (crop_size) {
    CHECK_EQ(crop_size, height);
    CHECK_EQ(crop_size, width);
  } else {
    CHECK_EQ(datum_height, height);
    CHECK_EQ(datum_width, width);
  }

  Dtype* transformed_data = transformed_blob->mutable_cpu_data();
  Transform(datum, transformed_data);
}

template<typename Dtype>
void DataTransformer<Dtype>::Transform(const vector<Datum> & datum_vector,
                                       Blob<Dtype>* transformed_blob) {
  const int datum_num = datum_vector.size();
  const int num = transformed_blob->num();
  const int channels = transformed_blob->channels();
  const int height = transformed_blob->height();
  const int width = transformed_blob->width();

  CHECK_GT(datum_num, 0) << "There is no datum to add";
  CHECK_LE(datum_num, num) <<
    "The size of datum_vector must be no greater than transformed_blob->num()";
  Blob<Dtype> uni_blob(1, channels, height, width);
  for (int item_id = 0; item_id < datum_num; ++item_id) {
    int offset = transformed_blob->offset(item_id);
    uni_blob.set_cpu_data(transformed_blob->mutable_cpu_data() + offset);
    Transform(datum_vector[item_id], &uni_blob);
  }
}

#ifdef USE_OPENCV
template<typename Dtype>
void DataTransformer<Dtype>::Transform(const vector<cv::Mat> & mat_vector,
                                       Blob<Dtype>* transformed_blob) {
  const int mat_num = mat_vector.size();
  const int num = transformed_blob->num();
  const int channels = transformed_blob->channels();
  const int height = transformed_blob->height();
  const int width = transformed_blob->width();

  CHECK_GT(mat_num, 0) << "There is no MAT to add";
  CHECK_EQ(mat_num, num) <<
    "The size of mat_vector must be equals to transformed_blob->num()";
  Blob<Dtype> uni_blob(1, channels, height, width);
  for (int item_id = 0; item_id < mat_num; ++item_id) {
    int offset = transformed_blob->offset(item_id);
    uni_blob.set_cpu_data(transformed_blob->mutable_cpu_data() + offset);
    Transform(mat_vector[item_id], &uni_blob);
  }
}

template<typename Dtype>
void DataTransformer<Dtype>::Transform(const cv::Mat& cv_img,
                                       Blob<Dtype>* transformed_blob) {
  const int crop_size = param_.crop_size();
  const int img_channels = cv_img.channels();
  const int img_height = cv_img.rows;
  const int img_width = cv_img.cols;

  // Check dimensions.
  const int channels = transformed_blob->channels();
  const int height = transformed_blob->height();
  const int width = transformed_blob->width();
  const int num = transformed_blob->num();

  CHECK_EQ(channels, img_channels);
  CHECK_LE(height, img_height);
  CHECK_LE(width, img_width);
  CHECK_GE(num, 1);

  CHECK(cv_img.depth() == CV_8U) << "Image data type must be unsigned byte";

  const Dtype scale = param_.scale();
  const bool do_mirror = param_.mirror() && Rand(2);
  const bool has_mean_file = param_.has_mean_file();
  const bool has_mean_values = mean_values_.size() > 0;

  CHECK_GT(img_channels, 0);
  CHECK_GE(img_height, crop_size);
  CHECK_GE(img_width, crop_size);

  Dtype* mean = NULL;
  if (has_mean_file) {
    CHECK_EQ(img_channels, data_mean_.channels());
    CHECK_EQ(img_height, data_mean_.height());
    CHECK_EQ(img_width, data_mean_.width());
    mean = data_mean_.mutable_cpu_data();
  }
  if (has_mean_values) {
    CHECK(mean_values_.size() == 1 || mean_values_.size() == img_channels) <<
     "Specify either 1 mean_value or as many as channels: " << img_channels;
    if (img_channels > 1 && mean_values_.size() == 1) {
      // Replicate the mean_value for simplicity
      for (int c = 1; c < img_channels; ++c) {
        mean_values_.push_back(mean_values_[0]);
      }
    }
  }

  int h_off = 0;
  int w_off = 0;
  cv::Mat cv_cropped_img = cv_img;
  if (crop_size) {
    CHECK_EQ(crop_size, height);
    CHECK_EQ(crop_size, width);
    // We only do random crop when we do training.
    if (phase_ == TRAIN) {
      h_off = Rand(img_height - crop_size + 1);
      w_off = Rand(img_width - crop_size + 1);
    } else {
      h_off = (img_height - crop_size) / 2;
      w_off = (img_width - crop_size) / 2;
    }
    cv::Rect roi(w_off, h_off, crop_size, crop_size);
    cv_cropped_img = cv_img(roi);
  } else {
    CHECK_EQ(img_height, height);
    CHECK_EQ(img_width, width);
  }

  CHECK(cv_cropped_img.data);

  Dtype* transformed_data = transformed_blob->mutable_cpu_data();
  int top_index;
  for (int h = 0; h < height; ++h) {
    const uchar* ptr = cv_cropped_img.ptr<uchar>(h);
    int img_index = 0;
    for (int w = 0; w < width; ++w) {
      for (int c = 0; c < img_channels; ++c) {
        if (do_mirror) {
          top_index = (c * height + h) * width + (width - 1 - w);
        } else {
          top_index = (c * height + h) * width + w;
        }
        // int top_index = (c * height + h) * width + w;
        Dtype pixel = static_cast<Dtype>(ptr[img_index++]);
        if (has_mean_file) {
          int mean_index = (c * img_height + h_off + h) * img_width + w_off + w;
          transformed_data[top_index] =
            (pixel - mean[mean_index]) * scale;
        } else {
          if (has_mean_values) {
            transformed_data[top_index] =
              (pixel - mean_values_[c]) * scale;
          } else {
            transformed_data[top_index] = pixel * scale;
          }
        }
      }
    }
  }
}
#endif  // USE_OPENCV

template<typename Dtype>
void DataTransformer<Dtype>::Transform(Blob<Dtype>* input_blob,
                                       Blob<Dtype>* transformed_blob) {
  const int crop_size = param_.crop_size();
  const int input_num = input_blob->num();
  const int input_channels = input_blob->channels();
  const int input_height = input_blob->height();
  const int input_width = input_blob->width();

  if (transformed_blob->count() == 0) {
    // Initialize transformed_blob with the right shape.
    if (crop_size) {
      transformed_blob->Reshape(input_num, input_channels,
                                crop_size, crop_size);
    } else {
      transformed_blob->Reshape(input_num, input_channels,
                                input_height, input_width);
    }
  }

  const int num = transformed_blob->num();
  const int channels = transformed_blob->channels();
  const int height = transformed_blob->height();
  const int width = transformed_blob->width();
  const int size = transformed_blob->count();

  CHECK_LE(input_num, num);
  CHECK_EQ(input_channels, channels);
  CHECK_GE(input_height, height);
  CHECK_GE(input_width, width);


  const Dtype scale = param_.scale();
  const bool do_mirror = param_.mirror() && Rand(2);
  const bool has_mean_file = param_.has_mean_file();
  const bool has_mean_values = mean_values_.size() > 0;

  int h_off = 0;
  int w_off = 0;
  if (crop_size) {
    CHECK_EQ(crop_size, height);
    CHECK_EQ(crop_size, width);
    // We only do random crop when we do training.
    if (phase_ == TRAIN) {
      h_off = Rand(input_height - crop_size + 1);
      w_off = Rand(input_width - crop_size + 1);
    } else {
      h_off = (input_height - crop_size) / 2;
      w_off = (input_width - crop_size) / 2;
    }
  } else {
    CHECK_EQ(input_height, height);
    CHECK_EQ(input_width, width);
  }

  Dtype* input_data = input_blob->mutable_cpu_data();
  if (has_mean_file) {
    CHECK_EQ(input_channels, data_mean_.channels());
    CHECK_EQ(input_height, data_mean_.height());
    CHECK_EQ(input_width, data_mean_.width());
    for (int n = 0; n < input_num; ++n) {
      int offset = input_blob->offset(n);
      caffe_sub(data_mean_.count(), input_data + offset,
            data_mean_.cpu_data(), input_data + offset);
    }
  }

  if (has_mean_values) {
    CHECK(mean_values_.size() == 1 || mean_values_.size() == input_channels) <<
     "Specify either 1 mean_value or as many as channels: " << input_channels;
    if (mean_values_.size() == 1) {
      caffe_add_scalar(input_blob->count(), -(mean_values_[0]), input_data);
    } else {
      for (int n = 0; n < input_num; ++n) {
        for (int c = 0; c < input_channels; ++c) {
          int offset = input_blob->offset(n, c);
          caffe_add_scalar(input_height * input_width, -(mean_values_[c]),
            input_data + offset);
        }
      }
    }
  }

  Dtype* transformed_data = transformed_blob->mutable_cpu_data();

  for (int n = 0; n < input_num; ++n) {
    int top_index_n = n * channels;
    int data_index_n = n * channels;
    for (int c = 0; c < channels; ++c) {
      int top_index_c = (top_index_n + c) * height;
      int data_index_c = (data_index_n + c) * input_height + h_off;
      for (int h = 0; h < height; ++h) {
        int top_index_h = (top_index_c + h) * width;
        int data_index_h = (data_index_c + h) * input_width + w_off;
        if (do_mirror) {
          int top_index_w = top_index_h + width - 1;
          for (int w = 0; w < width; ++w) {
            transformed_data[top_index_w-w] = input_data[data_index_h + w];
          }
        } else {
          for (int w = 0; w < width; ++w) {
            transformed_data[top_index_h + w] = input_data[data_index_h + w];
          }
        }
      }
    }
  }
  if (scale != Dtype(1)) {
    DLOG(INFO) << "Scale: " << scale;
    caffe_scal(size, scale, transformed_data);
  }
}

template<typename Dtype>
void DataTransformer<Dtype>::TransformImgAndSeg(
    const std::vector<cv::Mat>& cv_img_seg,
    Blob<Dtype>* transformed_data_blob,
    Blob<Dtype>* transformed_label_blob,
    const int ignore_label) {
  CHECK(cv_img_seg.size() == 2) << "Input must contain image and seg.";

  const int img_channels = cv_img_seg[0].channels();
  // height and width may change due to pad for cropping
  int img_height   = cv_img_seg[0].rows;
  int img_width    = cv_img_seg[0].cols;

  const int seg_channels = cv_img_seg[1].channels();
  int seg_height   = cv_img_seg[1].rows;
  int seg_width    = cv_img_seg[1].cols;

  const int data_channels = transformed_data_blob->channels();
  const int data_height   = transformed_data_blob->height();
  const int data_width    = transformed_data_blob->width();

  const int label_channels = transformed_label_blob->channels();
  const int label_height   = transformed_label_blob->height();
  const int label_width    = transformed_label_blob->width();

  CHECK_EQ(seg_channels, 1);
  CHECK_EQ(img_channels, data_channels);
  CHECK_EQ(img_height, seg_height);
  CHECK_EQ(img_width, seg_width);

  CHECK_EQ(label_channels, 1);
  CHECK_EQ(data_height, label_height);
  CHECK_EQ(data_width, label_width);

  CHECK(cv_img_seg[0].depth() == CV_8U)
      << "Image data type must be unsigned byte";
  CHECK(cv_img_seg[1].depth() == CV_8U)
      << "Seg data type must be unsigned byte";

  //const int crop_size = param_.crop_size();
  int crop_width = 0;
  int crop_height = 0;
  CHECK((!param_.has_crop_size() && param_.has_crop_height() && param_.has_crop_width())
	|| (!param_.has_crop_height() && !param_.has_crop_width()))
    << "Must either specify crop_size or both crop_height and crop_width.";
  if (param_.has_crop_size()) {
    crop_width = param_.crop_size();
    crop_height = param_.crop_size();
  } 
  if (param_.has_crop_height() && param_.has_crop_width()) {
    crop_width = param_.crop_width();
    crop_height = param_.crop_height();
  }

  const Dtype scale = param_.scale();
  const bool do_mirror = param_.mirror() && Rand(2);
  const bool has_mean_file = param_.has_mean_file();
  const bool has_mean_values = mean_values_.size() > 0;

  CHECK_GT(img_channels, 0);

  Dtype* mean = NULL;
  if (has_mean_file) {
    CHECK_EQ(img_channels, data_mean_.channels());
    CHECK_EQ(img_height, data_mean_.height());
    CHECK_EQ(img_width, data_mean_.width());
    mean = data_mean_.mutable_cpu_data();
  }
  if (has_mean_values) {
    CHECK(mean_values_.size() == 1 || mean_values_.size() == img_channels) <<
     "Specify either 1 mean_value or as many as channels: " << img_channels;
    if (img_channels > 1 && mean_values_.size() == 1) {
      // Replicate the mean_value for simplicity
      for (int c = 1; c < img_channels; ++c) {
        mean_values_.push_back(mean_values_[0]);
      }
    }
  }

  // start to perform transformation
  cv::Mat cv_cropped_img;
  cv::Mat cv_cropped_seg;

  // perform scaling
  if (scale_factors_.size() > 0) {
    int scale_ind = Rand(scale_factors_.size());
    Dtype scale   = scale_factors_[scale_ind];
 //   LOG(INFO)<<"scale**************************"<<scale;

    if (scale != 1) {
      img_height *= scale;
      img_width  *= scale;
      cv::resize(cv_img_seg[0], cv_cropped_img, cv::Size(img_width, img_height), 0, 0, 
		 cv::INTER_LINEAR);
      cv::resize(cv_img_seg[1], cv_cropped_seg, cv::Size(img_width, img_height), 0, 0, 
		 cv::INTER_NEAREST);
    } else {
      cv_cropped_img = cv_img_seg[0];
      cv_cropped_seg = cv_img_seg[1];
    }
  } else {
    cv_cropped_img = cv_img_seg[0];
    cv_cropped_seg = cv_img_seg[1];
  }
  //

  int h_off = 0;
  int w_off = 0;

  // transform to double, since we will pad mean pixel values
  cv_cropped_img.convertTo(cv_cropped_img, CV_64F);

  // Check if we need to pad img to fit for crop_size
  // copymakeborder
  int pad_height = std::max(crop_height - img_height, 0);
  int pad_width  = std::max(crop_width - img_width, 0);
  if (pad_height > 0 || pad_width > 0) {
    cv::copyMakeBorder(cv_cropped_img, cv_cropped_img, 0, pad_height,
          0, pad_width, cv::BORDER_CONSTANT,
          cv::Scalar(mean_values_[0], mean_values_[1], mean_values_[2]));
    cv::copyMakeBorder(cv_cropped_seg, cv_cropped_seg, 0, pad_height,
          0, pad_width, cv::BORDER_CONSTANT,
          cv::Scalar(ignore_label));
    // update height/width
    img_height   = cv_cropped_img.rows;
    img_width    = cv_cropped_img.cols;

    seg_height   = cv_cropped_seg.rows;
    seg_width    = cv_cropped_seg.cols;
  }

  // crop img/seg
  if (crop_width && crop_height) {
    CHECK_EQ(crop_height, data_height);
    CHECK_EQ(crop_width, data_width);
    // We only do random crop when we do training.
    if (phase_ == TRAIN) {
      h_off = Rand(img_height - crop_height + 1);
      w_off = Rand(img_width - crop_width + 1);
    } else {
      // CHECK: use middle crop
      h_off = (img_height - crop_height) / 2;
      w_off = (img_width - crop_width) / 2;
    }
    cv::Rect roi(w_off, h_off, crop_width, crop_height);
    cv_cropped_img = cv_cropped_img(roi);
    cv_cropped_seg = cv_cropped_seg(roi);
  }

  CHECK(cv_cropped_img.data);
  CHECK(cv_cropped_seg.data);

  Dtype* transformed_data  = transformed_data_blob->mutable_cpu_data();
  Dtype* transformed_label = transformed_label_blob->mutable_cpu_data();

  int top_index;
  const double* data_ptr;
  const uchar* label_ptr;

  for (int h = 0; h < data_height; ++h) {
    data_ptr = cv_cropped_img.ptr<double>(h);
    label_ptr = cv_cropped_seg.ptr<uchar>(h);

    int data_index = 0;
    int label_index = 0;

    for (int w = 0; w < data_width; ++w) {
      // for image
      for (int c = 0; c < img_channels; ++c) {
        if (do_mirror) {
          top_index = (c * data_height + h) * data_width + (data_width - 1 - w);
        } else {
          top_index = (c * data_height + h) * data_width + w;
        }
        Dtype pixel = static_cast<Dtype>(data_ptr[data_index++]);
        if (has_mean_file) {
          int mean_index = (c * img_height + h_off + h) * img_width + w_off + w;
          transformed_data[top_index] =
            (pixel - mean[mean_index]) * scale;
        } else {
          if (has_mean_values) {
            transformed_data[top_index] =
              (pixel - mean_values_[c]) * scale;
          } else {
            transformed_data[top_index] = pixel * scale;
          }
        }
      }

      // for segmentation
      if (do_mirror) {
        top_index = h * data_width + data_width - 1 - w;
      } else {
        top_index = h * data_width + w;
      }
      Dtype pixel = static_cast<Dtype>(label_ptr[label_index++]);
      transformed_label[top_index] = pixel;
    }
  }
}

template<typename Dtype>
vector<int> DataTransformer<Dtype>::InferBlobShape(const Datum& datum) {
  if (datum.encoded()) {
#ifdef USE_OPENCV
    CHECK(!(param_.force_color() && param_.force_gray()))
        << "cannot set both force_color and force_gray";
    cv::Mat cv_img;
    if (param_.force_color() || param_.force_gray()) {
    // If force_color then decode in color otherwise decode in gray.
      cv_img = DecodeDatumToCVMat(datum, param_.force_color());
    } else {
      cv_img = DecodeDatumToCVMatNative(datum);
    }
    // InferBlobShape using the cv::image.
    return InferBlobShape(cv_img);
#else
    LOG(FATAL) << "Encoded datum requires OpenCV; compile with USE_OPENCV.";
#endif  // USE_OPENCV
  }
  const int crop_size = param_.crop_size();
  const int datum_channels = datum.channels();
  const int datum_height = datum.height();
  const int datum_width = datum.width();
  // Check dimensions.
  CHECK_GT(datum_channels, 0);
  CHECK_GE(datum_height, crop_size);
  CHECK_GE(datum_width, crop_size);
  // Build BlobShape.
  vector<int> shape(4);
  shape[0] = 1;
  shape[1] = datum_channels;
  shape[2] = (crop_size)? crop_size: datum_height;
  shape[3] = (crop_size)? crop_size: datum_width;
  return shape;
}

template<typename Dtype>
vector<int> DataTransformer<Dtype>::InferBlobShape(
    const vector<Datum> & datum_vector) {
  const int num = datum_vector.size();
  CHECK_GT(num, 0) << "There is no datum to in the vector";
  // Use first datum in the vector to InferBlobShape.
  vector<int> shape = InferBlobShape(datum_vector[0]);
  // Adjust num to the size of the vector.
  shape[0] = num;
  return shape;
}

#ifdef USE_OPENCV
template<typename Dtype>
vector<int> DataTransformer<Dtype>::InferBlobShape(const cv::Mat& cv_img) {
  const int crop_size = param_.crop_size();
  const int img_channels = cv_img.channels();
  const int img_height = cv_img.rows;
  const int img_width = cv_img.cols;
  // Check dimensions.
  CHECK_GT(img_channels, 0);
  CHECK_GE(img_height, crop_size);
  CHECK_GE(img_width, crop_size);
  // Build BlobShape.
  vector<int> shape(4);
  shape[0] = 1;
  shape[1] = img_channels;
  shape[2] = (crop_size)? crop_size: img_height;
  shape[3] = (crop_size)? crop_size: img_width;
  return shape;
}

template<typename Dtype>
vector<int> DataTransformer<Dtype>::InferBlobShape(
    const vector<cv::Mat> & mat_vector) {
  const int num = mat_vector.size();
  CHECK_GT(num, 0) << "There is no cv_img to in the vector";
  // Use first cv_img in the vector to InferBlobShape.
  vector<int> shape = InferBlobShape(mat_vector[0]);
  // Adjust num to the size of the vector.
  shape[0] = num;
  return shape;
}
#endif  // USE_OPENCV

template <typename Dtype>
void DataTransformer<Dtype>::InitRand() {
  const bool needs_rand = param_.mirror() ||
      (phase_ == TRAIN && param_.crop_size());
  if (needs_rand) {
    const unsigned int rng_seed = caffe_rng_rand();
    rng_.reset(new Caffe::RNG(rng_seed));
  } else {
    rng_.reset();
  }
}
template <typename Dtype>
void DataTransformer<Dtype>::InitRand2() {
    LOG(INFO)<< "Initializing rng";
    const unsigned int rng_seed = caffe_rng_rand();
    rng_.reset(new Caffe::RNG(rng_seed));
  
}
template <typename Dtype>
int DataTransformer<Dtype>::Rand(int n) {
  CHECK(rng_);
  CHECK_GT(n, 0);
  caffe::rng_t* rng =
      static_cast<caffe::rng_t*>(rng_->generator());
  return ((*rng)() % n);
}

template <typename Dtype>
float DataTransformer<Dtype>::Uniform(const float min, const float max) {
  CHECK(rng_);
  Dtype d[1];
  caffe_rng_uniform<Dtype>(1, Dtype(min), Dtype(max), d);
  return (float)d[0];
}
template<typename Dtype>
void DataTransformer<Dtype>::TransformImgSegDepthandnorm(
    const Dtype* img, const Dtype*depth, const Dtype*norm,const Dtype*mask,
    Dtype* img_data,
    Dtype* depth_data,
    Dtype* norm_data,
    Dtype* mask_data,
    int height_in, int width_in, int channels_in, int height_out,int width_out,
    bool do_jittering, bool do_mirror, vector<Dtype> *scale_factors) {

  // img -------- img
  // depth ------ depth
  // norm --------- 3d point
  // mask --------- label

  //CHECK(cv_img_seg.size() == 2) << "Input must contain image and seg.";
// LOG(INFO)<< "TransformImgSegDepthandnorm";
  const int img_channels = channels_in;
  // height and width may change due to pad for cropping
  int img_height   = height_in;
  int img_width    = width_in;
  //const int depth_channels = depth->channels();
  //int depth_height   = depth->height();
  //int depth_width    = depth->width();
  bool mirror = do_mirror && Rand(2);
  //const int norm_channels = norm->channels();
  //int norm_height = norm->height();
  //int norm_width =  norm->width();

  //const int data_channels = transformed_img_blob->channels();
  const int data_height   = height_out;
  const int data_width    = width_out;

  //LOG(INFO) << img_height<< ' '<< img_width<< ' '<< img_channels<< ' '<< data_height<<' '<<data_width;
  //const int label_channels = transformed_label_blob->channels();
  //const int label_height   = transformed_label_blob->height();
  //const int label_width    = transformed_label_blob->width();

 // const int crop_size = param_.crop_size();
  //const Dtype scale = param_.scale();
 // const bool do_mirror = param_.mirror() && Rand(2);
  //const bool has_mean_file = param_.has_mean_file();
 // const bool has_mean_values = mean_values_.size() > 0;

  // transform data into mat files
  cv::Mat cv_img(img_height, img_width, CV_32FC3);
  cv::Mat cv_depth(img_height, img_width, CV_32FC1);
  cv::Mat cv_norm(img_height, img_width, CV_32FC3);
  cv::Mat cv_mask(img_height, img_width, CV_32FC1);

   float* img_ptr_tmp;
   float* depth_ptr_tmp;
   float* norm_ptr_tmp;
   float* mask_ptr_tmp;
   int bottom_img_index;
   int tmp_bottom_img_index;
   int bottom_depth_index;
  /*const Dtype* img_data  = img->cpu_data();
  const Dtype* depth_data = depth->cpu_data();
  const Dtype* norm_data  = norm->cpu_data();
  const Dtype* mask_data = mask->cpu_data();*/
 // cv:: FileStorage file("img.txt",cv::FileStorage::WRITE);
  for (int h = 0; h < img_height; ++h) {
    img_ptr_tmp = cv_img.ptr<float>(h);
    depth_ptr_tmp = cv_depth.ptr<float>(h);
    norm_ptr_tmp = cv_norm.ptr<float>(h);
    mask_ptr_tmp = cv_mask.ptr<float>(h);

    int img_index = 0;
    int depth_index = 0;
    int norm_index = 0;
    int mask_index = 0;

    for (int w = 0; w < img_width; ++w) {
      // for image
      for (int c = 0; c < img_channels; ++c) {
        
          bottom_img_index = (c * img_height + h) * img_width + w;
          if(c==0) tmp_bottom_img_index = (2 * img_height + h) * img_width + w;
          if(c==1) tmp_bottom_img_index = (1 * img_height + h) * img_width + w;
          if(c==2) tmp_bottom_img_index = (0 * img_height + h) * img_width + w;
          if(c==0) img_ptr_tmp[img_index++] = img[tmp_bottom_img_index] + 122.675;
          if(c==1) img_ptr_tmp[img_index++] = img[tmp_bottom_img_index] + 116.669;
          if(c==2) img_ptr_tmp[img_index++] = img[tmp_bottom_img_index] + 104.008;
          //img_ptr_tmp[img_index++] = img[tmp_bottom_img_index] ;
          norm_ptr_tmp[norm_index++] = norm[bottom_img_index];
        //  LOG(INFO) << img_ptr_tmp[img_index-1]<< ' ' << norm_ptr_tmp[norm_index-1];

        }
        bottom_depth_index = h * img_width + w;
        depth_ptr_tmp[depth_index++] = depth[bottom_depth_index];
        mask_ptr_tmp[mask_index++] = mask[bottom_depth_index];
       // LOG(INFO) << depth_ptr_tmp[depth_index-1]<< ' ' << mask_ptr_tmp[depth_index-1];
    }
  }
  //file<<"img"<<cv_img;
 //LOG(INFO) << "Mat file parsing successful";
  //const int crop_size = param_.crop_size();
  int crop_width = width_out;
  int crop_height = height_out;
 /* LOG(INFO)<<  crop_width <<  crop_height;
  LOG(INFO)<<  param_.has_crop_height() <<  param_.has_crop_height();
  CHECK((!param_.has_crop_size() && param_.has_crop_height() && param_.has_crop_width())
  || (!param_.has_crop_height() && !param_.has_crop_width()))
    << "Must either specify crop_size or both crop_height and crop_width.";
  if (param_.has_crop_size()) {
    crop_width = param_.crop_size();
    crop_height = param_.crop_size();
  } 
  if (param_.has_crop_height() && param_.has_crop_width()) {
    LOG(INFO)<< crop_width<< ' '<<crop_width;
    crop_width = param_.crop_width();
    crop_height = param_.crop_height();
  }*/
  //LOG(INFO)<<crop_height<< ' '<< crop_width;

  //const Dtype scale = param_.scale();
  //const bool do_mirror = param_.mirror() && Rand(2);
  //const bool has_mean_file = 0;
  //const bool has_mean_values = mean_values_.size() > 0;
  //const bool do_jittering = param_.pca_jittering();
  Dtype noise[3];
  if(do_jittering)
  {
    
    caffe_rng_gaussian(3,Dtype(0.0),Dtype(0.1),noise);
  }

  CHECK_GT(img_channels, 0);

  //Dtype* mean = NULL;

  // start to perform transformation
  cv::Mat cv_cropped_img;
  cv::Mat cv_cropped_depth;
  cv::Mat cv_cropped_norm;
  cv::Mat cv_cropped_mask;

  // perform scaling
  Dtype scale = 1.0;
  //LOG(INFO)<< Rand(2);
  //LOG(INFO) << do_jittering;
  //LOG(INFO) << "start_scaling";
  //LOG(INFO) << scale_factors->size();
  if (scale_factors->size() > 0) {
   // LOG(INFO)<< "start";
   // LOG(INFO) << scale_factors->size();
    //LOG(INFO)<< Rand(2);
    int scale_ind = Rand(scale_factors->size());
   // LOG(INFO)<< scale_ind;
     scale   = scale_factors->at(scale_ind);
   // LOG(INFO)<<"scale**************************"<<scale;

    if (scale != 1) {
      img_height *= scale;
      img_width  *= scale;
      cv::resize(cv_img, cv_cropped_img, cv::Size(img_width, img_height), 0, 0, 
     cv::INTER_LINEAR);

      cv::resize(cv_depth, cv_cropped_depth, cv::Size(img_width, img_height), 0, 0, 
     cv::INTER_LINEAR);
      cv::resize(cv_norm, cv_cropped_norm, cv::Size(img_width, img_height), 0, 0, 
     cv::INTER_NEAREST);
      cv::resize(cv_mask, cv_cropped_mask, cv::Size(img_width, img_height), 0, 0, 
     cv::INTER_NEAREST);
    } else {
      cv_cropped_img = cv_img;
      cv_cropped_depth = cv_depth;
      cv_cropped_norm = cv_norm;
      cv_cropped_mask = cv_mask;
    }
  } else {
    cv_cropped_img = cv_img;
    cv_cropped_depth = cv_depth;
    cv_cropped_norm = cv_norm;
    cv_cropped_mask = cv_mask;
  }
  //
//LOG(INFO) << "end_scaling"<< img_height<<' '<<img_width;
  int h_off = 0;
  int w_off = 0;

  // transform to double, since we will pad mean pixel values
  //cv_cropped_img.convertTo(cv_cropped_img, CV_64F);

  // Check if we need to pad img to fit for crop_size
  // copymakeborder
  int pad_height = std::max(crop_height - img_height, 0);
  int pad_width  = std::max(crop_width - img_width, 0);
  if (pad_height > 0 || pad_width > 0) {
    cv::copyMakeBorder(cv_cropped_img, cv_cropped_img, 0, pad_height,
          0, pad_width, cv::BORDER_CONSTANT,
          cv::Scalar(Dtype(104.008), Dtype(116.669), Dtype(122.675))); // img
    cv::copyMakeBorder(cv_cropped_norm, cv_cropped_norm, 0, pad_height,
          0, pad_width, cv::BORDER_CONSTANT,
          cv::Scalar(Dtype(0.0), Dtype(0.0), Dtype(0.0))); // 3d point
    cv::copyMakeBorder(cv_cropped_depth, cv_cropped_depth, 0, pad_height,
          0, pad_width, cv::BORDER_CONSTANT,
          cv::Scalar(Dtype(0.0))); // depth
    cv::copyMakeBorder(cv_cropped_mask, cv_cropped_mask, 0, pad_height,
          0, pad_width, cv::BORDER_CONSTANT,
          cv::Scalar(Dtype(255))); // label
    // update height/width
    img_height   = cv_cropped_img.rows;
    img_width    = cv_cropped_img.cols;

    //depth_height   = cv_cropped_depth.rows;
    //depth_width    = cv_cropped_depth.cols;
  }

  // crop img/seg
  if (crop_width && crop_height) {
    CHECK_EQ(crop_height, data_height);
    CHECK_EQ(crop_width, data_width);
    // We only do random crop when we do training.
    if (phase_ == TRAIN) {
      h_off = Rand(img_height - crop_height + 1);
      w_off = Rand(img_width - crop_width + 1);
    } else {
      // CHECK: use middle crop
      h_off = (img_height - crop_height) / 2;
      w_off = (img_width - crop_width) / 2;
    }
    cv::Rect roi(w_off, h_off, crop_width, crop_height);
    cv_cropped_img = cv_cropped_img(roi);
    cv_cropped_depth = cv_cropped_depth(roi);
    cv_cropped_norm = cv_cropped_norm(roi);
    cv_cropped_mask = cv_cropped_mask(roi);
  }

  CHECK(cv_cropped_img.data);
  CHECK(cv_cropped_depth.data);
  CHECK(cv_cropped_norm.data);
  CHECK(cv_cropped_mask.data);

 /* Dtype* transformed_img_data  = transformed_img_blob->mutable_cpu_data();
  Dtype* transformed_depth_data = transformed_depth_blob->mutable_cpu_data();
  Dtype* transformed_norm_data  = transformed_norm_blob->mutable_cpu_data();
  Dtype* transformed_mask_data = transformed_mask_blob->mutable_cpu_data();
*/
  int top_index;
  const float* img_ptr;
  const float* depth_ptr;
  const float* norm_ptr;
  const float* mask_ptr;
  //const uchar* label_ptr;

  for (int h = 0; h < data_height; ++h) {
    img_ptr = cv_cropped_img.ptr<float>(h);
    depth_ptr = cv_cropped_depth.ptr<float>(h);
    norm_ptr = cv_cropped_norm.ptr<float>(h);
    mask_ptr = cv_cropped_mask.ptr<float>(h);

    int img_index = 0;
    int depth_index = 0;
    int norm_index = 0;
    int mask_index = 0;

    for (int w = 0; w < data_width; ++w) {
      // for image
      for (int c = 0; c < img_channels; ++c) {
        if (mirror) {
          top_index = (c * data_height + h) * data_width + (data_width - 1 - w);
        } else {
          top_index = (c * data_height + h) * data_width + w;
        }
         // top_index = (c * data_height + h) * data_width + w;
        
        Dtype pixel = static_cast<Dtype>(img_ptr[img_index++]);
        Dtype pixel_norm = static_cast<Dtype>(norm_ptr[norm_index++]);
        //if(c==0 && mirror)
        //{
        //  pixel_norm = -1.0*pixel_norm ;
        //}
        if(do_jittering)
        {   
          pixel = pixel/255.0;
          if(c==0)
            pixel = pixel - 104.008/255.0 + noise[0]*(-0.4481) +noise[1]*(-0.0768)+noise[2]*(0.0173);
          else if(c==1)
             pixel = pixel- 116.669/255.0 + noise[0]*(-0.4621) +noise[1]*(-0.0066)+noise[2]*(-0.0341);
          else if(c==2)
            pixel = pixel - 122.675/255.0 + noise[0]*(-0.5241) +noise[1]*(0.0719)+noise[2]*(0.0154);
          pixel = pixel*255.0;
        }

            img_data[top_index] = pixel ;
            norm_data[top_index] = pixel_norm;
         }

      // for segmentation
       if (mirror) {
          top_index = h * data_width + (data_width - 1 - w);
        } else {
          top_index = h * data_width + w;
        }
       //top_index = h * data_width + w;
      
      Dtype pixel = static_cast<Dtype>(depth_ptr[depth_index++]);
      Dtype pixel_mask = static_cast<Dtype>(mask_ptr[mask_index++]);
      depth_data[top_index] = pixel/scale;
      mask_data[top_index] = pixel_mask;
    }
  }
}


template<typename Dtype>
void DataTransformer<Dtype>::TransformImgSegDepthandnormimg(
    const Dtype* img, 
    const Dtype* depth_pred, 
    const Dtype* norm_pred,
    const Dtype* label,
    const Dtype* depth, 
    const Dtype* norm,
    const Dtype* mask,
    //const Dtype* semantic,
    Dtype* img_data,
    Dtype* depth_pred_data,
    Dtype* norm_pred_data,
    Dtype* label_data,
    Dtype* depth_data,
    Dtype* norm_data,
    Dtype* mask_data,
    //Dtype* semantic_data,
    int height_in, int width_in, int channels_in, int height_out,int width_out,
    bool do_jittering, bool do_mirror, vector<Dtype> *scale_factors) {
  const int img_channels = channels_in;
  // height and width may change due to pad for cropping
  int img_height   = height_in;
  int img_width    = width_in;
  
  bool mirror = do_mirror && Rand(2);
  
  const int data_height   = height_out;
  const int data_width    = width_out;


  // transform data into mat files
  cv::Mat cv_img(img_height, img_width, CV_32FC3);
  cv::Mat cv_depth(img_height, img_width, CV_32FC1);
  cv::Mat cv_norm(img_height, img_width, CV_32FC3);
  cv::Mat cv_mask(img_height, img_width, CV_32FC1);
  cv::Mat cv_depth_pred(img_height, img_width, CV_32FC1);
  cv::Mat cv_norm_pred(img_height, img_width, CV_32FC3);
  cv::Mat cv_label(img_height, img_width, CV_32FC1);
  //cv::Mat cv_semantic(img_height, img_width, CV_32FC(40));

   float* img_ptr_tmp;
   float* depth_ptr_tmp;
   float* norm_ptr_tmp;
   float* mask_ptr_tmp;
   float* label_ptr_tmp;
   float* depth_pred_ptr_tmp;
   float* norm_pred_ptr_tmp;
   //float* semantic_ptr_tmp;
   int bottom_img_index;
   int bottom_depth_index;
   //int bottom_semantic_index;
   int bottom_img_index_tmp;
   //LOG(INFO)<< "imgintomat";
  
  for (int h = 0; h < img_height; ++h) {
    img_ptr_tmp = cv_img.ptr<float>(h);
    depth_ptr_tmp = cv_depth.ptr<float>(h);
    norm_ptr_tmp = cv_norm.ptr<float>(h);
    mask_ptr_tmp = cv_mask.ptr<float>(h);
    label_ptr_tmp = cv_label.ptr<float>(h);
    norm_pred_ptr_tmp = cv_norm_pred.ptr<float>(h);
    depth_pred_ptr_tmp = cv_depth_pred.ptr<float>(h);
    //semantic_ptr_tmp  = cv_semantic.ptr<float>(h);
    

    int img_index = 0;
    int depth_index = 0;
    int norm_index = 0;
    int mask_index = 0;
    int label_index = 0;
    int depth_pred_index = 0;
    int norm_pred_index = 0;
    //int semantic_index = 0;

    for (int w = 0; w < img_width; ++w) {
      // for image
      for (int c = 0; c < img_channels; ++c) {
          
          bottom_img_index = (c * img_height + h) * img_width + w;
          if(c==0) bottom_img_index_tmp = (2 * img_height + h) * img_width + w;
          if(c==1) bottom_img_index_tmp = (1 * img_height + h) * img_width + w;
          if(c==2) bottom_img_index_tmp = (0 * img_height + h) * img_width + w;
          img_ptr_tmp[img_index++] = img[bottom_img_index_tmp]*255.0 ;
          norm_ptr_tmp[norm_index++] = norm[bottom_img_index];
          norm_pred_ptr_tmp[norm_pred_index++] = norm_pred[bottom_img_index];
        //  LOG(INFO) << img_ptr_tmp[img_index-1]<< ' ' << norm_ptr_tmp[norm_index-1];

        }
        //LOG(INFO) << h<<w;
      /*for(int c = 0; c< 40; ++c)

      {
         
         bottom_semantic_index = (c * img_height + h) * img_width + w;
       // if(h==12&&w==86) LOG(INFO) <<semantic[bottom_semantic_index];
         semantic_ptr_tmp[semantic_index++] = semantic[bottom_semantic_index];


      }*/
        //LOG(INFO)<< h << w;
        bottom_depth_index = h * img_width + w;
        depth_ptr_tmp[depth_index++] = depth[bottom_depth_index];
        mask_ptr_tmp[mask_index++] = mask[bottom_depth_index];
        depth_pred_ptr_tmp[depth_pred_index++] = depth_pred[bottom_depth_index];
        label_ptr_tmp[label_index++] = label[bottom_depth_index];
       // LOG(INFO) << depth_ptr_tmp[depth_index-1]<< ' ' << mask_ptr_tmp[depth_index-1];
    }
  }
 // LOG(INFO)<<"endimgtomat";
  //file<<"img"<<cv_img;
 //LOG(INFO) << "Mat file parsing successful";
  //const int crop_size = param_.crop_size();
  int crop_width = width_out;
  int crop_height = height_out;
 
  Dtype noise[3];
  if(do_jittering)
  {
    
    caffe_rng_gaussian(3,Dtype(0.0),Dtype(0.1),noise);
  }

  CHECK_GT(img_channels, 0);

  //Dtype* mean = NULL;

  // start to perform transformation
  cv::Mat cv_cropped_img;
  cv::Mat cv_cropped_depth;
  cv::Mat cv_cropped_norm;
  cv::Mat cv_cropped_mask;
  cv::Mat cv_cropped_depth_pred;
  cv::Mat cv_cropped_norm_pred;
  cv::Mat cv_cropped_label;
  //cv::Mat cv_cropped_semantic;

  // perform scaling
  Dtype scale = 1.0;
  
  if (scale_factors->size() > 0) {
   // LOG(INFO)<< "start";
   // LOG(INFO) << scale_factors->size();
    //LOG(INFO)<< Rand(2);
    int scale_ind = Rand(scale_factors->size());
   // LOG(INFO)<< scale_ind;
     scale   = scale_factors->at(scale_ind);
   // LOG(INFO)<<"scale**************************"<<scale;

    if (scale != 1) {
      img_height *= scale;
      img_width  *= scale;
      cv::resize(cv_img, cv_cropped_img, cv::Size(img_width, img_height), 0, 0, 
     cv::INTER_LINEAR);
      cv::resize(cv_depth, cv_cropped_depth, cv::Size(img_width, img_height), 0, 0, 
     cv::INTER_LINEAR);
      cv::resize(cv_norm, cv_cropped_norm, cv::Size(img_width, img_height), 0, 0, 
     cv::INTER_NEAREST);
      cv::resize(cv_mask, cv_cropped_mask, cv::Size(img_width, img_height), 0, 0, 
     cv::INTER_NEAREST);
    cv::resize(cv_depth_pred,cv_cropped_depth_pred,cv::Size(img_width, img_height), 0, 0, 
     cv::INTER_LINEAR);
    cv::resize(cv_norm_pred, cv_cropped_norm_pred, cv::Size(img_width, img_height), 0, 0, 
     cv::INTER_NEAREST);
    cv::resize(cv_label, cv_cropped_label, cv::Size(img_width, img_height), 0, 0, 
     cv::INTER_NEAREST);
    //cv::resize(cv_semantic, cv_cropped_semantic, cv::Size(img_width, img_height), 0, 0, 
     //cv::INTER_LINEAR);
    } else {
      cv_cropped_img = cv_img;
      cv_cropped_depth = cv_depth;
      cv_cropped_norm = cv_norm;
      cv_cropped_mask = cv_mask;
      cv_cropped_norm_pred = cv_norm_pred;
      cv_cropped_depth_pred = cv_depth_pred;
      cv_cropped_label = cv_label;
     // cv_cropped_semantic = cv_semantic;
    }
  } else {
    cv_cropped_img = cv_img;
    cv_cropped_depth = cv_depth;
    cv_cropped_norm = cv_norm;
    cv_cropped_mask = cv_mask;
    cv_cropped_depth_pred = cv_depth_pred;
    cv_cropped_norm_pred = cv_norm_pred;
    cv_cropped_label = cv_label;
   // cv_cropped_semantic = cv_semantic;
  }
  //
//LOG(INFO) << "end_scaling"<< img_height<<' '<<img_width;
  int h_off = 0;
  int w_off = 0;

  // transform to double, since we will pad mean pixel values
  //cv_cropped_img.convertTo(cv_cropped_img, CV_64F);

  // Check if we need to pad img to fit for crop_size
  // copymakeborder
  int pad_height = std::max(crop_height - img_height, 0);
  int pad_width  = std::max(crop_width - img_width, 0);
  if (pad_height > 0 || pad_width > 0) {
    cv::copyMakeBorder(cv_cropped_img, cv_cropped_img, 0, pad_height,
          0, pad_width, cv::BORDER_CONSTANT,
          cv::Scalar(Dtype(104.008), Dtype(116.669), Dtype(122.675)));
    cv::copyMakeBorder(cv_cropped_norm, cv_cropped_norm, 0, pad_height,
          0, pad_width, cv::BORDER_CONSTANT,
          cv::Scalar(Dtype(0.0), Dtype(0.0), Dtype(0.0)));
    cv::copyMakeBorder(cv_cropped_depth, cv_cropped_depth, 0, pad_height,
          0, pad_width, cv::BORDER_CONSTANT,
          cv::Scalar(Dtype(0.0)));
    cv::copyMakeBorder(cv_cropped_mask, cv_cropped_mask, 0, pad_height,
          0, pad_width, cv::BORDER_CONSTANT,
          cv::Scalar(Dtype(0.0)));
    cv::copyMakeBorder(cv_cropped_depth_pred, cv_cropped_depth_pred, 0, pad_height,
          0, pad_width, cv::BORDER_CONSTANT,
          cv::Scalar(Dtype(0.0)));
   cv::copyMakeBorder(cv_cropped_norm_pred, cv_cropped_norm_pred, 0, pad_height,
          0, pad_width, cv::BORDER_CONSTANT,
          cv::Scalar(Dtype(0.0), Dtype(0.0), Dtype(0.0)));
   cv::copyMakeBorder(cv_cropped_label, cv_cropped_label, 0, pad_height,
          0, pad_width, cv::BORDER_CONSTANT,
          cv::Scalar(Dtype(255)));
   //cv::copyMakeBorder(cv_cropped_semantic, cv_cropped_semantic, 0, pad_height,
    //      0, pad_width, cv::BORDER_REPLICATE
       //    );
    // update height/width
    img_height   = cv_cropped_img.rows;
    img_width    = cv_cropped_img.cols;

    //depth_height   = cv_cropped_depth.rows;
    //depth_width    = cv_cropped_depth.cols;
  }

  // crop img/seg
  if (crop_width && crop_height) {
    CHECK_EQ(crop_height, data_height);
    CHECK_EQ(crop_width, data_width);
    // We only do random crop when we do training.
    if (phase_ == TRAIN) {
      h_off = Rand(img_height - crop_height + 1);
      w_off = Rand(img_width - crop_width + 1);
    } else {
      // CHECK: use middle crop
      h_off = (img_height - crop_height) / 2;
      w_off = (img_width - crop_width) / 2;
    }
    cv::Rect roi(w_off, h_off, crop_width, crop_height);
    cv_cropped_img = cv_cropped_img(roi);
    cv_cropped_depth = cv_cropped_depth(roi);
    cv_cropped_norm = cv_cropped_norm(roi);
    cv_cropped_mask = cv_cropped_mask(roi);
    cv_cropped_depth_pred = cv_cropped_depth_pred(roi);
    cv_cropped_norm_pred = cv_cropped_norm_pred(roi);
    cv_cropped_label = cv_cropped_label(roi);
   // cv_cropped_semantic = cv_cropped_semantic(roi);
  }

  CHECK(cv_cropped_img.data);
  CHECK(cv_cropped_depth.data);
  CHECK(cv_cropped_norm.data);
  CHECK(cv_cropped_mask.data);
  CHECK(cv_cropped_depth_pred.data);
  CHECK(cv_cropped_norm_pred.data);
  CHECK(cv_cropped_label.data);
  //CHECK(cv_cropped_semantic.data);

  int top_index;
  const float* img_ptr;
  const float* depth_ptr;
  const float* norm_ptr;
  const float* mask_ptr;
  const float* depth_pred_ptr;
  const float* norm_pred_ptr;
  const float* label_ptr;
  //const float* semantic_ptr;
  //const uchar* label_ptr;

  for (int h = 0; h < data_height; ++h) {
    img_ptr = cv_cropped_img.ptr<float>(h);
    depth_ptr = cv_cropped_depth.ptr<float>(h);
    norm_ptr = cv_cropped_norm.ptr<float>(h);
    mask_ptr = cv_cropped_mask.ptr<float>(h);
    depth_pred_ptr = cv_cropped_depth_pred.ptr<float>(h);
    norm_pred_ptr = cv_cropped_norm_pred.ptr<float>(h);
    label_ptr = cv_cropped_label.ptr<float>(h);
    //semantic_ptr = cv_cropped_semantic.ptr<float>(h);

    int img_index = 0;
    int depth_index = 0;
    int norm_index = 0;
    int mask_index = 0;
    int depth_pred_index = 0;
    int norm_pred_index = 0;
    int label_index = 0;
    //int semantic_index = 0;

    for (int w = 0; w < data_width; ++w) {
      // for image
      for (int c = 0; c < img_channels; ++c) {
        if (mirror) {
          top_index = (c * data_height + h) * data_width + (data_width - 1 - w);
        } else {
          top_index = (c * data_height + h) * data_width + w;
        }
         // top_index = (c * data_height + h) * data_width + w;
        
        Dtype pixel = static_cast<Dtype>(img_ptr[img_index++]);
        Dtype pixel_norm = static_cast<Dtype>(norm_ptr[norm_index++]);
        Dtype pixel_norm_pred = static_cast<Dtype>(norm_pred_ptr[norm_pred_index++]);
        if(c==0 && mirror)
        {
          pixel_norm = -1.0*pixel_norm ;
          pixel_norm_pred = -1.0* pixel_norm_pred;
        }
        if(do_jittering)
        {   
          pixel = pixel/255.0;
          if(c==0)
            pixel = pixel - 104.008/255.0 + noise[0]*(-0.4481) +noise[1]*(-0.0768)+noise[2]*(0.0173);
          else if(c==1)
             pixel = pixel- 116.669/255.0 + noise[0]*(-0.4621) +noise[1]*(-0.0066)+noise[2]*(-0.0341);
          else if(c==2)
            pixel = pixel - 122.675/255.0 + noise[0]*(-0.5241) +noise[1]*(0.0719)+noise[2]*(0.0154);
          pixel = pixel*255.0;
        }

            img_data[top_index] = pixel ;
            norm_data[top_index] = pixel_norm;
            norm_pred_data[top_index] = pixel_norm_pred;
         }
         /*for(int c = 0; c< 40; ++c)
         {
            if (mirror) {
             top_index = (c * data_height + h) * data_width + (data_width - 1 - w);
             } else {
              top_index = (c * data_height + h) * data_width + w;
            }
            Dtype pixel_semantic = static_cast<Dtype>(semantic_ptr[semantic_index++]);
            semantic_data[top_index] = pixel_semantic;
         }*/

      // for segmentation
       if (mirror) {
          top_index = h * data_width + (data_width - 1 - w);
        } else {
          top_index = h * data_width + w;
        }
       //top_index = h * data_width + w;
      
      Dtype pixel = static_cast<Dtype>(depth_ptr[depth_index++]);
      Dtype pixel_mask = static_cast<Dtype>(mask_ptr[mask_index++]);
      Dtype pixel_depth_pred = static_cast<Dtype>(depth_pred_ptr[depth_pred_index++]);
      Dtype pixel_label = static_cast<Dtype>(label_ptr[label_index++]);
      depth_data[top_index] = pixel/scale;
      mask_data[top_index] = pixel_mask;
      depth_pred_data[top_index] = pixel_depth_pred/scale;
      label_data[top_index] = pixel_label;
    }
  }
}

INSTANTIATE_CLASS(DataTransformer);

}  // namespace caffe
