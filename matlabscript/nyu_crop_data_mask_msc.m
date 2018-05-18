%clear all, clc;

addpath(genpath('/research/leojia2/xjqi/deeplab_lstm_end_3d_knn/matlab'));
addpath(genpath('./crop_test/'));
caffe.set_mode_gpu();
gpu_id = 3;
caffe.set_device(gpu_id);
result = [];

  

net_model ='../model/nyu_40/train_lstm.prototxt';
net_weights = '../model/nyu_40/train_iter_30000.caffemodel';

phase = 'test'; % run with phase test (sclearo that dropout isn't applied)
if ~exist(net_weights, 'file')
  error('Please download CaffeNet from Model Zoo before you run this demo');
end

% Initialize a network
net = caffe.Net(net_model, net_weights, phase);

%result save folder
save_png_folder = '../result/nyu_40/';
mkdir(save_png_folder);
% number of class and cropsize
num_class = 40;
img_cha = 40;
load('cityscapes_colormap.mat');
%scales = [1.0]
show_image = false;

%crop size and stride, for nyu dataset we directly use the original image
%and do not use crop
crop_size_h = 481;
crop_size_w  = 641;
batch_size = 1;
stride_h = 240;
stride_w = 320;
img_row = 480;
img_col = 640;
mean_value = [122.675,116.669,104.008];

% input path
feature_path = '../testdata/nyu_40/';

list = load('./splits.mat');
list = list.testNdxs;
list = unique(list);
scales = [0.5, 0.8,1.0,1.2,1.5];
 for i = 1:1:numel(list)
    
    fprintf(1, 'processing %d (%d)...\n', list(i), numel(list));
    data_all = zeros(img_row,img_col,img_cha,'single');
    
    % load data
   % data0 = load([feature_path num2str(list(i)) '.mat']);
   data0 = load([feature_path num2str(list(i)) '.mat']);
  %  data0 = load('0001.mat');
    for s = 1:size(scales,2)
        img = single(squeeze(data0.image));
        tmp = zeros(img_row+1,img_col+1,3)+128;
        tmp(1:img_row,1:img_col,:) = img;
        img = tmp;
        img = imresize(img,scales(s),'bilinear');

        depth = data0.depth;
        tmp = zeros(img_row+1,img_col+1,'single');
        tmp(1:img_row,1:img_col) = depth;
        depth = tmp;
        depth = imresize(depth,scales(s),'bilinear');
        
        img(:,:,1) = img(:,:,1) - mean_value(1);
        img(:,:,2) = img(:,:,2) - mean_value(2);
        img(:,:,3) = img(:,:,3) - mean_value(3);
        tmp_img = img;
        tmp_img(:,:,1) = img(:,:,3);
        tmp_img(:,:,2) = img(:,:,2);
        tmp_img(:,:,3) = img(:,:,1);
        img = tmp_img;
        input_row = size(img,1);
        input_col = size(img,2);
        if(size(img,1) <= crop_size_h && size(img,2) <= crop_size_w)
            img = padarray(img,[crop_size_h - size(img,1), crop_size_w - size(img,2),0],'post');
            depth = padarray(depth,[crop_size_h - size(depth,1), crop_size_w - size(depth,2),0],'post');
            feature = cat(3, img, depth);
            feature = permute(feature,[2 1 3]);
            score = net.forward({feature});
            score = score{1};
            score = permute(score,[2 1 3]);
        else
          if(size(img,1) > crop_size_h && size(img,2) <= crop_size_w)
              img = padarray(img,[0, crop_size_w - size(img,2),0],'post');
              depth = padarray(depth,[0, crop_size_w - size(depth,2),0],'post');
          end
          if(size(img,1) <= crop_size_h && size(img,2) > crop_size_w)
              img = padarray(img,[crop_size_h - size(img,1), 0,0],'post');
              depth = padarray(depth,[crop_size_h - size(depth,1),0,0],'post');
          end
            feature = cat(3,img,depth);
            feature = permute(feature,[2 1 3]);
            score = caffe_process_batch_score(net,feature,1,stride_w,stride_h,crop_size_w,crop_size_h,...
                batch_size,num_class);
            score = permute(score,[2 1 3]);
        
        end
        data_all = data_all + imresize(score(1:input_row,1:input_col,:),[img_row, img_col], 'bilinear');
        
    end 
    

    data = data_all;
  
    data = data(1:img_row,1:img_col,:);
    
   
    [~,predict] = max(data,[],3); 
    predict = uint8(predict);
    imwrite(predict-1,colormap,[save_png_folder num2str(list(i)), '.png']);
    if show_image
        
        
        figure(1),
        imshow(predict-1,colormap);
        pause;
    end
   
end
caffe.reset_all();
 
 ['evaluation...']
  iou = val_score;
  acc = val_accuracy;
 ['iou:']
  iou
  ['accuracy:']
  acc
%end