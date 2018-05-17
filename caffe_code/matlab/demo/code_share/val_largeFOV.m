clear all, clc;
if exist('../+caffe', 'dir')
  addpath('..');
else
  error('Please run this demo from caffe/matlab/demo');
end
caffe.set_mode_gpu();
gpu_id = 3;
caffe.set_device(gpu_id);

% Initialize the network using BVLC CaffeNet for image classification
% Weights (parameter) file needs to be downloaded from Model Zoo.
model_dir = '/home/xjqi/Research/deeplab_v2/prototxt_and_model/Resnet_50/';
net_model = [model_dir 'deploy_sg.prototxt'];
net_weights = ['/home/xjqi/Research/deeplab_v2/prototxt_and_model/Resnet_50/model/train_iter_120000.caffemodel'];

% model_dir = '/home/hszhao/research/caffe/models/bvlc_reference_caffenet/';
% net_model = [model_dir 'deploy.prototxt'];
% net_weights = [model_dir 'bvlc_reference_caffenet.caffemodel'];

phase = 'test'; % run with phase test (so that dropout isn't applied)
if ~exist(net_weights, 'file')
  error('Please download CaffeNet from Model Zoo before you run this demo');
end

% Initialize a network
net = caffe.Net(net_model, net_weights, phase);

%open
save_png_folder = '/home/xjqi/Research/deeplab_v2/prototxt_and_model/Resnet_50/test/';
mkdir(save_png_folder);
save_bin_folder = '/mnt/sdb1/hszhao/ResNet50_largeFOV/fc8tobin/';
root_folder = '/mnt/sdd1/hszhao/cityscapes';
list = importdata(fullfile(root_folder,'list/fine_val.txt'));
if(~isdir(save_png_folder))
    mkdir(save_png_folder);
end
if(~isdir(save_bin_folder))
    mkdir(save_bin_folder);
end

img_row = 1024;
img_col = 2048;
img_cha = 19;
CROPPED_DIM = 513;%1025;%609;
load('cityscapes_colormap.mat');
show_image = false;
save_bin = false;%148M per
scale_array = [1]%[0.5 1 1.5 2]


for i = 1:numel(list)
    fprintf(1, 'processing %d (%d)...\n', i, numel(list));
    data_all = zeros(img_row,img_col,img_cha,'single');
    str = strsplit(list{i});
    img_path = str{1};
    img = imread(fullfile(root_folder,img_path));
    for j = 1:size(scale_array,2)
        data_all = data_all + scale_process(net,img,scale_array(j),img_cha,CROPPED_DIM);
    end
    data_all = data_all/size(scale_array,2);
    
    data = data_all;%already exp process
    
    img_fn = strsplit(img_path,'/');
    img_fn = img_fn{end};
    img_fn = img_fn(1:end-4);
    img_fn = strrep(img_fn,'leftImg8bit','gtFine_labelTrainIds');
    if save_bin
        save_fn = [save_bin_folder img_fn '.bin'];
        SaveBinFile(data, save_fn, 'float');
    end
    
    [~,predict] = max(data,[],3); 
    predict = uint8(predict);
    imwrite(predict-1,colormap,[save_png_folder img_fn, '.png']);
    if show_image
        %str = strsplit(list{i});
        %img_path = str{1};
        gt_path = str{2};
        %img = imread(fullfile(root_folder,img_path));
        gt = imread(fullfile(root_folder, gt_path));
        figure(1), 
        subplot(221),imshow(img), title('img');
        subplot(222),imshow(gt, colormap), title('gt');
        subplot(224), imshow(predict-1,colormap), title('predict');
        %pause;
    end
end
caffe.reset_all();

val_acc;