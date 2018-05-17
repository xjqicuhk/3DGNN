clear all, clc;
if exist('../+caffe', 'dir')
  addpath('..');
else
  error('Please run this demo from caffe/matlab/demo');
end
caffe.set_mode_gpu();
gpu_id = 0;
caffe.set_device(gpu_id);

% Initialize the network using BVLC CaffeNet for image classification
% Weights (parameter) file needs to be downloaded from Model Zoo.
%model_dir = '/home/xjqi/Research/deeplab_v2/prototxt_and_model/Resnet_50/';
phase = 'test';
model_original = '/home/xjqi/Research/deeplab_v2/Resnet/ResNet-50-deploy.prototxt';
original_weights = '/home/xjqi/Research/deeplab_v2/Resnet/ResNet-50-model.caffemodel';
model_new      = '/home/xjqi/Research/deeplab_v2/Resnet/resnet-50-bn_deploy.prototxt';
net_original =  caffe.Net(model_original, original_weights, phase);
net_new =  caffe.Net(model_new, original_weights, phase);
bn_index_new = [];
for  i =1 :size(net_new.layer_names,1)
    if(net_new.layer_names{i}(1)=='b'&net_new.layer_names{i}(2)=='n')
        bn_index_new(end+1) = i;
    end
end
bn_index = [];
for i = 1:size(net_original.layer_names,1)
    if(net_original.layer_names{i}(1)=='b'&net_original.layer_names{i}(2)=='n')
        bn_index(end+1) = i;
    end
end
for i  = 1:1:size(bn_index,2)
    i
    scale = net_original.layer_vec(bn_index(i)+1).params(1,1).get_data();
    shift = net_original.layer_vec(bn_index(i)+1).params(1,2).get_data();
    mean = net_original.layer_vec(bn_index(i)).params(1,1).get_data();
    variance = net_original.layer_vec(bn_index(i)).params(1,2).get_data();
    scale = reshape(scale,[1 1 size(scale,1)]);
    shift = reshape(shift,[1 1 size(shift,1)]);
    mean  = reshape(mean,[1 1 size(mean,1)]);
    variance = reshape(variance,[1,1,size(variance,1)]);
    net_new.layer_vec(bn_index_new(i)).params(1,1).set_data(scale);
    net_new.layer_vec(bn_index_new(i)).params(1,2).set_data(shift);
    net_new.layer_vec(bn_index_new(i)).params(1,3).set_data(mean);
    net_new.layer_vec(bn_index_new(i)).params(1,4).set_data(variance);
end
net_new.save('resnet_50_bn.caffemodel');
pause;



