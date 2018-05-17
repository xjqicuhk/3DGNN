clc, clear all;
file_name = '/home/xjqi/Research/deeplab_v2/deeplab-public-ver2_bn_layer/prototxts/cityscape/train.log';
%file_name = '/home/hszhao/research/deeplab/models/cityscapes/multiscale_largeFOV/former/train2.log';
%file_name = '/home/hszhao/research/deeplab/models/cityscapes/MSc_multiscale/train.log';
%file_name = '/home/hszhao/research/caffe_sg/models/ResNet50/train.log';
%file_name = '/home/hszhao/Downloads/caffe.bin.IDC1-10-10-10-105.shijianping.log.INFO.20160510-220939.10246';
%file_name = '/home/hszhao/research/deeplab/models/cityscapes/half_deeplab_largeFOV/train.log';
fid = fopen(file_name,'r');
key1 = 'Iteration';
key2 = 'loss';
key3 = '#2:';
key4 = '#0:';
iter = [];
loss = [];
acc =  [];
iou = [];
line_num = 1;
tline = fgets(fid);
while ischar(tline)
    line_num;
    if(~isempty(findstr(key1,tline)) && ~isempty(findstr(key2,tline)))
        %loss = {loss,tline(findstr(key1,tline):end)};
        line_str = tline(findstr(key1,tline):end);
        iter_loss = sscanf(line_str,'%*s %d %*c %*s %*c %f');
        iter = [iter;iter_loss(1)];
        loss = [loss;iter_loss(2)];
    end
     if(~isempty(findstr(key4,tline)))
         line_str = tline(findstr(key4,tline):end);
         iter_acc = sscanf(line_str,'%*s %*s %*c %f');
         acc = [acc;iter_acc];
     end
    if(~isempty(findstr(key3,tline)))
         line_str = tline(findstr(key3,tline):end);
         iter_iou = sscanf(line_str,'%*s %*s %*c %f');
         iou = [iou;iter_iou];
     end
    tline = fgets(fid);
    line_num = line_num + 1;
end
fclose(fid);
disp('finished');
figure;
plot(iter(100:end),loss(100:end));
title('loss');
figure;
plot(iter(100:end-1),acc(100:end));
title('acc');
figure;
plot(iter(100:end-1),iou(100:end));
title('iou');

% clc, clear all;
% test_name = '/home/xjqi/research/caffe_master_hszhao/scene/model_map60/60_test95000.txt';
% cluster_name = '/home/xjqi/research/dataset/places/trainvalsplit_places205/60_clusters.txt';
% acc = importdata(test_name);
% fid = fopen(cluster_name,'r');
% line_num = 1;
% tline = fgets(fid);
% mean_score = zeros(60,1);
% while ischar(tline)
%     clusters = textscan(tline,'%d');
%     clusters = clusters{1};
%     score = 0;
%     for i=1:size(clusters,1)
%         score = score + acc(clusters(i)+1);
%     end
%     score = score / size(clusters,1);
%     mean_score(line_num) = score;
%     tline = fgets(fid);
%     line_num = line_num + 1;
% end
% fclose(fid);
% figure;
% plot([1:1:60],mean_score);




