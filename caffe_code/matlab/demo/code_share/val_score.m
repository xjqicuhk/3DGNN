function d = val_score()
list = load('../data/splits.mat');
list = list.testNdxs;
%list_path = '/mnt/sdd1/hszhao/cityscapes/list/fine_val_id.txt';
%list_path = '/mnt/sdc1/hszhao/cityscapes/list/fine_train_id.txt';
%predict_folder = '/mnt/sdb1/hszhao/DeepLab_MSc_LargeFOV/crfbintopng/';
predict_folder = './result_lstm/';
%predict_folder = '/mnt/sdd1/xjqi/nyu_code/networkresult/semantic_png/';
root_folder = '../data/label/';
%list_gt = importdata(fullfile(root_folder,'list/fine_val.txt'));
%list_gt = importdata(fullfile(root_folder,'list/fine_train.txt'));
num_class = 40;
eps  = 0.0;
%list = importdata(list_path);
confusionmatrix =zeros(num_class,num_class);
load('cityscapes_colormap.mat');


for i = 1:size(list,1)
    i
%     str = strsplit(list_gt{i});
%     gt_path = str{2};
%     img_path = str{1};
%     imLab = imread(fullfile(root_folder, gt_path));
     imLab = imread([root_folder num2str(list(i)) '.png']);
     imLab(imLab ~= 255) = imLab(imLab ~= 255) + 1;
     imPred = imread([predict_folder num2str(list(i)) '.png']);
     imPred = imPred + 1;
%    data = load(['/mnt/sdd1/xjqi/nyu_code/networkresult/combine_module/' sprintf('%04d',list(i)) '.mat']);
%    data = data.feature;
%    data = data(:,:,1:40);
%    [~,imPred] = max(data,[],3);
    %gt(gt==255) = 19;
    imLab (imLab == 255) = 0;
%     imLab(imLab == 38) = 0;
%     imLab(imLab==39) = 0;
%     imLab(imLab==40) = 0;
%     imPred(imPred==38) = imLab(imPred==38);
%     imPred(imPred==39) = imLab(imPred==39);
%     imPred(imPred==40) = imLab(imPred==40);
    
    imLab = imLab(45:471, 41:601,:);
    imPred = imPred(45:471, 41:601,:);
    %imPred = imresize(imLab,0.375,'nearest');
    %imPred = imresize(imPred,[size(imLab,1), size(imLab,2)],'nearest');
    %predict = imresize(gt,0.5,'nearest');
    %predict = imresize(predict,2,'nearest');
    %predict = imread([predict_folder list{i}(1:end-14) '.png']);
    %gt= gt(1:256,1:256);
    %predict = predict(1:256,1:256);
    %confusionmatrix(gt+1,predict+1) = confusionmatrix(gt+1,predict+1)+1;
     [area_intersection(:,i), area_union(:,i)]=intersectionAndUnion(imPred, imLab,num_class);
 end
 IoU = sum(area_intersection,2)./sum(eps+area_union,2);
 d = sum(IoU(:)) / num_class
end