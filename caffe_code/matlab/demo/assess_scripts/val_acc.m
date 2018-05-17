clear all, clc;
list_path = '/mnt/sdc1/hszhao/cityscapes/list/fine_val_id.txt';
%list_path = '/mnt/sdc1/hszhao/cityscapes/list/fine_train_id.txt';
predict_folder = '/mnt/sdb1/hszhao/DeepLab_MSc_LargeFOV/crfbintopng/';
predict_folder = '/mnt/sdb1/hszhao/ResNet50_largeFOV/fc8topng_val_jpshi513_70000/';
root_folder = '/mnt/sdc1/hszhao/cityscapes';
list_gt = importdata(fullfile(root_folder,'list/fine_val.txt'));
%list_gt = importdata(fullfile(root_folder,'list/fine_train.txt'));
num_class = 19;

list = importdata(list_path);
confusionmatrix =zeros(num_class,num_class);
tic
for i=1:1:size(list,1)
    i
    str = strsplit(list_gt{i});
    gt_path = str{2};
    gt = imread(fullfile(root_folder, gt_path));
    %gt(gt==255) = 19;
    predict = imread([predict_folder list{i} '.png']);
    %predict = imread([predict_folder list{i}(1:end-14) '.png']);
    %gt= gt(1:256,1:256);
    %predict = predict(1:256,1:256);
    %confusionmatrix(gt+1,predict+1) = confusionmatrix(gt+1,predict+1)+1;
    
    label = unique(gt);
    label(label==255) = [];
%     myCluster = parcluster('local');
%     myCluster.NumWorkers = 16;
%     saveProfile(myCluster);
%     parfor k = 1:num_class
%        for c = 1:num_class
%            confusionmatrix(k,c) =  confusionmatrix(k,c) + sum(sum(gt==k-1&predict==c-1));
%        end
%     end
    
    for k = 1:size(label,1)
        idx = find(gt==label(k));
        sublabel = label(k) + 1;
        for c = 1:num_class
            confusionmatrix(sublabel,c) = confusionmatrix(sublabel,c) + sum(predict(idx) == c-1);
        end
    end

%     for j=1:1:size(predict,1)
%         %j
%         for l=1:1:size(predict,2)
%             if(gt(j,l)~=255)
%                 confusionmatrix(gt(j,l)+1,predict(j,l)+1)= ...
%                 confusionmatrix(gt(j,l)+1,predict(j,l)+1)+1;
%             end
%         end
%     end
end

result =zeros(num_class,1);
%confusionmatrix(num_class+1,:) = [];
%confusionmatrix(:,num_class+1) = [];
for i=1:1:num_class
    tempa=sum(confusionmatrix(i,:));
    tempb=sum(confusionmatrix(:,i));
    result(i)=confusionmatrix(i,i)/(tempa+tempb-confusionmatrix(i,i));
end
 
acc = sum(result(:))/num_class
toc