function d = val_score_accuracy()
list = load('./data/splits.mat');
list = list.testNdxs;

%predict_folder = './label//result/';
root_folder = './data/label/';
%root_folder = '/data/ssd/public/xjqi/SUNRGBD/label/';
predict_folder = './validate/validation_unroll_step_5/result/';

%list_gt = importdata(fullfile(root_folder,'list/fine_val.txt'));
%list_gt = importdata(fullfile(root_folder,'list/fine_train.txt'));
num_class = 40;
eps  = 0.0;
%list = importdata(list_path);
confusionmatrix =zeros(num_class,num_class);
load('cityscapes_colormap.mat');
%list = 1:5050;
%list = list';

for i = 1:size(list,1)
    i
    imLab = imread([root_folder num2str(list(i)) '.png']);
    imLab(imLab ~= 255) = imLab(imLab ~= 255) + 1;
    imPred = imread([predict_folder num2str(list(i)) '.png']);
    imPred = imPred + 1;
    imLab (imLab == 255) = 0;
    %imLab = imLab(45:471, 41:601,:);
    [area_intersection(:,i), area_union(:,i)]=intersectionAndUnion_accuracy(imPred, imLab,num_class);
 end
 IoU = sum(area_intersection,2)./sum(eps+area_union,2);
 d = sum(IoU(:)) / num_class
end