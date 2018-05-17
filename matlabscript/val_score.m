function d = val_score()
list = load('./splits.mat');
list = list.testNdxs;
predict_folder = '../result/nyu_40_msc/';
root_folder = '../provided_data/label_color/';
num_class = 40;
eps  = 0.0;

confusionmatrix =zeros(num_class,num_class);
load('cityscapes_colormap.mat');


for i = 1:size(list,1)
    i

     imLab = imread([root_folder num2str(list(i)) '.png']);
     imLab(imLab ~= 255) = imLab(imLab ~= 255) + 1;
     imPred = imread([predict_folder num2str(list(i)) '.png']);
     imPred = imPred + 1;
    
    imLab (imLab == 255) = 0;
    
    imLab = imLab(45:471, 41:601,:);
    imPred = imPred(45:471, 41:601,:);
    
     [area_intersection(:,i), area_union(:,i)]=intersectionAndUnion(imPred, imLab,num_class);
 end
 IoU = sum(area_intersection,2)./sum(eps+area_union,2);
 d = sum(IoU(:)) / num_class
end
