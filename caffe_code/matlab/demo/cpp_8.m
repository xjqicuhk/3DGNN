clear all, clc;
save_subpng_folder = '/home/hszhao/research/caffe_sg/result_crf/';
save_predict_folder = '/mnt/sdb1/hszhao/ResNet50_largeFOV/val_crfvallabel_90000/';
save_png_folder = '/mnt/sdb1/hszhao/ResNet50_largeFOV/val_crfcolorpng_90000/';

if(~isdir(save_predict_folder))
    mkdir(save_predict_folder);
end
if(~isdir(save_png_folder))
    mkdir(save_png_folder);
end

%CROPPED_DIM = 513;%1025;%609;
show_image = false;
load('cityscapes_colormap.mat');

root_folder = '/mnt/sdc1/hszhao/cityscapes';
list = importdata(fullfile(root_folder,'list/fine_val.txt'));

for j = 1:numel(list)
    j
    str = strsplit(list{j});
    img_path = str{1};
    img_name = strsplit(img_path,'/');
    img_name = img_name{end};
    img_name = img_name(1:end-4);
    img_name = strrep(img_name,'leftImg8bit','gtFine_labelTrainIds');
    
    img = zeros(1024,2048,'uint8');
    for i = 1:8
       img_sub = imread(fullfile(save_subpng_folder,[img_name '_' int2str(i) '.png']));
        if i==1
            img(1:512,1:512) = img_sub(1:512,1:512);
        elseif i==2
            img(1:512,513:1024) = img_sub(1:512,50:561);
        elseif i==3
            img(1:512,1025:1536) = img_sub(1:512,50:561);
        elseif i==4
            img(1:512,1537:2048) = img_sub(1:512,98:609);
        elseif i==5
            img(513:1024,1:512) = img_sub(98:609,1:512);
        elseif i==6
            img(513:1024,513:1024) = img_sub(98:609,50:561);
        elseif i==7
            img(513:1024,1025:1536) = img_sub(98:609,50:561);
        elseif i==8
            img(513:1024,1537:2048) = img_sub(98:609,98:609);
        end
    end
    
    %save_name = strrep(img_name,'leftImg8bit','gtFine_labelTrainIds');
    imwrite(img,fullfile(save_predict_folder,[img_name '.png']));
    imwrite(img,colormap,fullfile(save_png_folder,[img_name '.png']));
    if show_image
        gt_path = str{2};
        imgsrc = imread(fullfile(root_folder,img_path));
        gt = imread(fullfile(root_folder, gt_path));
        figure(1), 
        subplot(221),imshow(imgsrc), title('img');
        subplot(222),imshow(gt, colormap), title('gt');
        subplot(224), imshow(img,colormap), title('predict');
    end
end

