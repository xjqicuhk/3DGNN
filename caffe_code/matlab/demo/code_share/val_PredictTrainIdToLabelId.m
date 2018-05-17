clear all, clc;
png_folder = '/mnt/sdb1/hszhao/ResNet50_largeFOV/val_crfvallabel_90000/';
png_dir = dir([png_folder '*.png']);
save_label_folder = '/mnt/sdb1/hszhao/ResNet50_largeFOV/val_crftrainlabel_90000/';
if(~isdir(save_label_folder))
    mkdir(save_label_folder);
end

map_id = [255 , 255 , 255 , 255 , 255 , 255 , 255 , ...
          0 ,   1 , 255 , 255 ,   2 ,   3 ,   4 , ...
          255 , 255 , 255 ,   5 , 255 ,   6 ,   7 , ...
          8 ,   9 ,  10 ,  11 ,  12 ,  13 ,  14 ,  15 , ...
          255 , 255 ,  16 ,  17 ,  18 ,  255];
map_id = uint8(map_id);
map_id = map_id + 1;
[~,index] = sort(map_id);
for i = 1:size(png_dir,1)
    i
    str = png_dir(i).name;
    img = imread(fullfile(png_folder,str));
    result = uint8(index(img+1)-1);
    
%     row = size(img,1);
%     col = size(img,2);
%     img_reshape = reshape(img,row*col,1);
%     map_id_rep = repmat(map_id,row*col,1);
%     img_rep = repmat(img_reshape,1,size(map_id,2));
%     %index = (map_id_rep==img_rep);
%     index = find(map_id_rep==img_rep);
%     result = zeros(row*col,1,'uint8');
%     row_index = int32(mod(index,1024*2048));
%     row_last = find(row_index==0);
%     row_index(row_last) = 1024*2048;
%     col_index = uint8(floor(index/(1024*2048)));
%     col_index(row_last) = col_index(row_last) - 1;
%     result(row_index) = col_index;
% %     for j=1:row*col
% %         result(j,1) = find(index(j,:));
% %     end
%     result = reshape(result,row,col);
    save_dir = fullfile(save_label_folder,str);
    imwrite(result,save_dir);
end