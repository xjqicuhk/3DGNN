function [result] = caffe_process_batch_score(net,img,scale,stride_h,stride_w,crop_size_h,crop_size_w,batch_size,num_class);
 img = imresize(img,scale,'bilinear');
 [Image_patch,row_num,col_num] = ImageToGrid_score(img,stride_h,stride_w,crop_size_h,crop_size_w);
 score = zeros([size(Image_patch,1) size(Image_patch,2) num_class size(Image_patch,4)]);
 batch_data = zeros(crop_size_h,crop_size_w,size(img,3),batch_size);
 num_iter = ceil(size(Image_patch,4)/batch_size);
 for i = 1:num_iter
     batch_data = zeros(crop_size_h,crop_size_w,size(img,3),batch_size);
    if(i~=num_iter||(i==num_iter&mod(size(Image_patch,4),batch_size)==0))
        batch_data = Image_patch(:,:,:,(i-1)*batch_size+1:1:i*batch_size);
    else batch_data(:,:,:,1:mod(size(Image_patch,4),batch_size)) = Image_patch(:,:,:,(num_iter-1)*batch_size+1:end); 
    end
    %batch_data = preImgData(batch_data,crop_size_h,crop_size_w,mean_value);
    if(i~=num_iter||(i==num_iter&mod(size(Image_patch,4),batch_size)==0))
    score(:,:,:,(i-1)*batch_size+1:1:i*batch_size) = caffe_process_data_score(net,batch_data,num_class);
    else tmp = caffe_process_data_score(net,batch_data,num_class);
         score(:,:,:,(num_iter-1)*batch_size+1:end) = tmp(:,:,:,1:mod(size(Image_patch,4),batch_size));
    end   
 end
 result = recover_score(score,size(img,1),size(img,2),stride_h,stride_w,crop_size_h,crop_size_w,row_num,col_num);
 
end

