function data = caffe_process_data(net,img,CROPPED_DIM_h,CROPPED_DIM_w,mean_value)
    %addpath(genpath('/home/hszhao/research/caffe_sg/matlab/caffe'));
   % img = preImgData(img,CROPPED_DIM,mean_value);
    input_data = {img};
    
   % crops_data = zeros(CROPPED_DIM, CROPPED_DIM, 3, 2, 'single');
    %crops_data(:,:,:,1) = img;
    %crops_data(:,:,:,2) = img;
    %input_data = {input_data};
    data = net.forward(input_data);
    data = data{1};
    %%
%     img_cha = size(data,3);
%     data_output = zeros(CROPPED_DIM,CROPPED_DIM,img_cha,'single');
%     for i=1:img_cha
%         data_output(:,:,i) = imresize(data(:,:,i),[CROPPED_DIM CROPPED_DIM],'bilinear');
%     end
%     data = data_output;
    %%
    tmp_data = zeros(size(img,2),size(img,1),size(data,3),size(data,4),'single');
    for j = 1:size(data,4)
    tmp = data(:,:,:,j);
    
    tmp = permute(tmp, [2 1 3]);
    tmp = exp(tmp);
    tmp = bsxfun(@rdivide, tmp, sum(tmp, 3));
    tmp = imresize(tmp,[size(img,2) size(img,1)],'bilinear');
    tmp_data(:,:,:,j) = tmp;
    end
    data = tmp_data;
end