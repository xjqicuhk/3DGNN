function data = caffe_process(net,img,CROPPED_DIM)
    %addpath(genpath('/home/hszhao/research/caffe_sg/matlab/caffe'));
    img = preImg(img,CROPPED_DIM);
    input_data = {img};
    
    crops_data = zeros(CROPPED_DIM, CROPPED_DIM, 3, 2, 'single');
    crops_data(:,:,:,1) = img;
    crops_data(:,:,:,2) = img;
    input_data = {crops_data};
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
    data = data(:,:,:,1);
    
    data = permute(data, [2 1 3]);
    data = exp(data);
    data = bsxfun(@rdivide, data, sum(data, 3));
    data = imresize(data,[size(img,1) size(img,2)],'bilinear');
end