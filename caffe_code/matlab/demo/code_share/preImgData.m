function im_pad = preImgData(im,CROPPED_DIM_h,CROPPED_DIM_w,mean_value)
    mean_r = mean_value(1,1);
    mean_g = mean_value(1,2);
    mean_b = mean_value(1,3);
    row = size(im,1);
    col = size(im,2);
    im_pad = zeros(CROPPED_DIM_w,CROPPED_DIM_h,size(im,3),size(im,4),'single');
   for j  =1:size(im,4)
       flag = 0;
    if row < CROPPED_DIM_h
        im_r = padarray(im(:,:,1,j),[CROPPED_DIM_h-row,0],mean_r,'post');
        im_g = padarray(im(:,:,2,j),[CROPPED_DIM_h-row,0],mean_g,'post');
        im_b = padarray(im(:,:,3,j),[CROPPED_DIM_h-row,0],mean_b,'post');
        flag = true;
       % im_pad(:,:,:,j) = cat(3,im_r,im_g,im_b);
    end
    if col < CROPPED_DIM_w
        im_r = padarray(im_r,[0,CROPPED_DIM_w-col],mean_r,'post');
        im_g = padarray(im_g,[0,CROPPED_DIM_w-col],mean_g,'post');
        im_b = padarray(im_b,[0,CROPPED_DIM_w-col],mean_b,'post');
        flag = true;
        %im_pad(:,:,:,j) = cat(3,im_r,im_g,im_b);
    end
    if(flag)
    tmp = cat(3,im_r,im_g,im_b);
    else
        tmp = im(:,:,:,j);
    end
    im_mean = zeros(CROPPED_DIM_h,CROPPED_DIM_w,3,'single');
    im_mean(:,:,1) = mean_r;
    im_mean(:,:,2) = mean_g;
    im_mean(:,:,3) = mean_b;
    %im_pad(:,:,:,j) = single(im_pad(:,:,:,j)) - im_mean;
    tmp = single(tmp);
    tmp =tmp - im_mean;
    tmp = tmp(:,:,[3 2 1]);
    tmp = permute(tmp,[2 1 3]);
    im_pad(:,:,:,j) = tmp;
   end
end
