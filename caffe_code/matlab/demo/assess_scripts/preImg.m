function im_pad = preImg(im,CROPPED_DIM)
    mean_r = 122.675;
    mean_g = 116.669;
    mean_b = 104.008;
    row = size(im,1);
    col = size(im,2);
    im_pad = single(im);
    if row < CROPPED_DIM
        im_r = padarray(im_pad(:,:,1),[CROPPED_DIM-row,0],mean_r,'post');
        im_g = padarray(im_pad(:,:,2),[CROPPED_DIM-row,0],mean_g,'post');
        im_b = padarray(im_pad(:,:,3),[CROPPED_DIM-row,0],mean_b,'post');
        im_pad = cat(3,im_r,im_g,im_b);
    end
    if col < CROPPED_DIM
        im_r = padarray(im_pad(:,:,1),[0,CROPPED_DIM-col],mean_r,'post');
        im_g = padarray(im_pad(:,:,2),[0,CROPPED_DIM-col],mean_g,'post');
        im_b = padarray(im_pad(:,:,3),[0,CROPPED_DIM-col],mean_b,'post');
        im_pad = cat(3,im_r,im_g,im_b);
    end
    im_mean = zeros(CROPPED_DIM,CROPPED_DIM,3,'single');
    im_mean(:,:,1) = mean_r;
    im_mean(:,:,2) = mean_g;
    im_mean(:,:,3) = mean_b;
    im_pad = single(im_pad) - im_mean;
    im_pad = im_pad(:,:,[3 2 1]);
    im_pad = permute(im_pad,[2 1 3]);
end
