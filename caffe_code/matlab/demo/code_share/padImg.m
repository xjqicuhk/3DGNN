function im_pad = padImg(im,PadSize,mean_value)
    mean_r = mean_value(1,1);
    mean_g = mean_value(1,2);
    mean_b = mean_value(1,3);
    row = size(im,1);
    col = size(im,2);
    im_pad = single(im);
    if row < PadSize(1,1);
        im_r = padarray(im_pad(:,:,1),[PadSize(1,1)-row,0],mean_r,'post');
        im_g = padarray(im_pad(:,:,2),[PadSize(1,1)-row,0],mean_g,'post');
        im_b = padarray(im_pad(:,:,3),[PadSize(1,1)-row,0],mean_b,'post');
        im_pad = cat(3,im_r,im_g,im_b);
    end
    if col < PadSize(1,2)
        im_r = padarray(im_pad(:,:,1),[0,PadSize(1,2)-col],mean_r,'post');
        im_g = padarray(im_pad(:,:,2),[0,PadSize(1,2)-col],mean_g,'post');
        im_b = padarray(im_pad(:,:,3),[0,PadSize(1,2)-col],mean_b,'post');
        im_pad = cat(3,im_r,im_g,im_b);
    end
