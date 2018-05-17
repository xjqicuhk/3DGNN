function data_output = scale_process(net,img,scale,img_cha,CROPPED_DIM)
    r_max = scale*2;
    c_max = scale*4;
    img_scale = imresize(img,scale);
    data_output = zeros(size(img,1),size(img,2),img_cha,'single');
    data_scale = zeros(size(img_scale,1),size(img_scale,2),img_cha,'single');
    count_scale = zeros(size(img_scale,1),size(img_scale,2),img_cha,'single');
    if(r_max==1)
        for c=1:c_max
            if(c==c_max)
                img_sub = img_scale(:,end-608:end,:);
                count_scale(:,end-608:end,:) = count_scale(:,end-608:end,:) + 1;
            else
                img_sub = img_scale(:,(c-1)*512+1:(c-1)*512+609,:);
                count_scale(:,(c-1)*512+1:(c-1)*512+609,:) = count_scale(:,(c-1)*512+1:(c-1)*512+609,:) + 1;
            end
            data = caffe_process(net,img_sub,CROPPED_DIM);
            if(c==c_max)
                data_scale(:,end-608:end,:) = data_scale(:,end-608:end,:) + data(1:512,end-608:end,:);
            else
                data_scale(:,(c-1)*512+1:(c-1)*512+609,:) = data_scale(:,(c-1)*512+1:(c-1)*512+609,:) + data(1:512,(c-1)*512+1:(c-1)*512+609,:);
            end
        end
    else
        for r=1:r_max
           for c=1:c_max
               if((r==r_max) && (c==c_max))
                   img_sub = img_scale(end-608:end,end-608:end,:);
                   count_scale(end-608:end,end-608:end,:) = count_scale(end-608:end,end-608:end,:) + 1;
               elseif((r==r_max) && (c~=c_max))
                   img_sub = img_scale(end-608:end,(c-1)*512+1:(c-1)*512+609,:);
                   count_scale(end-608:end,(c-1)*512+1:(c-1)*512+609,:) = count_scale(end-608:end,(c-1)*512+1:(c-1)*512+609,:) + 1;
               elseif((r~=r_max) && (c==c_max))
                   img_sub = img_scale((r-1)*512+1:(r-1)*512+609,end-608:end,:);
                   count_scale((r-1)*512+1:(r-1)*512+609,end-608:end,:) = count_scale((r-1)*512+1:(r-1)*512+609,end-608:end,:) + 1;
               else
                   img_sub = img_scale((r-1)*512+1:(r-1)*512+609,(c-1)*512+1:(c-1)*512+609,:);
                   count_scale((r-1)*512+1:(r-1)*512+609,(c-1)*512+1:(c-1)*512+609,:) = count_scale((r-1)*512+1:(r-1)*512+609,(c-1)*512+1:(c-1)*512+609,:) + 1;
               end
               data = caffe_process(net,img_sub,CROPPED_DIM);
               if((r==r_max) && (c==c_max))
                   data_scale(end-608:end,end-608:end,:) = data_scale(end-608:end,end-608:end,:) + data(end-608:end,end-608:end,:);
               elseif((r==r_max) && (c~=c_max))
                   data_scale(end-608:end,(c-1)*512+1:(c-1)*512+609,:) = data_scale(end-608:end,(c-1)*512+1:(c-1)*512+609,:) + data(end-608:end,1:609,:);
               elseif((r~=r_max) && (c==c_max))
                   data_scale((r-1)*512+1:(r-1)*512+609,end-608:end,:) = data_scale((r-1)*512+1:(r-1)*512+609,end-608:end,:) + data(1:609,end-608:end,:);
               else
                   data_scale((r-1)*512+1:(r-1)*512+609,(c-1)*512+1:(c-1)*512+609,:) = data_scale((r-1)*512+1:(r-1)*512+609,(c-1)*512+1:(c-1)*512+609,:) + data(1:609,1:609,:);
               end
           end
        end
    end
    
    data_scale = data_scale./count_scale;
    for i=1:img_cha
       data_output(:,:,i) = imresize(data_scale(:,:,i),[size(img,1) size(img,2)],'bilinear');%1/single(scale));
    end
    data_output = bsxfun(@rdivide, data_output, sum(data_output, 3));
end