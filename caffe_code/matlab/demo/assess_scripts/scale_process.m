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
                img_sub = img_scale(:,end-CROPPED_DIM+1:end,:);
                count_scale(:,end-CROPPED_DIM+1:end,:) = count_scale(:,end-CROPPED_DIM+1:end,:) + 1;
            else
                img_sub = img_scale(:,(c-1)*512+1:(c-1)*512+CROPPED_DIM,:);
                count_scale(:,(c-1)*512+1:(c-1)*512+CROPPED_DIM,:) = count_scale(:,(c-1)*512+1:(c-1)*512+CROPPED_DIM,:) + 1;
            end
            data = caffe_process(net,img_sub,CROPPED_DIM);
            if(c==c_max)
                data_scale(:,end-CROPPED_DIM+1:end,:) = data_scale(:,end-CROPPED_DIM+1:end,:) + data(1:512,end-CROPPED_DIM+1:end,:);
            else
                data_scale(:,(c-1)*512+1:(c-1)*512+CROPPED_DIM,:) = data_scale(:,(c-1)*512+1:(c-1)*512+CROPPED_DIM,:) + data(1:512,(c-1)*512+1:(c-1)*512+CROPPED_DIM,:);
            end
        end
    else
        for r=1:r_max
           for c=1:c_max
               if((r==r_max) && (c==c_max))
                   img_sub = img_scale(end-CROPPED_DIM+1:end,end-CROPPED_DIM+1:end,:);
                   count_scale(end-CROPPED_DIM+1:end,end-CROPPED_DIM+1:end,:) = count_scale(end-CROPPED_DIM+1:end,end-CROPPED_DIM+1:end,:) + 1;
               elseif((r==r_max) && (c~=c_max))
                   img_sub = img_scale(end-CROPPED_DIM+1:end,(c-1)*512+1:(c-1)*512+CROPPED_DIM,:);
                   count_scale(end-CROPPED_DIM+1:end,(c-1)*512+1:(c-1)*512+CROPPED_DIM,:) = count_scale(end-CROPPED_DIM+1:end,(c-1)*512+1:(c-1)*512+CROPPED_DIM,:) + 1;
               elseif((r~=r_max) && (c==c_max))
                   img_sub = img_scale((r-1)*512+1:(r-1)*512+CROPPED_DIM,end-CROPPED_DIM+1:end,:);
                   count_scale((r-1)*512+1:(r-1)*512+CROPPED_DIM,end-CROPPED_DIM+1:end,:) = count_scale((r-1)*512+1:(r-1)*512+CROPPED_DIM,end-CROPPED_DIM+1:end,:) + 1;
               else
                   img_sub = img_scale((r-1)*512+1:(r-1)*512+CROPPED_DIM,(c-1)*512+1:(c-1)*512+CROPPED_DIM,:);
                   count_scale((r-1)*512+1:(r-1)*512+CROPPED_DIM,(c-1)*512+1:(c-1)*512+CROPPED_DIM,:) = count_scale((r-1)*512+1:(r-1)*512+CROPPED_DIM,(c-1)*512+1:(c-1)*512+CROPPED_DIM,:) + 1;
               end
               data = caffe_process(net,img_sub,CROPPED_DIM);
               if((r==r_max) && (c==c_max))
                   data_scale(end-CROPPED_DIM+1:end,end-CROPPED_DIM+1:end,:) = data_scale(end-CROPPED_DIM+1:end,end-CROPPED_DIM+1:end,:) + data(end-CROPPED_DIM+1:end,end-CROPPED_DIM+1:end,:);
               elseif((r==r_max) && (c~=c_max))
                   data_scale(end-CROPPED_DIM+1:end,(c-1)*512+1:(c-1)*512+CROPPED_DIM,:) = data_scale(end-CROPPED_DIM+1:end,(c-1)*512+1:(c-1)*512+CROPPED_DIM,:) + data(end-CROPPED_DIM+1:end,1:CROPPED_DIM,:);
               elseif((r~=r_max) && (c==c_max))
                   data_scale((r-1)*512+1:(r-1)*512+CROPPED_DIM,end-CROPPED_DIM+1:end,:) = data_scale((r-1)*512+1:(r-1)*512+CROPPED_DIM,end-CROPPED_DIM+1:end,:) + data(1:CROPPED_DIM,end-CROPPED_DIM+1:end,:);
               else
                   data_scale((r-1)*512+1:(r-1)*512+CROPPED_DIM,(c-1)*512+1:(c-1)*512+CROPPED_DIM,:) = data_scale((r-1)*512+1:(r-1)*512+CROPPED_DIM,(c-1)*512+1:(c-1)*512+CROPPED_DIM,:) + data(1:CROPPED_DIM,1:CROPPED_DIM,:);
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