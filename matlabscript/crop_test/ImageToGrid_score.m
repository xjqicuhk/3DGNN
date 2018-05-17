function [Image_patch,row_num,col_num] = ImageToGrid_score(img,stride_h,stride_w,crop_size_h,crop_size_w)
      row_num = ceil((size(img,1)- crop_size_h)/stride_h +1) ;
      col_num = ceil((size(img,2) - crop_size_w)/stride_w +1);
      %row_real = (row_num-1)*stride + crop_size;
      %col_real = (col_num-1)*stride + crop_size;
      %img = padImg(img,[row_real col_real],mean_value);
      Image_patch = zeros(crop_size_h,crop_size_w,size(img,3),row_num*col_num);
      for i = 1:row_num
          for j = 1:col_num
              if(i~=row_num&&j~=col_num)
             Image_patch(:,:,:,(i-1)*col_num+j) = img(stride_h*(i-1)+1:stride_h*(i-1)+crop_size_h,...
                                                       stride_w*(j-1)+1:stride_w*(j-1)+crop_size_w,:);
              elseif(i==row_num&&j~=col_num)
              Image_patch(:,:,:,(i-1)*col_num+j) = img(end-crop_size_h+1:end,...
                                                       stride_w*(j-1)+1:stride_w*(j-1)+crop_size_w,:);  
              elseif(i~=row_num&&j==col_num)
              Image_patch(:,:,:,(i-1)*col_num+j) = img(stride_h*(i-1)+1:stride_h*(i-1)+crop_size_h,...
                                                       end-crop_size_w+1:end,:); 
              else
              Image_patch(:,:,:,(i-1)*col_num+j) = img(end-crop_size_h+1:end,...
                                                       end-crop_size_w+1:end,:);    
              end
         end
      end
end