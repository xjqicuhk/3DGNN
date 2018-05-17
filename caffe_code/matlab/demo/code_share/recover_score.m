function [result] = recover_score(score,row,col,stride_h,stride_w,crop_size_h,crop_size_w,row_num,col_num)
%       row_num = ceil((size(img,1)-crop_size)/stride +1) ;
%       col_num = ceil((size(img,2) - crop_size)/stride +1);
%       row_real = (row_num-1)*stride + crop_size;
%       col_real = (col_num-1)*stride + crop_size;
%       img = padImg(img,[row_real col_real],mean_value);
%       Image_patch = zeros(crop_size,crop_size,size(img,3),row_num*col_num);
        tmp_result = zeros(row,col,size(score,3),'single');
%       for i = 1:row_num
%           for j = 1:col_num
%            tmp_result(stride*(i-1)+1:stride*(i-1)+crop_size,stride*(j-1)+1:stride*(j-1)+crop_size,:) =...
%              tmp_result(stride*(i-1)+1:stride*(i-1)+crop_size,...
%               stride*(j-1)+1:stride*(j-1)+crop_size,:)+score(:,:,:,(i-1)*col_num+j);
%           end
%       end
      for i = 1:row_num
          for j = 1:col_num
              if(i~=row_num&&j~=col_num)
             tmp_result(stride_h*(i-1)+1:stride_h*(i-1)+crop_size_h,stride_w*(j-1)+1:stride_w*(j-1)+crop_size_w,:)=...
                 score(:,:,:,(i-1)*col_num+j) + tmp_result(stride_h*(i-1)+1:stride_h*(i-1)+crop_size_h,...
                                                       stride_w*(j-1)+1:stride_w*(j-1)+crop_size_w,:);
              elseif(i==row_num&&j~=col_num)
              tmp_result(end-crop_size_h+1:end,stride_w*(j-1)+1:stride_w*(j-1)+crop_size_w,:)=...
                  score(:,:,:,(i-1)*col_num+j) + tmp_result(end-crop_size_h+1:end,...
                                                       stride_w*(j-1)+1:stride_w*(j-1)+crop_size_w,:);  
              elseif(i~=row_num&&j==col_num)
              tmp_result(stride_h*(i-1)+1:stride_h*(i-1)+crop_size_h,end-crop_size_w+1:end,:)=...
                  score(:,:,:,(i-1)*col_num+j) + tmp_result(stride_h*(i-1)+1:stride_h*(i-1)+crop_size_h,...
                                                       end-crop_size_w+1:end,:); 
              elseif(i==row_num&&j==col_num)
              tmp_result(end-crop_size_h+1:end,end-crop_size_w+1:end,:)=...
                  score(:,:,:,(i-1)*col_num+j) + tmp_result(end-crop_size_h+1:end,...
                                                       end-crop_size_w+1:end,:);    
              end
         end
      end

      result = tmp_result(1:row,1:col,:);

end

