dirs = dir('/mnt/sda1/xjqi/Imagenet/annotation/training/*.png');
dirs = struct2cell(dirs);
for i = 1:size(dirs,2)
    i
    img = imread(['/mnt/sda1/xjqi/Imagenet/annotation/training/' dirs{1,i}]);
    img(img==0) = 151;
    img = img-1;
    img(img==150) = 255;
    imwrite(img,['/mnt/sda1/xjqi/Imagenet/annotations/training/' dirs{1,i}]);
end