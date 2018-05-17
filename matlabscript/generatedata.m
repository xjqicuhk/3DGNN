training_data = false;
load('./splits.mat');
if(training_data)
    list = trainNdxs;
    label_path = '../provided_data/label_color/';
    save_path = '../testdata/nyu_40/traindata/';
else
    list = testNdxs;
    save_path = '../testdata/nyu_40/testdata/';
end
mkdir(save_path);

depths = load('../provided_data/depths.mat');
depths = depths.depths;
img_path = '../provided_data/images/';


for i = 1:size(list,1)
    i
    depth = depths(:,:,list(i));
    image = imread([img_path num2str(list(i)) '.png']);
    if(training_data)
     label = imread([label_path num2str(list(i)) '.png']);
    end
    point = depth_plane2depth_world(depth);
    point = reshape(point,[480,640,3]);
    
    if(training_data)
        save([save_path num2str(list(i)) '.mat'],'image','depth','label','point');
    else save([save_path num2str(list(i)) '.mat'],'image','depth','point');
    end
end