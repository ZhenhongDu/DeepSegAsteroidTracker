img_path = 'G:\PROJECT\Asteroid_proj\Telescope\telescope\data2\new_asteroid_raw_data\';
img_dir = dir([img_path,'*.fits']);
save_path = 'E:/data/telescope/data2/enhanced_data/';

for i = 1:length(img_dir) %1:length(img_dir)
    img = fitsread([img_path, img_dir(i).name]);
    low_light_img = uint16(img);
%     input_low_light_img = sqrtStretch(low_light_img);
     input_low_light_img = imadjust(low_light_img,[0.005, 0.035]);
    
    
    
    AInv = imcomplement(input_low_light_img);
    BInv = imreducehaze(AInv,'ContrastEnhancement','none');
    BImp = imcomplement(BInv);
    
%     res = imguidedfilter(BImp, "NeighborhoodSize",[6 6]);
    out = medfilt2(BImp,[3 3]);
    
%     montage({input_low_light_img,BImp});
    
%   save_name = strrep(img_dir(i).name,'.fits','.tif');
    save_name = ['Star_img_', num2str(i-1), '.tif'];
    
%     imwrite(out,[save_path, save_name]);
    sprintf('The star image %d have been saved!',i)

end
