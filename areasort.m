function [ output_image, areas_sorted, n_areas ] = areasort( input_image, num_to_keep )
%AREASORT takes a 2D image and keep the top n regions with areas in
%descending order. It also returns the sorted areas and the number of areas
%found
%   [ output_image, areas_sorted, n_areas ] = areasort( input_image, 
%   num_to_keep )

% Just to make sure the images are binarized
im = input_image > 0;

% Label
[im_lab, n_areas] = bwlabel (im, 4);

% Determine areas
areas = regionprops(im_lab,'Area');
areas = cell2mat({areas.Area}');

% Sort the areas
[areas_sorted, order] = sort(areas, 'descend');

% Generate output image
output_image = zeros(size(input_image));

for i = 1 : min(num_to_keep , n_areas)
    output_image( im_lab == order(i) ) = i;
end

end

