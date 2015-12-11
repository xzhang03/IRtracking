function [ labeledimage, areasfound, thesholds_used ]...
    = locallabel( image, imagegroup, threshold )
%locallabel segment images locally
%   Detailed explanation goes here

% Determine the number of arenas
n_arenas2 = length(unique(imagegroup)) - 1;

% initialize image
im2 = zeros(size(image));

% initialize n_areasfound
areasfound = zeros(n_arenas2, 1);

% Flags 1 = success
thesholds_used = zeros(n_arenas2, 1);

for i = 1 : n_arenas2

    % localized image
    im = double(image > threshold) .* (imagegroup == i); 
    
    if sum(im(:))>0
        % label image
        im2(areasort(im, 1)>0) = i;
        areasfound(i) = 1;
        
        % Flag success
        thesholds_used(i) = threshold;
    else
        % Prepare to apply dynamic threshold
        threshold_tmp = threshold;
        
        while sum(im(:)) == 0 && threshold > 0
            % Decrease threshold
            threshold_tmp = threshold_tmp - 1;
            
            % Apply decreased threshold
            im = double(image > threshold_tmp) .* (imagegroup == i);
            
        end
        
        % Label the tresholded image
        if sum(im(:))>0
            % label image
            im2(areasort(im, 1)>0) = i;
            areasfound(i) = 1;

            % Flag success
            thesholds_used(i) = threshold_tmp;
        end
    end
    
end

labeledimage = im2;


end

