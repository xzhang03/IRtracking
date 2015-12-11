%% IR tracking - a most-automated way to tracking locomotion in IR videos
% Stephen Zhang 11-24-2015

%% Set parameters
% Set target fps
targetfps = 3.5;

% Set frame-gaps used for background calculation
bg_frame_gaps = 1;

% First frame to load (for tracking and background calculation)

firstframe2load = 27;

% Last frame to load (a debugging variable)
lastframe2load = 120;

% Last frame used for background
bg_lastframe2load = 120;

% Max tunning threshold
Max_threshold = 100;

% Channel to choose: red = 1, blue = 2, or green = 3
RGBchannel = 1;

% The size of erosion
erosionsize = 1;

% Direction 1 = fly moving horizontally  2 = vertically
flydirection = 2;

% Frames used to tune threshold2
nframesthresh2 = 3;

% Arena threshold
threshold = 0.2;

%% Load video
% Specify video name and path
[filename, vidpath] = uigetfile('E:\Dropbox\Michelle\Video-data\*.avi','Select the video file');
addpath(vidpath);

% Get common parameters
VidObj = VideoReader(fullfile(vidpath,filename));

nVidFrame = VidObj.NumberOfFrames;
vidHeight = VidObj.Height;
vidWidth = VidObj.Width;
vidfps = VidObj.FrameRate;
vidDuration = VidObj.Duration;

%% Crop-out the ROI
% Read out the first frame

sampleframe = read(VidObj , firstframe2load);

% Only use the Red channel
sampleframe = sampleframe(:,:,RGBchannel);

% Manually select ROI
[sampleframe_cr, cropindices] = imcrop(sampleframe);
cropindices = floor(cropindices);
close(gcf)


%% Calculate background
% Calculate the frames used for background computation
bg_frames = firstframe2load : bg_frame_gaps : bg_lastframe2load;
n_bg_frames = length(bg_frames);

% Determine the size of each frame after cropping
all_arena_size = size(sampleframe_cr);

% Create background stack
bg_stack = uint8( zeros( all_arena_size(1), all_arena_size(2), n_bg_frames ) );

% Use text progress bar
textprogressbar('Loading background: ');


% Load the background stack
for i = 1 : n_bg_frames
    % Load a tempporary frame
    im = read(VidObj , i);
    
    % Apply manual cropping
    bg_stack(:,:,i) = im(cropindices(2):cropindices(2)+cropindices(4),...
        cropindices(1):cropindices(1)+cropindices(3), RGBchannel);
    
    % Update text progress bar
    textprogressbar(i/n_bg_frames*100);
end

% Use median to apply background
background = single(median(bg_stack, 3));

% Finish text progress bar
textprogressbar('Done!');

%% Determine which flies to keep
% Show background
imshow(background,[])

fly2ignore = input('Input which arenas to ignore (e.g. [1 3 4] from left to right): ');


%% Set threshold for finding wells
%{
    figure(101)
    set(101,'Position',[100 50 1000 600])

    % Showcase all the threshold levels
    for i = 1 : 8
        subplot(2,4,i);
        imshow(im2bw(sampleframe_cr, i/10));
        text(10,15,num2str(i/10),'Color',[1 0 0]);
    end

    % Input the threshold
    threshold = input('Threshold=');
    close(101)
%}

%% Find and sort arenas from top to bottom (or left to right)
% Apply threshold to find the arenas

[all_arenas , n_arenas] = bwlabel(im2bw(sampleframe_cr, threshold));
disp(['Found ', num2str(n_arenas), ' arenas.'])
disp(['Ignore Arenas: ', mat2str(fly2ignore)])

% Determine how many arenas count during thresholding
n_arenas2 = n_arenas - length(fly2ignore);

% Use the centroids of the arenas to sort them from top to bottom
centroids = regionprops(all_arenas,'Centroid');
centroids = round(cell2mat({centroids.Centroid}'));

if flydirection == 1
    % Top to down
    centroids_y = centroids(:,2);
    [~, arena_order] = sort(centroids_y,'ascend');
else
    % Left to right
    centroids_x = centroids(:,1);
    [~, arena_order] = sort(centroids_x,'ascend');

end
% 

% Apply the sorted arena order to relabel the new arenas order (1 - top, 2 
% - second from top..., n - bottom)
all_arenas_new = zeros(size(all_arenas));

% Initiate arenaind
arenaind = 1;

% If the arena is ignored, don't count that arena
for i = 1 : n_arenas
    if sum(fly2ignore == i) == 0
        all_arenas_new(all_arenas==arena_order(i)) = arenaind;
        % Update arenaind
        arenaind = arenaind + 1; 
    end
end

all_arenas_new = imfill(all_arenas_new);

% Find the boundries of the new arenas
% extremas = regionprops(all_arenas_new,'Extrema');

%% Loading arenas
% For reference, 15 fps for 1 hour at 640x480 is about 16 GB
% Ajudt how many frames to skip during loading
frames2skip = round(vidfps/targetfps);

% Generate a vector of which frames to load
frames2load_vec = firstframe2load : frames2skip : lastframe2load;

% Calculate how many frames to load
nframe2load = length(frames2load_vec);

% Create arena stack
arena = uint8( zeros( all_arena_size(1), all_arena_size(2), nframe2load ) );

% A stack used to detect gross movements of the arena (debug)
% arena_gross = arena;

% A erosoin mask used to remove the arena edges
erosion_mask = all_arenas > 0;
erosion_mask = uint8(imerode(erosion_mask,strel('disk',erosionsize)));

% Initialize text progress bar
textprogressbar('Loading arena stack: ');

for i = 1 : nframe2load
    % Loading
    im = single(read(VidObj , frames2load_vec(i)));

    % Apply manual cropping and background
    arena(:,:,i) = uint8(-im(cropindices(2):cropindices(2)+cropindices(4),...
    cropindices(1):cropindices(1)+cropindices(3), RGBchannel)...
    + background) .* erosion_mask;

    % Update arena gross (debug)
    % arena_gross(:,:,i) = uint8(im2bw(im(cropindices(2):cropindices(2)+cropindices(4),...
    % cropindices(1):cropindices(1)+cropindices(3), RGBchannel),threshold));

    % Update progress bar
    textprogressbar(i/nframe2load*100);
end

% Terminate text progress bar
textprogressbar('Done!');


%% Tune the threshold to pickout flies.
% Define the imaged used to tune threshold2
tuning_image = repmat(arena(:,:,1), [1 1 nframesthresh2]);

% Load the images, which are equally spaced apart in the entire stack
for i = 1 : nframesthresh2
    tuning_image(:,:,i) = arena(:,:,...
        round(nframe2load/(nframesthresh2 + 1) * i)) .* uint8(all_arenas_new>0);
end

% Prepare to tuning threshold
Tuning_vec = zeros(Max_threshold , 3 , nframesthresh2);

% Initiate textprogressbar
textprogressbar('Preparing to tune threshold2: ');

for j = 1 : nframesthresh2
    for i = 1 : Max_threshold
        % use locallabel to label images
        [sorted_areas , areasfound] =...
        locallabel( tuning_image(:,:,j), all_arenas_new, i );

        % Calculate precision
        Tuning_vec(i,1,j) = sum(sorted_areas(:)>0)...
            / sum(sum(tuning_image(:,:,j) > i));

        % Calculate recall
        % If no fly is detected in any well, set recall to 0;

        if sum(areasfound) == n_arenas2
            Tuning_vec(i,2,j) = 1;
        end

        % Calculate F1 score
        Tuning_vec(i,3,j) = 2 * Tuning_vec(i,1,j) * Tuning_vec(i,2,j)...
            / (Tuning_vec(i,1,j) + Tuning_vec(i,2,j));
        
        % Update progress bar
        textprogressbar(((j-1)*Max_threshold+i)...
            /nframesthresh2/Max_threshold*100);
    end
end

% Terminate text progress bar
textprogressbar('Done!');

% Make figure
figure(101)
plot(squeeze(Tuning_vec(:,3,:)), 'o-','LineWidth',3)
xlabel('Threshold')
ylabel('F1 score')
legend({'Sample Frame 1', 'Sample Frame 2', 'Sample Frame 3'})

grid on

% Input the threshold
threshold2 = input('Threshold2=');
close(101)

%% Final segmentation
% Create arena stack to store the segmented arena stack
arena_final = arena;

% Prime a thresholds vector for debugging
thresholds_final = zeros(nframe2load,n_arenas2);

% Initiate textprogressbar
textprogressbar('Final segmentation: ');


for i = 1 : nframe2load
    % Use locallabel
    [ im2, areasfound, thresholds_final(i,:) ] =...
        locallabel(arena(:,:,i), all_arenas_new, threshold2);
    
    % Test how many flies are detected
    n_flies_detected = sum(areasfound);

    % Log the final arena
    % It's only worth logging if the software did find a threshold
    if sum(thresholds_final(i,:) ==0) < 1
        arena_final(:,:,i) = uint8(double(im2>0) .* all_arenas_new);
    end
    % Update progress bar
    textprogressbar(i/nframe2load*100);
end

% Terminate textprogressbar
textprogressbar('Done!')

%% Output workspace and the image if needed
% saveastiff(uint8(arena_final),'Michelletracking.tif');

save(fullfile(vidpath,[filename(1:end-4),'.mat']))

%% Obtain the coordinates of the flies
% Prime the matrix to store the coordinates
flycoords = zeros(n_arenas2, 2, nframe2load);

% Initiate textprogressbar
textprogressbar('Determining centroids: ')

for ii = 1 : nframe2load
    % If the software never found a threshold, it's not worth calculating
    if sum(thresholds_final(i,:) ==0) < 1
        % Obrain the centroids
        centroids = regionprops(arena_final(:,:,ii),'Centroid');

        % For mat the centroids and load it to the flycoords matrix
        flycoords(:,:,ii) = cell2mat({centroids.Centroid}');

        % Update text progress bar
        textprogressbar(i/nframe2load * 100);
    else
        % Load NaN to coordinates
        flycoords(:,:,ii) = NaN;
        
        % Update text progress bar
        textprogressbar(i/nframe2load * 100);
    end
end

% Terminate textprogressbar
textprogressbar('Done!')

%% Zero initial coordinate and make the plot
% Zero initial location
flycoords_zeroed = flycoords - repmat(flycoords(:,:,1),[1,1,nframe2load]);

% flycoords_zeroed2 (to be fixed by MF)
% flycoords_zeroed2 = flycoords_zeroed(fly2keep,:,:);

% Plot the zeroed traces
figure('Position',[50, 200, 1500, 350], 'Color', [1 1 1])

% Left subplot shows the zeroed traces
subplot(1,4,1:3)
% plot(squeeze(flycoords_zeroed(:,1,:))' , squeeze(flycoords_zeroed(:,2,:))',...
%     'LineWidth', 2);

if flydirection == 1
    % Plot horizontal
    plot(squeeze(flycoords_zeroed(:,1,:))')
else
    % Plot vertical
    plot(squeeze(flycoords_zeroed(:,2,:))')
end

% Create lines to label quadrants
% ylimits = get(gca,'ylim');
% xlimits = get(gca,'xlim');

% line([0 0], ylimits, 'Color',[0 0 0])
% line(xlimits, [0 0], 'Color', [0 0 0])

% Label x and y
xlabel('Time (frame)')

if flydirection ==1
    ylabel('X location')
else
    ylabel('Y location')
end

% Right subplot shows the polar plot of the net displacement of each fly
subplot(1,4,4)
compass(flycoords_zeroed(:,:,end))

%% Print average summary plots

% Compute mean and sem
mean_coords = squeeze(nanmean(flycoords_zeroed2,1));
semcoords = squeeze(nanstd(flycoords_zeroed2,1) / sqrt(n_arenas));

% Plot mean
figure('color',[1 1 1]); 
% This part will differ based on whether flies are moving vertically or
% horizontally
shadedErrorBar((1:length(mean_coords(1,:)))/targetfps, mean_coords(1,:),...
    semcoords(1,:), {'Color', [0 .4 0.7]});
title('Mean Position', 'FontWeight', 'bold');
ylabel('Mean Position (pixels)');
xlabel('Time (seconds)');
set(gca,'Box','off');

% Compute mean velocity
diff_coords = diff(flycoords_zeroed2,1,3);
mean_vel = squeeze(nanmean(diff_coords,1));
sem_vel = squeeze(nanstd(diff_coords,1) / sqrt(n_arenas));

% Plot mean velocity
figure('color',[1 1 1]); 
% This part will differ based on whether flies are moving vertically or
% horizontally
shadedErrorBar((1:length(mean_vel(1,:)))/targetfps, mean_vel(1,:),...
    sem_vel(1,:), {'Color', [0.5 0 0.5]});
title('Mean Velocity', 'FontWeight', 'bold');
ylabel('Velocity (pixels/frame)');
xlabel('Time (seconds)');
set(gca,'Box','off');

% Print out the number of nans in the computed array (as a sanity check on
% how good your thresholding was)
num_nans = sum(isnan(flycoords(:)));
display('Number of NaNs:'), display(num_nans);
%% Keep and save data
keep all_arena_size all_arenas_new arena_final flycoords flycoords_zeroed...
    filename n_arenas vidpath vidfps nframe2load
save(fullfile(vidpath,[filename(1:end-4),'_clean.mat']))
