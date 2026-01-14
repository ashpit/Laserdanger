%% Lidar Pipeline

%%-----------------------------------------------------------------------%%
%           WAVE RESOLVING POINT CLOUD PROCESSING PIPELINE (MAY 2025)     %
%------------------------------------------------------------------------%%

%%--[0] SETUP & PARAMETERS
% ---> Define input/output folders, transformation matrix, bounds
%       -> load json configuration file
% ---> Load existing L1 structure (if any) to resume processing

%%--[1] IDENTIFY VALID FOLDERS
% ---> Get list of folders with hourly timestamps (POSIX)
% ---> Filter folders to match target date range

%%--[2] FOR EACH VALID FOLDER:
%     [2.1] Read and randomly sample up to 5min of data
%     [2.2] For each sampled file:
%         --> Convert POSIX time to datetime
%         --> Read point cloud and apply homogeneous transformation
%         --> Store transformed XYZ points
%         --> Keep points with Intensity < 100 and within set x,y boundary

%%--[3] BINNING STAGE — INITIAL NOISE FILTERING and Rasterize
% ---> Round x/y to spatial bins [xRes, yRes]
% ---> Accumulate z statistics:
%    --> Accumarray points to bin size, create 50th percentile filter
%    --> Cell function to find count, mean, std, min, mode of filtered bins
%       -> zcount: number of points per bin
%       -> zmean: mean of all points in bin 
%       -> zmode: mode of 5 cm binned elevations
%       -> zmin:  min elevation
%       -> zstd:  standard deviation of z in each bin
% ---> Use Signal/noise ratio to remove bad bins
%       -> SNR = bin mean / Standard Error
%       -> Standard Error = sigma/sqrt(count)
%       -> also remove bins with low count (will have a high snr)

%%--[4] Fit planes to raster, remove points-above-plane
% ---> Delauney Triangulation of the pointcloud grid 
%       -> makes triangles that fill up bounds of pointcloud
%       -> iteratively finds points within those bounds
% ---> for each triangle, remove residual points greater than 50cm
% ---> returns point indices for 'groun' points

%%--[5] STRUCTURE EXPORT
% ---> Store binned stats in DO(n).*
%       -> Dates, X, Y, Zmean, Zmin, Zmode, Zstdv
% ---> Save updated DO to .mat file
% ---> Print progress message

%%--[6] Export figure
% ---> use Get3_1Dprofiles.m to produce a 1D profile along shorenormal
% transect
% ---> choose data from previous 5 days; for every 6 hours.
% ---> create plot data: 
%      -> date, x, y, color
% ---> export .json file containing plot data


%%-----------------------------------------------------------------------%%
%                              END OF PIPELINE                            %
%------------------------------------------------------------------------%%


%% process 8-14
% Load config JSON
config = jsondecode(fileread('livox_config_L2.json'));
dataFolder = config.dataFolder;
ProcessFolder = config.processFolder;
tmatrix = config.transformMatrix;
bounds = config.LidarBoundary;
% Load existing LO structure if available
% Get LiDAR files

%% load in test file in 
% dataFolder = '/Users/ashton/Library/CloudStorage/OneDrive-UCSanDiego/CPG/Lidar/data';
fileList = dir(fullfile(dataFolder, 'do-lidar_*.laz'));
fileNames = {fileList.name};
%%
epochStrings = erase(erase(fileNames, 'do-lidar_'), '.laz');
% lidar filenames are in UTC? matlab adds 7 hours to get in UTC
fileDates = datetime(str2double(epochStrings), 'ConvertFrom', 'posixtime', 'TimeZone','local');
%
% Sort files by timestamp
[fileDates, sortIdx] = sort(fileDates);
fileNames = fileNames(sortIdx);
%%
% Just grab files within the range of high tide on specific date.
t1 = datetime([2025,07,01,00,00,00], 'TimeZone', 'local'); t2 = datetime([2025,07,08,00,00,00], 'TimeZone', 'local');
fileselect = fileDates >= t1 & fileDates <= t2;
keepfileNames = fileNames(fileselect);
keepfileDates = fileDates(fileselect);
% keepfileDates.TimeZone = 'UTC';
%% Get high tides from list of high tide times; 
% % Define reference time (latest file)
% latestDate = fileDates(end);
% % Define time threshold (4 days prior)
% cutoffDate = latestDate - days(1);
% % Find indices of files within the past 2 days
% recentIdx = fileDates >= cutoffDate;
% % Subset filenames and datetimes
% recentFileDates = fileDates(recentIdx);
% recentFileNames = fileNames(recentIdx);
% num_to_process = numel(recentFileDates);
% % Round fileDates to day (or half-hour) for grouping
% fileDatesRounded = dateshift(recentFileDates, 'start', 'day');
% uniqueDays = unique(fileDatesRounded);
%% Enter the loop 
%if isempty(validFiles)
    % start the loop
    %fprintf('no pointclouds to process')
%end
num_to_process = numel(keepfileDates);
datapath = '/Users/ashton/Library/CloudStorage/OneDrive-UCSanDiego/CPG/Laserdanger/rawtimestacks';
% i = 1;
%%
for i = 46:num_to_process
% i = 2; % an hour of data (5 min or 20min)
    currentFile = fullfile(dataFolder, keepfileNames{i});
    thisHour = roundToHalfHour(keepfileDates(i)); 
% %
    lasreader = lasFileReader(currentFile);
    [ptCloud, Attributes] = readPointCloud(lasreader, 'Attributes','GPSTimeStamp');

    numPoints = ptCloud.Count;
    points = ptCloud.Location;
    homogeneousPoints = [points, ones(numPoints, 1)];  
% % Apply the Transformation
    transformedPoints = (tmatrix * homogeneousPoints')';
    xyz = transformedPoints(:, 1:3);
    Ii = ptCloud.Intensity;
% create time array
    gpstime = Attributes.GPSTimeStamp;
    gpsEpoch = datetime(1980, 1, 6, 0, 0, 0, 'TimeZone', 'UTC');
    utcTime = gpsEpoch + gpstime;
    timeelapsed = seconds(gpstime - gpstime(1));
% % pick a freq and round to nearest Hz
    ff = 1/2;
    fftime = ff*round(timeelapsed/ff); %+1/ff;
% Load 3D point Data into columns X, Y, Z
    % quick Intensity filter, boundary filter, and time filter (5 minutes)
    % find fftime < 4minutes = 60*4
    [in, ~] = inpolygon(xyz(:,1), xyz(:,2), bounds(:,1), bounds(:,2));
    selectinds = find(xyz(:,3) < 4 & Ii < 100 & in);
    points = xyz(selectinds,:); I = Ii(selectinds);
    X = points(:,1); Y = points(:,2); Z = points(:,3);
    % % Add point boundary 
    % addpath /Users/ashton/Library/CloudStorage/OneDrive-UCSanDiego/CPG/Lidar
    % bounds = DrawLidarBounds(X,Y,Z);
    % %%
    % save('L2_lidarbounds.mat', 'bounds');
    % %
    ttime = fftime(selectinds);
    res = 0.05; 
    [x1d, Z_xt, I_xt] = Get1D_profiles_swash_accum(points, ttime, I,res);
    time_vec = unique(ttime);

    Current_time = utcTime(1) - seconds(18); % remove 18seconds for leap years
    % %
    Time = Current_time + seconds(time_vec);
    dt = ff; %seconds
    dx = res; % m
    % Save data to struct;
    savefilename = ['timestackraw_' datestr(roundToHalfHour(Current_time), 'mmdd_HHMM') '.mat'];
    save(fullfile(datapath,savefilename), 'Z_xt', 'I_xt', 'x1d', 'Current_time', 'time_vec' )

end


%%
% remove flag positions
% %%%%%  if statement for times that have flags on the beach  %%%%%%%%%
addpath '/Users/ashton/Library/CloudStorage/OneDrive-UCSanDiego/CPG/terrace'
% I_filt = I_xt; Z_filt = Z_xt;
% flags = ((x1d >= 11.65 & x1d <= 11.85) | (x1d >= 16.65 & x1d <= 17.05) | (x1d >= 20.20 & x1d <= 20.5) | ... 
%     (x1d >= 24.40 & x1d <= 24.60) | (x1d >= 27.75 & x1d <= 27.95));
% % I_filt(flags,:) = NaN; 
% Z_filt(flags,:) = NaN;
% I_filt = fillmissing(I_filt, 'linear', 1,'MaxGap', 10);
% Z_filt = fillmissing(Z_filt, 'linear', 1,'MaxGap', 10);

% calculate the minimum profile and smooth
% Z_min = minbeach(Z_filt,2);
IGlength = 60; ff = 1/2;
IGfilt = round(IGlength/ff);
nt = length(time_vec);
Z_end = Z_xt(:,end-IGfilt:end); Z_end = fliplr(Z_end);
Z_concat = horzcat(Z_xt,Z_end);
Z_min = movmin(Z_concat', IGfilt, 1, 'omitnan')';  % Transpose to work in time, then back
Z_min_s = movmean(Z_min, round(50/ff), 2);  % Smooth along time axis
Z_min_s = Z_min_s(:,1:nt);
Z_min = Z_min(:,1:nt);

I_min = movmin(I_xt', IGfilt, 1, 'omitnan')';  % Transpose to work in time, then back
I_min_s = movmean(I_min, round(50/ff), 2);  % Smooth along time axis

% Calculate the change from minimum - gives sea surface displacement
% Z_diff = Z_xt-Z_min_s;
% Z_filt(Z_diff < 0.05)=NaN;
% I_filt(Z_diff < 0.05)=NaN;
% I_max = maxbeach(I_xt,2);
Z_std = stdnd(Z_xt,2);
I_std = stdnd(I_xt,2);
Z_min_m = nanmean(Z_min_s,2);
I_min_m = nanmean(I_min_s,2);

%% plot 1D 

figure(1);clf
subplot(1,2,1)
plot(x1d, Z_min_m,'k-');hold on;
ylim([0.7 3.5]);
yyaxis right
plot(x1d, Z_std, 'r-');
xlim([4 50]);


subplot(1,2,2)
plot(x1d, I_min_m, 'k-'); hold on;
ylim([0 50]);
yyaxis right
plot(x1d, I_std, 'r-');

xlim([4 50]);
%% Visualize the space-time matrix
figure(2);clf
set(gcf,'Position', [100, 100, 1400, 800]);
% Hovmöller diagram (x-t plot)
subplot(2,1,1)
pcolor(Time, x1d, Z_xt-Z_min_s);
shading flat;
xlabel('Time (s)'); ylabel('Cross-shore Position (m)');

title('Difference of Elevation from Minimum surface (m)');
colorbar(); colormap('jet');
% clim([0.05 0.1]);
clim([0 1])
ylim([10 60]);
% xlim([200 360])
ax = gca; ax.FontSize = 14;
% Same plot but with better aspect ratio

subplot(2,1,2)
imagesc(Time, x1d, I_xt); shading flat;
xlabel('Time (s)'); ylabel('Cross-shore Position (m)');
title('Intensity of LiDAR pts');
colorbar(); colormap('jet');
set(gca, 'YDir', 'normal');
clim([0 40]);
ylim([10 60]);
% xlim([200 360])
ax = gca; ax.FontSize = 14;
% Time series at a specific location
% x_idx = round(length(x1d));  % 60% up the beach
% z_timeseries = Z_xt(x_idx, :);
sgtitle('Hovmöller plot of LiDAR data');
%% Plot water surface elevation at each flag
% stacks = [find(x1d == 27.75); find(x1d == 24.50); find(x1d == 20.50); ...
%     find(x1d == 17.75);find(x1d == 11.50)]; 
% figure(4);clf
% colors = turbo(5);
% for i = 1:5
%     stack = stacks(i);
%     plot(time_vec, Z_filt(stack,:)+ Z_min(stack), '-', 'color', colors(i,:), 'linewidth',1.7,'displayname', ['Stack ' num2str(i)]); hold on;
% end
% legend(); grid on
% ylabel('Elevation (m Navd88)');
% xlabel('Time (sec)')
% title('Lidar Sand & Water Surface elevation above Stack positions ')
% ax = gca; ax.FontSize = 14;
% % xlim([200 360])

%%
[isOutlier, Z_filt] = detect_outliers_conv2D(Z_xt, x1d, time_vec, 'thresholdStd', 4,'kernel_size', [40, 20], 'use_gradient', true, 'gradient_threshold', 4,'fillmethod', 'inpaint');
% Z_filt = Z_xt; Z_filt(isOutlier) = NaN;
%% recalculate Z_min_s
% Z_min = movmin(Z_filt', IGfilt, 1, 'omitnan')';  % Transpose to work in time, then back
% Z_min_s = movmean(Z_min, round(50/ff), 2);  % Smooth along time axis
% Z_min_m = nanmean(Z_min_s,2);
% Z_filt(isnan(Z_xt)) = 0;
% Z_filt = medfilt2(Z_filt + Z_min_s,[5,1]);

%% fill intensity outliers
I_filt = I_xt;
% I_filt(isnan(I_xt)) = 0;
I_filt(isOutlier) = NaN;
I_filt = medfilt2(I_filt,[3,1]);


%% Process data for runup timeseries
[Spec,Info,Bulk,Tseries] = get_runupStats_L2(Z_filt, I_filt, x1d, time_vec, ...
    'threshold', 0.05, ...        % Water depth threshold
    'windowlength', 2, ...        % Minutes for spectral windows
    'IGlength', 100, ...          % IG filter length (seconds)
    'use_intensity', false);      % Set to true to use intensity

%%

% median filter the runup

figure(4);clf
subplot(2,3,1:2)
plot(Time, medfilt1(Tseries.Zrunup,5/dt), 'r-', 'linewidth',1.5);

subplot(2,3,4:5)
pcolor(Time, x1d, I_filt); hold on; shading flat
colorbar(); colormap('gray'); clim([0 30]); 
plot(Time, medfilt1(Tseries.Xrunup,5/dt), 'c-', 'linewidth',1.5);
ylim([4 50])
subplot(2,3, [3 6])
plot(x1d, Z_min_m, 'k-'); hold on;
plot(Bulk.foreshoreX, Bulk.foreshore, 'r.');
%% Find Exit point from intensity data
% sample the dry sand intensity to make sure scaling is correct. 

% get the upper runup position;
UpRunup = prctile(Tseries.Xrunup, 25);
I_medfilt = fillmissing(medfilt2(I_filt, [10,10]), 'linear','EndValues','none');
Isand = mode(I_medfilt(x1d < UpRunup,:), 'all');
Iwat = mode(I_medfilt(x1d > UpRunup,:), 'all');
%%
Ithresh1 = 30;
Ithresh2 = 20;
Ithresh3 = 10;
MeanX = nanmean(Tseries.Xrunup);
thresholds = [Ithresh1, Ithresh2, Ithresh3];

% Extract contours
[contours, contour_stats] = Get_intensity_contours(I_medfilt, x1d, time_vec, thresholds, 'UseRunup', true, 'mean_runupX', MeanX);

% Get Elevation of Contours
%%
figure(1);clf
pcolor(time_vec, x1d, I_medfilt); shading flat;
colormap('jet'); colorbar(); clim([Iwat Isand]); hold on;
yline(MeanX,'w--','LineWidth',2);
ylim([4 50]);



%% Invert depth from phase speed
% I_filt(I_filt > 30) = NaN;
[depth, cphase, results] = Get_depth_from_gradient(Z_filt, x1d, time_vec, ...
    'I_xt', I_filt, ...
    'use_intensity', false, ...
    'intensity_method', 'gradient', ...
    'gradient_threshold', 15);  % Adjust based on your data
%%

figure(8);clf
subplot(3,1,1)
plot(x1d,medfilt1(mean(cphase,2,'omitnan')),'r.-'); xlim([4 50]); ylim([0 1]);grid on;
xlabel('X coords (m offshore)')
ylabel('Phase Speed (m/s)')
title('Phase speed of bores')
ax = gca; ax.FontSize = 12;
subplot(3,1,2)
plot(x1d, Z_min, 'k-', 'LineWidth',2); hold on
plot(x1d, Z_min + results.depth_corrected, 'r.'); xlim([4 50]); grid on
legend('Minimum Profile', 'Depth correction')
xlabel('X coords (m offshore)')
ylabel('Elevation (m navd88)')
sgtitle('Phase Speed Depth Correction'); 
ax = gca; ax.FontSize = 12;



%% _-------___--------___--------__----------___--------___--------__-----%

% load in all the processed data

dataFolder = '/Users/ashton/Library/CloudStorage/OneDrive-UCSanDiego/CPG/Laserdanger/rawtimestacks';
fileList = dir(fullfile(dataFolder, 'timestackraw_*.mat'));
fileNames = {fileList.name};
%
epochStrings = erase(erase(fileNames, 'timestackraw_'), '.mat');

% lidar filenames are in UTC? matlab adds 7 hours to get in UTC
% fileDates = datetime(str2double(epochStrings), 'ConvertFrom', 'posixtime', 'TimeZone','local');

fileDates = datetime(epochStrings', 'InputFormat', 'MMdd_HHmm');
%
% Sort files by timestamp
[fileDates, sortIdx] = sort(fileDates);
fileNames = fileNames(sortIdx);

addpath /Users/ashton/Library/CloudStorage/OneDrive-UCSanDiego/CPG/terrace
keepFileNames = fileNames(fileDates >= datetime([2025,07,21,15,00,00]) & fileDates <= datetime([2025,07,24,15,00,00]) );

numfiles = numel(keepFileNames);
%%
% L2 = struct();

% runupZ = []; runupX = [];
% Icontour10 = []; Icontour20 = [];
for n = 1:numfiles

    load(fullfile(dataFolder, keepFileNames{n}));

    % cut off last index (bad for some reason?)
    nt = numel(time_vec);
    % Z_xt = Z_xt(:,1:nt-1);
    % I_xt = I_xt(:,1:nt-1);
    % time_vec = time_vec(1:nt-1);
    % Current_time = Current_time;% - seconds(18); % remove 18seconds for leap years

    [isOutlier, Z_filt] = detect_outliers_conv2D(Z_xt, x1d, time_vec, 'thresholdStd', 4,'kernel_size', [40, 20], 'use_gradient', true, 'gradient_threshold', 3,'fillmethod', 'inpaint');

    % Z_filt = Z_xt; 
    % Z_filt(isOutlier) = NaN;
    I_filt = I_xt;
    % I_filt(isnan(I_xt)) = 0;
    I_filt(isOutlier) = NaN;

    flags = ((x1d >= 11.65 & x1d <= 11.85) | (x1d >= 16.8 & x1d <= 17) | (x1d >= 20.25 & x1d <= 20.45) | ... 
    (x1d >= 24.40 & x1d <= 24.60) | (x1d >= 27.75 & x1d <= 27.95));
    I_filt(flags,:) = NaN; 
    Z_filt(flags,:) = NaN;
    % linear interpolate just the portion that has flags. 
    flagrange = (x1d <= 30);
    Z_fill = Z_filt(flagrange,:); I_fill = I_filt(flagrange,:);
    Z_fill = fillmissing(Z_fill, 'linear',1,'MaxGap',8);
    I_fill = fillmissing(I_fill, 'linear',1,'MaxGap',8);
    Z_filt(flagrange,:) = Z_fill; I_filt(flagrange,:) = I_fill;
    Z_filt = medfilt2(Z_filt,[10,2]); I_filt = medfilt2(I_filt,[10,2]);
    Z_filt = Z_filt(:,1:nt-1);
    I_filt = I_filt(:,1:nt-1);
    time_vec = time_vec(1:nt-1);
    % get runup statistics
    [Spec,Info,Bulk,Tseries] = get_runupStats_L2(Z_filt, I_filt, x1d, time_vec, ...
        'threshold', 0.07, ...        % Water depth threshold
        'windowlength', 2, ...        % Minutes for spectral windows
        'IGlength', 120, ...          % IG filter length (seconds)
        'use_intensity', false, ...   % Set to true to use intensity
        'plot', false);

    % get intensity threshhold statistsics
    MeanX = nanmean(Tseries.Xrunup);
    thresholds = [20, 10, 5]; 

    % Extract contours
    [contours, contour_stats] = Get_intensity_contours(medfilt2(I_filt, [10,5]), x1d, time_vec, thresholds, ...
        'UseRunup', true, ...
        'mean_runupX', MeanX, ...
        'plot', false);

    % [contours2, contour_stats2] = Get_intensity_contours(I_filt, x1d, time_vec, thresholds, ...
    %     'UseRunup', true, ...
    %     'mean_runupX', MeanX, ...
    %     'plot', true);
    
    % save data to struct
    L2(n).Z_filt = Z_filt;
    L2(n).I_filt = I_filt;
    L2(n).x1d = x1d;
    L2(n).timesec = time_vec;
    half_seconds_per_day = 86400/.5;
    current_datenum = datenum(Current_time);
    current_date = round(current_datenum * half_seconds_per_day) / half_seconds_per_day;
    L2(n).timedate = datetime(current_date, 'convertfrom','datenum') + seconds(time_vec);

    L2(n).runupZ = Tseries.Zrunup;
    L2(n).runupX = Tseries.Xrunup;
    L2(n).I5 = contours{3}.x;
    L2(n).I10 = contours{2}.x;
    L2(n).I20 = contours{1}.x;

    % clear Z_xt I_xt Current_time time_vec x1d current_datenum current_date 

end

Z_all = horzcat(L2(:).Z_filt);
I_all = horzcat(L2(:).I_filt);
T_all = vertcat(L2(:).timedate);
x1d = L2(1).x1d;
T_full = (T_all(1):seconds(1/2):T_all(end))';

[~,missinginds] = ismember(T_all,T_full);

nx = numel(x1d);
nt = numel(T_full);
Z_full = NaN(nx, nt); I_full = NaN(nx,nt);
Z_full(:,missinginds) = Z_all; I_full(:,missinginds) = I_all;
%%
IGlength = 120; dt = 1/2;
IGfilt = round(IGlength/dt);
Z_min = movmin(Z_full', IGfilt, 1, 'omitnan')';  % Transpose to work in time, then back
Z_min_s = movmean(Z_min, IGlength, 2);  % Smooth along time axis

runupZ_full = NaN(1,nt); runupX_full = NaN(1,nt);
runupZ_all = horzcat(L2(:).runupZ); runupZ_full(missinginds) = runupZ_all;
runupX_all = horzcat(L2(:).runupX); runupX_full(missinginds) = runupX_all;

I5_full = NaN(1,nt); I10_full = NaN(1,nt);
I5_all = horzcat(L2(:).I10); I5_full(missinginds) = I5_all;
I10_all = horzcat(L2(:).I20); I10_full(missinginds) = I10_all;



%%
save('timestackprocessed_0721.mat', 'L2');
%%
figure(1);clf

subplot(1,2,1)
colors = jet(30);

k = round(size(Z_min_s,2)/30);

for i = 1:30

    n = i.*k;

    plot(x1d, movmean(Z_min_s(:,n),4), '-', 'color', colors(i,:), 'linewidth',2); hold on;

end

subplot(1,2,2)
bar(x1d, Z_min_s(:,end) - Z_min_s(:,1), 'r' ); hold on
plot(x1d, stdnd(Z_min_s,2), 'k-');
figure(2);clf
colors = turbo(3);
subplot(2,1,1)
plot(T_full, runupZ_full, 'color', colors(3,:), 'linewidth',2); hold on;
subplot(2,1,2)
plot(T_full, runupX_full, 'color', colors(2,:), 'linewidth',2);


figure(3);clf
colors = turbo(3);
subplot(2,1,1)
plot(T_full, I5_full, 'color', colors(3,:), 'linewidth',2); hold on;
plot(T_full, I10_full, 'color', colors(2,:), 'linewidth',2);

%%
figure(5);clf
xinds = find(x1d >= 10 & x1d <= 35);
tinds = find(T_full >= datetime([2025,07,20,22,00,00]) & T_full <= datetime([2025,07,21,04,00,00]));
% Z_diff = diff(Z_min_s(xinds,:),120,2);

pcolor(T_full(tinds), x1d(xinds), Z_full(xinds,tinds) - meannd(Z_min_s(xinds,tinds(1:120)),2)); shading flat;
cb=colorbar();  colormap(flipud(redblue(50)));
cb.Label.String = 'Zchange from initial (m)';
clim([-0.2 0.2])
hold on;
plot(T_full(tinds), I10_full(tinds), '-','color', colors(2,:),'linewidth',2, 'DisplayName','Threshold=10'); hold on;
% clim([-10 10])
ylabel('Xcoord')
set(gca,'Ydir','reverse')

title(' Lidar Elevations smoothed with 60sec minimum')
ax = gca; ax.FontSize = 14;
legend('Lidar pts', 'Intensity contour for runup edge')
text( T_full(tinds(1)), 12, 'Onshore of runup', 'HorizontalAlignment','right', 'FontSize',15);
%% 
x22 = find(x1d == 20);

[isOutlier, zFiltered] = persistent_random_walk_outliers(T_full, Z_full(x22,:), 1);

figure(1);clf
plot(T_full, Z_full(x22,:), 'k-'); hold on;
plot(T_full, medfilt1(zFiltered,5), 'c-', 'linewidth',2);
f = polyfit(1:nt,zFiltered);
p = polyval(f,T_full);

plot(T_full, 1:nt*p(2)+p(2),'r-')