%% Lidar Pipeline

%%-----------------------------------------------------------------------%%
%           HOURLY LIDAR POINT CLOUD PROCESSING PIPELINE (April 2025)     %
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


%% Load config JSON
config = jsondecode(fileread('livox_config.json'));
dataFolder = config.dataFolder;
ProcessFolder = config.processFolder;
outputPath = fullfile(ProcessFolder, config.outputFile);
tmatrix = config.transformMatrix;
bounds = config.LidarBoundary;
%% Load existing LO structure if available
if isfile(outputPath)
    L1 = load(outputPath);
    % varNames = fieldnames(S);
    % L0_varname = varNames{contains(varNames, 'L0')};
    % L0 = S.(L0_varname);
    
    % Start from next hour after last timestamp
    N = numel(L1);
    Start = dateshift(L1(end).Dates + hours(1), 'start','hour');
else
    % No existing DO — start fresh
    L1 = struct('Dates', {}, 'X', {}, 'Y', {}, 'Zmean', {}, 'Zmin', {},...
                'Zstdv', {}, 'Zmode', {}, 'SNR', {});
    N = 0;
    % Default earliest start time (first of April as fallback)
    Start = datetime(2025, 05, 01);
end

% get the files
fileList = dir(fullfile(dataFolder, 'do-lidar_*.laz'));

% Extract just the epoch part from filenames
fileNames = {fileList.name};
epochStrings = erase(fileNames, 'do-lidar_');   % remove prefix
epochStrings = erase(epochStrings, '.laz');     % remove extension

% Convert epoch strings to numbers
epochNumbers = str2double(epochStrings);

% Convert to datetime
fileDates = datetime(epochNumbers, 'ConvertFrom', 'posixtime');
validDates = fileDates(fileDates > Start);
validFiles = fileNames(fileDates > Start);
% Example: get nearest hour for last file
End = dateshift(fileDates(end), 'start', 'hour', 'nearest');
num_to_process = numel(validDates);
%% Enter the loop 

for n = 1:num_to_process
    currentFile = fullfile(dataFolder, validFiles{n});
    Xtime = validDates(n);

    lasreader = lasFileReader(currentFile);
    [ptCloud, Attributes] = readPointCloud(lasreader, 'Attributes','GPSTimeStamp');

    numPoints = ptCloud.Count;
    points = ptCloud.Location;
    homogeneousPoints = [points, ones(numPoints, 1)];        
% Apply the Transformation
    transformedPoints = (tmatrix * homogeneousPoints')';
    xyz = transformedPoints(:, 1:3);
    Ii = ptCloud.Intensity;
% create time array
    gpstime = Attributes.GPSTimeStamp;
    gpsEpoch = datetime(1980, 1, 6, 0, 0, 0, 'TimeZone', 'UTC');
    utcTime = gpsEpoch + gpstime;
    timeelapsed = seconds(gpstime - gpstime(1));
    % pick a freq and round to nearest Hz
    ff = 1;
    fftime = ff*round(timeelapsed/ff)+1;

% Load 3D point Data into columns X, Y, Z
    % quick Intensity filter, boundary filter, and time filter (5 minutes)
    % find fftime < 4minutes = 60*4
    [in, ~] = inpolygon(xyz(:,1), xyz(:,2), bounds(:,1), bounds(:,2));
    selectinds = find(fftime <= 60*5 & Ii < 100 & in);
    points = xyz(selectinds,:);
    X = points(:,1); Y = points(:,2); Z = points(:,3);
% Rasterize pointcloud using accumarray script 
    binsize = 0.25; % in meters
    [Xutm, Yutm, Zmean, Zmin, Zstd, Zmode] = accumpts(points, binsize);
% Filter residual noise using svd triangle planar fitting 
    points3 = [Xutm, Yutm, Zmode];
    windowSize = 3; thresh = 0.2;
    [groundPoints, Z_interp] = ResidualKernelFilter(points3, windowSize, thresh);
    X_clean = Xutm(groundPoints);
    Y_clean = Yutm(groundPoints);
    Zmean_clean = Zmean(groundPoints);
    Zmode_clean = Zmode(groundPoints);
    Zmin_clean = Zmin(groundPoints);
    Zstd_clean = Zstd(groundPoints);
% save struct
    L1(n+N).Dates = utcTime(1);
    L1(n+N).X = X_clean;
    L1(n+N).Y = Y_clean;
    L1(n+N).Zmean = Zmean_clean;
    L1(n+N).Zmin = Zmin_clean;
    L1(n+N).Zstd = Zstd_clean;
    L1(n+N).Zmode = Zmode_clean;
    save(outputPath, 'L1', '-v7.3');
    fprintf('Processed hour %d/%d: %s\n', n, num_to_process, datestr(Xtime));

end

%%
% Define the filename for storing the plot data
jsonFilename = 'lidar_plot_data.json';
load('stackpos.mat');
plotIdx = 1:numel(L1); dt = 1;

if numel(L1) > 120
    plotIdx = find([L1.Dates] >= [L1(end-72).Dates]);
    dt = 6;

end
% --- STEP 4: Plot and store data ---
% figure(1); clf
% z1da = [];
plotData = struct('dates', {}, 'x', {}, 'z', {}, 'color', {});
colors = repmat([0.6 0.6 0.6], numel(plotIdx), 1);  % dull gray
colors(end, :) = [0 0.5 1];  % bold blue for latest

for j = 1:dt:numel(plotIdx)
    i = plotIdx(j);
    X = L1(i).X; Y = L1(i).Y; Z = L1(i).Zmode;

    if isempty(X)
        continue
    end

    [x1d, Z3D] = Get3_1Dprofiles(X, Y, Z);
    z1d = Z3D(5, x1d <= 50);  % Profile at fixed y-index
    x1d_crop = x1d(x1d <= 50);

    plot(x1d_crop, movmean(z1d,5), '-', 'Color', colors(j,:), 'LineWidth', round(numel(plotIdx)/(numel(plotIdx) - j/1.3)), 'DisplayName', datestr(L1(i).Dates)); hold on
    plot(x1d_crop, movmean(z1d,5), '-', 'Color', colors(j,:), 'LineWidth', round((j + numel(L1)+1)/numel(L1)), 'DisplayName', datestr(L1(i).Dates)); hold on
% % end
    % Save to structure for JSON
    plotData(j).dates = datestr(L1(i).Dates);
    plotData(j).x = x1d_crop;
    plotData(j).z = z1d;
    plotData(j).color = colors(j,:);
    % for i = 1:5
        scatter([stackplotData.x], [stackplotData.z], 100, 'r^', 'filled','HandleVisibility', 'off');
    % end
end


plotData.stackx = [stackplotData.x];
plotData.stackz = [stackplotData.z];
% Optional: add MHHW, MHW, MSL lines
MHHW = 1.566; MSL = 0.774; MHW = 1.344;
plot([0 50], [MHHW MHHW], 'k--', 'handlevisibility', 'off'); text(5, MHHW+0.05, 'MHHW')
plot([0 50], [MHW MHW], 'k--', 'handlevisibility', 'off'); text(5, MHW+0.05, 'MHW')
plot([0 50], [MSL MSL], 'k--', 'handlevisibility', 'off'); text(5, MSL+0.05, 'MSL')

grid on; legend show
title(['Recent 1D Profiles on' datestr(L1(end).Dates)] );
xlabel('Cross-shore Distance (m)');
ylabel('Elevation (NAVD88)');
ylim([0.5 3.5]); xlim([0 50]);
set(gcf, 'color', 'w')
ax1=gca; ax1.FontSize=14;
% --- STEP 5: Save the data to JSON file ---
fid = fopen(jsonFilename, 'w');
fprintf(fid, '%s', jsonencode(plotData));
fclose(fid);


    % %%
    % figure(2); set(gcf, 'position', [100 200 1000 1000]);clf
    % scatter3(X, Y, Z, 5, 'k', 'filled');hold on
    % title(['Raw Point Cloud for ' datestr(utcTime(1))])
    % xlabel('Xutm'); ylabel('Yutm');
    % set(gcf, 'Color', 'w');
    % ax1=gca; ax1.FontSize = 16;
    % %%
    % clf
    % scatter3(Xutm, Yutm, Zmean, 5, 'k', 'filled');
    % title(['Rasterized and fitlered Point Cloud for ' datestr(utcTime(1))])
    % xlabel('Xutm'); ylabel('Yutm');
    % set(gcf, 'Color', 'w');
    % ax1=gca; ax1.FontSize = 16;
    % %%
    % hold on
    % scatter3(X_clean, Y_clean, Zmean_clean, 5, 'r', 'filled');hold on
