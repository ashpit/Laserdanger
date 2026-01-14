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
% Load config
config = jsondecode(fileread('livox_config.json'));
dataFolder = config.dataFolder;
ProcessFolder = config.processFolder;
plotFolder = config.plotFolder;
tmatrix = config.transformMatrix;
bounds = config.LidarBoundary;
%%
% Get LiDAR files
fileList = dir(fullfile(dataFolder, 'do-lidar_*.laz'));
fileNames = {fileList.name};
epochStrings = erase(erase(fileNames, 'do-lidar_'), '.laz');
% lidar filenames are in UTC? matlab adds 7 hours to get in UTC
fileDates = datetime(str2double(epochStrings), 'ConvertFrom', 'posixtime', 'TimeZone','local');

% Sort files by timestamp
[fileDates, sortIdx] = sort(fileDates);
fileNames = fileNames(sortIdx);

% Define reference time (latest file)
latestDate = fileDates(end);
% Define time threshold (4 days prior)
cutoffDate = latestDate - days(10);
% Find indices of files within the past 2 days
recentIdx = fileDates >= cutoffDate;
% Subset filenames and datetimes
recentFileDates = fileDates(recentIdx);
recentFileNames = fileNames(recentIdx);
num_to_process = numel(recentFileDates);
% Round fileDates to day (or half-hour) for grouping
fileDatesRounded = dateshift(recentFileDates, 'start', 'day');
uniqueDays = unique(fileDatesRounded);
%% Enter the loop 
%if numel(validFiles) >= 1
    % start the loop
% for n = 1:num_to_process
% n = num_to_process;
n = 68;
    currentFile = fullfile(dataFolder, validFiles{n});
    Xtime = validDates(63);

    lasreader = lasFileReader(currentFile);
    [ptCloud, Attributes] = readPointCloud(lasreader, 'Attributes','GPSTimeStamp');

    numPoints = ptCloud.Count;
    points = ptCloud.Location;
    homogeneousPoints = [points, ones(numPoints, 1)];        
%% Apply the Transformation
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
    binsize = 0.10; % in meters
    [Xutm, Yutm, Zmean, Zmax, Zmin, Zstd, Zmode] = accumpts(points, binsize);
%% Filter residual noise using svd triangle planar fitting 
    points3 = [Xutm, Yutm, Zmode];
    windowSize = 10; thresh = 0.2;
    [groundPoints, ~] = ResidualKernelFilter(points3, windowSize, thresh);
    Xutmc = Xutm(groundPoints);
    Yutmc = Yutm(groundPoints);
    Zmeanc = Zmean(groundPoints);
    Zmaxc = Zmax(groundPoints);
    Zminc = Zmin(groundPoints);
    Zmodec = Zmode(groundPoints);
    Zstdc = Zstd(groundPoints);
% Filter again with smaller triangles
    points3 = [Xutmc, Yutmc, Zmodec];
    windowSize = 3; thresh = 0.1;
    [groundPoints, Z_interp] = ResidualKernelFilter(points3, windowSize, thresh);
    X_clean = Xutmc(groundPoints);
    Y_clean = Yutmc(groundPoints);
    Zmean_clean = Zmeanc(groundPoints);
    Zmax_clean = Zmaxc(groundPoints);
    Zmin_clean = Zminc(groundPoints);
    Zmode_clean = Zmodec(groundPoints);
    Zstd_clean = Zstdc(groundPoints);

%% save struct
    L1(n+N).Dates = utcTime(1);
    L1(n+N).X = X_clean;
    L1(n+N).Y = Y_clean;
    L1(n+N).Zmean = Zmean_clean;
    L1(n+N).Zmax = Zmax_clean;
    L1(n+N).Zmin = Zmin_clean;
    L1(n+N).Zstdv = Zstd_clean;
    L1(n+N).Zmode = Zmode_clean;
    % save(outputPath, 'L1', '-v7.3');
    % fprintf('Processed hour %d/%d: %s\n', n, num_to_process, datestr(Xtime));

% end
%end
%%
% Define the filename for storing the plot data
jsonFilename = 'lidar_plot_data.json';
load('stackpos.mat');
plotIdx = 1:numel(L1); dt = 1;

if numel(L1) > 24
    plotIdx = find([L1.Dates] >= [L1(end-240).Dates]);
    dt = 48;

end
% --- STEP 4: Plot and store data ---
% figure(1); clf
z1da = [];
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
   
    % plot(x1d_crop, movmean(z1d,5), '-', 'Color', colors(j,:), 'LineWidth', 4*(j/numel(plotIdx)), 'DisplayName', datestr(L1(i).Dates)); hold on

    % Save to structure for JSON
    plotData(j).dates = datestr(L1(i).Dates);
    plotData(j).x = x1d_crop;
    plotData(j).z = z1d;
    plotData(j).color = colors(j,:);
end

% % Optional: add MHHW, MHW, MSL lines
% MHHW = 1.566; MSL = 0.774; MHW = 1.344;
% plot([0 50], [MHHW MHHW], 'k--', 'handlevisibility', 'off'); text(5, MHHW+0.05, 'MHHW')
% plot([0 50], [MHW MHW], 'k--', 'handlevisibility', 'off'); text(5, MHW+0.05, 'MHW')
% plot([0 50], [MSL MSL], 'k--', 'handlevisibility', 'off'); text(5, MSL+0.05, 'MSL')
% grid on; legend show
% title(['Recent 1D Profiles since' datestr(L1(end-numel(plotIdx)).Dates)] );
% xlabel('Cross-shore Distance (m)');
% ylabel('Elevation (NAVD88)');
% ylim([0.5 3.5]); xlim([0 50]);
% set(gcf, 'color', 'w')
% ax1=gca; ax1.FontSize=14;

% scatter([stackplotData.x], [stackplotData.z], 100, 'r^', 'filled','HandleVisibility', 'off');
outData = struct();
outData.profiles = plotData;  % this is the array of profiles
outData.stackx = [stackplotData.x];
outData.stackz = [stackplotData.z];
% outData.MHHW = [1.566 1.566]; outData.MSL = [0.774 0.774]; outData.MHW = [1.344 1.344];
% outData.datumsx = [ 0 50]; 

% --- STEP 5: Save the data to JSON file ---
fid = fopen(jsonFilename, 'w');
fprintf(fid, '%s', jsonencode(outData));
fclose(fid);
