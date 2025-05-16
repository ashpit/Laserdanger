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

% Load config
config = jsondecode(fileread('livox_config2.json'));
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
fileDates = datetime(str2double(epochStrings), 'ConvertFrom', 'posixtime', 'TimeZone','UTC');

% Sort files by timestamp
[fileDates, sortIdx] = sort(fileDates);
fileNames = fileNames(sortIdx);

% Define reference time (latest file)
latestDate = fileDates(end);
% Define time threshold (2 days prior)
cutoffDate = latestDate - days(2);
% Find indices of files within the past 2 days
recentIdx = fileDates >= cutoffDate;
% Subset filenames and datetimes
recentFileDates = fileDates(recentIdx);
recentFileNames = fileNames(recentIdx);
num_to_process = numel(recentFileDates);
%
% Round fileDates to day (or half-hour) for grouping
fileDatesRounded = dateshift(recentFileDates, 'start', 'day');

uniqueDays = unique(fileDatesRounded);
%%
for d = 1:numel(uniqueDays)
    thisDay = uniqueDays(d);
    
    % Find all files from this day
    idxToday = find(fileDatesRounded == thisDay);
    num_to_process = numel(idxToday);
    % Construct output filename for the day
    outName = ['L1_' datestr(thisDay, 'yyyymmdd') '.mat'];
    outPath = fullfile(ProcessFolder, outName);
    
    % Load existing day's data or initialize empty
    if isfile(outPath)
        load(outPath, 'L1_day'); N = numel(L1_day);
    else
        L1_day = struct('Dates', {}, 'X', {}, 'Y', {}, 'Zmean', {}, ...
                    'Zmax', {}, 'Zmin', {}, 'Zmode', {}, 'Zstd', {});
        N = 0;
    end

    % Process files for this day
    j = 1;
    for i = idxToday
        currentFile = fullfile(dataFolder, recentFileNames{i});
        
        % Round datenum for existence check
        % currentDatenum = round(datenum(recentFileDates(i)) / (0.5/24)) * (0.5/24);
        currentDvec = datevec(recentFileDates(i));
        currentDvec(:,5) = round(currentDvec(:,5)/30)*30;currentDvec(:,6) = 0;
        currentDatenum = datenum(currentDvec);
        % Extract all existing rounded datenums from L1_day
        % existingDatenums = round(datenum([L1_day.Dates]) / (0.5/24)) * (0.5/24);
        existingDvecs = datevec([L1_day.Dates]);
        existingDvecs(:,5) = round(existingDvecs(:,5)/30)*30;existingDvecs(:,6) = 0;
        existingDatenums = datenum(existingDvecs);
        % Check if current file has already been processed
        if ~isempty(existingDatenums) && any(abs(existingDatenums - currentDatenum) < 1e-6)
                fprintf('Skipping already processed file: %s\n', recentFileNames{i});
            continue;
        end
        
        try
            % Process lidar file
            L1_append = process_lidar_L1(currentFile, tmatrix, bounds);  % Your processing function
            % remove fields I don't want in there
            if isfield(L1_day, 'Zstdv')
                L1_day = rmfield(L1_day, 'Zstdv');
            end
            % Append new data
            L1_day(N+1).Dates = L1_append.Dates;
            L1_day(N+1).X = L1_append.X;
            L1_day(N+1).Y = L1_append.Y;
            L1_day(N+1).Zmean = L1_append.Zmean;
            L1_day(N+1).Zmax = L1_append.Zmax;
            L1_day(N+1).Zmin = L1_append.Zmin;
            L1_day(N+1).Zmode = L1_append.Zmode;
            L1_day(N+1).Zstd = L1_append.Zstd;
            save(outPath, 'L1_day');
            fprintf('Processed hour %d/%d: %s\n', j, num_to_process, datestr(currentDatenum));
        catch ME
            warning('Failed processing %s: %s', recentFileNames{i}, ME.message);
            continue;
        end
        j = j + 1;
    end

    % delete duplicate surveys
    Dvecs = datevec([L1_day.Dates]);Dvecs(:,5) = round(Dvecs(:,5)/30)*30;Dvecs(:,6) = 0;
    Dnums = datenum(Dvecs);
    [uniqueDnums, uniqueIdx] = unique(Dnums, 'stable');  % keep first occurrence
    % Keep only unique entries
    L1_day = L1_day(uniqueIdx);
    % Save day's data once after all files processed
    % save(outPath, 'L1_day')
    fprintf('Processed day %d/%d: %s\n', d, numel(uniqueDays), datestr(thisDay));
end

%% --- STEP 6: save Plot data for export to html webserver ---
% thisDay = datetime([2025, 05, 15]);
% Parameters
nDaysBack = 4;
thisDateDT = thisDay;  % ensure it's datetime

% Find all L1_*.mat files
allFiles = dir(fullfile(ProcessFolder, 'L1_*.mat'));

% Extract dates from filenames
fileDates = NaT(size(allFiles));
for i = 1:numel(allFiles)
    tokens = regexp(allFiles(i).name, 'L1_(\d{8})\.mat', 'tokens', 'once');
    if ~isempty(tokens)
        fileDates(i) = datetime(tokens{1}, 'InputFormat', 'yyyyMMdd');
    end
end
% fileDates.TimeZone = 'UTC';
% Find files within the N-day window
validIdx = find(fileDates >= thisDateDT - day(nDaysBack - 1));
[~, sortOrder] = sort(fileDates(validIdx));
validIdx = validIdx(sortOrder);

% Load all L1 structs from those files
L1all = [];
for i = validIdx'
    fname = fullfile(ProcessFolder, allFiles(i).name);
    try
        tmp = load(fname, 'L1_day'); %L1_day = tmp.L1_day;
        % if isfield(L1_day, 'Datenum')
        %     L1_day = rmfield(L1_day, 'Datenum');
        %     save(fname, 'L1_day');
        % end     
        L1all = [L1all, tmp.L1_day];
    catch ME
        warning("Failed to load %s: %s", allFiles(i).name, ME.message);
    end
end
%%
% load the stack positions data
load('stackpos.mat');
plotIdx = 1:numel(L1all); dt = 24;
% make sure plotIdx is divisible by 24, if not shorten it from start
k = 1;

while mod(numel(plotIdx),dt) > 0
    k = k+1;
    plotIdx = k:numel(L1all);
end

%
% figure(1); clf
z1da = [];
plotData = struct('dates', {}, 'x', {}, 'z', {}, 'color', {});
colors = repmat([0.6 0.6 0.6], numel(plotIdx), 1);  % dull gray
colors(end, :) = [0 0.5 1];  % bold blue for latest

for j = 1:dt:numel(plotIdx)
    i = plotIdx(j);
    X = L1all(i).X; Y = L1all(i).Y; Z = L1all(i).Zmode;

    if isempty(X)
        continue
    end

    [x1d, Z3D] = Get3_1Dprofiles(X, Y, Z);
    z1d = Z3D(5, x1d <= 50);  % Profile at fixed y-index
    x1d_crop = x1d(x1d <= 50);
   
    % plot(x1d_crop, movmean(z1d,5), '-', 'Color', colors(j,:), 'LineWidth', 4*(j/numel(plotIdx)), 'DisplayName', datestr(L1(i).Dates)); hold on

    % Save to structure for JSON
    plotData(j).dates = datestr(L1all(i).Dates);
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

% --- Save the data to JSON file ---
jsonFilename = fullfile(plotFolder, 'lidar_plot_data.json');
fid = fopen(jsonFilename, 'w');
fprintf(fid, '%s', jsonencode(outData));
fclose(fid);

% clearvars