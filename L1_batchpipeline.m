%% Lidar Pipeline

%%-----------------------------------------------------------------------%%
%           HOURLY LIDAR POINT CLOUD PROCESSING PIPELINE (April 2025)     %
%------------------------------------------------------------------------%%

%%--[0] SETUP & PARAMETERS
% ---> Define input/output folders, transformation matrix, bounds
%       -> load json configuration file

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
%%
L1all = [];
% for d = 1:numel(uniqueDays)
d = 3;
% Find all files from this day
    thisDay = uniqueDays(d);
    idxToday = find(fileDatesRounded == thisDay);
    numToday = numel(idxToday);
% Construct output filename for the day
    outName = ['L1_' datestr(thisDay, 'yyyymmdd') '.mat'];
    outPath = fullfile(ProcessFolder, outName);
% Load existing day's data or initialize empty
    if isfile(outPath)
        load(outPath, 'L1_day'); 
        N = numel(L1_day); % number of files already processed
    else  % create empty struct
        L1_day = struct('Dates', {}, 'X', {}, 'Y', {}, 'Zmean', {}, ...
                    'Zmax', {}, 'Zmin', {}, 'Zmode', {}, 'Zstd', {});
        N = 0;
    end
%%
    % Process files for this day
    j = 1;
    % for i = idxToday
    i = 63;
        thisFile = fullfile(dataFolder, recentFileNames{i});
        % Round datenum for existence check
        thisHour = roundToHalfHour(recentFileDates(i)); 
        % Extract all existing rounded datenums from L1_day
        processedHours = roundToHalfHour([L1_day.Dates]);
        % Check if current file has already been processed
        % if ~isempty(processedHours) && any(abs(processedHours - thisHour) < 1e-6)
        %     fprintf('Skipping already processed file: %s\n', recentFileNames{i});
        %     L1all = [L1all L1_day];
        %     % continue;
        % end
        N = numel(L1_day);
        % try
            % Process lidar file
            L1_append = process_lidar_L1(thisFile, tmatrix, bounds);  % Your processing function
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
            Dnums = roundToHalfHour([L1_day.Dates]);
            [uniqueDnums, uniqueIdx] = unique(Dnums, 'stable');  % keep first occurrence
            % Keep only unique entries
            L1_day = L1_day(uniqueIdx);
            [~, sortIdx] = sort([L1_day.Dates]);
            L1_day = L1_day(sortIdx);
            L1all = [L1all L1_day];
            save(outPath, 'L1_day');
            fprintf('Processed hour %d/%d: %s\n', j, numToday, datestr(thisHour));
        % catch ME
        %     warning('Failed processing %s: %s', recentFileNames{i}, ME.message);
        %     continue;
        % end
        j = j + 1;
    % end
    fprintf('Processed day %d/%d: %s\n', d, numel(uniqueDays), datestr(thisDay));
% end
%% --- STEP 6: save Plot data for export to html webserver ---
% load the stack positions data
% adjust L1all
Dnums = roundToHalfHour([L1all.Dates]);
[uniqueDnums, uniqueIdx] = unique(Dnums, 'stable');  % keep first occurrence
% Keep only unique entries
L1all = L1all(uniqueIdx);
[~, sortIdx] = sort([L1all.Dates]);
L1all = L1all(sortIdx);

load('stackpos.mat');
plotIdx = 1:numel(L1all); dt = 24;
% make sure plotIdx is divisible by 24, if not shorten it from start
k = 1;

while mod(numel(plotIdx),dt) > 0
    k = k+1;
    plotIdx = k:numel(L1all);
end

% figure(1); clf
z1da = [];
plotData = struct('dates', {}, 'x', {}, 'z', {}, 'color', {});
stepIdx = 1:dt:numel(plotIdx);

% Ensure the last index is included
if stepIdx(end) < numel(plotIdx)
    stepIdx(end+1) = numel(plotIdx);
end

colors = repmat([0.6 0.6 0.6], numel(stepIdx), 1);
colors(end, :) = [0 0.5 1];  % bold blue for last profile

for j = 1:numel(stepIdx)
    i = plotIdx(stepIdx(j));
    X = L1all(i).X; Y = L1all(i).Y; Z = L1all(i).Zmode;

    if isempty(X)
        continue
    end

    [x1d, Z3D] = Get3_1Dprofiles(X, Y, Z);
    z1d = Z3D(5, x1d <= 50);  % Profile at fixed y-index
    x1d_crop = x1d(x1d <= 50);
   
    % plot(x1d_crop, movmean(z1d,5), '-', 'Color', colors(j,:), 'LineWidth', 4*(j/numel(stepIdx)), 'DisplayName', datestr(L1all(i).Dates)); hold on
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
% title(['Recent 1D Profiles since' datestr(L1all(end-numel(plotIdx)).Dates)] );
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

% --- Save the data to JSON file ---
jsonFilename = fullfile(plotFolder, 'lidar_plot_data.json');
fid = fopen(jsonFilename, 'w');
fprintf(fid, '%s', jsonencode(outData));
fclose(fid);

% clearvars
% delete the lockfile

%%


function dn = roundToHalfHour(dt)
    vec = datevec(dt);
    vec(:,5) = round(vec(:,5)/30)*30;  % Round minutes
    vec(:,6) = 0;  % Zero out seconds
    dn = datenum(vec);
end