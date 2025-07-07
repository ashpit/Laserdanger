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


%%
% Load config JSON
config = jsondecode(fileread('livox_config.json'));
dataFolder = config.dataFolder;
ProcessFolder = config.processFolder;
outputPath = fullfile(ProcessFolder, config.outputFile);
tmatrix = config.transformMatrix;
bounds = config.LidarBoundary;
%% Load existing LO structure if available
if isfile(outputPath)
    load(outputPath); % loads struct labeled (L1)
    % Start from next hour after last timestamp
    N = numel(L1);
    Start = L1(end).Dates; 
else
    % No existing DO — start fresh
    L1 = struct('Dates', {}, 'X', {}, 'Y', {}, 'Zmean', {},'Zmax', {}, ...
                'Zmin', {}, 'Zstd', {}, 'Zmode', {});
    N = 0;
    % Default earliest start time (first of April as fallback)
    Start = datetime(2025, 05, 01); Start.TimeZone = 'UTC';
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
fileDates = datetime(epochNumbers, 'ConvertFrom', 'posixtime', 'TimeZone','UTC');
validDates = fileDates(fileDates > Start);
validFiles = fileNames(fileDates > Start);
% Example: get nearest hour for last file
End = dateshift(fileDates(end), 'start', 'hour', 'nearest');
num_to_process = numel(validDates);
%% Enter the loop 
%if isempty(validFiles)
    % start the loop
    %fprintf('no pointclouds to process')
%end
% for n = 1:num_to_process
n = 30; % an hour of data (5 min or 20min)
    currentFile = fullfile(dataFolder, validFiles{n});
    Xtime = validDates(n);

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
    selectinds = find(fftime <= 60*2 & Ii < 100 & in);
    points = xyz(selectinds,:);
    X = points(:,1); Y = points(:,2); Z = points(:,3);
% Rasterize pointcloud using accumarray script 
    % binsize = 0.10; % in meters

    %%

x = points(:,1); y = points(:,2); z = points(:,3); t = fftime(selectinds);
res = 0.1;
xr = res*round(x/res);
yr = res*round(y/res);

% bin rounded survey data
[ux, ~, xidx] = unique(xr);
[uy, ~, yidx] = unique(yr);
zs = accumarray([xidx(:), yidx(:)], z.', [], @(V) {V}, {});
xs = accumarray([xidx(:), yidx(:)], x.', [], @(V) {V}, {});
ts = accumarray([xidx(:), yidx(:)], t.', [], @(V) {V}, {});
%%
Zs = zs(:);
Ts = ts(:);
Xs = xs(:);

%%
figure(1); set(gcf, 'position', [200 300 1500 500]);clf
scatter(t, -(x - 476190), 10, z, 'filled');
grid on; grid minor
xlabel('Seconds');
ylabel('Xposition')
colorbar()
clim([1 3])
%%
res = 0.1;
ttime = fftime(selectinds);
[Xutm, Yutm, Zmean, Zmax, Zmin, Zstd, Zmode, xr,zr,tr] = accumpts_L2(points,ttime, res);


%%
figure(2); set(gcf, 'position', [200 300 1500 500]);clf
scatter(tr, -(xr - 476190), 10, zr, 'filled');
grid on; grid minor
xlabel('Seconds');
ylabel('Xposition')
colorbar()
clim([1 3])

%%


%%

    points3 = [Xutm, Yutm, Zmin];
    windowSize = 2; thresh = 0.5;
    [groundPoints, ~] = ResidualKernelFilter(points3, windowSize, thresh);
    Xutmc = Xutm(groundPoints);
    Yutmc = Yutm(groundPoints);
    Zmeanc = Zmean(groundPoints);
    Zmaxc = Zmax(groundPoints);
    Zminc = Zmin(groundPoints);
    Zmodec = Zmode(groundPoints);
    Zstdc = Zstd(groundPoints);

    figure(1);clf
scatter3(Xutmc, Yutmc, Zminc, 5, 'b', 'filled');
%%
figure(1);set(gcf, 'position', [300 600 700 700]);clf
for n = 1:80
    scatter(L1(n).X, L1(n).Y, 10, L1(n).Zmean - L1(n).Zmin, 'filled');
    colorbar()
    clim([0 0.2]); 
    xlim([476145 476190]); ylim([3636310 3636370]);
    pause(0.1)
end

%%
    % Save a struct with updated indexing
    L0(n+N).Dates = Xtime;% + minutes(n) - minutes(1);
    L0(n+N).X = Xutm;
    L0(n+N).Y = Yutm;
    L0(n+N).Zmean = Zavg;
    L0(n+N).Zmin = Zmin;
    L0(n+N).Zmode = Zmode;
    L0(n+N).Zstdv = Zstd;
    L0(n+N).SNR = SNR;
    % DO(n+N).RawZ = zp;
    % save the data to output file and print message
    save(outputPath, 'L0', '-v7.3');
    fprintf('Processed hour %d/%d: %s\n', n, numel(validFolders), datestr(Xtime));

    % DO(p).Zminmean = mean(Zmin,2);
%
% end    

%%



figure(2);clf
subplot(2,2,1)
q = pcolor(ux, uy, snr'); set(q, 'EdgeColor','none')
colorbar()
clim([0 1000])
title('Signal to Noise Ratio', 'FontSize',14)
ylim([3636325 3636345]); xlim([476155 476187]);
subplot(2,2,2)
q = pcolor(ux, uy, stds'); set(q, 'EdgeColor','none')
colorbar()
clim([0 1])
title('Standard Deviation', 'FontSize',14)
ylim([3636325 3636345]); xlim([476155 476187]);
subplot(2,2,3)
q = pcolor(ux, uy, zcount'); set(q, 'EdgeColor','none')
colorbar()
title('Bin count over 2 minutes of scans', 'FontSize',14)
ylim([3636325 3636345]); xlim([476155 476187]);
clim([0 60])
subplot(2,2,4)
q = pcolor(ux, uy, zmode'); set(q, 'EdgeColor','none')
colorbar()
title('Bin Average filtered with SNR < 1e3*Res & bin count > 3', 'FontSize',14)
ylim([3636325 3636345]); xlim([476155 476187]);
clim([1 3]);
set(gcf, 'color', [1 1 1])
%% Plot those bad boys (bad boys = 14 most recent low tide surveys

fun = @(s) all(structfun(@isempty,s));
idx = arrayfun(fun,DO);
DO(idx)=[];

% Define the filename for storing the plot data
jsonFilename = 'lidar_plot_data.json';
%
% --- STEP 1: Find low-tide entries ---
% Replace this logic with your actual tide logic if needed
% lowTideIdx = find(arrayfun(@(d) isfield(d, 'TideHeight') && d.TideHeight < 0.5, DO));
% [~,lowTideIdx]= findpeaks(-tidehgt);
% lowtidetime = tidetime(lowTideIdx);
% 
% % --- STEP 2: Get the most recent 7 low-tide surveys ---
% recentLowTideIdx = lowTideIdx(lowtidetime <= DO(end).Dates);
% %   - Convert low-tide idx to hourly idx from DO
% % --- STEP 3: Always include the most recent survey ---
% [~, mostRecentIdx] = max([DO.Dates]);
% if ~ismember(mostRecentIdx, recentLowTideIdx)
%     plotIdx = [recentLowTideIdx, mostRecentIdx];
% else
%     plotIdx = recentLowTideIdx;
% end
plotIdx = find([DO(:).Dates] >= [DO(end-120).Dates]);
% --- STEP 4: Plot and store data ---
figure(1); clf
z1da = [];
plotData = struct('dates', {}, 'x', {}, 'z', {}, 'color', {});
colors = repmat([0.6 0.6 0.6], numel(plotIdx), 1);  % dull gray
colors(end, :) = [0 0.5 1];  % bold blue for latest

for j = 1:6:numel(plotIdx)
    i = plotIdx(j);
    X = DO(i).X; Y = DO(i).Y; Z = DO(i).Zmode;

    if isempty(X)
        continue
    end

    [x1d, Z3D] = Get3_1Dprofiles(X, Y, Z);
    z1d = Z3D(5, x1d <= 50);  % Profile at fixed y-index
    x1d_crop = x1d(x1d <= 50);

    plot(x1d_crop, movmean(z1d,5), '-', 'Color', colors(j,:), 'LineWidth', round(numel(plotIdx)/(numel(plotIdx) - j/1.3)), 'DisplayName', datestr(DO(i).Dates)); hold on
    % plot(x1d_crop, movmean(z1d,5), '-', 'Color', colors(j,:), 'LineWidth', round((j + numel(DO)+1)/numel(DO)), 'DisplayName', datestr(DO(i).Dates)); hold on
% end
    % Save to structure for JSON
    plotData(j).dates = datestr(DO(i).Dates);
    plotData(j).x = x1d_crop;
    plotData(j).z = z1d;
    plotData(j).color = colors(j,:);
end

% Optional: add MHHW, MHW, MSL lines
MHHW = 1.566; MSL = 0.774; MHW = 1.344;
plot([0 50], [MHHW MHHW], 'k--', 'handlevisibility', 'off'); text(5, MHHW+0.05, 'MHHW')
plot([0 50], [MHW MHW], 'k--', 'handlevisibility', 'off'); text(5, MHW+0.05, 'MHW')
plot([0 50], [MSL MSL], 'k--', 'handlevisibility', 'off'); text(5, MSL+0.05, 'MSL')

grid on; legend show
title(['Recent 1D Profiles on' datestr(DO(end).Dates)] );
xlabel('Cross-shore Distance (m)');
ylabel('Elevation (NAVD88)');
ylim([0.5 3.5]); xlim([0 50]);

% --- STEP 5: Save the data to JSON file ---
fid = fopen(jsonFilename, 'w');
fprintf(fid, '%s', jsonencode(plotData));
fclose(fid);
set(gcf, 'color', 'w')
%%

figure(1);
grid on; grid minor

%%
% Your plotting code
% figure(1); clf
% z1da = [];
% colors = cool(numel(DO));
% 
% % Define the filename for storing the plot data
% jsonFilename = 'lidar_plot_data.json';
% 
% % Load existing plot data if available
% if isfile(jsonFilename)
%     plotData = jsondecode(fileread(jsonFilename));
% else
%     plotData = struct('dates', {}, 'x', {}, 'z', {}, 'color', {});
% end
% 
% % For new plot data
% for i = 1:5
%     X = DO(i).X; Y = DO(i).Y; Z = DO(i).Zmode;
%     [x1d, Z3D] = Get3_1Dprofiles(X, Y, Z);
%     z1da(i, :) = Z3D(5, :);
% 
%     z1d = z1da(i,x1d <= 50);
%     x1d(x1d > 50) = [];
% 
%     % Store the new plot data
%     plotData(i).dates = datestr(DO(i).Dates);
%     plotData(i).x = x1d;
%     plotData(i).z = z1d;
%     plotData(i).color = colors(i, :);  % Store color for each plot
% end
% 
% % Save the plot data to JSON file
% fid = fopen(jsonFilename, 'w');
% fprintf(fid, '%s', jsonencode(plotData));
% fclose(fid);

%
% % 
% figure(1);clf
% z1da = [];
% colors = cool(numel(DO));
% for i = 1:5
%     X = DO(i).X; Y = DO(i).Y; Z = DO(i).Zmode;
%     [x1d,Z3D]=Get3_1Dprofiles(X,Y,Z);
%     z1da(i,:) = Z3D(6,:);
%     plot(x1d, z1da(i,:), '-', 'Color', colors(i,:), 'DisplayName', datestr(DO(i).Dates)); hold on
%     % plot(x1d, movmean(z1da(i,:),5,'omitnan'), '-', 'Color', colors(i,:), 'Linewidth', 2); hold on
% end
% %
% MHHW = 1.566; MSL = 0.774; MHW = 1.344;
% figure(1);hold on
% ylim([0.5 3.5])
% xlim([0 40])
% title('May 30 & August 30 & April 15-17 1D profiles');
% xlabel('Crosshore Position (m from sea wall) ');
% ylabel('Elevation (navd88)')
% grid on;
% grid minor;
% legend()
% hold on;
% plot([x1d(1) x1d(end)], [ MHHW MHHW], 'k--');
% text(10, MHHW+0.05, 'MHHW')
% plot([x1d(1) x1d(end)], [ MHW MHW], 'k--');
% text(10, MHW+0.05, 'MHW')
% plot([x1d(1) x1d(end)], [ MSL MSL], 'k--');
% text(10, MSL+0.05, 'MSL')
% ax1 = gca;
% ax1.FontSize = 14;
