function L1_append = process_lidar_L1(currentFile, tmatrix, bounds)

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

    if gpstime(1) < 1e9 
        gpstime = gpstime + 1e9;
    end
    utcTime = gpsEpoch + gpstime;
    timeelapsed = seconds(gpstime - gpstime(1));
    % pick a freq and round to nearest Hz
    ff = 1;
    fftime = ff*round(timeelapsed/ff)+1;

% Load 3D point Data into columns X, Y, Z
    % quick Intensity filter, boundary filter, and time filter (5 minutes)
    % find fftime < 4minutes = 60*4
    [in, ~] = inpolygon(xyz(:,1), xyz(:,2), bounds(:,1), bounds(:,2));
    selectinds = (fftime <= 60*5 & Ii < 100 & in);
    points = xyz(selectinds,:);
    % X = points(:,1); Y = points(:,2); Z = points(:,3);
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
    [groundPoints, ~] = ResidualKernelFilter(points3, windowSize, thresh);
    X_clean = Xutmc(groundPoints);
    Y_clean = Yutmc(groundPoints);
    Zmean_clean = Zmeanc(groundPoints);
    Zmax_clean = Zmaxc(groundPoints);
    Zmin_clean = Zminc(groundPoints);
    Zmode_clean = Zmodec(groundPoints);
    Zstd_clean = Zstdc(groundPoints);

%% save struct
    L1_append.Dates = utcTime(1);
    L1_append.X = X_clean;
    L1_append.Y = Y_clean;
    L1_append.Zmean = Zmean_clean;
    L1_append.Zmax = Zmax_clean;
    L1_append.Zmin = Zmin_clean;
    L1_append.Zstd = Zstd_clean;
    L1_append.Zmode = Zmode_clean;
    % save(outputPath, 'L1', '-v7.3');
    % fprintf('Processed hour %d/%d: %s\n', n, num_to_process, datestr(Xtime));