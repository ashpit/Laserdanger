function [Xutm, Yutm, Zmean, Zmax, Zmin, Zstd, Zmode, reconstructed_x, reconstructed_y, reconstructed_z, reconstructed_t] = accumpts_L2(xyz,t, res)

x = xyz(:,1); y = xyz(:,2); z = xyz(:,3);
% x = points(:,1); y = points(:,2); z = points(:,3);
% res = 0.1;
% round surveyx y utm to desired decimal place
if nargin < 2
    res = 0.1; % rounds to dplace decimal places
end
xr = res*round(x/res);
yr = res*round(y/res);

% bin rounded survey data
[ux, ~, xidx] = unique(xr);
[uy, ~, yidx] = unique(yr);

%create a list of the z that fall into each unique x/y combination
zs = accumarray([xidx(:), yidx(:)], z.', [], @(V) {V}, {});
ts = accumarray([xidx(:), yidx(:)], t.', [], @(V) {V}, {});
xs = accumarray([xidx(:), yidx(:)], x.', [], @(V) {V}, {});
ys = accumarray([xidx(:), yidx(:)], y.', [], @(V) {V}, {});

% Use 50th percentile filtering on low SNR bins
%   compare all points in zs with percentile50
%   remove points above the 50th percentile
%   recalculate mean, mode, etc.
percentabove = 70; % like a minumum surface,
stats = cellfun(@(v) filterStats(v, percentabove), zs, 'UniformOutput', false);
means = cellfun(@(s) s.mean, stats);
maxs = cellfun(@(s) s.max, stats);
mins  = cellfun(@(s) s.min, stats);
modes = cellfun(@(s) s.mode, stats);
stds  = cellfun(@(s) s.std, stats);
counts= cellfun(@(s) s.count, stats);

Error = stds./sqrt(counts);
% calculate signal to noise ratio for each x,y bin
snr = means./ Error ;%.* (sigma_ref ./ zstd); 

% Area = (max(ux)-min(ux))*(max(uy)-min(uy)); 
% numpoints = numel(x);
% fprintf('SNR threshold %f', snr_thresh);
invalid_idx = snr < 100 | counts <= 10;

% fprintf('Points removed by snr thresh and count < 10 %f', sum(invalid_idx));
counts(invalid_idx) = NaN;
means(invalid_idx) = NaN;
maxs(invalid_idx) = NaN;
mins(invalid_idx) = 0;
modes(invalid_idx) = NaN;
stds(invalid_idx) = NaN;
modes(invalid_idx) = NaN;

%%
% Initialize a cell array to store the difference
zs_diff = cell(size(zs));

% Subtract the mins surface from zs for each bin
for i = 1:numel(zs)
    % Get the corresponding minimum value for this (x, y) bin
    Zmin_bin = mins(i);  % mins contains the minimum z value for each (x, y) bin
    
    % Subtract the minimum value from each z value in the bin
    zs_diff{i} = zs{i} - Zmin_bin;
end

% 
% figure(2);clf
% pcolor(ux,uy, zs_diff');

% Set a threshold for what counts as "ground"
threshold = 0.1;  % Adjust this value based on your data

% Remove points where the difference is below the threshold
ground_points = cellfun(@(z_diff) abs(z_diff) < threshold, zs_diff, 'UniformOutput', false);

% Now remove the ground points by setting them to NaN or empty
filtered_zs = cellfun(@(z, ground) z(~ground), zs, ground_points, 'UniformOutput', false);
filtered_ts = cellfun(@(t, ground) t(~ground), ts, ground_points, 'UniformOutput', false);
filtered_xs = cellfun(@(x, ground) x(~ground), xs, ground_points, 'UniformOutput', false);
filtered_ys = cellfun(@(y, ground) y(~ground), ys, ground_points, 'UniformOutput', false);

ii=isnan(counts(:)) == 0; % 1d indices of valid data
[i,j]=find(isnan(counts) == 0); % 2d indices of valid data

Xutm=ux(i);Yutm=uy(j);
Zmean=means(ii);
Zmax=maxs(ii);
Zmin=mins(ii); 
Zstd=stds(ii);
Zmode=modes(ii);
% Zs=zs(ii);
% Ts=ts(ii);


reconstructed_x = [];
reconstructed_y = [];
reconstructed_z = [];
reconstructed_t = [];

for i = 1:length(ux)
    for j = 1:length(uy)
        % Get filtered values for each (i, j) index
        x_values = filtered_xs{i, j};
        y_values = filtered_ys{i, j};
        z_values = filtered_zs{i, j};
        t_values = filtered_ts{i, j};
        
        % Append to the reconstructed arrays
        reconstructed_x = [reconstructed_x; x_values];
        reconstructed_y = [reconstructed_y; y_values];
        reconstructed_z = [reconstructed_z; z_values];
        reconstructed_t = [reconstructed_t; t_values];
    end
end


end
%%
% accumarray function
function s = filterStats(v, percentabove)
    threshold = prctile(v, percentabove);
    v_filtered = v(v <= threshold);
    if isempty(v_filtered)
        s.mean = NaN;
        s.max = NaN;
        s.min = NaN;
        s.mode = NaN;
        s.std = NaN;
        s.count = 0;
    else
        s.mean = mean(v_filtered);
        s.max = mean(v_filtered);
        s.min = min(v_filtered);
        s.mode = mode(v_filtered);
        s.std = std(v_filtered);
        s.count = numel(v_filtered);
    end
end

