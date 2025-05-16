function [Xutm, Yutm, Zmean, Zmax, Zmin, Zstd, Zmode] = accumpts(xyz, res)

x = xyz(:,1); y = xyz(:,2); z = xyz(:,3);
% round surveyx y utm to desired decimal place
if nargin < 2
    res = 0.1; % rounds to dplace decimal places
end
xr = res*round(x/res);
yr = res*round(y/res);

% bin rounded survey data
[ux, ~, xidx] = unique(xr);
[uy, ~, yidx] = unique(yr);

%count the number of points at each unique x/y combination
% counts = accumarray([xidx(:), yidx(:)], 1);  
% %average the z that fall into each unique x/y combination
% means = accumarray([xidx(:), yidx(:)], z.')./counts;
% stds = accumarray([xidx(:), yidx(:)], z.', [], @std); %Zstd = flipud(stds'); Zstd(Zstd == 0) = NaN;
% mins = accumarray([xidx(:), yidx(:)], z.', [], @min); %Zmin = flipud(mins'); Zmin(Zmin == 0) = NaN;
% maxs = accumarray([xidx(:), yidx(:)], z.', [], @max); %Zmax = flipud(maxs'); Zmax(Zmax == 0) = NaN;
% modes = accumarray([xidx(:), yidx(:)], z.', [], @mode); %Zmode = flipud(modes'); Zmode(Zmode == 0) = NaN;
% percentile = accumarray([xidx(:), yidx(:)], z.', [], @(v) prctile(v,percentabove));

%create a list of the z that fall into each unique x/y combination
zs = accumarray([xidx(:), yidx(:)], z.', [], @(V) {V}, {});
% ts = accumarray([xidx(:), yidx(:)], t.', [], @(V) {V}, {});

% Use 50th percentile filtering on low SNR bins
%   compare all points in zs with percentile50
%   remove points above the 50th percentile
%   recalculate mean, mode, etc.
percentabove = 50; % like a minumum surface, 
% filterStats = @(v) struct( ...
%     'mean', mean(v(v <= prctile(v,percentabove))), ...
%     'min', min(v(v <= prctile(v,percentabove))), ...
%     'mode', mode(v(v <= prctile(v,percentabove))), ...
%     'std',  std(v(v <= prctile(v,percentabove))), ...
%     'count', numel(v(v <= prctile(v,percentabove))) ...
%     );
% stats = cellfun(filterStats, zs, 'UniformOutput', false);
% means = cellfun(@(s) s.mean, stats);
% mins = cellfun(@(s) s.min, stats);
% modes = cellfun(@(s) s.mode, stats);
% stds  = cellfun(@(s) s.std, stats);
% counts = cellfun(@(s) s.count, stats);

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
mins(invalid_idx) = NaN;
modes(invalid_idx) = NaN;
stds(invalid_idx) = NaN;
modes(invalid_idx) = NaN;

ii=isnan(counts(:)) == 0; % 1d indices of valid data
[i,j]=find(isnan(counts) == 0); % 2d indices of valid data

Xutm=ux(i);Yutm=uy(j);
Zmean=means(ii);
Zmax=maxs(ii);
Zmin=mins(ii); 
Zstd=stds(ii);
Zmode=modes(ii);
% Zs=zs(ii,:);
% Ts=ts(ii,:);
end

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

