function [contours, contour_stats] = Get_intensity_contours(I_xt, x1d, time_vec, thresholds, varargin)
% Extract contour lines from intensity data at multiple threshold levels
% Useful for tracking foam/water boundaries in lidar intensity data
%
% Input:
%   I_xt:       Intensity matrix (cross-shore x time)
%   x1d:        Cross-shore positions (m)
%   time_vec:   Time vector (datetime or seconds)
%   thresholds: Array of intensity thresholds to extract [Ithresh1, Ithresh2, ...]
%   varargin:   Optional parameters
%
% Output:
%   contours:       Cell array of contour data for each threshold
%   contour_stats:  Statistics for each contour (mean, std, etc.)

% Default options
options.smooth_contours = true;      % Smooth extracted contours
options.smooth_window = 5;           % Moving average window
options.min_points = 10;             % Minimum points for valid contour
options.interpolate_gaps = true;     % Fill gaps in contours
options.max_gap = 5;                 % Maximum gap to interpolate (time steps)
options.method = 'linear';           % 'linear' or 'contour' extraction
options.UseRunup = false;
options.mean_runupX = 40;            % Mean runup Xcoord, for an offshore bound
options.plot = true;
options = parseOptions(options, varargin);

fprintf('Extracting intensity contours...\n');
fprintf('Intensity range: %.1f to %.1f\n', nanmin(I_xt(:)), nanmax(I_xt(:)));
fprintf('Number of thresholds: %d\n', length(thresholds));

%% Initialize output
n_thresh = length(thresholds);
contours = cell(n_thresh, 1);
contour_stats = struct();

%% Extract contours for each threshold
for k = 1:n_thresh
    thresh = thresholds(k);
    fprintf('\nProcessing threshold %d: I = %.1f\n', k, thresh);
    
    switch options.method
        case 'linear'
            % Method 1: Linear interpolation approach (fast, robust)
            contour_x = extract_contour_linear(I_xt, x1d, time_vec, thresh, options);
            
        case 'contour'
            % Method 2: MATLAB contour function (more accurate but can be jumpy)
            contour_x = extract_contour_matlab(I_xt, x1d, time_vec, thresh, options);
    end
    
    % Store results
    contours{k}.threshold = thresh;
    contours{k}.x = contour_x;
    contours{k}.t = time_vec;
    
    % Calculate statistics
    valid = ~isnan(contour_x);
    if sum(valid) > 0
        contour_stats(k).threshold = thresh;
        contour_stats(k).mean_position = nanmean(contour_x);
        contour_stats(k).std_position = nanstd(contour_x);
        contour_stats(k).min_position = nanmin(contour_x);
        contour_stats(k).max_position = nanmax(contour_x);
        contour_stats(k).valid_fraction = sum(valid) / length(contour_x);
        
        fprintf('  Mean position: %.2f m\n', contour_stats(k).mean_position);
        fprintf('  Std: %.2f m\n', contour_stats(k).std_position);
        fprintf('  Valid points: %.1f%%\n', 100*contour_stats(k).valid_fraction);
    else
        fprintf('  Warning: No valid contour points found!\n');
        contour_stats(k).threshold = thresh;
        contour_stats(k).mean_position = NaN;
        contour_stats(k).std_position = NaN;
        contour_stats(k).valid_fraction = 0;
    end
end

%% Visualization
if options.plot == true
    visualize_contours(I_xt, x1d, time_vec, contours, contour_stats);
end
%% Helper Functions

    function x_contour = extract_contour_linear(I_xt, x1d, time_vec, thresh, opts)
        % Extract contour using linear interpolation at each time step
        
        nt = length(time_vec);
        x_contour = nan(1, nt);

        if opts.UseRunup && opts.mean_runupX > 20
            RunupX = opts.mean_runupX;
            idxoffshore = x1d > RunupX;
            I_xt(idxoffshore,:) = NaN;
        end


        % else 
        %     RunupX = 40;
        % end
        
        for j = 1:nt
            I_profile = I_xt(:, j);
            % skip NaNs
            valid_idx = ~isnan(I_profile);
            x_valid = x1d(valid_idx);
            I_profile = I_profile(valid_idx);
            
            % Skip if too many NaNs
            if sum(~isnan(I_profile)) < 3
                continue;
            end
            
            % Find where intensity crosses threshold
            % Look for transition from high to low intensity (sand to water)
            I_diff = I_profile - thresh;
            
            % Find zero crossings (sign changes)
            sign_changes = find(diff(sign(I_diff)) ~= 0);
            
            if ~isempty(sign_changes)
                % Take the last crossing (seaward-most)
                % Or could take last crossing (shoreward-most)
                idx = sign_changes(end);
                
                % Linear interpolation for sub-grid accuracy
                I1 = I_profile(idx);
                x1 = x_valid(idx);
                % I2 = I_profile(idx+1);
                % x2 = x_valid(idx+1);
                
                if ~isnan(I1) %&& ~isnan(I2) && abs(I2 - I1) > 1e-6
                    % Linear interpolation: x = x1 + (thresh - I1)/(I2 - I1) * (x2 - x1)
                    x_contour(j) = x1;% + (thresh - I1) / (I2 - I1) * (x2 - x1);
                end
            end
        end
        
        % % Remove outliers
        % x_contour = filloutliers(x_contour, NaN, 'movmedian', 10);
        % 
        % % Smooth contours
        % if opts.smooth_contours
        %     valid = ~isnan(x_contour);
        %     if sum(valid) > opts.smooth_window
        %         x_contour(valid) = movmean(x_contour(valid), opts.smooth_window, 'omitnan');
        %     end
        % end
        % 
        % % Interpolate small gaps
        % if opts.interpolate_gaps
        %     gaps = gapsize(x_contour);
        %     small_gaps = gaps > 0 & gaps <= opts.max_gap;
        %     x_contour = fillmissing(x_contour, 'linear', ...
        %         'SamplePoints', 1:length(x_contour), ...
        %         'EndValues', 'none');
        %     % Don't fill large gaps
        %     x_contour(gaps > opts.max_gap) = NaN;
        % end
        % 
        % % Remove if too few points
        % if sum(~isnan(x_contour)) < opts.min_points
        %     x_contour(:) = NaN;
        % end
    end

    function x_contour = extract_contour_matlab(I_xt, x1d, time_vec, thresh, opts)
        % Extract contour using MATLAB's contour function
        
        nt = length(time_vec);
        x_contour = nan(1, nt);
        
        % Use contour to extract the threshold line
        [C, ~] = contour(time_vec, x1d, I_xt, [thresh, thresh]);
        
        if size(C, 2) > 1
            % Parse contour matrix
            % Format: [level, level, ...; numpoints, x1, x2, ...; numpoints, y1, y2, ...]
            idx = 1;
            all_x = [];
            all_t = [];
            
            while idx < size(C, 2)
                level = C(1, idx);
                n_pts = C(2, idx);
                
                if level == thresh && n_pts > 0
                    t_seg = C(1, idx+1:idx+n_pts);
                    x_seg = C(2, idx+1:idx+n_pts);
                    
                    all_t = [all_t, t_seg];
                    all_x = [all_x, x_seg];
                end
                
                idx = idx + n_pts + 1;
            end
            
            % Interpolate to regular time grid
            if ~isempty(all_t)
                if isdatetime(time_vec)
                    t_numeric = datenum(all_t);
                    t_grid = datenum(time_vec);
                else
                    t_numeric = all_t;
                    t_grid = time_vec;
                end
                
                x_contour = interp1(t_numeric, all_x, t_grid, 'linear');
            end
        end
        
        % Apply same post-processing as linear method
        if opts.smooth_contours
            valid = ~isnan(x_contour);
            if sum(valid) > opts.smooth_window
                x_contour(valid) = movmean(x_contour(valid), opts.smooth_window, 'omitnan');
            end
        end
    end

    function visualize_contours(I_xt, x1d, time_vec, contours, stats)
        % Visualization of extracted contours
        
        figure('Position', [100, 100, 1400, 800]);
        
        % Plot 1: Intensity with all contours overlaid
        subplot(2,1,1)
        pcolor(time_vec, x1d, I_xt);
        shading interp;
        hold on;
        
        % Color scheme for multiple contours
        colors = lines(length(contours));
        
        for k = 1:length(contours)
            plot(contours{k}.t, contours{k}.x, '.-', ...
                'Color', colors(k,:), 'LineWidth', 2);
        end
        ylim([4 50]);
        xlabel('Time (sec)'); ylabel('Cross-shore (m)');
        title('Intensity with Extracted Contours');
        colorbar(); colormap(gca, 'jet'); clim([0 40]);
        
        % Add legend
        legend_str = arrayfun(@(c) sprintf('I = %.1f', c.threshold), ...
            [contours{:}], 'UniformOutput', false);
        legend([{'Intensity'} legend_str], 'Location', 'best');
        
        % Plot 2: Individual contour time series
        subplot(2,1,2)
        for k = 1:length(contours)
            plot(contours{k}.t, contours{k}.x, '.-', ...
                'Color', colors(k,:), 'LineWidth', 1.5);
            hold on;
        end
        xlabel('Time (sec)'); ylabel('Contour Position (m)');
        title('Contour Positions vs Time');
        legend(legend_str, 'Location', 'best');
        grid on;
        
    end

    function sz = gapsize(x)
        % Calculate size of NaN gaps in vector
        sz = zeros(size(x));
        if isempty(x) || all(~isnan(x))
            return;
        end
        
        nan_idx = isnan(x);
        d = diff([0; nan_idx(:); 0]);
        start_idx = find(d == 1);
        end_idx = find(d == -1) - 1;
        
        for i = 1:length(start_idx)
            gap_length = end_idx(i) - start_idx(i) + 1;
            sz(start_idx(i):end_idx(i)) = gap_length;
        end
    end

    function options = parseOptions(defaults, cellString)
        p = inputParser;
        p.KeepUnmatched = true;
        names = fieldnames(defaults);
        for ii = 1:length(names)
            addOptional(p, names{ii}, defaults.(names{ii}));
        end
        parse(p, cellString{:});
        options = p.Results;
    end

end