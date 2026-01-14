function [isOutlier, Z_filtered] = detect_outliers_conv2D(Z_xt, x1d, time_vec, varargin)
% Fast 2D outlier detection using convolutional filters
% Much more efficient than loop-based approach
%
% Input:
%   Z_xt:       Elevation matrix (cross-shore x time)
%   x1d:        Cross-shore positions (m)
%   time_vec:   Time vector (datetime or seconds)
%   varargin:   Optional parameters
%
% Output:
%   isOutlier:  Logical matrix same size as Z_xt (true = outlier)
%   Z_filtered: Cleaned elevation matrix

% Default options
options.thresholdStd = 2.5;          % Std threshold for outlier detection
options.kernel_size = [3, 3];        % Size of spatial kernel [x, t]
options.fillmethod = 'inpaint';      % 'inpaint', 'linear', 'median'
options.gradient_threshold = 2.5;    % Threshold for gradient spikes
options.dilate_outliers = true;      % Expand outlier regions
options.dilation_sigma = 1.5;        % Gaussian sigma for dilation (in grid units)
options.dilation_threshold = 0.3;    % Threshold for expanded mask (0-1)
options.plot = false;
options = parseOptions(options, varargin);

fprintf('Fast Convolutional 2D Outlier Detection\n');
fprintf('Input size: %d x %d (x, t)\n', size(Z_xt));

%% Remove minimum surface
IGlength = 60;
dt = mean(diff(time_vec));
dx = mean(diff(time_vec));
IGfilt = round(IGlength/dt);
Z_min = movmin(Z_xt', IGfilt, 1, 'omitnan')';  % Transpose to work in time, then back
Z_min_s = movmean(Z_min, round(50/dt), 2);  % Smooth along time axis

Z_diff = medfilt2(Z_xt - Z_min_s,[round(0.3/dx),2]);
%% Method 2: Gradient-Based Spike Detection Using Convolution

% Sobel filters for gradients
sobel_x = [-1 0 1; -2 0 2; -1 0 1];  % Spatial gradient
sobel_t = [-1 -2 -1; 0 0 0; 1 2 1];  % Temporal gradient

% Compute gradients
grad_x = conv2(Z_diff, sobel_x, 'same');
grad_t = conv2(Z_diff, sobel_t, 'same');

% Gradient magnitude
grad_mag = sqrt(grad_x.^2 + grad_t.^2);

% Detect spikes: high gradient isolated points
% Use Laplacian to find points with high curvature
laplacian_kernel = [0 1 0; 1 -4 1; 0 1 0];
laplacian = conv2(Z_diff, laplacian_kernel, 'same');

% Outliers have both high gradient AND high laplacian (sharp peaks)
grad_threshold = nanmean(grad_mag(:)) + options.gradient_threshold * nanstd(grad_mag(:));
lap_threshold = nanmean(abs(laplacian(:))) + options.gradient_threshold * nanstd(laplacian(:));

isOutlier = (grad_mag > grad_threshold) & (abs(laplacian) > lap_threshold);
isOutlier(isnan(Z_diff)) = false;

se = strel('rectangle',[6 2]);
isOutlier = imdilate(isOutlier, se);

fprintf('Gradient-based outliers: %d (%.2f%%)\n', ...
    sum(isOutlier(:)), 100*sum(isOutlier(:))/numel(isOutlier));

%% Fill Outliersfig
fprintf('Filling outliers using method: %s\n', options.fillmethod);

Z_filtered = Z_xt;
% Z_filtered(isnan(Z_xt)) = 0;
Z_filtered(isOutlier) = NaN;
% Z_filtered = medfilt2(Z_filtered, [4,2]);
% switch options.fillmethod
%     case 'inpaint'
%         % Use inpainting (regionfill or inpaint_nans)
%         % if exist('inpaint_nans', 'file')
%             % Z_filtered = inpaint_nans(Z_filtered);
%         % else
%             % Fallback: iterative diffusion-based inpainting
%             Z_filtered = inpaint_diffusion(Z_filtered, isOutlier);
%         % end
% 
%     case 'linear'
%         % Linear interpolation in x only 
%         % for i = 1:size(Z_filtered, 1)
%         %     row = Z_filtered(i, :);
%         %     valid = ~isnan(row);
%         %     if sum(valid) > 1
%         %         Z_filtered(i, :) = interp1(find(valid), row(valid), 1:length(row), 'linear', 'extrap');
%         %     end
%         % end
%         for j = 1:size(Z_filtered, 2)
%             col = Z_filtered(:, j);
%             valid = ~isnan(col);
%             if sum(valid) > 1
%                 Z_filtered(:, j) = interp1(find(valid), col(valid), 1:length(col), 'linear', 'extrap');
%             end
%         end
% 
%     case 'median'
%         % Median filter to fill gaps
%         Z_filtered = medfilt2(Z_filtered, [3 3], 'symmetric');
% 
%         % For remaining NaNs, use local median
%         for i = 1:size(Z_filtered, 1)
%             for j = 1:size(Z_filtered, 2)
%                 if isnan(Z_filtered(i, j))
%                     % Get 5x5 neighborhood
%                     i_range = max(1, i-2):min(size(Z_filtered,1), i+2);
%                     j_range = max(1, j-2):min(size(Z_filtered,2), j+2);
%                     local = Z_filtered(i_range, j_range);
%                     Z_filtered(i, j) = nanmedian(local(:));
%                 end
%             end
%         end
% 
%     case 'conv'
%         % Iterative convolution-based filling
%         Z_filtered = inpaint_conv(Z_filtered, isOutlier);
% end
% 
% Z_filtered(isnan(Z_xt)) = NaN;
% fill in small holes in X-range
% Z_filtered = medfilt2(Z_filtered, [6,1]);

%% Visualization
if options.plot == true
figure('Position', [100, 100, 1600, 1000]);

% Original data
subplot(2,2,1)
pcolor(time_vec, x1d, Z_xt);
shading interp;
xlabel('Time'); ylabel('Cross-shore (m)');
title('Original Z(x,t)');
colorbar(); colormap(gca, 'jet'); clim([1 3]); ylim([4 60]);
xlim([0 400])
% Gradient Mag
subplot(2,2,2)
pcolor(time_vec, x1d, grad_mag);
shading flat;
xlabel('Time'); ylabel('Cross-shore (m)');
title('Gradient Outliers');
colorbar(); colormap(gca, 'hot'); ylim([4 60]); clim([0 3]);
xlim([0 200])

% Gradient outliers
subplot(2,2,3)
pcolor(time_vec, x1d, double(isOutlier));
shading flat;
xlabel('Time'); ylabel('Cross-shore (m)');
title('Gradient Outliers');
colormap(gca, 'gray'); ylim([4 60])


% Filtered data
subplot(2,2,4)
pcolor(time_vec, x1d, Z_filtered);
shading interp;
xlabel('Time'); ylabel('Cross-shore (m)');
title('Filtered Z(x,t)');
colorbar(); colormap(gca, 'jet'); clim([ 1 3]); ylim([4 60])
xlim([0 100])
% % Gradient magnitude
% subplot(2,2,4)
% pcolor(time_vec, x1d, double(isOutlier_sm));
% shading flat;
% xlabel('Time'); ylabel('Cross-shore (m)');
% title('Gradient Magnitude');
% colorbar(); colormap(gca, 'gray');  ylim([4 40]);%clim([ 0 4]);
end
%% Helper Functions

    function Z_filled = inpaint_diffusion(Z, mask)
        % Simple diffusion-based inpainting
        Z_filled = Z;
        max_iter = 50;
        
        for iter = 1:max_iter
            Z_old = Z_filled;
            
            % Diffusion kernel
            kernel = [0 1 0; 1 0 1; 0 1 0] / 4;
            Z_smoothed = conv2(Z_filled, kernel, 'same');
            
            % Update only the outlier locations
            Z_filled(mask) = Z_smoothed(mask);
            
            % Check convergence
            if max(abs(Z_filled(mask) - Z_old(mask))) < 1e-4
                break;
            end
        end
        fprintf('  Inpainting converged in %d iterations\n', iter);
    end

    function Z_filled = inpaint_conv(Z, mask)
        % Fast convolution-based gap filling
        Z_filled = Z;
        
        % Gaussian kernel for smooth interpolation
        sigma = 1.5;
        k_size = 10;
        [x, y] = meshgrid(-floor(k_size/2):floor(k_size/2));
        kernel = exp(-(x.^2 + y.^2)/(2*sigma^2));
        kernel = kernel / sum(kernel(:));
        
        % Weight matrix (0 for outliers, 1 for valid data)
        W = double(~mask);
        W(isnan(Z)) = 0;
        
        % Iterative weighted averaging
        max_iter = 50;
        for iter = 1:max_iter
            % Convolve data and weights
            Z_conv = conv2(Z_filled .* W, kernel, 'same');
            W_conv = conv2(W, kernel, 'same');
            
            % Update outlier locations
            Z_new = Z_conv ./ (W_conv + eps);
            Z_filled(mask) = Z_new(mask);
            
            % Gradually increase weights at filled locations
            W(mask) = min(1, W(mask) + 0.1);
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