function [x1d, Z_xt, I_xt] = Get1D_profiles_swash_accum(xyz, t, I,res)
% Combined function that bins data and extracts transect profiles
%
% INPUTS:
%   xyz         - Nx3 matrix of coordinates [x, y, z]
%   t           - Nx1 vector of time values
%   I           - Nx1 vector of intensity values
%   res_grid    - Grid resolution for binning (default: 0.1 m)
%   res_transect- Cross-shore resolution for transect (default: 0.05 m)
%
% OUTPUTS:
%   x1d    - Cross-shore positions along transect
%   Z_xt   - Elevation matrix (cross-shore x time)
%   I_xt   - Intensity matrix (cross-shore x time)

if nargin < 4, res = 0.1; end
% res = 0.05;
% res = 0.1; 

%%  Bin the data using accumarray
fprintf('Binning data with %.2f m resolution...\n', res);

x = xyz(:,1); y = xyz(:,2); z = xyz(:,3);

% Round to grid resolution
xr = res * round(x / res);
yr = res * round(y / res);

% Get unique grid positions
[ux, ~, xidx] = unique(xr);
[uy, ~, yidx] = unique(yr);

% Create cell arrays with all data at each grid cell
zs = accumarray([xidx(:), yidx(:)], z.', [], @(V) {V}, {});
ts = accumarray([xidx(:), yidx(:)], t.', [], @(V) {V}, {});
Is = accumarray([xidx(:), yidx(:)], I.', [], @(V) {V}, {});

fprintf('Created %d x %d grid (%d unique positions)\n', ...
    length(ux), length(uy), sum(~cellfun(@isempty, zs(:))));

%%  Transect geometry
x1 = 476190.0618275550;   y1 = 3636333.442333425;
x2 = 475620.6132784432;   y2 = 3636465.645345889;

% Compute transect angle
ang = atan2(y2 - y1, x2 - x1);

% Transect parameters
ExtendLine = [0 -300];
tol = 2;  % Tolerance (m)

% Define transect endpoints
xs = x1; ys = y1;
xe = x2; ye = y2;

% Create cross-shore distance vector
dist = pdist([xs, ys; xe, ye]);
ExtendOff = ExtendLine(2) + dist;

% Cross-shore positions
x1d = ExtendLine(1):res:dist+ExtendLine(2);

% UTM coordinates along transect
xt = xs + ExtendLine(1)*cos(ang):res*cos(ang):xs + ExtendOff*cos(ang);
yt = ys + ExtendLine(1)*sin(ang):res*sin(ang):ys + ExtendOff*sin(ang);

% Transect line vector
lineVec = [xe - xs, ye - ys];
lineVec = lineVec / norm(lineVec);

%% Find grid cells near transect
% fprintf('Identifying grid cells near transect...\n');
% 
% % Find which cells actually contain data and check if near transect
% nearTransect = false(size(zs));
% 
% for i = 1:length(ux)
%     for j = 1:length(uy)
%         if ~isempty(zs{i,j})
%             % This cell has data, check distance to transect
%             cell_point = [ux(i), uy(j)];
% 
%             % Project onto transect
%             lineToPointVec = cell_point - [xs, ys];
%             projLength = dot(lineToPointVec, lineVec);
%             projPoint = [xs, ys] + projLength * lineVec;
% 
%             % Distance to transect line
%             distToLine = norm(cell_point - projPoint);
% 
%             nearTransect(i,j) = (distToLine <= tol);
%         end
%     end
% end
% 
% fprintf('Found %d grid cells within %.1f m of transect\n', sum(nearTransect(:)), tol);

% --- Optimized Section: Find grid cells near transect ---
fprintf('Identifying grid cells near transect (Optimized)...\n');

% 1. Get coordinates of all non-empty grid cell centers
[I, J] = find(~cellfun(@isempty, zs));
if isempty(I), nearTransect = false(size(zs)); return; end

all_cell_centers = [ux(I), uy(J)];
start_point = [xs, ys];
lineVec = [xe - xs, ye - ys];
lineVec = lineVec / norm(lineVec);

% 2. Vectorized calculation of projected length (dot product)
% The difference vector from the line start to all cell points
lineToPointVec = all_cell_centers - start_point; 
projLength = lineToPointVec * lineVec'; 

% 3. Vectorized calculation of projected point coordinates
projPoint = start_point + projLength .* lineVec;

% 4. Vectorized calculation of distance to the line (Euclidean distance)
distToLine = sqrt(sum((all_cell_centers - projPoint).^2, 2));

% 5. Identify which cells are near the transect
is_near = (distToLine <= tol);

% 6. Map back to the cell array index (nearTransect is the same size as zs)
nearTransect = false(size(zs));
nearTransect(sub2ind(size(zs), I(is_near), J(is_near))) = true;

fprintf('Found %d grid cells within %.1f m of transect\n', sum(nearTransect(:)), tol);
% --- End Optimized Section ---

% Extract time series for each grid cell near transect
time_vec = unique(t);
nt = length(time_vec);

fprintf('Processing %d time steps...\n', nt);

% Initialize output matrices
Z_xt = nan(length(x1d), nt);
I_xt = nan(length(x1d), nt);

%% Loop through time and extract profiles
% for t_idx = 1:nt
%     current_time = time_vec(t_idx);
% 
%     % Collect all points at this time from cells near transect
%     x_t = []; y_t = []; z_t = []; i_t = [];
% 
%     for i = 1:size(zs, 1)
%         for j = 1:size(zs, 2)
%             if nearTransect(i, j) && ~isempty(ts{i,j})
%                 % Find indices where time matches
%                 time_match = ts{i,j} == current_time;
% 
%                 if any(time_match)
%                     % Use the gridded x/y position for this cell
%                     n_pts = sum(time_match);
%                     x_t = [x_t; repmat(ux(i), n_pts, 1)];
%                     y_t = [y_t; repmat(uy(j), n_pts, 1)];
%                     z_t = [z_t; zs{i,j}(time_match)];
%                     i_t = [i_t; Is{i,j}(time_match)];
%                 end
%             end
%         end
%     end
% 
%     if isempty(x_t)
%         continue;
%     end
% 
%     % Map points to transect positions
%     [~, NearIdx] = pdist2([xt(:), yt(:)], [x_t, y_t], 'euclidean', 'smallest', 1);
%     [~, col] = ind2sub(size(xt), NearIdx);
%     X = x1d(col);
% 
%     % Bin by cross-shore position and take mean
%     Xuniq = unique(X);
%     Zproj = nan(size(Xuniq));
%     Iproj = nan(size(Xuniq));
% 
%     for n = 1:length(Xuniq)
%         Zproj(n) = nanmean(z_t(X == Xuniq(n)));
%         Iproj(n) = nanmean(i_t(X == Xuniq(n)));
%     end
% 
% --- Optimized Pre-Loop Section: Extract all near-transect data ---
% --- Optimized Pre-Loop Section: Extract all near-transect data ---

% 1. Get the indices of the non-empty, near-transect cells (I_near and J_near)
[I_near, J_near] = find(nearTransect);

% 2. Extract the data for the near cells

% CORRECTED: ux, uy are numeric arrays, index with parentheses.
% These will be N_near x 1 arrays of the cell center coordinates.
x_cell_centers = ux(I_near); 
y_cell_centers = uy(J_near);

% The data in zs, ts, Is are the points themselves (cell arrays), 
% so they must be indexed with parentheses and flattened with cell2mat.
z_near = cell2mat(zs(nearTransect)); 
t_near = cell2mat(ts(nearTransect));
i_near = cell2mat(Is(nearTransect));

% --- Important: Create the point coordinates (x_near, y_near) ---
% The point clouds themselves need their coordinates. 
% You need to replicate the cell center coordinates for every point within that cell.
num_points_in_cell = cellfun(@numel, zs(nearTransect));

x_near = repelem(x_cell_centers, num_points_in_cell);
y_near = repelem(y_cell_centers, num_points_in_cell);

% --- End Optimized Pre-Loop Section ---
% The mapping to the transect position needs to be done once for each unique cell center.
% However, you use the gridded x/y position (ux(i), uy(j)) for the projection,
% so you need to identify the cross-shore position for each point's grid cell.

% Let's stick to the current structure, but make the inner logic faster:
% --- Optimized Time Loop Section: Replace inner loops with fast filtering ---
for t_idx = 1:nt
    current_time = time_vec(t_idx);
    
    % Use logical indexing on the pre-flattened data
    time_filter = t_near == current_time;
    
    if ~any(time_filter), continue; end
    
    % Select points for the current time step
    x_t = x_near(time_filter); 
    y_t = y_near(time_filter);
    z_t = z_near(time_filter);
    i_t = i_near(time_filter);
    
    % --- Map points to transect positions (Pdist2 is good, keep it) ---
    [~, NearIdx] = pdist2([xt(:), yt(:)], [x_t, y_t], 'euclidean', 'smallest', 1);
    [~, col] = ind2sub(size(xt), NearIdx);
    X = x1d(col);
    
    % --- Bin by cross-shore position and take mean (Use ACCUMARRAY!) ---
    % This is the most critical change in this section.
    [Xuniq, ~, Xidx] = unique(X);
    
    % Replace the inner loop (for n = 1:length(Xuniq)) with accumarray
    Zproj = accumarray(Xidx, z_t, [], @nanmean);
    Iproj = accumarray(Xidx, i_t, [], @nanmean);
    
    % ... (rest of the mapping back to the full x1d grid remains the same)

    % Map to full cross-shore grid
    ndx = ismember(x1d, Xuniq);
    z1d = nan(size(x1d));
    z1d(ndx) = Zproj;
    i1d = nan(size(x1d));
    i1d(ndx) = Iproj;
    
    % Store in matrix
    Z_xt(:, t_idx) = z1d;
    I_xt(:, t_idx) = i1d;
end

%% Step 6: Trim spatial extent
xcut = (x1d > 4 & x1d < 100);
Z_xt = Z_xt(xcut, :);
I_xt = I_xt(xcut, :);
x1d = x1d(xcut);

fprintf('Complete! Output size: %d cross-shore positions x %d time steps\n', ...
    length(x1d), nt);

end