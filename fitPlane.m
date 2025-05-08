function [zplane, zfit] = fitPlane(points)

X = points(:,1); Y = points(:,2); Z = points(:,3);
meanPoint = mean(points, 1);

% Step 2: Subtract mean (center the data)
centeredPoints = points - meanPoint;

% Step 3: Singular Value Decomposition
[~, ~, V] = svd(centeredPoints, 0);

% Step 4: The normal vector is the last column of V
normal = V(:, 3);

% Step 5: Compute the plane equation
% Equation: a*(X - x0) + b*(Y - y0) + c*(Z - z0) = 0 → expand to aX + bY + cZ + d = 0
d = -dot(normal, meanPoint);

% Output
planeNormal = normal(:)';
planePoint = meanPoint;
planeEq = [planeNormal, d];
%
% % Define grid over your x and y range
[xGrid, yGrid] = meshgrid(linspace(min(X), max(X), 1000), linspace(min(Y), max(Y), 1000));
% [in, on] = inpolygon(xGrid, yGrid, bounds(:,1), bounds(:,2));
%
% Compute z using the plane equation: z = (-d - a*x - b*y)/c
a = planeEq(1);
b = planeEq(2);
c = planeEq(3);
d = planeEq(4);

zfit = (-d - a * X - b * Y) / c;

zGrid = (-d - a * xGrid - b * yGrid) / c;
% Plot points
% figure(1); set(gcf, 'position', [100 100 1000 1000])
% clf
% scatter3(X, Y, Z, 10, 'k.'); hold on;
% xlim([475950 476200]); ylim([3636250 3636500]);
% view(2);

% zGrid(~in & ~on) = NaN;
zplane = zGrid;
% % Plot plane surface
% surf(xGrid, yGrid, zplane, 'FaceAlpha', 0.5, 'EdgeColor', 'none', 'FaceColor', 'cyan');
% zlim([-5 10]);
% xlabel('X'); ylabel('Y'); zlabel('Z'); grid on; axis equal;
% title('Best-fit plane to lidar points');

% kernel = fspecial('gaussian', [15 15], 5);  % 15x15 Gaussian kernel, sigma=5
% zGridSmooth = conv2(zGrid, kernel, 'same');
% 
% % Keep NaNs outside polygon
% zGridSmooth(~in & ~on) = NaN;
% 
% surf(xGrid, yGrid, zGridSmooth, 'FaceAlpha', 0.5, 'EdgeColor', 'none', 'FaceColor', 'cyan');

end