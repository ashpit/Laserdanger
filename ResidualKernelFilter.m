function [groundPoints, Z_interp] = ResidualKernelFilter(points, l, thresh)

if nargin <3
    l=2;
    thresh=0.5;
end
h = 3*l;

overlap = 0.1;
step_x = l * (1 - overlap);
step_y = h * (1 - overlap);

X = points(:,1);
Y = points(:,2);
Z = points(:,3);

x_min = min(X);
x_max = max(X);
y_min = min(Y);
y_max = max(Y);

% Generate grid of points with overlapping spacing
xv = x_min : step_x : x_max + l;  % add extra to ensure coverage
yv = y_min : step_y : y_max + h;

% Create grid points
[Xgrid, Ygrid] = meshgrid(xv, yv);


% Triangulate the grid
DT = delaunayTriangulation(Xgrid(:), Ygrid(:));
% figure(1);clf
% triplot(DT);
% hold on;
% scatter(X, Y, 'r.');
% xlabel('Xutm'); ylabel('Yutm');
% title('DelauneyTesselation and Kernel Noise Removal');
% set(gcf, 'color', 'w')

keepIdx = false(size(X));
Z_interp = NaN(size(X));

% Loop over each triangle
for i = 1:size(DT.ConnectivityList,1)
    tri_idx = DT.ConnectivityList(i,:);
    triX = DT.Points(tri_idx,1);
    triY = DT.Points(tri_idx,2);

    % Check which points fall inside triangle
    [inTri, ~] = inpolygon(X, Y, triX, triY);

    section = points(inTri,:);

    if isempty(section) || size(section,1) < 10
        continue
    end

    % Fit plane
    [~, Z_fit] = fitPlane(section);

    % figure(1);hold on
    % scatter3(X(inTri), Y(inTri), Z(inTri), 'k.');hold on
    % scatter3(X(inTri), Y(inTri), Z_fit, 10,'ro');

    residuals = section(:,3) - Z_fit;
    keepInTriangle = abs(residuals) < thresh;
    tempIdx = find(inTri);
    keepIdx(tempIdx(keepInTriangle)) = true;
    Z_interp(inTri) = Z_fit;
end

groundPoints = keepIdx;
end
