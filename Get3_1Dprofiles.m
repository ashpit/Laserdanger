function [x1d,Z3D]=Get3_1Dprofiles(Xutm,Yutm,Z)
%
% Z3D is 3xX. index position 
% transects are at positions:
%   1 for south
%   2 for middle
%   3 for north
% figure(1);clf
% subplot(1,2,1)
% plot3(Xutm, Yutm, Z, 'k.'); hold on
% 
% if CSF == 1
%     % Set parameters
%     gridResolution = 1;      % Grid size for partitioning the data (10 cm)
%     clothResolution = 1;     % Cloth resolution (adjust as needed)
%     maxIterations = 25;      % Number of iterations for cloth simulation
% 
%     % Call the CSF function (Cloth Simulation Filtering)
%     groundMask = csf_filtering(Xutm, Yutm, Z, gridResolution, clothResolution, maxIterations);
%     % remove the bad data
%     Z(~groundMask) = [];
%     Xutm(~groundMask) = []; Yutm(~groundMask)=[];
% end

% subplot(1,2,2)
% plot3(Xutm, Yutm, Z, 'b*');

points = [Xutm, Yutm];

% Data centered shore normal transect
x1 = 476190.0618275550;   y1 = 3636333.442333425; 
x2 = 475620.6132784432;   y2 = 3636465.645345889;

alongshore_spacings = [ -8, -6, -4, -2, 0, 2, 4, 6, 8, 10]; % Customize as needed

% Compute unit vector in the alongshore direction (perpendicular to x1-x2)
ang = atan2(y2 - y1, x2 - x1); % Angle of original transect
alongshore_angle = ang + pi/2; % Rotate by 90 degrees for alongshore direction
dx_alongshore = cos(alongshore_angle);
dy_alongshore = sin(alongshore_angle);

dist=pdist([x1,y1;x2,y2]); % distance between back and offshore point
% adjust offshore extension so all transects have same xshore length
ExtendLine = [0 -300];
res = 0.25;


% Initialize arrays to store multiple transect coordinates
num_transects = length(alongshore_spacings);
x_start = zeros(num_transects, 1);
y_start = zeros(num_transects, 1);
x_end = zeros(num_transects, 1);
y_end = zeros(num_transects, 1);

% Generate additional transects
for i = 1:num_transects
    offset = alongshore_spacings(i);
    xs = x1 + offset * dx_alongshore;
    ys = y1 + offset * dy_alongshore;
    xe = x2 + offset * dx_alongshore;
    ye = y2 + offset * dy_alongshore;

    xline = [xs,xe]; yline = [ys, ye];
    dist=pdist([xs,ys;xe,ye]);
    x1d=ExtendLine(1):res:dist+ExtendLine(2);
    ExtendOff=ExtendLine(2)+(dist);
    xt=x1+ExtendLine(1)*cos(ang):res*cos(ang):x2+ExtendOff*cos(ang);
    yt=y1+ExtendLine(1)*sin(ang):res*sin(ang):y2+ExtendOff*sin(ang);
    lineVec = [xe - xs, ye - ys];
    lineVec = lineVec / norm(lineVec);
    lineToPointVec = points - [xline(1), yline(1)]; 
    projLength = dot(lineToPointVec, repmat(lineVec, size(points, 1), 1), 2);
    projPoints = [xline(1), yline(1)] + projLength .* lineVec;
    distToLine = sqrt(sum((points - projPoints).^2, 2));
    tol = 1; 
    closePointsIdx = distToLine <= tol;
    x_proj = projPoints(closePointsIdx, 1); % Projected x-coordinates
    y_proj = projPoints(closePointsIdx, 2); % Projected y-coordinates
    z_values = Z(closePointsIdx);
    [dp,NearIdx]= pdist2([xt(:),yt(:)],[double(x_proj),double(y_proj)],'euclidean','smallest',1);
    [row,col] = ind2sub(size(xt),NearIdx); 
    X=x1d(col);

    if isempty(row)
        Z1D = x1d*NaN;
        Z3D(i,:) = Z1D;
        continue
    end

    A = [X'.^2, X', ones(size(X'))];
    y = A\z_values; %y(1) is the slope, y(2) is the intercept
    %
    z_fit = X'.^2*y(1) + y(2)*X' + y(3);
    
    % get the residuals
    residuals = z_values - z_fit;
    % find stdv between the fit and points above the line. 
    stdR = std(residuals);
    r_values = nan(size(z_values));
    r_values(abs(residuals) >= 0.4) = z_values(abs(residuals) >= 0.4);
    z_values(abs(residuals) >= 0.4)=nan;

    Xuniq=unique(X); Zproj = NaN*Xuniq;
    n=0;
    for x=Xuniq
        n=n+1;
        %find all z_values for
        Zproj(n) = nanmean(z_values(X==x));
        % Zproj(n) = min(z_values(X==x),[],1);
    end
    ndx=find(ismember(x1d,Xuniq));
    z1d=x1d*NaN;z1d(ndx)=Zproj;

    % interpolate to fill gaps
    dx = 4;
    sz=gapsize(z1d); 
    x1dv=x1d;x1dv(isnan(z1d) & sz <= dx)=[];z1dv=z1d;z1dv(isnan(z1d) & sz <= dx)=[]; 
    % interpolate to fill the small gaps
    z1di=interp1(x1dv,z1dv,x1d);
    Z1D=interp1(x1dv,z1dv,x1d);

    Z3D(i,:) = Z1D;

end
end






%%




% 
% % Define the original transect line (MOP line)
% xline = [x1, x2]; xline1 = [x11, x21];  xline2 = [x12, x22]; % x coordinates of the transect line
% yline = [y1, y2]; yline1 = [y11, y21];  yline2 = [y12, y22]; % y coordinates of the transect line
% %optional additional lines
% ExtendLine = [0 -300];
% dist=pdist([x1,y1;x2,y2]);dist1=pdist([x11,y11;x21,y21]);dist2=pdist([x12,y12;x22,y22]);
% res = 0.25; %resolution
% % xshore distance m from back beach point
% x1d=ExtendLine(1):res:dist+ExtendLine(2); %x1d1=ExtendLine(1):res:dist1+ExtendLine(2); x1d2=ExtendLine(1):res:dist2+ExtendLine(2);
% %
% % Mop transect 1m spaced xshore points
% ang=atan2(y2-y1,x2-x1); % radian angle between back and offshore point
% dist=pdist([x1,y1;x2,y2]); % distance between back and offshore point
% % adjust offshore extension so all transects have same xshore length
% ExtendOff=ExtendLine(2)+(dist);
% 
% % x coords of 1m spaced points along transect
% xt=x1+ExtendLine(1)*cos(ang):res*cos(ang):x2+ExtendOff*cos(ang); xt1=x11+ExtendLine(1)*cos(ang):res*cos(ang):x21+ExtendOff*cos(ang); xt2=x12+ExtendLine(1)*cos(ang):res*cos(ang):x22+ExtendOff*cos(ang);
% % y coords of 1m spaced points along transect
% yt=y1+ExtendLine(1)*sin(ang):res*sin(ang):y2+ExtendOff*sin(ang); yt1=y11+ExtendLine(1)*sin(ang):res*sin(ang):y21+ExtendOff*sin(ang); yt2=y12+ExtendLine(1)*sin(ang):res*sin(ang):y22+ExtendOff*sin(ang); 
% 
% % Create a vector representing the direction of the transect line
% lineVec = [x2 - x1, y2 - y1]; lineVec1 = [x21 - x11, y21 - y11]; lineVec2 = [x22 - x12, y22 - y12];% Direction vector of the transect
% lineVec = lineVec / norm(lineVec); lineVec1 = lineVec1 / norm(lineVec1); lineVec2 = lineVec2 / norm(lineVec2); % Normalize the vector
% 
% % Define the input points (X, Y, Z)
% points = [Xutm, Yutm]; % Nx2 matrix of input points (X, Y)
% 
% % Vector from the first point on the line to the points
% lineToPointVec = points - [xline(1), yline(1)]; lineToPointVec1 = points - [xline1(1), yline1(1)]; lineToPointVec2 = points - [xline2(1), yline2(1)]; % Nx2 matrix
% 
% % Projection of each point onto the line
% projLength = dot(lineToPointVec, repmat(lineVec, size(points, 1), 1), 2); % Nx1 vector of projection lengths
% projLength1 = dot(lineToPointVec1, repmat(lineVec1, size(points, 1), 1), 2); % Nx1 vector of projection lengths
% projLength2 = dot(lineToPointVec2, repmat(lineVec2, size(points, 1), 1), 2); % Nx1 vector of projection lengths
% 
% % Get the projected points on the line
% projPoints = [xline(1), yline(1)] + projLength .* lineVec; projPoints1 = [xline1(1), yline1(1)] + projLength1 .* lineVec1; projPoints2 = [xline2(1), yline2(1)] + projLength2 .* lineVec2; % Nx2 matrix of projected points
% % Compute the distance from each point to its projection on the line
% distToLine = sqrt(sum((points - projPoints).^2, 2)); distToLine1 = sqrt(sum((points - projPoints1).^2, 2)); distToLine2 = sqrt(sum((points - projPoints2).^2, 2)); % Nx1 vector of distances
% 
% % Define tolerance for distance to the line (e.g., 2 meters)
% tol = 1; 
% 
% % Find points that are within tolerance of the line
% closePointsIdx = distToLine <= tol; closePointsIdx1 = distToLine1 <= tol; closePointsIdx2 = distToLine2 <= tol;
% 
% % Get the corresponding projected x and z values
% x_proj = projPoints(closePointsIdx, 1); x_proj1 = projPoints1(closePointsIdx1, 1); x_proj2 = projPoints2(closePointsIdx2, 1);% Projected x-coordinates
% y_proj = projPoints(closePointsIdx, 2); y_proj1 = projPoints1(closePointsIdx1, 2); y_proj2 = projPoints2(closePointsIdx2, 2); % Projected y-coordinates
% z_values = Z(closePointsIdx); z_values1 = Z(closePointsIdx1); z_values2 = Z(closePointsIdx2); % Corresponding Z values
% 
% % find nearest subtransect line point to the input location
% [dp,NearIdx]= pdist2([xt(:),yt(:)],[double(x_proj),double(y_proj)],'euclidean','smallest',1);
% [dp1,NearIdx1]= pdist2([xt1(:),yt1(:)],[double(x_proj1),double(y_proj1)],'euclidean','smallest',1);
% [dp2,NearIdx2]= pdist2([xt2(:),yt2(:)],[double(x_proj2),double(y_proj2)],'euclidean','smallest',1);
% 
% 
% % define X based on the nearest transect line point
% %   row=nearest subtransect number; col = xshore distance indice on
% %   the nearest subtransect
% [row,col] = ind2sub(size(xt),NearIdx); [row1,col1] = ind2sub(size(xt1),NearIdx1); [row2,col2] = ind2sub(size(xt2),NearIdx2); 
% 
% % xshore distance (5cm xshore resolution) along the Mop transect for each survey point
% X=x1d(col); X1=x1d(col1); X2=x1d(col2);
% 
% %
% % figure(2);clf
% % plot3(Xutm,Yutm,Z, 'k.'); hold on
% % plot3(x_proj1, y_proj1, z_values1,'r*');
% % plot3(x_proj, y_proj, z_values,'b*');
% % plot3(x_proj2, y_proj2, z_values2,'g*');
% % hold on;
% % plot(xt, yt, 'b-');
% % plot([x11 x21], [y11 y21], 'r-', 'LineWidth',2);
% % plot([x1 x2], [y1 y2], 'b-', 'LineWidth',2);
% % plot([x12 x22], [y12 y22], 'g-', 'LineWidth',2);
% % view(2)
% % ylim([3636325 3636345]); xlim([476155 476195]);
% %
% 
% 
% A = [X'.^2, X', ones(size(X'))];
% y = A\z_values; %y(1) is the slope, y(2) is the intercept
% %
% z_fit = X'.^2*y(1) + y(2)*X' + y(3);
% 
% % get the residuals
% residuals = z_values - z_fit;
% % find stdv between the fit and points above the line. 
% stdR = std(residuals);
% r_values = nan(size(z_values));
% r_values(abs(residuals) >= 0.4) = z_values(abs(residuals) >= 0.4);
% z_values(abs(residuals) >= 0.4)=nan;
% 
% 
% A1 = [X1'.^2, X1', ones(size(X1'))];
% y1 = A1\z_values1; %y(1) is the slope, y(2) is the intercept
% %
% z_fit1 = X1'.^2*y1(1) + y1(2)*X1' + y1(3);
% 
% % get the residuals
% residuals1 = z_values1 - z_fit1;
% % find stdv between the fit and points above the line. 
% stdR1 = std(residuals1);
% r_values1 = nan(size(z_values1));
% r_values1(abs(residuals1) >= 0.4) = z_values1(abs(residuals1) >= 0.4);
% z_values1(abs(residuals1) >= 0.4)=nan;
% 
% 
% A2 = [X2'.^2, X2', ones(size(X2'))];
% y2 = A2\z_values2; %y(1) is the slope, y(2) is the intercept
% %
% z_fit2 = X2'.^2*y2(1) + y2(2)*X2' + y2(3);
% 
% % get the residuals
% residuals2 = z_values2 - z_fit2;
% % find stdv between the fit and points above the line. 
% stdR2 = std(residuals2);
% r_values2 = nan(size(z_values2));
% r_values2(abs(residuals2) >= 0.4) = z_values2(abs(residuals2) >= 0.4);
% z_values2(abs(residuals2) >= 0.4)=nan;
% 
% 
% Xuniq=unique(X); Zproj = NaN*Xuniq;
% n=0;
% for x=Xuniq
%     n=n+1;
%     %find all z_values for
%     Zproj(n) = nanmean(z_values(X==x));
%     % Zproj(n) = min(z_values(X==x),[],1);
% end
% ndx=find(ismember(x1d,Xuniq));
% z1d=x1d*NaN;z1d(ndx)=Zproj;
% 
% Xuniq1=unique(X1); Zproj1 = NaN*Xuniq1;
% n=0;
% for x=Xuniq1
%     n=n+1;
%     %find all z_values for
%     Zproj1(n) = nanmean(z_values1(X1==x));
%     % Zproj(n) = min(z_values(X==x),[],1);
% end
% ndx1=find(ismember(x1d,Xuniq1));
% z1d1=x1d*NaN;z1d1(ndx1)=Zproj1;
% 
% Xuniq2=unique(X2); Zproj2 = NaN*Xuniq2;
% n=0;
% for x=Xuniq2
%     n=n+1;
%     %find all z_values for
%     Zproj2(n) = nanmean(z_values2(X2==x));
%     % Zproj(n) = min(z_values(X==x),[],1);
% end
% ndx2=find(ismember(x1d,Xuniq2));
% z1d2=x1d*NaN;z1d2(ndx2)=Zproj2;
% 
% % interpolate to fill gaps
% dx = 4;
% sz=gapsize(z1d); sz1=gapsize(z1d1); sz2=gapsize(z1d2); 
% 
% x1dv=x1d;x1dv(isnan(z1d) & sz <= dx)=[];z1dv=z1d;z1dv(isnan(z1d) & sz <= dx)=[]; 
% x1dv1=x1d;x1dv1(isnan(z1d1) & sz1 <= dx)=[];z1dv1=z1d1;z1dv1(isnan(z1d1) & sz1 <= dx)=[];
% x1dv2=x1d;x1dv2(isnan(z1d2) & sz2 <= dx)=[];z1dv2=z1d2;z1dv2(isnan(z1d2) & sz2 <= dx)=[];
% 
% % interpolate to fill the small gaps
% z1di=interp1(x1dv,z1dv,x1d);Z1D=interp1(x1dv,z1dv,x1d);
% z1di1=interp1(x1dv1,z1dv1,x1d);Z1D1=interp1(x1dv1,z1dv1,x1d);
% z1di2=interp1(x1dv2,z1dv2,x1d);Z1D2=interp1(x1dv2,z1dv2,x1d);
% Z3D = [Z1D1; Z1D; Z1D2]; % south to north 
% 
% end