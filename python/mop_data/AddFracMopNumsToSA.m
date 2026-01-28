function SA=AddFracMopNumsToSA(SA,Mop)

%% returns a fractional mop number and Mop cross-shore location X relative to the
% Mop back beach point line, at for SA.X SA.Y point as the new struct array
% fields: SA.MopFracNums and SA.MopXshore
   
%% put all SA.X/Y data into a grid

%% -- Make 2d elevation and slope grids of the survey and 

%% put x,y,z and mop number SA data into a 2d grid format

x=vertcat(SA.X);
y=vertcat(SA.Y);

xmin=min(x);xmax=max(x);
ymin=min(y);ymax=max(y);

% make 2d X,Y grid indice arrays for the entire area using meshgrid
[X,Y]=meshgrid(xmin:xmax,ymin:ymax);

% initialize the frac mopnum and mop xshore grids as NaNs 
FM=nan(size(X));
XM=nan(size(X));

% 1d grid indices with data
gdx=sub2ind(size(X),round(y)-ymin+1,round(x)-xmin+1); 

% reduce to unique ones
gdx=unique(gdx);

% get FracMopNums and mop xshore locations for the unique grid points
[FmopNum,Xmop]=utm2MopxshoreX(X(gdx),Y(gdx),Mop);

FM(gdx)=FmopNum; % overlay on grid
XM(gdx)=Xmop;

%% Now loop through individual SA surveys and assign fractional mop
%   numbers to their survey points from the grids
for n=1:size(SA,2)
   
    SA(n).FracMopNums=NaN*SA(n).X;
    SA(n).MopXshore=NaN*SA(n).X;
    x=vertcat(SA(n).X);
    y=vertcat(SA(n).Y);
    gdx=sub2ind(size(X),round(y)-ymin+1,round(x)-xmin+1);     
    SA(n).FracMopNums=FM(gdx);
    SA(n).MopXshore=XM(gdx);
end
    
end