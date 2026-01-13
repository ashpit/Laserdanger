function [Spec,Info,Bulk,Tseries] = get_runupStats_L2(Z_xt, I_xt, x1d, time_vec, varargin)
% Adapted from get_runupStatsLidar_L1 to work with Z_xt and I_xt matrices
%
% Input:
%   Z_xt:       Elevation matrix (cross-shore x time)
%   I_xt:       Intensity matrix (cross-shore x time) [optional]
%   x1d:        Cross-shore positions (m)
%   time_vec:   Time vector (datetime or seconds)
%   varargin:   Additional options
%
% Output:
%   Spec:       Frequency spectra and confidence intervals
%   Info:       Processing metadata
%   Bulk:       Bulk statistics (Sig, Sinc, eta, beta)
%   Tseries:    Time series of runup position and elevation

% Default options
options.threshold = 0.1;        % Water depth threshold (m)
options.windowlength = 5;       % Minutes for spectral windows
options.g = 9.81;
options.use_intensity = false;  % Whether to use intensity for detection
options.IGlength = 100;         % Infragravity filter length (seconds)
options.plot = false;
options = parseOptions(options, varargin);

%% Setup
xx = x1d(:)';  % Cross-shore positions (row vector)
t = time_vec(:);  % Time vector (column)
nt = length(t);

% Calculate sampling rate
if isdatetime(t)
    dt = mean(seconds(diff(t)));
else
    dt = mean(diff(t));
end

fprintf('Processing runup from Z_xt matrix...\n');
fprintf('Grid size: %d x %d (cross-shore x time)\n', length(xx), nt);
fprintf('Sampling rate: %.2f Hz\n', 1/dt);

%% Create reference beach surface (moving minimum for "dry beach")
IGlength = options.IGlength;  % seconds
IGfilt = round(IGlength/dt);

% Moving minimum to get dry beach surface on IG frequency
% pad Z with IG length on the end
% Z_end = Z_xt(:,end-IGfilt:end); Z_end = fliplr(Z_end);
% Z_concat = horzcat(Z_xt,Z_end);
M = movmin(Z_xt', IGfilt, 1, 'omitnan')';  % Transpose to work in time, then back
M = fillmissing(M,'linear',1,'MaxGap',6);
M = movmean(M, round(50/dt), 2);  % Smooth along time axis
% M = M(:,1:nt);
%% Initialize outputs
RunupImage = nan(1, nt);
Zrunup = nan(1, nt);
idxrunup = ones(1, nt);

% get offshore bound
addpath '/Users/ashton/Library/CloudStorage/OneDrive-UCSanDiego/CPG/terrace'
Zcount = sumnd(Z_xt,2);
offshore = find(Zcount < 0.25*nt*dt,1, 'first'); % at least 1/4th of the timeseries has data.
Xoff = x1d(offshore);


% Intersection parameters
L2 = [0 Xoff; options.threshold options.threshold];
dx = xx(2) - xx(1);
msize = 0.5/dx;  % Search window size (0.5m on each side)
%% Intensity filtering: 
% use the intensity gradient to capture indiv. waves

% if options.use_intensity
% 
% 
% 
% end


%% Main loop: Find runup line at each time step
fprintf('Detecting runup line...\n');

for ii = 1:nt
    
    % Get dry beach surface at this time
    sand = medfilt1(M(:, ii), 5);
    
        % Option: Use intensity to help identify water
    if options.use_intensity && ~isempty(I_xt)
        % Low intensity often indicates water (adjust threshold as needed)
        intensity_mask = I_xt(:, ii) > 30;  % Adjust this!
        % wlevtemp(~intensity_mask) = wlevtemp(~intensity_mask) * 0.5;
        Z_xt(intensity_mask,ii) = nan;
    end

    % Calculate water level (elevation - dry beach)
    if ii > 2 && ii < nt
        % Use median over 3 time steps for stability
        watline = median(Z_xt(:, ii-1:ii+1), 2, 'omitnan');
        wlevtemp = movmean(watline - sand,5,'omitnan');
    else
        wlevtemp = movmean(Z_xt(:, ii) - sand,5,'omitnan');
    end
    
    
    % Create line for intersection
    L1 = [xx; wlevtemp'];
    
    % Adaptive search window based on previous runup positions
    if ii > 4 && ~isnan(RunupImage(ii-1))
        prev_avg = max(4,nanmean(RunupImage(ii-4:ii-1)));
        L2 = [prev_avg-msize, prev_avg+msize; ...
              options.threshold, options.threshold];
    end
    
    % Find intersection (runup line)
    runupline = InterX(L1, L2);
    
    if ~isempty(runupline)
        RunupImage(ii) = runupline(1, 1);
    else
        % Expand search if no intersection found
        if ii > 5 && ~isnan(nanmean(RunupImage(ii-4:ii-1)))
            prev_avg = nanmean(RunupImage(ii-4:ii-1));
            L2 = [prev_avg-2*msize, prev_avg+2*msize; ...
                  options.threshold+0.01, options.threshold+0.01];
        else
            L2 = [0 Xoff; options.threshold options.threshold];
        end
        
        runupline = InterX(L1, L2);
        if ~isempty(runupline)
            RunupImage(ii) = runupline(1, 1);
        else
            RunupImage(ii) = NaN;
        end
    end
    
    % Progress indicator
    if mod(ii, 500) == 0
        fprintf('  Processed %d/%d time steps\n', ii, nt);
    end
end

fprintf('Runup detection complete!\n');

%% Filter and interpolate runup position
nfilt = round(1/dt);  % 1-second median filter

R = RunupImage;%medfilt1(RunupImage, nfilt, 'omitnan');
RR = inpaint_nans(R);  % Fill NaN gaps

% Find indices in x grid
idxR = nan(size(RR));
for i = 1:nt
    Rint = round(RR(i) * 10) / 10;
    if Rint < max(xx) && Rint > min(xx)
        [~, idxR(i)] = min(abs(xx - Rint));
    elseif Rint < min(xx)
        idxR(i) = 1;
    else
        idxR(i) = length(xx);
    end
end

% Extract elevation at runup position
for i = 1:nt
    Zrunup(i) = Z_xt(idxR(i), i);
end

% Clean up
Xrunup = RR;
Zrunup = medfilt1(Zrunup, nfilt, 'omitnan', 'truncate');

% Fill any remaining NaNs
ZZ = inpaint_nans(Zrunup);

%% Remove artifacts (jumps at riprap or other features)
% jump = find(diff(Zrunup) > 1);  % Large jumps > 1m
% riprap = find(Zrunup > 3.3);   % Adjust for your site!
% % 
% riprapjump = intersect(jump+1, riprap);
% % 
% if ~isempty(riprapjump)
%     fprintf('Removing %d riprap artifacts\n', length(riprapjump));
%     RunupImage(RunupImage > min(RunupImage(riprapjump))) = NaN;
% end

%% Compute spectra
fprintf('Computing spectra...\n');

nfft = options.windowlength * 60 / dt;  % Window length in samples
[f, S, Slo, Sup, ~, dof] = get_spectrum(detrend(ZZ), nfft, 1/dt, 0.05);

% Frequency bands
nIG = find(f > 0.004 & f <= 0.04);   % Infragravity
nINC = find(f > 0.04 & f <= 0.25);   % Incident (sea-swell)
df = f(2) - f(1);

% Mean water level
eta = nanmean(Zrunup(:));

%% Estimate Ocean Water Level
% OWL is the mean z offshore of the runup line that intersects 20cm. 
Xrmean = nanmean(Xrunup);
Zowl = movmin(Z_xt(x1d > Xrmean,:),10/dt,1,'omitnan');
OWL = mean(Zowl(:), 'omitnan');
if isnan(OWL)
    OWL = 0.774;
end

%% Estimate foreshore slope (beta)
fprintf('Estimating foreshore slope...\n');

stdEta = nanstd(Zrunup);
maxEta = eta + 2*stdEta;
minEta = eta - 2*stdEta;

% Use low-variability regions to define foreshore
Md = mean(M, 2);  % Time-averaged dry beach
foreshore_variance = var(M, 0, 2, 'includemissing');

% Select stable foreshore region
foreshore_idx = foreshore_variance < 0.2;  % Low temporal variance
foreshore = Md(foreshore_idx);
foreshorex = xx(foreshore_idx);

% Remove offshore and riprap regions
rmInd = foreshorex < 4 | foreshorex > Xoff;
foreshore(rmInd) = [];
foreshorex(rmInd) = [];

% Ensure continuous x positions
jumps = find(diff(foreshorex) > 1);
if ~isempty(jumps)
    foreshore(jumps) = [];
    foreshorex(jumps) = [];
end

% Select points in active runup range
botRange = find(foreshore >= OWL & foreshore <= maxEta);

% now just get below bermcrest.
linevec = linspace(foreshore(botRange(1)), foreshore(botRange(end)), numel(botRange))';
residual = foreshore(botRange) - linevec;
[bermcrest, bermcrestx] = max(residual);
if bermcrest > 0.1 && (botRange(end) - bermcrestx) > 0 % greater than 10cm and bermcrest is onshore of the lower bound
    botRange = botRange(bermcrestx:end);
end

% Another check for continuity
% jumps = find(diff(foreshorex(botRange)) > 1);
% if ~isempty(jumps)
%     botRange = botRange(jumps(end)+1:end);
% end

% If there is a berm crest, take the slope offshore of the crest. 

% Fit linear slope
if length(botRange) > 5
    fitvars = polyfit(foreshorex(botRange), foreshore(botRange), 1);
    beta = fitvars(1);
else
    beta = NaN;
    fprintf('Warning: Could not estimate foreshore slope\n');
end

%% Calculate bulk parameters
[Sig, Sig_lo, Sig_up, ~] = getSWHebounds(S(nIG), dof, 0.95, df);
[Sinc, Sinc_lo, Sinc_up, ~] = getSWHebounds(S(nINC), dof, 0.95, df);

swashparams = [Sig Sinc eta];
swashparamsLO = [Sig_lo Sinc_lo eta];
swashparamsUP = [Sig_up Sinc_up eta];

%% Package outputs
Spec.f = f;
Spec.S = S;
Spec.Slo = Slo;
Spec.Sup = Sup;
Spec.dof = dof;

Info.Hz = 1/dt;
Info.threshold = options.threshold;
Info.datahour = t(1);
Info.duration = (t(end) - t(1));
if isdatetime(Info.duration)
    Info.duration = minutes(Info.duration);
end

Bulk.swashparams = swashparams;
Bulk.swashparamsLO = swashparamsLO;
Bulk.swashparamsUP = swashparamsUP;
Bulk.swashParamsNames = {'Sig','Sinc','eta'};
Bulk.beta = beta;
Bulk.foreshore = foreshore;
Bulk.foreshoreX = foreshorex;

if isdatetime(t)
    Tseries.T = datenum(t);
else
    Tseries.T = t;
end
Tseries.Zrunup = Zrunup;
Tseries.Xrunup = Xrunup;
Tseries.idxrunup = idxR;

%% Print summary
fprintf('\n=== Runup Statistics Summary ===\n');
fprintf('Sig (IG):       %.3f m (%.3f - %.3f)\n', Sig, Sig_lo, Sig_up);
fprintf('Sinc (SS):      %.3f m (%.3f - %.3f)\n', Sinc, Sinc_lo, Sinc_up);
fprintf('Mean eta:       %.3f m\n', eta);
fprintf('Beach slope:    %.4f (1:%.1f)\n', beta, abs(1/beta));
fprintf('Peak IG freq:   %.4f Hz (T = %.1f s)\n', f(nIG(S(nIG)==max(S(nIG)))), 1/f(nIG(S(nIG)==max(S(nIG)))));
fprintf('Peak SS freq:   %.4f Hz (T = %.1f s)\n', f(nINC(S(nINC)==max(S(nINC)))), 1/f(nINC(S(nINC)==max(S(nINC)))));

%% Visualization

if options.plot == true
figure('Position', [100, 100, 1400, 900]);

% Runup time series
subplot(3,2,1)
plot(t, Xrunup, 'b-', 'LineWidth', 1);
xlabel('Time'); ylabel('Runup Position (m)');
title('Runup Position Time Series');
grid on; ylim([4 Xoff])

subplot(3,2,2)
plot(t, Zrunup, 'r-', 'LineWidth', 1);
xlabel('Time'); ylabel('Runup Elevation (m)');
title('Runup Elevation Time Series');
grid on;
yline(eta, 'k--', 'Mean'); ylim([1 3])

% Spectrum
subplot(3,2,3)
loglog(f, S, 'b-', 'LineWidth', 2); hold on;
loglog(f, Slo, 'k--', 'LineWidth', 0.5);
loglog(f, Sup, 'k--', 'LineWidth', 0.5);
xlabel('Frequency (Hz)'); ylabel('S(f) (m^2/Hz)');
title('Runup Spectrum');
grid on;
xlim([0.004, 0.5]);
xline(0.04, 'r--', 'IG/SS boundary');

% Hovmöller with runup overlay
subplot(3,2,4)
pcolor(t, xx, Z_xt);
shading interp;
hold on;
plot(t, Xrunup, 'k-', 'LineWidth', 2);
xlabel('Time'); ylabel('Cross-shore Position (m)');
title('Hovmöller with Detected Runup');
cb=colorbar(); colormap(gca, 'jet');clim([1 3]); ylim([4 Xoff])
cb.Label.String = 'Elevation (m)';
text(-50, 0, 'SeaWall', 'HorizontalAlignment', 'right', 'FontSize', 12);
xlim([0 600])
% Foreshore slope
subplot(3,2,5)
plot(xx, Md, 'k.', 'MarkerSize', 4); hold on;
plot(foreshorex(botRange), foreshore(botRange), 'b.', 'MarkerSize', 10);
if ~isnan(beta)
    x_fit = [min(foreshorex(botRange)), max(foreshorex(botRange))];
    y_fit = polyval(fitvars, x_fit);
    plot(x_fit, y_fit, 'r-', 'LineWidth', 2);
end
ylim([1 3.5])
xlabel('Cross-shore (m)'); ylabel('Elevation (m)');
title(sprintf('Foreshore (slope = %.4f)', beta));
grid on;
legend('All points', 'Fit region', 'Linear fit', 'Location', 'best');

% Energy by frequency band
subplot(3,2,6)
pcolor(t,xx,I_xt); shading flat;
cb=colorbar(); colormap(gca, 'gray'); clim([0 30]);
hold on;
plot(t,Xrunup, 'c-', 'LineWidth',1.5);
title('Hovmoller of Intensity with Detected Runup')
ylabel('Cross-shore position (m)')
xlabel('Time (sec)')
cb.Label.String = 'Intensity';
xlim([0 200]); ylim([4 Xoff])

% E_IG = sum(S(nIG)) * df;
% E_SS = sum(S(nINC)) * df;
% bar([1, 2], [E_IG, E_SS]);
% set(gca, 'XTickLabel', {'IG (0.004-0.04 Hz)', 'SS (0.04-0.25 Hz)'});
% ylabel('Energy (m^2)');
% title('Energy by Frequency Band');
% grid on;
end
%% HELPER FUNCTIONS
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

    % function P = InterX(L1, varargin)
    %     % Intersection of curves (from original code)
    %     narginchk(1,2);
    %     if nargin == 1
    %         L2 = L1; hF = @lt;
    %     else
    %         L2 = varargin{1}; hF = @le;
    %     end
    % 
    %     x1 = L1(1,:)'; x2 = L2(1,:);
    %     y1 = L1(2,:)'; y2 = L2(2,:);
    %     dx1 = diff(x1); dy1 = diff(y1);
    %     dx2 = diff(x2); dy2 = diff(y2);
    % 
    %     S1 = dx1.*y1(1:end-1) - dy1.*x1(1:end-1);
    %     S2 = dx2.*y2(1:end-1) - dy2.*x2(1:end-1);
    % 
    %     C1 = feval(hF,D(bsxfun(@times,dx1,y2)-bsxfun(@times,dy1,x2),S1),0);
    %     C2 = feval(hF,D((bsxfun(@times,y1,dx2)-bsxfun(@times,x1,dy2))',S2'),0)';
    % 
    %     [i,j] = find(C1 & C2);
    %     if isempty(i), P = zeros(2,0); return; end
    % 
    %     i=i'; dx2=dx2'; dy2=dy2'; S2 = S2';
    %     L = dy2(j).*dx1(i) - dy1(i).*dx2(j);
    %     i = i(L~=0); j=j(L~=0); L=L(L~=0);
    % 
    %     P = unique([dx2(j).*S1(i) - dx1(i).*S2(j), ...
    %         dy2(j).*S1(i) - dy1(i).*S2(j)]./[L L],'rows')';
    % 
    %     function u = D(x,y)
    %         u = bsxfun(@minus,x(:,1:end-1),y).*bsxfun(@minus,x(:,2:end),y);
    %     end
    % end

   function P = InterX(L1, varargin)
        % Intersection of curves with spike filtering
        % Returns only intersections with smooth curves, not narrow peaks
        
        narginchk(1,2);
        
        % Parse inputs
        if nargin == 1
            L2 = L1; hF = @lt;
        else
            L2 = varargin{1}; hF = @le;
        end
        
        % Original intersection algorithm
        x1 = L1(1,:)'; x2 = L2(1,:);
        y1 = L1(2,:)'; y2 = L2(2,:);
        dx1 = diff(x1); dy1 = diff(y1);
        dx2 = diff(x2); dy2 = diff(y2);
        
        S1 = dx1.*y1(1:end-1) - dy1.*x1(1:end-1);
        S2 = dx2.*y2(1:end-1) - dy2.*x2(1:end-1);
        
        C1 = feval(hF,D(bsxfun(@times,dx1,y2)-bsxfun(@times,dy1,x2),S1),0);
        C2 = feval(hF,D((bsxfun(@times,y1,dx2)-bsxfun(@times,x1,dy2))',S2'),0)';
        
        [i,j] = find(C1 & C2);
        if isempty(i), P = zeros(2,0); return; end
        
        i=i'; dx2=dx2'; dy2=dy2'; S2 = S2';
        L = dy2(j).*dx1(i) - dy1(i).*dx2(j);
        i = i(L~=0); j=j(L~=0); L=L(L~=0);
        
        P = unique([dx2(j).*S1(i) - dx1(i).*S2(j), ...
            dy2(j).*S1(i) - dy1(i).*S2(j)]./[L L],'rows')';
        
        % Filter out spikes (narrow peaks)
        if ~isempty(P)
            P = filterSpikes(P, L1, L2);
        end
        % 
        function u = D(x,y)
            u = bsxfun(@minus,x(:,1:end-1),y).*bsxfun(@minus,x(:,2:end),y);
        end
        
        function P_filtered = filterSpikes(P, L1, L2)
            % Filter intersections that occur at narrow spikes
            % windowsize: number of points to check on either side
            
            windowsize = 15;  % Check +/- 10 points around intersection
            
            if isempty(P)
                P_filtered = P;
                return;
            end
            
            x1 = L1(1,:);
            y1 = L1(2,:);
            x2 = L2(1,:);
            y2 = L2(2,:);
            
            % Store valid intersections
            valid = true(1, size(P, 2));
            
            for k = 1:size(P, 2)
                px = P(1, k);  % Intersection x-coordinate
                py = P(2, k);  % Intersection y-coordinate
                
                % Find nearest point in L1 to this intersection
                [~, idx1] = min(abs(x1 - px));
                
                % Define window around intersection in L1
                idx_start = max(1, idx1 - windowsize);
                idx_end = min(length(x1), idx1 + windowsize);
                
                % Extract local segment of L1
                x1_local = x1(idx_start:idx_end);
                y1_local = y1(idx_start:idx_end);
                
                % Interpolate L2 to same x-coordinates as local L1
                if max(x2) >= min(x1_local) && min(x2) <= max(x1_local)
                    % Only interpolate where there's overlap
                    x_common = x1_local(x1_local >= min(x2) & x1_local <= max(x2));
                    
                    if length(x_common) > 2
                        y2_interp = interp1(x2, y2, x_common, 'linear', 'extrap');
                        y1_common = interp1(x1_local, y1_local, x_common);
                        
                        % Calculate difference between curves
                        diff_curve = y1_common - y2_interp;
                        
                        % Check for sign changes (additional intersections)
                        sign_changes = diff(sign(diff_curve));
                        num_crossings = sum(abs(sign_changes) > 0);
                        
                        % If there are multiple crossings in the window, it's a spike
                        if num_crossings > 1
                            valid(k) = false;
                        end
                    else
                        % Not enough overlap to check
                        valid(k) = false;
                    end
                else
                    % No overlap between curves in this window
                    valid(k) = false;
                end
            end
            
            % Return only valid intersections
            P_filtered = P(:, valid);
        end
    end

    function [fm, Spp, Spplo, Sppup, nens, dof] = get_spectrum(P, nfft, fs, alpha)
        % Spectral analysis (from original code)
        [m,n] = size(P);
        if m<n, P = P'; end
        
        [Amp,nens] = calculate_fft2(P(:,1), nfft, fs);
        
        df = fs/(nfft-1);
        nnyq = nfft/2 + 1;
        
        fm = (0:nnyq-1)*df;
        Spp = mean(Amp .* conj(Amp)) / (nnyq * df);
        Spp = Spp(1:nnyq);
        Spp = Spp(:);
        
        dof = 8/3*nens;
        chi2 = [chi2inv(alpha/2,dof) chi2inv(1-alpha/2,dof)];
        CI_chi2 = [(1/chi2(1)).*(dof*Spp) (1/chi2(2)).*(dof*Spp)];
        Spplo = CI_chi2(:,1);
        Sppup = CI_chi2(:,2);
    end

 function [ Hs, lb, ub, edof ] = getSWHebounds( e, dof, q, df )
        %[ lb, ub ] = getSWHebounds( e, dof )
        %   e - Energy(time,freq), units: m^2
        %   dof - degree of freedom per freq-band
        %   q - percentile to determine ub,lb (e.g. 0.90 or 0.68)
        %   lb - lower Hs bound
        %   ub - upper Hs
        %   Hs - Hsig
        %   edof - Effective dof used
        % SEE ELGAR 1987, for details of un-biased EDOF estimate
        
        % percent on either side
        a = (1-q)/2;
        
        % Get Hsig
        Hs = 4*sqrt(sum(e.*df));
        
        % Determine edof
        edof = dof*sum(e).^2./sum(e.^2);    %estimate effective DOF
        edof = edof/(1+2/dof);                    %UNBIAS THE ESTIMATE
        
        % Generate normalized ub,lb
        
        lb = edof/chi2inv(1-a,edof);
        ub = edof/chi2inv(a,edof);
        
        % Multiply by Hsig
        lb = sqrt(lb).*Hs;
        ub = sqrt(ub).*Hs;
        
        % % Multiply by Hsig
        % lb = lb.*Hs;
        % ub = ub.*Hs;
        
    end

    function [A,nens] = calculate_fft2(X,nfft,fs)
            % needs commenting
            % X:
            % nfft: chunk size in fourier space
            % fs: somehow not used?
            
            [n,~]=size(X);
            
            num = floor(2*n/nfft)-1;
            nens = num;
            %[n nfft num]
            
            X = X-ones(n,1)*mean(X); %demean
            
            X(isnan(X)) = 0;
            X = detrend(X);
            X = X-ones(n,1)*mean(X); %demean
            
            sumXt = (X'*X)/n; %get fourier coeffs
            
            %WIN = hanningwindow(@hamming,nfft);
            jj = [0:nfft-1]';
            WIN = 0.5 * ( 1 - cos(2*pi*jj/(nfft-1)));
            
            
            A = zeros(num,nfft);
            
            % set it up so that SQR(|A(i,:)|^2) = sum(X^2) (Parseval's Thm)
            
            varXwindtot = 0;
            
            for i=1:num
                istart = (i-1)*(nfft/2)+1;
                istop = istart+nfft-1;
                Xwind = X(istart:istop);
                Xwind = Xwind - mean(Xwind);  % demean.   detrend?
                Xwind = detrend(Xwind); %detrend
                lenX = length(Xwind); %why is this not evident from nfft
                varXwind =( Xwind'*Xwind)/lenX; %get variance, normalized
                varXwindtot = varXwindtot + varXwind; %add to total variance
                Xwind = Xwind .* WIN; %window it
                tmp = ( Xwind'*Xwind)/lenX; % get windowed covariance
                if (tmp == 0)
                    Xwind = Xwind * 0.0;
                else
                    Xwind = Xwind*sqrt(varXwind/tmp); %parseval's thm
                end
                A(i,:) = fft(Xwind')/sqrt(nfft);
                meanA2 = mean( A(i,:) .* conj(A(i,:)));
            end
        end



end