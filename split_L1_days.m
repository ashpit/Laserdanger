config = jsondecode(fileread('livox_config.json'));
dataFolder = config.dataFolder;
ProcessFolder = config.processHFolder;
ProcessDFolder = config.processDFolder;
tmatrix = config.transformMatrix;
bounds = config.LidarBoundary;
%%
% --- SETTINGS ---
inputFile = 'M00511L1.mat';  % file with full L1 struct
currentFile = fullfile(ProcessFolder, inputFile);


%%
% --- LOAD MASTER STRUCT ---
load(currentFile);
if ~isstruct(L1) || ~isfield(L1, 'Dates')
    error('L1 must be a struct with a "Dates" field');
end
%%
% --- EXTRACT UNIQUE DAYS ---
allDates = [L1all.Dates];  % datetime vector
allDays = dateshift(allDates, 'start', 'day');
uniqueDays = unique(allDays);

fprintf('Found %d unique days in input.\n', numel(uniqueDays));


%% remove non-unique files

roundedDnums = round(datenum([L1all.Dates]) / (0.5/24)) * (0.5/24);
[uniqueDnums, uniqueIdx] = unique(roundedDnums, 'stable');  % keep first occurrence
% Keep only unique entries
% L1all = L1all(uniqueIdx);


%%
% --- SPLIT AND SAVE ---
for i = 1:numel(uniqueDays)
    thisDay = uniqueDays(i);
    idx = allDays == thisDay;
    L1_0 = L1all(idx);

    roundedDnums = round([L1_0.Datenum] / (0.5/24)) * (0.5/24);
    [uniqueDnums, uniqueIdx] = unique(roundedDnums, 'stable');  % keep first occurrence
    % Keep only unique entries
    L1_day = L1_0(uniqueIdx);

    outName = ['L1_' datestr(thisDay, 'yyyymmdd') '.mat'];
    outPath = fullfile(ProcessDFolder, outName);
    save(outPath, 'L1_day');
    fprintf('Saved %d entries to %s\n', numel(L1_day), outName);
end

fprintf('Done splitting.\n');
